"""Deterministic geometric feasibility validation for 2D wheel-rim spoke designs.

This module implements the deterministic (non-learned) geometric feasibility
validation algorithm described in Sections 3.3 and 3.4 of:

    "Generative AI and CAD Automation for Diverse and Novel Mechanical
     Component Designs Under Data Constraints"

The paper generates candidate wheel-rim spoke contours with a learned diffusion
model and then filters them through a set of purely geometric, deterministic
feasibility checks. This module reproduces ONLY those deterministic checks; no
learned component (diffusion / LoRA) is included.

A design is described purely by its 2D contours: closed polygons given as lists
of (x, y) tuples, already expressed in CAD coordinates centered at the rim
center. Polygons may be ordered clockwise or counter-clockwise.

Equations referenced (paper numbering):

  Eq 9  -- polygon area via the shoelace formula:
             A = |0.5 * sum_j (x_j * y_{j+1} - y_j * x_{j+1})|
  Eq 10 -- centroid x:
             x_c = (1 / (6A)) * sum_j (x_j + x_{j+1}) (x_j y_{j+1} - y_j x_{j+1})
  Eq 11 -- centroid y:
             y_c = (1 / (6A)) * sum_j (y_j + y_{j+1}) (x_j y_{j+1} - y_j x_{j+1})
           (A here is the SIGNED area.)
  Eq 12 -- polar radius:  r = sqrt(x_c^2 + y_c^2)
  Eq 13 -- polar angle:   theta = atan2(y_c, x_c)
  Eq 7  -- spoke inner-bound feasibility (must clear bolt circle + reserve):
             dist > pcd/2 + bolt_radius + reserved_eps
  Eq 8  -- spoke outer-bound feasibility (must stay inside the drop well):
             dist < D/2 - H - t_c
  Eq 14 -- rotational symmetry angular step:
             |theta_{i+1} - theta_i - 2*pi/N| < tol_angle
  Eq 15 -- rotational symmetry contour self-similarity:
             1 - Jaccard(rotate(contour_i, 2*pi/N), contour_{i+1}) < tol_jaccard

Section 3.3(a) additionally requires that a filled design be bounded by exactly
one outer contour.  A rim without spoke features is considered naturally
balanced (Section 3.4).

The module is deterministic and stdlib-only.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

Point = Tuple[float, float]
Polygon = Sequence[Point]


# ---------------------------------------------------------------------------
# Core geometric primitives
# ---------------------------------------------------------------------------

def signed_area(poly: Polygon) -> float:
    """Signed polygon area via the shoelace formula (basis for paper Eq 9).

    Positive for counter-clockwise vertex order, negative for clockwise.
    """
    n = len(poly)
    if n < 3:
        return 0.0
    s = 0.0
    for j in range(n):
        x_j, y_j = poly[j]
        x_k, y_k = poly[(j + 1) % n]
        s += x_j * y_k - y_j * x_k
    return 0.5 * s


def polygon_area(poly: Polygon) -> float:
    """Absolute polygon area (paper Eq 9): |0.5 * sum(x_j y_{j+1} - y_j x_{j+1})|."""
    return abs(signed_area(poly))


def polygon_centroid(poly: Polygon) -> Point:
    """Polygon centroid (paper Eqs 10, 11), using the SIGNED area.

    Falls back to the arithmetic mean of vertices when the area is ~0 (a
    degenerate / collinear polygon).
    """
    a = signed_area(poly)
    n = len(poly)
    if n == 0:
        return (0.0, 0.0)
    if abs(a) < 1e-12:
        sx = sum(p[0] for p in poly)
        sy = sum(p[1] for p in poly)
        return (sx / n, sy / n)
    cx = 0.0
    cy = 0.0
    for j in range(n):
        x_j, y_j = poly[j]
        x_k, y_k = poly[(j + 1) % n]
        cross = x_j * y_k - y_j * x_k
        cx += (x_j + x_k) * cross
        cy += (y_j + y_k) * cross
    factor = 1.0 / (6.0 * a)
    return (cx * factor, cy * factor)


def to_polar(xc: float, yc: float) -> Tuple[float, float]:
    """Convert a centroid to polar coordinates (paper Eqs 12, 13).

    Returns (r, theta) with r = sqrt(xc^2 + yc^2) and theta = atan2(yc, xc).
    """
    r = math.hypot(xc, yc)
    theta = math.atan2(yc, xc)
    return (r, theta)


def rotate_polygon(poly: Polygon, angle_rad: float,
                   about: Point = (0.0, 0.0)) -> List[Point]:
    """Rotate a polygon by angle_rad (radians) about the given pivot point."""
    ax, ay = about
    ca = math.cos(angle_rad)
    sa = math.sin(angle_rad)
    out: List[Point] = []
    for x, y in poly:
        dx = x - ax
        dy = y - ay
        rx = dx * ca - dy * sa
        ry = dx * sa + dy * ca
        out.append((rx + ax, ry + ay))
    return out


def point_in_polygon(x: float, y: float, poly: Polygon) -> bool:
    """Even-odd ray-casting point-in-polygon test.

    Returns True if (x, y) lies inside the polygon.
    """
    n = len(poly)
    if n < 3:
        return False
    inside = False
    j = n - 1
    for i in range(n):
        xi, yi = poly[i]
        xj, yj = poly[j]
        # Does the horizontal ray from (x, y) cross edge (j -> i)?
        if (yi > y) != (yj > y):
            x_cross = (xj - xi) * (y - yi) / (yj - yi) + xi
            if x < x_cross:
                inside = not inside
        j = i
    return inside


def _bbox_of_points(points: Sequence[Point]) -> Tuple[float, float, float, float]:
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    return (min(xs), min(ys), max(xs), max(ys))


def jaccard_similarity(poly_a: Polygon, poly_b: Polygon, grid: int = 64) -> float:
    """Jaccard similarity of two polygons via rasterization (paper Eq 15).

    Both polygons are rasterized onto a shared square grid covering their
    combined bounding box. A cell is "filled" for a polygon if its center is
    inside that polygon (even-odd test). Returns |intersection| / |union| of
    the filled cell sets, or 0.0 if the union is empty.
    """
    if len(poly_a) < 3 or len(poly_b) < 3:
        return 0.0
    combined = list(poly_a) + list(poly_b)
    min_x, min_y, max_x, max_y = _bbox_of_points(combined)
    w = max_x - min_x
    h = max_y - min_y
    if w <= 0.0 and h <= 0.0:
        return 0.0
    # Guard against degenerate (zero-width or zero-height) bboxes.
    if w <= 0.0:
        w = 1.0
    if h <= 0.0:
        h = 1.0
    inter = 0
    union = 0
    for gy in range(grid):
        cy = min_y + (gy + 0.5) * h / grid
        for gx in range(grid):
            cx = min_x + (gx + 0.5) * w / grid
            in_a = point_in_polygon(cx, cy, poly_a)
            in_b = point_in_polygon(cx, cy, poly_b)
            if in_a or in_b:
                union += 1
                if in_a and in_b:
                    inter += 1
    if union == 0:
        return 0.0
    return inter / union


# ---------------------------------------------------------------------------
# Validation checks
# ---------------------------------------------------------------------------

def single_outer_contour(contours: Sequence[Polygon]) -> bool:
    """Paper Section 3.3(a): exactly one outer contour bounds everything.

    Approximated as: exactly one contour whose axis-aligned bounding box
    contains every point of all other contours. Returns True iff exactly one
    such enclosing contour exists.
    """
    n = len(contours)
    if n == 0:
        return False
    if n == 1:
        return True

    enclosers = 0
    for i, ci in enumerate(contours):
        if len(ci) < 3:
            continue
        min_x, min_y, max_x, max_y = _bbox_of_points(ci)
        encloses_all = True
        for k, ck in enumerate(contours):
            if k == i:
                continue
            for (x, y) in ck:
                if not (min_x <= x <= max_x and min_y <= y <= max_y):
                    encloses_all = False
                    break
            if not encloses_all:
                break
        if encloses_all:
            enclosers += 1
    return enclosers == 1


def spoke_position_ok(contour_points: Sequence[Point], pcd: float,
                      bolt_radius: float, rim_diameter_D: float,
                      well_depth_H: float, rim_thickness_tc: float,
                      reserved_eps: Optional[float] = None) -> bool:
    """Paper Eqs 7 & 8: every spoke point must lie in the feasible annulus.

    For every point (x, y) with dist = sqrt(x^2 + y^2):
        Eq 7 (inner):  dist > pcd/2 + bolt_radius + reserved_eps
        Eq 8 (outer):  dist < D/2 - H - t_c

    reserved_eps defaults to 2 * bolt_radius. Returns True only if all points
    satisfy both bounds.
    """
    if reserved_eps is None:
        reserved_eps = 2.0 * bolt_radius
    inner = pcd / 2.0 + bolt_radius + reserved_eps
    outer = rim_diameter_D / 2.0 - well_depth_H - rim_thickness_tc
    if not contour_points:
        return False
    for (x, y) in contour_points:
        dist = math.hypot(x, y)
        if not (dist > inner and dist < outer):
            return False
    return True


def _angular_diff(a: float, b: float) -> float:
    """Smallest signed difference a - b wrapped to (-pi, pi]."""
    d = a - b
    while d <= -math.pi:
        d += 2.0 * math.pi
    while d > math.pi:
        d -= 2.0 * math.pi
    return d


def rotational_symmetry_ok(spoke_contours: Sequence[Polygon],
                           tol_angle: float = 0.05,
                           tol_jaccard: float = 0.15,
                           radius_tol: float = 5.0,
                           area_tol_frac: float = 0.15) -> bool:
    """Paper Eqs 14 & 15: rotational-symmetry feasibility of spoke contours.

    Steps:
      1. Compute (area, centroid, r, theta) per spoke contour.
      2. Group contours whose centroid radius r and area A are similar
         (within radius_tol absolute and area_tol_frac relative).
      3. In the largest group of size N >= 2: sort by theta; the expected
         angular step is 2*pi/N. Each adjacent pair (wrap-around aware) must
         satisfy Eq 14, and rotating contour i by 2*pi/N about the origin must
         match contour i+1 within Eq 15 (1 - Jaccard < tol_jaccard).

    Returns True if a symmetric group of N >= 2 passes, or if there are fewer
    than two spoke contours (a plain rim is naturally balanced, Section 3.4).
    """
    contours = [c for c in spoke_contours if len(c) >= 3]
    if len(contours) < 2:
        # No spoke features -> naturally balanced rim.
        return True

    feats = []
    for c in contours:
        area = polygon_area(c)
        cx, cy = polygon_centroid(c)
        r, theta = to_polar(cx, cy)
        feats.append({"poly": c, "area": area, "r": r, "theta": theta})

    # Greedy grouping by similar radius and area.
    used = [False] * len(feats)
    best_group: List[dict] = []
    for i in range(len(feats)):
        if used[i]:
            continue
        group = [feats[i]]
        for k in range(len(feats)):
            if k == i or used[k]:
                continue
            fi = feats[i]
            fk = feats[k]
            if abs(fi["r"] - fk["r"]) > radius_tol:
                continue
            ref_area = max(abs(fi["area"]), 1e-9)
            if abs(fi["area"] - fk["area"]) / ref_area > area_tol_frac:
                continue
            group.append(fk)
        if len(group) > len(best_group):
            best_group = group

    n = len(best_group)
    if n < 2:
        return False

    group = sorted(best_group, key=lambda f: f["theta"])
    expected_step = 2.0 * math.pi / n

    for idx in range(n):
        cur = group[idx]
        nxt = group[(idx + 1) % n]
        # Angular step (Eq 14), wrap-around aware.
        step = _angular_diff(nxt["theta"], cur["theta"])
        if step < 0:
            step += 2.0 * math.pi
        if abs(_angular_diff(step, expected_step)) >= tol_angle:
            return False
        # Contour self-similarity under rotation (Eq 15).
        rotated = rotate_polygon(cur["poly"], expected_step, about=(0.0, 0.0))
        jac = jaccard_similarity(rotated, nxt["poly"])
        if (1.0 - jac) >= tol_jaccard:
            return False
    return True


# ---------------------------------------------------------------------------
# Top-level report
# ---------------------------------------------------------------------------

@dataclass
class ValidationReport:
    """Structured result of validate_design.

    Attributes:
        single_contour: Section 3.3(a) single outer contour check.
        position_ok:    Eqs 7 & 8 spoke position feasibility.
        symmetry_ok:    Eqs 14 & 15 rotational symmetry feasibility.
        feasible:       Conjunction of the three checks above.
        reasons:        Human-readable notes about any failing check.
    """
    single_contour: bool
    position_ok: bool
    symmetry_ok: bool
    feasible: bool
    reasons: List[str] = field(default_factory=list)


def validate_design(contours: Sequence[Polygon],
                    spoke_contours: Sequence[Polygon],
                    spec: Dict[str, float]) -> ValidationReport:
    """Run the full deterministic feasibility validation (Sections 3.3, 3.4).

    Args:
        contours: All closed contours of the design (outer rim + spokes/holes).
        spoke_contours: The subset of contours that represent spoke features.
        spec: Dict with keys pcd, bolt_radius, rim_diameter_D, well_depth_H,
              and optional rim_thickness_tc (defaults to 4.0 per the paper) and
              reserved_eps.

    Returns:
        ValidationReport with the three boolean checks, their conjunction, and
        a list of reasons for any failures.
    """
    pcd = spec["pcd"]
    bolt_radius = spec["bolt_radius"]
    rim_diameter_D = spec["rim_diameter_D"]
    well_depth_H = spec["well_depth_H"]
    rim_thickness_tc = spec.get("rim_thickness_tc", 4.0)
    reserved_eps = spec.get("reserved_eps", None)

    reasons: List[str] = []

    single_c = single_outer_contour(contours)
    if not single_c:
        reasons.append(
            "single_outer_contour failed: design is not bounded by exactly "
            "one enclosing contour (Section 3.3a)."
        )

    position_ok = True
    for i, sc in enumerate(spoke_contours):
        if not spoke_position_ok(sc, pcd, bolt_radius, rim_diameter_D,
                                 well_depth_H, rim_thickness_tc, reserved_eps):
            position_ok = False
            reasons.append(
                "spoke_position_ok failed for spoke contour %d: violates the "
                "feasible annulus (Eqs 7, 8)." % i
            )
    if not spoke_contours:
        # No spokes: position constraint is vacuously satisfied.
        position_ok = True

    symmetry_ok = rotational_symmetry_ok(spoke_contours)
    if not symmetry_ok:
        reasons.append(
            "rotational_symmetry_ok failed: no balanced N-fold symmetric spoke "
            "group found (Eqs 14, 15)."
        )

    feasible = single_c and position_ok and symmetry_ok
    return ValidationReport(
        single_contour=single_c,
        position_ok=position_ok,
        symmetry_ok=symmetry_ok,
        feasible=feasible,
        reasons=reasons,
    )
