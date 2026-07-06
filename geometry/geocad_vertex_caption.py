"""Vertex-based captioning of *simple* local CAD parts (GeoCAD, Zhang et al. 2025).

GeoCAD ("Local Geometry-Controllable CAD Generation with Large Language Models",
NeurIPS 2025) annotates ~221k local loops with *geometric instructions* using a
**complementary captioning strategy** (paper Sec. 3.1, Fig. 2): complex parts are
rendered and captioned by a VLLM (learned / external -- out of scope here), while
**simple parts are captioned deterministically from their vertex coordinates**. This
module implements that deterministic vertex-based branch.

The categories of simple parts are enumerated verbatim in appendix C:

    acute triangle, right triangle, obtuse triangle, isosceles triangle,
    isosceles right triangle, quadrilateral, trapezoid, isosceles trapezoid,
    kite (two pairs of adjacent equal sides), parallelogram, rectangle, rhombus,
    square, circle, semicircle, quarter-circle, three-quarter circle,
    major-arc loop (arc longer than a semicircle), minor-arc loop (arc shorter).

Paper Sec. 3.1: "given a quadrilateral, we can calculate its side lengths and
inter-side angles based on its vertex coordinates. If it has four lines of equal
length, it is a rhombus; if it includes right angles, it is further categorized as a
square." Sec. 3.1 / Fig. 8 also fold *key dimensional parameters* into the caption
(radius of a circle, side length of a square, length & width of a rectangle).

Everything here is closed-form Euclidean geometry -- deterministic, no wall clock,
no learned model. Points are ``(x, y)`` numeric pairs; the caller supplies the
ordered polygon vertices (for polyline loops) or the arc sweep (for arc loops).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

# Tolerances. Lengths are compared with a relative tolerance; angles (in degrees)
# with an absolute tolerance. DeepCAD coordinates are quantised to an integer grid,
# so a small slack absorbs augmentation round-off (rotation/scaling).
_LEN_RTOL = 1e-6
_ANG_ATOL_DEG = 1.0


def _dist(p: tuple[float, float], q: tuple[float, float]) -> float:
    return math.hypot(p[0] - q[0], p[1] - q[1])


def _len_eq(a: float, b: float, rtol: float = _LEN_RTOL) -> bool:
    return abs(a - b) <= rtol * max(1.0, abs(a), abs(b))


def side_lengths(vertices: list[tuple[float, float]]) -> list[float]:
    """Consecutive edge lengths of a closed polygon (edge i = v_i -> v_{i+1})."""
    n = len(vertices)
    if n < 3:
        raise ValueError("a polygon needs at least 3 vertices")
    return [_dist(vertices[i], vertices[(i + 1) % n]) for i in range(n)]


def interior_angle_deg(a: tuple[float, float], b: tuple[float, float],
                       c: tuple[float, float]) -> float:
    """Angle at vertex ``b`` formed by rays b->a and b->c, in degrees [0, 180]."""
    ux, uy = a[0] - b[0], a[1] - b[1]
    vx, vy = c[0] - b[0], c[1] - b[1]
    nu = math.hypot(ux, uy)
    nv = math.hypot(vx, vy)
    if nu == 0.0 or nv == 0.0:
        raise ValueError("degenerate (repeated) vertex")
    cos = max(-1.0, min(1.0, (ux * vx + uy * vy) / (nu * nv)))
    return math.degrees(math.acos(cos))


def interior_angles_deg(vertices: list[tuple[float, float]]) -> list[float]:
    """Interior angle at each vertex of a closed polygon, in degrees."""
    n = len(vertices)
    return [
        interior_angle_deg(vertices[(i - 1) % n], vertices[i], vertices[(i + 1) % n])
        for i in range(n)
    ]


def _is_right(angle_deg: float) -> bool:
    return abs(angle_deg - 90.0) <= _ANG_ATOL_DEG


def _parallel(p0: tuple[float, float], p1: tuple[float, float],
              q0: tuple[float, float], q1: tuple[float, float]) -> bool:
    """Whether segment p0->p1 is parallel to q0->q1 (2D cross product ~ 0)."""
    ux, uy = p1[0] - p0[0], p1[1] - p0[1]
    vx, vy = q1[0] - q0[0], q1[1] - q0[1]
    cross = ux * vy - uy * vx
    scale = max(1.0, math.hypot(ux, uy) * math.hypot(vx, vy))
    return abs(cross) <= 1e-6 * scale


# --- triangle -------------------------------------------------------------
def caption_triangle(vertices: list[tuple[float, float]]) -> str:
    """Caption a 3-vertex loop per appendix C (acute/right/obtuse x isosceles)."""
    if len(vertices) != 3:
        raise ValueError("a triangle needs exactly 3 vertices")
    s = sorted(side_lengths(vertices))  # s[0] <= s[1] <= s[2]
    a, b, c = s
    # Largest angle is opposite the longest side; sign of a^2+b^2-c^2 classifies it.
    disc = a * a + b * b - c * c
    tol = _LEN_RTOL * max(1.0, c * c)
    if abs(disc) <= max(tol, 1e-9 * c * c):
        angle_kind = "right"
    elif disc < 0:
        angle_kind = "obtuse"
    else:
        angle_kind = "acute"

    n_equal_pairs = sum(
        1 for (x, y) in ((a, b), (b, c), (a, c)) if _len_eq(x, y)
    )
    equilateral = _len_eq(a, b) and _len_eq(b, c)
    isosceles = n_equal_pairs >= 1

    if equilateral:
        return "an equilateral triangle"
    if angle_kind == "right" and isosceles:
        return "an isosceles right triangle"
    if angle_kind == "right":
        return "a right triangle"
    if isosceles:
        return "an isosceles triangle"
    if angle_kind == "obtuse":
        return "an obtuse triangle"
    return "an acute triangle"


# --- quadrilateral --------------------------------------------------------
def caption_quadrilateral(vertices: list[tuple[float, float]]) -> str:
    """Caption a 4-vertex loop per appendix C (square/rectangle/rhombus/... )."""
    if len(vertices) != 4:
        raise ValueError("a quadrilateral needs exactly 4 vertices")
    a, b, c, d = side_lengths(vertices)  # AB, BC, CD, DA
    angles = interior_angles_deg(vertices)
    all_right = all(_is_right(t) for t in angles)
    all_sides_eq = _len_eq(a, b) and _len_eq(b, c) and _len_eq(c, d)
    opp_sides_eq = _len_eq(a, c) and _len_eq(b, d)

    v0, v1, v2, v3 = vertices
    ab_par_cd = _parallel(v0, v1, v3, v2)  # AB || DC
    bc_par_ad = _parallel(v1, v2, v0, v3)  # BC || AD

    if all_sides_eq and all_right:
        return "a square"
    if opp_sides_eq and all_right:
        return "a rectangle"
    if all_sides_eq:
        return "a rhombus"
    if opp_sides_eq and ab_par_cd and bc_par_ad:
        return "a parallelogram"

    # Trapezoid: exactly one pair of parallel sides.
    n_parallel = int(ab_par_cd) + int(bc_par_ad)
    if n_parallel == 1:
        # Isosceles trapezoid: the two non-parallel legs are equal.
        if ab_par_cd and _len_eq(b, d):
            return "an isosceles trapezoid"
        if bc_par_ad and _len_eq(a, c):
            return "an isosceles trapezoid"
        return "a trapezoid"

    # Kite: two distinct pairs of adjacent equal sides.
    if (_len_eq(a, b) and _len_eq(c, d) and not _len_eq(a, d)) or (
        _len_eq(b, c) and _len_eq(d, a) and not _len_eq(a, b)
    ):
        return "a kite"

    return "a quadrilateral"


# --- arc / circle loops ---------------------------------------------------
CIRCLE = "circle"
SEMICIRCLE = "semicircle"
QUARTER = "quarter-circle"
THREE_QUARTER = "three-quarter circle"
MAJOR_ARC = "major-arc loop"
MINOR_ARC = "minor-arc loop"


def caption_arc_loop(sweep_deg: float) -> str:
    """Caption an arc-bounded loop from its arc sweep angle (degrees).

    Per appendix C: a full 360 sweep is a circle, 90 a quarter-circle, 180 a
    semicircle, 270 a three-quarter circle; otherwise an arc longer than a
    semicircle is a *major-arc loop* and one shorter a *minor-arc loop*.
    """
    if sweep_deg <= 0 or sweep_deg > 360 + _ANG_ATOL_DEG:
        raise ValueError(f"arc sweep out of range: {sweep_deg}")
    if abs(sweep_deg - 360.0) <= _ANG_ATOL_DEG:
        return "a " + CIRCLE
    if abs(sweep_deg - 90.0) <= _ANG_ATOL_DEG:
        return "a " + QUARTER
    if abs(sweep_deg - 180.0) <= _ANG_ATOL_DEG:
        return "a " + SEMICIRCLE
    if abs(sweep_deg - 270.0) <= _ANG_ATOL_DEG:
        return "a " + THREE_QUARTER
    if sweep_deg > 180.0:
        return "a " + MAJOR_ARC
    return "a " + MINOR_ARC


def caption_polygon(vertices: list[tuple[float, float]]) -> str:
    """Dispatch a straight-sided simple part (3 -> triangle, 4 -> quadrilateral)."""
    n = len(vertices)
    if n == 3:
        return caption_triangle(vertices)
    if n == 4:
        return caption_quadrilateral(vertices)
    raise ValueError(f"vertex-based captioning handles 3- or 4-gons, got {n}")


# --- key dimensional parameters (Fig. 8) ----------------------------------
@dataclass(frozen=True)
class Dimensions:
    """Key dimensional parameters folded into a caption (paper Fig. 8)."""

    kind: str
    values: dict[str, float]


def circle_dimensions(radius: float) -> Dimensions:
    return Dimensions("circle", {"radius": float(radius)})


def square_dimensions(vertices: list[tuple[float, float]]) -> Dimensions:
    if len(vertices) != 4:
        raise ValueError("square needs 4 vertices")
    return Dimensions("square", {"side": side_lengths(vertices)[0]})


def rectangle_dimensions(vertices: list[tuple[float, float]]) -> Dimensions:
    if len(vertices) != 4:
        raise ValueError("rectangle needs 4 vertices")
    a, b, _, _ = side_lengths(vertices)
    length, width = (a, b) if a >= b else (b, a)
    return Dimensions("rectangle", {"length": length, "width": width})


def caption_with_dimensions(base_caption: str, dims: Dimensions) -> str:
    """Append rounded dimensional parameters to a caption (paper Fig. 8)."""
    parts = ", ".join(
        f"{k} {_fmt(v)}" for k, v in sorted(dims.values.items())
    )
    return f"{base_caption} with {parts}"


def _fmt(v: float) -> str:
    r = round(v, 3)
    return str(int(r)) if r == int(r) else str(r)
