"""Deterministic geometric relations for mrCAD curves (design.py predicate suite).

The mrCAD reference (`mrcad/design.py`) attaches a family of exact,
control-point-based geometric predicates and distance queries to its `Line`,
`Arc` and `Circle` classes that the harness's :mod:`editing.mrcad_schema`
(which stores only ``kind`` + control points) never carried over:

* ``parallel`` / ``perpendicular`` -- slope comparison with vertical handling;
* ``parallel_distance`` -- perpendicular gap between two parallel segments,
  gated on the segments actually overlapping when projected onto each other
  (returns ``None`` when they do not overlap, so distant collinear segments are
  not treated as "the same wall");
* ``meeting_ends`` -- whether two curves share an endpoint (arc endpoints are the
  first and last control point, not the middle);
* ``concentric`` -- arcs/circles sharing a centre;
* ``point_to_curve_distance`` -- the exact analytic distance from a point to a
  line segment / circle / arc.  The arc case tests whether the target's bearing
  from the centre falls inside the arc's sweep (the paper's ``sxxe`` / ``exxs``
  angular-ordering trick); inside, the distance is ``|r_target - r|``, outside it
  is the distance to the nearer endpoint.

This exact point-to-curve distance is what the paper's ``Design.design_distance``
is built on (see :mod:`bench.mrcad2_design_distance`) and differs from the
sampled point-to-point chamfer in :mod:`bench.mrcad_metrics`.

Pure standard library, deterministic. Curves are :class:`editing.mrcad_schema.Curve`.
"""
from __future__ import annotations

import math
from typing import Optional, Tuple

from harnesscad.domain.editing.mrcad_schema import Curve

Point = Tuple[float, float]

#: Relative tolerance matching the paper's ``rel_tol=1e-3`` predicate checks.
REL_TOL = 1e-3
#: Absolute tolerance for degenerate/near-zero denominators.
ABS_TOL = 1e-10


def _isclose(a: float, b: float) -> bool:
    return math.isclose(a, b, rel_tol=REL_TOL, abs_tol=ABS_TOL)


def _pts_close(p: Point, q: Point) -> bool:
    return _isclose(p[0], q[0]) and _isclose(p[1], q[1])


# ---------------------------------------------------------------------------
# Circle / arc geometry from three (or two) control points.
# ---------------------------------------------------------------------------
def _circumcircle(p0: Point, p1: Point, p2: Point) -> Optional[Tuple[Point, float]]:
    """Return ``(center, radius)`` of the circle through 3 points, or ``None``
    if they are (near) collinear."""
    ax, ay = p0
    bx, by = p1
    cx, cy = p2
    d = 2.0 * (ax * (by - cy) + bx * (cy - ay) + cx * (ay - by))
    if abs(d) < ABS_TOL:
        return None
    a2 = ax * ax + ay * ay
    b2 = bx * bx + by * by
    c2 = cx * cx + cy * cy
    ux = (a2 * (by - cy) + b2 * (cy - ay) + c2 * (ay - by)) / d
    uy = (a2 * (cx - bx) + b2 * (ax - cx) + c2 * (bx - ax)) / d
    center = (ux, uy)
    return center, math.dist(center, p0)


def curve_center(curve: Curve) -> Optional[Point]:
    """Centre of a circle (midpoint of diameter) or arc (circumcentre)."""
    if curve.kind == "circle":
        (x1, y1), (x2, y2) = curve.points
        return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)
    if curve.kind == "arc":
        res = _circumcircle(*curve.points)
        return None if res is None else res[0]
    return None


def curve_radius(curve: Curve) -> Optional[float]:
    """Radius of a circle or arc (``None`` for a line / degenerate arc)."""
    if curve.kind == "circle":
        a, b = curve.points
        return math.dist(a, b) / 2.0
    if curve.kind == "arc":
        res = _circumcircle(*curve.points)
        return None if res is None else res[1]
    return None


def endpoints(curve: Curve) -> Tuple[Point, ...]:
    """The connectable endpoints: line ``(p0, p1)``, arc ``(p0, p2)``, circle ``()``."""
    if curve.kind == "line":
        return curve.points
    if curve.kind == "arc":
        return (curve.points[0], curve.points[2])
    return ()


# ---------------------------------------------------------------------------
# Line relations.
# ---------------------------------------------------------------------------
def _require_line(curve: Curve) -> None:
    if curve.kind != "line":
        raise ValueError(f"expected a line, got {curve.kind!r}")


def parallel(line_a: Curve, line_b: Curve) -> bool:
    """True if two line segments are parallel (vertical pair handled)."""
    _require_line(line_a)
    _require_line(line_b)
    (x1, y1), (x2, y2) = line_a.points
    (x3, y3), (x4, y4) = line_b.points
    dx1, dx2 = x2 - x1, x4 - x3
    if dx1 == 0 or dx2 == 0:
        return dx1 == dx2
    return _isclose((y2 - y1) / dx1, (y4 - y3) / dx2)


def perpendicular(line_a: Curve, line_b: Curve) -> bool:
    """True if two line segments are perpendicular (vertical/horizontal handled)."""
    _require_line(line_a)
    _require_line(line_b)
    (x1, y1), (x2, y2) = line_a.points
    (x3, y3), (x4, y4) = line_b.points
    if x2 - x1 == 0:
        return y4 - y3 == 0
    if x4 - x3 == 0:
        return y2 - y1 == 0
    slope1 = (y2 - y1) / (x2 - x1)
    slope2 = (y4 - y3) / (x4 - x3)
    return _isclose(slope1 * slope2, -1.0)


def parallel_distance(line_a: Curve, line_b: Curve) -> Optional[float]:
    """Perpendicular distance between two parallel, overlapping segments.

    Returns ``None`` when the lines are not parallel, or when neither endpoint of
    ``line_b`` projects onto the span of ``line_a`` (i.e. the segments do not
    overlap along their shared direction).
    """
    _require_line(line_a)
    _require_line(line_b)
    if not parallel(line_a, line_b):
        return None
    (x1, y1), (x2, y2) = line_a.points
    (x3, y3), (x4, y4) = line_b.points
    dx, dy = x2 - x1, y2 - y1
    denom = dx * dx + dy * dy
    if denom < ABS_TOL:
        return None
    t1 = (dx * (x3 - x1) + dy * (y3 - y1)) / denom
    t2 = (dx * (x4 - x1) + dy * (y4 - y1)) / denom
    if not (0.0 <= t1 <= 1.0) and not (0.0 <= t2 <= 1.0):
        return None
    return abs((y2 - y1) * x3 - (x2 - x1) * y3 + x2 * y1 - y2 * x1) / math.sqrt(denom)


# ---------------------------------------------------------------------------
# Cross-curve relations.
# ---------------------------------------------------------------------------
def meeting_ends(curve_a: Curve, curve_b: Curve) -> bool:
    """True if the two curves share an endpoint (within tolerance)."""
    for p in endpoints(curve_a):
        for q in endpoints(curve_b):
            if _pts_close(p, q):
                return True
    return False


def concentric(curve_a: Curve, curve_b: Curve) -> bool:
    """True if two arcs/circles share a centre (lines are never concentric)."""
    c1 = curve_center(curve_a)
    c2 = curve_center(curve_b)
    if c1 is None or c2 is None:
        return False
    return _pts_close(c1, c2)


# ---------------------------------------------------------------------------
# Exact point-to-curve distance.
# ---------------------------------------------------------------------------
def _point_to_segment(p0: Point, p1: Point, point: Point) -> float:
    x1, y1 = p0
    x2, y2 = p1
    x0, y0 = point
    dx, dy = x2 - x1, y2 - y1
    denom = dx * dx + dy * dy
    if denom < ABS_TOL:
        return math.dist(p0, point)
    t = ((x0 - x1) * dx + (y0 - y1) * dy) / denom
    if t <= 0.0:
        return math.dist(p0, point)
    if t >= 1.0:
        return math.dist(p1, point)
    foot = (x1 + t * dx, y1 + t * dy)
    return math.dist(foot, point)


def _arc_contains_bearing(
    center: Point, start: Point, mid: Point, end: Point, target: Point
) -> bool:
    """Whether the bearing centre->target lies within the arc's sweep.

    Uses the paper's angular-ordering trick: sort the four bearings (and their
    ``+2pi`` copies), collapse mid/target to a wildcard, and look for the arc
    endpoints straddling the target (``s..e`` with the target between).
    """
    def ang(p: Point) -> float:
        return math.atan2(p[1] - center[1], p[0] - center[0])

    labelled = [("s", ang(start)), ("m", ang(mid)), ("e", ang(end)), ("t", ang(target))]
    doubled = labelled + [(n, a + 2 * math.pi) for n, a in labelled]
    doubled.sort(key=lambda x: x[1])
    order = "".join(n for n, _ in doubled).replace("m", "x").replace("t", "x")
    return "sxxe" in order or "exxs" in order


def point_to_curve_distance(curve: Curve, point: Point) -> float:
    """Exact analytic distance from ``point`` to ``curve``.

    * line   -- distance to the clamped segment;
    * circle -- ``|dist(point, center) - radius|``;
    * arc    -- ``|dist(point, center) - radius|`` when the target's bearing lies
      inside the sweep, else distance to the nearer endpoint (falls back to a
      chord segment when the three control points are collinear).
    """
    pt = (float(point[0]), float(point[1]))
    if curve.kind == "line":
        return _point_to_segment(curve.points[0], curve.points[1], pt)
    if curve.kind == "circle":
        center = curve_center(curve)
        radius = curve_radius(curve)
        return abs(math.dist(pt, center) - radius)
    # arc
    start, mid, end = curve.points
    res = _circumcircle(start, mid, end)
    if res is None:
        return _point_to_segment(start, end, pt)
    center, radius = res
    if _arc_contains_bearing(center, start, mid, end, pt):
        return abs(math.dist(pt, center) - radius)
    return min(math.dist(start, pt), math.dist(end, pt))
