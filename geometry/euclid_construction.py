"""Ruler-and-compass construction engine for CAD profile generation.

Deterministic, closed-form geometric constructions, implementing the domain
specific language of construction steps described in Li et al., "Draw It Like
Euclid: Teaching Transformer Models to Generate CAD Profiles Using Ruler and
Compass Construction Steps" (Autodesk Research). See Tables 1 and 7 and
Appendix B.3.6 of the paper.

The learned transformer / reinforcement-learning parts of the paper are out of
scope here; this module supplies the fully deterministic geometric substrate:
the entity types (points, directed infinite lines, oriented circles, arcs and
bounded segments) and every atomic construction step, each of which has a
closed-form solution.

Conventions
-----------
* All geometry lives in the unit square centred on the origin, coordinates in
  ``[-0.5, 0.5]`` (matching the paper's normalisation), but nothing here
  enforces that range -- it is purely a modelling convention.
* Directed infinite lines are stored in Hessian normal form ``(phi, rho)``:
  ``phi`` is the direction angle of the line (angle with the x-axis) and
  ``rho`` is the signed distance from the origin measured along the line's
  *left* normal ``n = (-sin phi, cos phi)`` (``u`` rotated by +90 degrees).
  "Left-hand side" offsets therefore move ``rho`` in the ``+n`` direction.
* Oriented circles carry a ``ccw`` flag (counter-clockwise when True).

All functions are pure and free of wall-clock / RNG state.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Optional, Tuple

TWO_PI = 2.0 * math.pi

# Default coincidence tolerance: one quantisation bin in the unit square
# (1/127 model units), matching Appendix B.1 of the paper.
TOL = 1.0 / 127.0
# Angular tolerance: the paper treats vectors as parallel within 1 degree.
ANGLE_TOL = math.radians(1.0)


def _norm_angle(a: float) -> float:
    """Wrap an angle into ``[0, 2*pi)``."""
    a = math.fmod(a, TWO_PI)
    if a < 0.0:
        a += TWO_PI
    # Guard against fmod returning exactly TWO_PI due to rounding.
    if a >= TWO_PI:
        a -= TWO_PI
    return a


# ---------------------------------------------------------------------------
# Entity types
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class Point:
    x: float
    y: float

    def __add__(self, o: "Point") -> "Point":
        return Point(self.x + o.x, self.y + o.y)

    def __sub__(self, o: "Point") -> "Point":
        return Point(self.x - o.x, self.y - o.y)

    def scaled(self, s: float) -> "Point":
        return Point(self.x * s, self.y * s)

    def dist(self, o: "Point") -> float:
        return math.hypot(self.x - o.x, self.y - o.y)

    def almost_equals(self, o: "Point", tol: float = TOL) -> bool:
        return self.dist(o) <= tol


@dataclass(frozen=True)
class Line:
    """A directed infinite line in Hessian normal form ``(phi, rho)``."""

    phi: float
    rho: float

    # -- derived vectors ---------------------------------------------------
    def direction(self) -> Tuple[float, float]:
        return (math.cos(self.phi), math.sin(self.phi))

    def normal(self) -> Tuple[float, float]:
        """Left normal ``n = (-sin phi, cos phi)`` (direction rotated +90)."""
        return (-math.sin(self.phi), math.cos(self.phi))

    def base_point(self) -> Point:
        """Point on the line closest to the origin (``rho * n``)."""
        nx, ny = self.normal()
        return Point(self.rho * nx, self.rho * ny)

    def point_at(self, t: float) -> Point:
        b = self.base_point()
        ux, uy = self.direction()
        return Point(b.x + t * ux, b.y + t * uy)

    def signed_distance(self, p: Point) -> float:
        """Signed distance of ``p`` from the line (positive on the left)."""
        nx, ny = self.normal()
        return nx * p.x + ny * p.y - self.rho

    def contains(self, p: Point, tol: float = TOL) -> bool:
        return abs(self.signed_distance(p)) <= tol

    def is_parallel(self, o: "Line", tol: float = ANGLE_TOL) -> bool:
        d = _norm_angle(self.phi - o.phi)
        return d <= tol or abs(d - math.pi) <= tol or abs(d - TWO_PI) <= tol


@dataclass(frozen=True)
class Circle:
    center: Point
    radius: float
    ccw: bool = True

    def point_at_angle(self, a: float) -> Point:
        return Point(self.center.x + self.radius * math.cos(a),
                     self.center.y + self.radius * math.sin(a))


@dataclass(frozen=True)
class Arc:
    """An arc defined by ordered start, mid and end points (as in the DSL)."""

    start: Point
    mid: Point
    end: Point


@dataclass(frozen=True)
class Segment:
    start: Point
    end: Point

    def length(self) -> float:
        return self.start.dist(self.end)


# ---------------------------------------------------------------------------
# Classic ruler-and-compass primitives
# ---------------------------------------------------------------------------
def line_through_points(p1: Point, p2: Point) -> Line:
    """Directed infinite line running from ``p1`` towards ``p2``."""
    if p1.almost_equals(p2, tol=1e-12):
        raise ValueError("cannot build a line through two coincident points")
    phi = _norm_angle(math.atan2(p2.y - p1.y, p2.x - p1.x))
    nx, ny = (-math.sin(phi), math.cos(phi))
    rho = nx * p1.x + ny * p1.y
    return Line(phi, rho)


def circle_center_radius(center: Point, radius: float, ccw: bool = True) -> Circle:
    if radius <= 0.0:
        raise ValueError("radius must be positive")
    return Circle(center, radius, ccw)


def circle_through_point(center: Point, on: Point, ccw: bool = True) -> Circle:
    return circle_center_radius(center, center.dist(on), ccw)


def perpendicular_line(line: Line, through: Point) -> Line:
    """Line through ``through`` perpendicular to ``line``."""
    phi = _norm_angle(line.phi + math.pi / 2.0)
    nx, ny = (-math.sin(phi), math.cos(phi))
    return Line(phi, nx * through.x + ny * through.y)


def perpendicular_bisector(p1: Point, p2: Point) -> Line:
    """Perpendicular bisector of the segment ``p1 p2``."""
    mid = Point((p1.x + p2.x) / 2.0, (p1.y + p2.y) / 2.0)
    base = line_through_points(p1, p2)
    return perpendicular_line(base, mid)


def parallel_line(line: Line, through: Point) -> Line:
    """Line parallel to ``line`` passing through ``through`` (same direction)."""
    nx, ny = line.normal()
    return Line(line.phi, nx * through.x + ny * through.y)


def angle_bisector(vertex: Point, a: Point, b: Point) -> Line:
    """Interior angle bisector of the rays ``vertex->a`` and ``vertex->b``."""
    da = a - vertex
    db = b - vertex
    la = math.hypot(da.x, da.y)
    lb = math.hypot(db.x, db.y)
    if la <= 1e-12 or lb <= 1e-12:
        raise ValueError("degenerate angle: coincident points")
    ux = da.x / la + db.x / lb
    uy = da.y / la + db.y / lb
    if abs(ux) <= 1e-12 and abs(uy) <= 1e-12:
        # Rays are anti-parallel; bisector is perpendicular to them.
        base = line_through_points(vertex, a)
        return perpendicular_line(base, vertex)
    phi = _norm_angle(math.atan2(uy, ux))
    nx, ny = (-math.sin(phi), math.cos(phi))
    return Line(phi, nx * vertex.x + ny * vertex.y)


# ---------------------------------------------------------------------------
# Intersections
# ---------------------------------------------------------------------------
def line_line_intersection(l1: Line, l2: Line) -> Optional[Point]:
    """Intersection point of two infinite lines, or ``None`` if parallel."""
    n1x, n1y = l1.normal()
    n2x, n2y = l2.normal()
    det = n1x * n2y - n1y * n2x
    if abs(det) <= 1e-12:
        return None
    x = (l1.rho * n2y - l2.rho * n1y) / det
    y = (n1x * l2.rho - n2x * l1.rho) / det
    return Point(x, y)


def line_circle_intersection(line: Line, circle: Circle) -> List[Point]:
    """0, 1 or 2 intersection points of a line and a circle (ordered)."""
    d = line.signed_distance(circle.center)
    r = circle.radius
    if abs(d) > r + 1e-12:
        return []
    nx, ny = line.normal()
    foot = Point(circle.center.x - d * nx, circle.center.y - d * ny)
    half_sq = r * r - d * d
    if half_sq <= 1e-15:
        return [foot]
    half = math.sqrt(half_sq)
    ux, uy = line.direction()
    a = Point(foot.x - half * ux, foot.y - half * uy)
    b = Point(foot.x + half * ux, foot.y + half * uy)
    return [a, b]


def circle_circle_intersection(c1: Circle, c2: Circle) -> List[Point]:
    """0, 1 or 2 intersection points of two circles (ordered)."""
    dx = c2.center.x - c1.center.x
    dy = c2.center.y - c1.center.y
    d = math.hypot(dx, dy)
    if d <= 1e-12:
        return []
    if d > c1.radius + c2.radius + 1e-12:
        return []
    if d < abs(c1.radius - c2.radius) - 1e-12:
        return []
    a = (c1.radius ** 2 - c2.radius ** 2 + d * d) / (2.0 * d)
    h_sq = c1.radius ** 2 - a * a
    if h_sq < 0.0:
        h_sq = 0.0
    h = math.sqrt(h_sq)
    xm = c1.center.x + a * dx / d
    ym = c1.center.y + a * dy / d
    if h <= 1e-12:
        return [Point(xm, ym)]
    rx = -dy * (h / d)
    ry = dx * (h / d)
    return [Point(xm + rx, ym + ry), Point(xm - rx, ym - ry)]


# ---------------------------------------------------------------------------
# Paper construction steps (Table 1 and Table 7)
# ---------------------------------------------------------------------------
def circle_offset_circle(circle: Circle, offset: float) -> Circle:
    """CircleOffsetCircle: concentric circle with radius grown by ``offset``.

    The offset is applied outward for a CCW circle and inward for a CW circle,
    following the oriented-offset convention of the paper.
    """
    delta = offset if circle.ccw else -offset
    new_r = circle.radius + delta
    if new_r <= 0.0:
        raise ValueError("offset collapses the circle to non-positive radius")
    return Circle(circle.center, new_r, circle.ccw)


def line_x_line(l1: Line, l2: Line) -> Point:
    """LineXLine: intersection point of two lines (raises if parallel)."""
    p = line_line_intersection(l1, l2)
    if p is None:
        raise ValueError("LineXLine: input lines are parallel")
    return p


def line_offset_line(line: Line, offset: float) -> Line:
    """LineOffsetLine: line offset to the *left* by ``offset`` (same direction)."""
    return Line(line.phi, line.rho + offset)


def line_x_circle(line: Line, circle: Circle) -> List[Point]:
    """LineXCircle: intersection point(s) of a line and a circle."""
    return line_circle_intersection(line, circle)


def circle_reverse_circle(circle: Circle) -> Circle:
    """CircleReverseCircle: same circle with the opposite orientation."""
    return Circle(circle.center, circle.radius, not circle.ccw)


def circle_point_point_arc(circle: Circle, start: Point, end: Point) -> Arc:
    """CirclePointPointArc: midpoint of the arc that trims ``circle`` from
    ``start`` to ``end`` following the circle's orientation.

    ``start`` and ``end`` are assumed to lie on the circle. Returns an Arc with
    ordered start, mid and end points.
    """
    c = circle.center
    a0 = math.atan2(start.y - c.y, start.x - c.x)
    a1 = math.atan2(end.y - c.y, end.x - c.x)
    if circle.ccw:
        sweep = _norm_angle(a1 - a0)
        mid_a = a0 + sweep / 2.0
    else:
        sweep = _norm_angle(a0 - a1)
        mid_a = a0 - sweep / 2.0
    mid = circle.point_at_angle(mid_a)
    return Arc(start, mid, end)


def line_datum_parallel_line(line: Line, datum: Point) -> Line:
    """LineDatumParallelLine: line parallel to ``line`` through ``datum``."""
    return parallel_line(line, datum)


def line_line_fillet(l1: Line, l2: Line, radius: float) -> Arc:
    """LineLineFillet: fillet arc of ``radius`` tangent to two directed lines.

    The fillet centre is placed on the left of both directed lines (each line
    offset left by ``radius``); the tangent points are the feet of the
    perpendiculars from the centre onto each line and become the arc's ordered
    start (on ``l1``) and end (on ``l2``). The mid point bisects the arc.
    """
    if radius <= 0.0:
        raise ValueError("fillet radius must be positive")
    o = line_line_intersection(line_offset_line(l1, radius),
                               line_offset_line(l2, radius))
    if o is None:
        raise ValueError("LineLineFillet: input lines are parallel")
    # Tangent points: projection of the centre onto each original line.
    d1 = l1.signed_distance(o)
    n1x, n1y = l1.normal()
    start = Point(o.x - d1 * n1x, o.y - d1 * n1y)
    d2 = l2.signed_distance(o)
    n2x, n2y = l2.normal()
    end = Point(o.x - d2 * n2x, o.y - d2 * n2y)
    # Mid point: bisector direction from the centre out to the arc.
    vx = (start.x - o.x) + (end.x - o.x)
    vy = (start.y - o.y) + (end.y - o.y)
    vlen = math.hypot(vx, vy)
    if vlen <= 1e-12:
        # start and end diametrically opposite: pick the +normal of l1.
        vx, vy = n1x, n1y
        vlen = 1.0
    mid = Point(o.x + radius * vx / vlen, o.y + radius * vy / vlen)
    return Arc(start, mid, end)


def line_circle_parallel_line(line: Line, circle: Circle,
                              far_side: bool = False) -> Line:
    """LineCircleParallelLine: line parallel to ``line`` tangent to ``circle``.

    Two parallel tangents exist. By default the tangent on the same side as the
    line's left normal is returned (the circle centre lies at ``+radius``);
    ``far_side`` selects the other.
    """
    nx, ny = line.normal()
    center_proj = nx * circle.center.x + ny * circle.center.y
    if far_side:
        rho = center_proj + circle.radius
    else:
        rho = center_proj - circle.radius
    return Line(line.phi, rho)


def _reflect_point(p: Point, sym: Line) -> Point:
    d = sym.signed_distance(p)
    nx, ny = sym.normal()
    return Point(p.x - 2.0 * d * nx, p.y - 2.0 * d * ny)


def line_sym_line_line(line: Line, sym: Line) -> Line:
    """LineSymLineLine: image of ``line`` reflected across symmetry line ``sym``."""
    p0 = _reflect_point(line.base_point(), sym)
    ux, uy = line.direction()
    p1 = _reflect_point(Point(line.base_point().x + ux,
                              line.base_point().y + uy), sym)
    return line_through_points(p0, p1)


def point_line_sym_point(point: Point, sym: Line) -> Point:
    """PointLineSymPoint: image of ``point`` reflected across ``sym``."""
    return _reflect_point(point, sym)


def line_reverse_line(line: Line) -> Line:
    """LineReverseLine: same line with the direction reversed."""
    return Line(_norm_angle(line.phi + math.pi), -line.rho)


def line_axis_rotated_line(line: Line, pivot: Point, angle: float,
                           ccw: bool = True) -> Line:
    """LineAxisRotatedLine: ``line`` rotated about ``pivot`` by ``angle``."""
    a = angle if ccw else -angle
    ca, sa = math.cos(a), math.sin(a)

    def rot(p: Point) -> Point:
        dx, dy = p.x - pivot.x, p.y - pivot.y
        return Point(pivot.x + ca * dx - sa * dy,
                     pivot.y + sa * dx + ca * dy)

    b = line.base_point()
    ux, uy = line.direction()
    return line_through_points(rot(b), rot(Point(b.x + ux, b.y + uy)))


def point_radius_circle(center: Point, radius: float, ccw: bool = True) -> Circle:
    """PointRadiusCircle: create a circle from a centre point and radius."""
    return circle_center_radius(center, radius, ccw)


def symline_offset_line_line(sym: Line, offset: float) -> Tuple[Line, Line]:
    """SymlineOffsetLineLine: the symmetric pair of lines offset from ``sym``.

    Returns the two lines parallel to the symmetry line at ``+offset`` and
    ``-offset`` (each the mirror image of the other across ``sym``). Used when a
    pair of parallel lines is symmetric about a symmetry line.
    """
    return (Line(sym.phi, sym.rho + offset), Line(sym.phi, sym.rho - offset))
