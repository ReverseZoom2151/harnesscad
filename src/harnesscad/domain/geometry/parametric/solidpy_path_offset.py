"""Deterministic 2D path geometry: miter offsets, strokes and corner fillets.

Reimplementation of the 2D path toolkit in SolidPython's ``solid/utils.py``
(``offset_points``, ``offset_point``, ``cross_2d``, ``direction_of_bend``,
``perpendicular_vector``, ``path_2d``, ``fillet_2d``, ``arc_inverted``) with the
PyEuclid ``Line2.intersect`` dependency replaced by an explicit parametric
line-line intersection, and with the *points* returned instead of OpenSCAD
objects.

What is here that the harness did not have:

  * :func:`offset_points` -- miter-joined offset of an open polyline or a closed
    polygon by a signed distance.  Each segment is translated along its
    perpendicular and consecutive offset lines are intersected, so corners stay
    sharp (no round joins, no per-vertex normal averaging).  The offset side is
    keyed off the direction of the first bend, exactly as SolidPython does.
  * :func:`path_2d` -- turn a polyline into a closed polygon of a given width
    (a "stroke"), by offsetting to both sides and reversing one of them.  For a
    closed path this yields two rings (outer + inner), i.e. a polygon with a
    hole, and :func:`path_2d_paths` gives the matching OpenSCAD ``paths`` index
    lists.
  * :func:`fillet_corner` / :func:`round_polygon` -- exact tangent-arc corner
    rounding: the arc centre is the miter-offset of the corner by the fillet
    radius, the tangent points lie at distance ``r / tan(theta/2)`` from the
    corner, and the corner is replaced by a discretised arc.  Infeasible radii
    (arc longer than the adjacent segments) raise instead of self-intersecting.

Pure stdlib, deterministic.
"""

from __future__ import annotations

import math
from typing import List, Optional, Sequence, Tuple

__all__ = [
    "EPSILON",
    "LEFT_DIR",
    "RIGHT_DIR",
    "cross_2d",
    "direction_of_bend",
    "opposite_direction",
    "perpendicular_vector",
    "line_intersection",
    "signed_area",
    "is_ccw",
    "offset_point",
    "offset_points",
    "path_2d",
    "path_2d_paths",
    "arc_points",
    "fillet_corner",
    "round_polygon",
]

EPSILON = 1e-9

LEFT_DIR = 1
RIGHT_DIR = 2

Point2 = Tuple[float, float]


def _p2(p: Sequence[float]) -> Point2:
    return (float(p[0]), float(p[1]))


def cross_2d(a: Sequence[float], b: Sequence[float]) -> float:
    """Scalar cross product; its sign gives the direction of rotation a -> b."""
    return a[0] * b[1] - a[1] * b[0]


def direction_of_bend(a: Sequence[float], b: Sequence[float],
                      c: Sequence[float]) -> int:
    """``LEFT_DIR`` if the turn a->b->c is to the left, else ``RIGHT_DIR``.

    Colinear points report ``RIGHT_DIR`` (SolidPython's convention).
    """
    ab = (b[0] - a[0], b[1] - a[1])
    bc = (c[0] - b[0], c[1] - b[1])
    return LEFT_DIR if cross_2d(ab, bc) > 0 else RIGHT_DIR


def opposite_direction(direction: int) -> int:
    return LEFT_DIR if direction == RIGHT_DIR else RIGHT_DIR


def perpendicular_vector(v: Sequence[float], direction: int = RIGHT_DIR,
                         length: Optional[float] = None) -> Point2:
    """A vector perpendicular to ``v``, on the given side, optionally rescaled."""
    perp = (v[1], -v[0])
    if direction != RIGHT_DIR:
        perp = (-perp[0], -perp[1])
    if length is not None:
        mag = math.hypot(perp[0], perp[1])
        if mag < EPSILON:
            raise ValueError("cannot set the length of a zero-length vector")
        perp = (perp[0] / mag * length, perp[1] / mag * length)
    return perp


def line_intersection(p1: Sequence[float], v1: Sequence[float],
                      p2: Sequence[float], v2: Sequence[float]) -> Optional[Point2]:
    """Intersection of the infinite lines ``p1 + t*v1`` and ``p2 + u*v2``."""
    denom = cross_2d(v1, v2)
    if abs(denom) < EPSILON:
        return None  # parallel or colinear
    dx = (p2[0] - p1[0], p2[1] - p1[1])
    t = cross_2d(dx, v2) / denom
    return (p1[0] + t * v1[0], p1[1] + t * v1[1])


def signed_area(points: Sequence[Sequence[float]]) -> float:
    """Shoelace area; positive when the ring is counter-clockwise."""
    total = 0.0
    n = len(points)
    for i in range(n):
        a = points[i]
        b = points[(i + 1) % n]
        total += a[0] * b[1] - b[0] * a[1]
    return total / 2.0


def is_ccw(points: Sequence[Sequence[float]]) -> bool:
    return signed_area(points) > 0


def offset_point(a: Sequence[float], b: Sequence[float], c: Sequence[float],
                 offset: float, direction: int = LEFT_DIR) -> Point2:
    """Miter-offset the single corner ``b`` of the path a-b-c."""
    ab = (b[0] - a[0], b[1] - a[1])
    bc = (c[0] - b[0], c[1] - b[1])
    ab_perp = perpendicular_vector(ab, direction, length=offset)
    bc_perp = perpendicular_vector(bc, direction, length=offset)
    p1 = (a[0] + ab_perp[0], a[1] + ab_perp[1])
    p2 = (b[0] + bc_perp[0], b[1] + bc_perp[1])
    hit = line_intersection(p1, ab, p2, bc)
    if hit is None:
        # colinear: the offset corner is just b moved along the perpendicular
        return (b[0] + ab_perp[0], b[1] + ab_perp[1])
    return hit


def offset_points(points: Sequence[Sequence[float]], offset: float,
                  internal: bool = True, closed: bool = True) -> List[Point2]:
    """Offset a polyline/polygon by ``offset``, with mitered corners.

    ``internal`` selects the side, relative to the direction of the first bend
    (SolidPython's definition); for a non-convex first corner the sense flips.
    """
    src = [_p2(p) for p in points]
    if len(src) < 3 and closed:
        raise ValueError("a closed path needs at least 3 points")
    if len(src) < 2:
        raise ValueError("offset_points() needs at least 2 points")
    if closed:
        src = src + [src[0]]

    vecs = [(b[0] - a[0], b[1] - a[1]) for a, b in zip(src[:-1], src[1:])]
    if len(src) >= 3:
        direction = direction_of_bend(src[0], src[1], src[2])
    else:
        direction = RIGHT_DIR  # a single straight segment has no bend
    if not internal:
        direction = opposite_direction(direction)

    perps = [perpendicular_vector(v, direction, length=offset) for v in vecs]
    lines = [((a[0] + p[0], a[1] + p[1]), v)
             for p, a, v in zip(perps, src[:-1], vecs)]

    out: List[Point2] = []
    for (p1, v1), (p2, v2) in zip(lines[:-1], lines[1:]):
        hit = line_intersection(p1, v1, p2, v2)
        if hit is None:
            hit = (p1[0] + v1[0], p1[1] + v1[1])  # colinear segments
        out.append(hit)

    if closed:
        first = line_intersection(lines[0][0], lines[0][1],
                                  lines[-1][0], lines[-1][1])
        if first is None:
            first = lines[0][0]
        out.insert(0, first)
    else:
        out.insert(0, lines[0][0])
        last_p, last_v = lines[-1]
        out.append((last_p[0] + last_v[0], last_p[1] + last_v[1]))
    return out


def path_2d(points: Sequence[Sequence[float]], width: float = 1.0,
            closed: bool = False) -> List[Point2]:
    """A closed polygon of width ``width`` centred on the polyline ``points``."""
    a = offset_points(points, offset=width / 2.0, internal=True, closed=closed)
    b = list(reversed(
        offset_points(points, offset=width / 2.0, internal=False, closed=closed)))
    return a + b


def path_2d_paths(points: Sequence[Sequence[float]],
                  closed: bool = False) -> List[List[int]]:
    """The OpenSCAD ``paths`` index lists matching :func:`path_2d`'s output."""
    n = len(points)
    if closed:
        return [list(range(n)), list(range(n, 2 * n))]
    return [list(range(2 * (n + 1)))]


def arc_points(center: Sequence[float], radius: float, start_degrees: float,
               end_degrees: float, segments: int = 16) -> List[Point2]:
    """``segments + 1`` points along the arc from start to end (shorter way)."""
    if segments < 1:
        raise ValueError("segments must be >= 1")
    sweep = (end_degrees - start_degrees) % 360.0
    if sweep > 180.0:
        sweep -= 360.0
    out: List[Point2] = []
    for i in range(segments + 1):
        theta = math.radians(start_degrees + sweep * i / segments)
        out.append((center[0] + radius * math.cos(theta),
                    center[1] + radius * math.sin(theta)))
    return out


def fillet_corner(a: Sequence[float], b: Sequence[float], c: Sequence[float],
                  radius: float, segments: int = 8
                  ) -> Tuple[Point2, List[Point2]]:
    """Round the corner ``b``: return the arc centre and the arc's points.

    The centre is the miter-offset of ``b`` by ``radius`` toward the inside of
    the bend; the arc runs between the two tangent points on ``ba`` and ``bc``.
    """
    if radius <= 0:
        raise ValueError("radius must be positive")
    ba = (a[0] - b[0], a[1] - b[1])
    bc = (c[0] - b[0], c[1] - b[1])
    len_ba = math.hypot(*ba)
    len_bc = math.hypot(*bc)
    if len_ba < EPSILON or len_bc < EPSILON:
        raise ValueError("degenerate corner")

    cos_theta = (ba[0] * bc[0] + ba[1] * bc[1]) / (len_ba * len_bc)
    cos_theta = max(-1.0, min(1.0, cos_theta))
    theta = math.acos(cos_theta)  # interior angle at b
    if theta < EPSILON or abs(theta - math.pi) < EPSILON:
        raise ValueError("cannot fillet a straight or reversed corner")

    tangent_dist = radius / math.tan(theta / 2.0)
    if tangent_dist > len_ba + EPSILON or tangent_dist > len_bc + EPSILON:
        raise ValueError(
            "fillet radius %r too large for corner (needs %.4f of segment)"
            % (radius, tangent_dist))

    direction = direction_of_bend(a, b, c)
    center = offset_point(a, b, c, offset=radius, direction=direction)

    t_a = (b[0] + ba[0] / len_ba * tangent_dist, b[1] + ba[1] / len_ba * tangent_dist)
    t_c = (b[0] + bc[0] / len_bc * tangent_dist, b[1] + bc[1] / len_bc * tangent_dist)
    start = math.degrees(math.atan2(t_a[1] - center[1], t_a[0] - center[0]))
    end = math.degrees(math.atan2(t_c[1] - center[1], t_c[0] - center[0]))
    return center, arc_points(center, radius, start, end, segments)


def round_polygon(points: Sequence[Sequence[float]], radius: float,
                  segments: int = 8, closed: bool = True) -> List[Point2]:
    """Replace every corner of a polygon/polyline with a tangent arc."""
    src = [_p2(p) for p in points]
    n = len(src)
    if n < 3:
        raise ValueError("round_polygon() needs at least 3 points")
    out: List[Point2] = []
    if closed:
        indices = range(n)
    else:
        out.append(src[0])
        indices = range(1, n - 1)
    for i in indices:
        a = src[(i - 1) % n]
        b = src[i]
        c = src[(i + 1) % n]
        _, arc = fillet_corner(a, b, c, radius, segments)
        out.extend(arc)
    if not closed:
        out.append(src[-1])
    return out
