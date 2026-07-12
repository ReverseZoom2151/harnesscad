"""Ramer-Douglas-Peucker polyline decimation (from the ``arcs`` Rust CAD core).

``arcs-core/src/algorithms/line_simplification.rs`` decimates a polyline to a
"simpler" one with fewer points, where *simpler* is defined by the maximum
distance (``tolerance``) between the original curve and the simplified curve.
Unlike a tessellation/approximation pass, simplification never creates new
points -- it only removes existing ones.

Two implementation details are carried over verbatim from the Rust source
because they are what make the routine robust:

* the point/segment distance is the **perpendicular distance to the infinite
  line** through the segment, computed from the triangle area via the 2-D cross
  product (``area = |cross(start - p, end - p)| / 2``, ``distance = 2 * area /
  base``), with a degenerate fallback to ``|start - p|`` when the base length is
  below ``100 * DBL_EPSILON``;
* the split point is the *first* index attaining the strict maximum
  (``key > best``), which makes the output deterministic when several points tie.

The harness previously only mentioned Douglas-Peucker in the ``vision``
raster-vectoriser docstring; no implementation existed.

Pure standard library, deterministic. Points are ``(x, y)`` float tuples.
"""

from __future__ import annotations

import math
import sys
from typing import Iterable, List, Sequence, Tuple

Point = Tuple[float, float]

# arcs: `const SOME_SMALL_NUMBER: f64 = std::f64::EPSILON * 100.0;`
SOME_SMALL_NUMBER = sys.float_info.epsilon * 100.0

__all__ = [
    "SOME_SMALL_NUMBER",
    "perpendicular_distance",
    "polyline_length",
    "simplify",
    "simplify_indices",
    "max_deviation",
]


def _cross(ax: float, ay: float, bx: float, by: float) -> float:
    return ax * by - ay * bx


def perpendicular_distance(start: Point, end: Point, point: Point) -> float:
    """Distance from ``point`` to the infinite line through ``start``/``end``.

    Degenerate (zero-length) segments fall back to the distance to ``start``.
    """
    ax = start[0] - point[0]
    ay = start[1] - point[1]
    bx = end[0] - point[0]
    by = end[1] - point[1]

    area = _cross(ax, ay, bx, by) / 2.0
    base = math.hypot(end[0] - start[0], end[1] - start[1])

    if abs(base) < SOME_SMALL_NUMBER:
        return math.hypot(ax, ay)
    return abs(area) * 2.0 / base


def polyline_length(points: Sequence[Point]) -> float:
    """Total length of the polyline."""
    total = 0.0
    for i in range(1, len(points)):
        total += math.hypot(
            points[i][0] - points[i - 1][0], points[i][1] - points[i - 1][1]
        )
    return total


def _split(
    points: Sequence[Point],
    lo: int,
    hi: int,
    tolerance: float,
    keep: List[int],
) -> None:
    """Recursively keep the indices strictly between ``lo`` and ``hi``."""
    if hi - lo < 2:
        return

    best_index = -1
    best_distance = 0.0
    first_seen = False

    for i in range(lo + 1, hi):
        distance = perpendicular_distance(points[lo], points[hi], points[i])
        # strict `>` -> the FIRST maximum wins, as in the Rust `max_by_key`
        if not first_seen or distance > best_distance:
            first_seen = True
            best_index = i
            best_distance = distance

    if best_index >= 0 and best_distance > tolerance:
        _split(points, lo, best_index, tolerance, keep)
        keep.append(best_index)
        _split(points, best_index, hi, tolerance, keep)


def simplify_indices(points: Sequence[Point], tolerance: float) -> List[int]:
    """Indices of the points retained by :func:`simplify`, ascending."""
    n = len(points)
    if n <= 2:
        return list(range(n))

    keep: List[int] = [0]
    _split(points, 0, n - 1, tolerance, keep)
    keep.append(n - 1)
    return keep


def simplify(points: Iterable[Point], tolerance: float) -> List[Point]:
    """Decimate ``points`` so no removed point lies further than ``tolerance``.

    Polylines of 0, 1 or 2 points are returned unchanged. Endpoints are always
    preserved.
    """
    pts: List[Point] = [(float(p[0]), float(p[1])) for p in points]
    return [pts[i] for i in simplify_indices(pts, tolerance)]


def max_deviation(
    original: Sequence[Point], simplified: Sequence[Point]
) -> float:
    """Largest distance from a point of ``original`` to the ``simplified`` path.

    Distances are measured to the closest *segment* (clamped, not the infinite
    line), so this is a true Hausdorff-style check that a simplification
    honoured its tolerance.
    """
    if not original or len(simplified) < 2:
        return 0.0

    worst = 0.0
    for p in original:
        best = float("inf")
        for i in range(1, len(simplified)):
            best = min(
                best, _segment_distance(simplified[i - 1], simplified[i], p)
            )
        worst = max(worst, best)
    return worst


def _segment_distance(start: Point, end: Point, point: Point) -> float:
    dx = end[0] - start[0]
    dy = end[1] - start[1]
    squared = dx * dx + dy * dy
    if squared < SOME_SMALL_NUMBER:
        return math.hypot(point[0] - start[0], point[1] - start[1])
    t = ((point[0] - start[0]) * dx + (point[1] - start[1]) * dy) / squared
    t = min(1.0, max(0.0, t))
    return math.hypot(
        point[0] - (start[0] + t * dx), point[1] - (start[1] + t * dy)
    )
