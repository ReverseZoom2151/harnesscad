"""Spoke edge-coordinate processing for CAD spline generation.

Deterministic data-processing pipeline of the 3D-CAD-automation stage
(Section 4.4.2, "Data processing") of:

    Yoo et al., "Integrating deep learning into CAD/CAE system: generative design
    and evaluation of 3D conceptual wheel", Struct. Multidisc. Optim. 64 (2021)
    2725-2747.

After edge extraction, the raw spoke edge pixels must be turned into clean,
ordered point groups that a CAD tool can fit splines through.  The paper does
this deterministically:

    1. **Nearest-neighbour ordering.**  Starting from a fixed point, repeatedly
       hop to the nearest not-yet-visited point (shortest Euclidean distance),
       "so the points can be organized by the shortest distance".

    2. **Distance-threshold grouping.**  "If the point closest to the fixed point
       is greater than or equal to the threshold value, it is regarded as a
       different group."  A jump longer than the threshold starts a new group.

    3. **Group-size-based point reduction** (deletion rate by group size):
         * a group with more than 20 and fewer than 100 points is reduced to
           1/6 of its points,
         * a group with 100 points or more is reduced to 1/12,
         * a group with 20 or fewer points is kept as-is,
         * a group with 3 or fewer points is considered noise and deleted.
       Reduction keeps evenly-spaced points to preserve the curve shape.

    4. **Mean centering.**  All coordinates are moved so their centroid is at the
       origin.

    5. **Scalar scaling.**  Each point group is multiplied by a scalar (0.97 in
       the paper) to fit the target 18-inch wheel.

All functions are deterministic and stdlib-only (``math``).  Points are
``(x, y)`` tuples of floats.
"""

from __future__ import annotations

import math
from typing import List, Sequence, Tuple

Point = Tuple[float, float]

# Paper Section 4.4.2 reduction constants.
NOISE_MAX_POINTS = 3
SMALL_GROUP_MAX = 20
LARGE_GROUP_MIN = 100
MID_REDUCTION = 6
LARGE_REDUCTION = 12
WHEEL_SCALE = 0.97


def _dist(a: Point, b: Point) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def nearest_neighbour_order(points: Sequence[Point], start_index: int = 0) -> List[Point]:
    """Greedily order points by repeatedly hopping to the nearest unvisited one.

    Deterministic: ties are broken by the earliest original index.  Returns a new
    ordered list beginning at ``points[start_index]``.
    """
    n = len(points)
    if n == 0:
        return []
    if not (0 <= start_index < n):
        raise ValueError("start_index out of range")
    remaining = list(range(n))
    remaining.remove(start_index)
    order = [start_index]
    current = points[start_index]
    while remaining:
        best = None
        best_d = None
        for idx in remaining:
            d = _dist(current, points[idx])
            if best_d is None or d < best_d:
                best_d = d
                best = idx
        order.append(best)
        remaining.remove(best)
        current = points[best]
    return [tuple(points[i]) for i in order]


def group_by_distance(ordered: Sequence[Point], threshold: float) -> List[List[Point]]:
    """Split an ordered point path into groups at jumps ``>= threshold``.

    Consecutive points whose gap is below ``threshold`` stay in one group; a gap
    at or above ``threshold`` starts a new group (Section 4.4.2).
    """
    if threshold <= 0.0:
        raise ValueError("threshold must be positive")
    groups: List[List[Point]] = []
    current: List[Point] = []
    prev: Point = None
    for p in ordered:
        p = tuple(p)
        if prev is None:
            current = [p]
        elif _dist(prev, p) >= threshold:
            groups.append(current)
            current = [p]
        else:
            current.append(p)
        prev = p
    if current:
        groups.append(current)
    return groups


def _keep_indices(count: int, keep: int) -> List[int]:
    """Return ``keep`` evenly-spaced indices out of ``count`` (endpoints kept)."""
    if keep >= count:
        return list(range(count))
    if keep <= 1:
        return [0]
    return [round(i * (count - 1) / (keep - 1)) for i in range(keep)]


def reduce_group(group: Sequence[Point]) -> List[Point]:
    """Apply the group-size-based reduction rule (Section 4.4.2).

    Returns the reduced (evenly-subsampled) group, or an empty list if the group
    is noise (``<= NOISE_MAX_POINTS`` points).
    """
    n = len(group)
    if n <= NOISE_MAX_POINTS:
        return []
    if n <= SMALL_GROUP_MAX:
        return [tuple(p) for p in group]
    if n < LARGE_GROUP_MIN:
        keep = max(1, n // MID_REDUCTION)
    else:
        keep = max(1, n // LARGE_REDUCTION)
    idxs = _keep_indices(n, keep)
    return [tuple(group[i]) for i in idxs]


def reduce_groups(groups: Sequence[Sequence[Point]]) -> List[List[Point]]:
    """Reduce every group and drop groups deleted as noise (empty results)."""
    out: List[List[Point]] = []
    for g in groups:
        r = reduce_group(g)
        if r:
            out.append(r)
    return out


def centroid(points: Sequence[Point]) -> Point:
    """Arithmetic centroid ``(mean x, mean y)`` of a point set."""
    if not points:
        raise ValueError("cannot take centroid of empty point set")
    sx = math.fsum(p[0] for p in points)
    sy = math.fsum(p[1] for p in points)
    n = len(points)
    return (sx / n, sy / n)


def mean_center(groups: Sequence[Sequence[Point]]) -> List[List[Point]]:
    """Move all groups so the centroid of *all* points sits at the origin."""
    flat = [tuple(p) for g in groups for p in g]
    if not flat:
        return [list(g) for g in groups]
    cx, cy = centroid(flat)
    return [[(p[0] - cx, p[1] - cy) for p in g] for g in groups]


def scale_groups(groups: Sequence[Sequence[Point]], scale: float = WHEEL_SCALE) -> List[List[Point]]:
    """Multiply every coordinate in every group by ``scale`` (default 0.97)."""
    return [[(p[0] * scale, p[1] * scale) for p in g] for g in groups]


def process_spoke_points(
    points: Sequence[Point],
    threshold: float,
    scale: float = WHEEL_SCALE,
    start_index: int = 0,
) -> List[List[Point]]:
    """Run the full Section 4.4.2 pipeline on raw spoke edge points.

    order -> group -> reduce -> mean-center -> scale.  Returns the processed
    list of point groups ready for spline generation.
    """
    ordered = nearest_neighbour_order(points, start_index=start_index)
    groups = group_by_distance(ordered, threshold)
    reduced = reduce_groups(groups)
    centered = mean_center(reduced)
    return scale_groups(centered, scale=scale)
