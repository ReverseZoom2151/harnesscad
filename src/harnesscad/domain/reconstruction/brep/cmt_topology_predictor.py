"""Deterministic topology predictor for CMT (Sec. 4.4).

After the cascade generates edge tokens and surface tokens, CMT recovers the
edge-surface adjacency matrix ``R in R^{Ne x Ns}`` -- which edges contour which
surfaces -- with a single cross-attention layer, then thresholds the score
matrix at ``tau = 0.5``. The paper stresses this replaces an expensive
post-processing search (Point2CAD), being "over 4200 times faster".

The learned cross-attention is external, but the *task* it performs -- turning
generated edge and surface geometry into an adjacency matrix -- has a clean
deterministic geometric solution that we implement here: an edge contours a
surface when both of its endpoints lie on (within tolerance of) that surface's
bounding box. The score is a smooth closeness in ``[0, 1]`` so the same ``tau``
thresholding rule applies.
"""

from __future__ import annotations

Point = tuple[float, float, float]
Box = tuple[float, float, float, float, float, float]


def point_box_distance(point: Point, box: Box) -> float:
    """Euclidean distance from ``point`` to axis-aligned ``box`` (0 if inside)."""
    lo = box[:3]
    hi = box[3:]
    d2 = 0.0
    for i in range(3):
        if point[i] < lo[i]:
            gap = lo[i] - point[i]
            d2 += gap * gap
        elif point[i] > hi[i]:
            gap = point[i] - hi[i]
            d2 += gap * gap
    return d2 ** 0.5


def edge_surface_score(start: Point, end: Point, box: Box, tolerance: float) -> float:
    """Closeness in ``[0, 1]`` that the edge (start,end) contours the surface.

    Both endpoints must sit within ``tolerance`` of the surface box. The score
    is ``1 - mean_endpoint_distance / tolerance``, clamped to ``[0, 1]``; it is
    ``>= 0.5`` exactly when the mean endpoint distance is at most ``tolerance/2``.
    """
    if tolerance <= 0:
        raise ValueError("tolerance must be positive")
    ds = point_box_distance(start, box)
    de = point_box_distance(end, box)
    mean = (ds + de) / 2.0
    score = 1.0 - mean / tolerance
    if score < 0.0:
        return 0.0
    if score > 1.0:
        return 1.0
    return score


def topology_scores(edges: tuple[tuple[Point, Point], ...],
                    surfaces: tuple[Box, ...],
                    tolerance: float) -> tuple[tuple[float, ...], ...]:
    """Score matrix ``A[Ne][Ns]`` of edge-surface adjacency closeness."""
    return tuple(
        tuple(edge_surface_score(start, end, box, tolerance) for box in surfaces)
        for (start, end) in edges
    )


def predict_adjacency(scores: tuple[tuple[float, ...], ...],
                      tau: float = 0.5) -> tuple[tuple[bool, ...], ...]:
    """Threshold the score matrix at ``tau`` (paper default 0.5)."""
    return tuple(tuple(value > tau for value in row) for row in scores)


def predict(edges: tuple[tuple[Point, Point], ...],
            surfaces: tuple[Box, ...],
            tolerance: float,
            tau: float = 0.5) -> tuple[tuple[bool, ...], ...]:
    """End-to-end: geometry -> score matrix -> thresholded adjacency matrix."""
    return predict_adjacency(topology_scores(edges, surfaces, tolerance), tau)


def surface_edges(adjacency: tuple[tuple[bool, ...], ...]) -> tuple[tuple[int, ...], ...]:
    """Per-surface list of incident edge indices (the assembled B-Rep incidence)."""
    if not adjacency:
        return ()
    n_surfaces = len(adjacency[0])
    result: list[tuple[int, ...]] = []
    for s in range(n_surfaces):
        result.append(tuple(e for e, row in enumerate(adjacency) if row[s]))
    return tuple(result)
