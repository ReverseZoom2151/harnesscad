"""cadrille point-cloud input adapter: normalise + furthest point sampling.

cadrille (and CAD-Recode before it) feeds the LLM 256 *unordered* 3D points
without normals, sub-sampled from the surface with **furthest point sampling**
(FPS), after normalising the model into the unit cube ``[-0.5, 0.5]^3``. The
projection of those points into the shared embedding space is a single learned
linear layer and is therefore out of scope here; this module implements the
deterministic geometric preprocessing that precedes it.

Pure stdlib, deterministic (the FPS seed picks the first centre).
"""

from __future__ import annotations

from math import dist

NUM_POINTS = 256


def normalize_unit_cube(points):
    """Translate + uniformly scale ``points`` to fit inside ``[-0.5, 0.5]^3``.

    Centres on the bounding-box midpoint and scales by the largest extent so the
    aspect ratio is preserved. A degenerate (single-point / zero-extent) cloud
    is centred without scaling.
    """
    pts = [tuple(float(c) for c in p) for p in points]
    if not pts:
        raise ValueError("points must be non-empty")
    dims = len(pts[0])
    lo = [min(p[d] for p in pts) for d in range(dims)]
    hi = [max(p[d] for p in pts) for d in range(dims)]
    center = [(lo[d] + hi[d]) / 2.0 for d in range(dims)]
    extent = max(hi[d] - lo[d] for d in range(dims))
    scale = (1.0 / extent) if extent > 0 else 1.0
    return [tuple((p[d] - center[d]) * scale for d in range(dims)) for p in pts]


def furthest_point_sampling(points, k: int = NUM_POINTS, seed: int = 0):
    """Greedy FPS: iteratively pick the point farthest from the chosen set.

    The first centre is chosen deterministically from ``seed`` (modulo the cloud
    size). Returns ``(sampled_points, indices)``. When ``k`` exceeds the cloud
    size every point is returned once, in FPS order.
    """
    pts = [tuple(float(c) for c in p) for p in points]
    n = len(pts)
    if n == 0:
        raise ValueError("points must be non-empty")
    if k <= 0:
        raise ValueError("k must be positive")
    k = min(k, n)
    start = seed % n
    chosen = [start]
    min_d = [dist(pts[i], pts[start]) for i in range(n)]
    while len(chosen) < k:
        # farthest point from the current set; ties break by lowest index.
        best_i, best_d = -1, -1.0
        for i in range(n):
            if min_d[i] > best_d:
                best_d, best_i = min_d[i], i
        chosen.append(best_i)
        for i in range(n):
            d = dist(pts[i], pts[best_i])
            if d < min_d[i]:
                min_d[i] = d
    return [pts[i] for i in chosen], chosen


def prepare_point_input(points, k: int = NUM_POINTS, seed: int = 0):
    """Full adapter: normalise to the unit cube, then FPS to ``k`` points."""
    normalized = normalize_unit_cube(points)
    sampled, _ = furthest_point_sampling(normalized, k, seed)
    return sampled
