"""Dataset-level point-cloud normalisation from LION (Zeng et al., 2022).

LION follows PointFlow's data preprocessing, which normalises the ShapeNet point
clouds **globally across the whole dataset** rather than per-shape. Concretely,
PointFlow computes a single per-axis mean and a single global standard deviation
over *all points of all training shapes*, then applies ``(x - mean) / std`` to
every shape. This "global normalisation" preserves the relative scale between
different objects (a car stays bigger than a mug), which is what LION's Table 1
uses. Some baselines instead require *per-shape* normalisation into ``[-1, 1]``
(LION's Table 2 -- "data normalised individually into [-1, 1]"), so both variants
are provided here.

This is deliberately distinct from the existing per-shape unit-cube helper
``reconstruction.cadrille_pointcloud_adapter.normalize_unit_cube`` (which centres
each shape on its own bounding box and scales into ``[-0.5, 0.5]^3`` independently
of any dataset). The pieces here are dataset statistics, not per-shape geometry.

Pure stdlib, deterministic.
"""

from __future__ import annotations

from math import sqrt
from typing import List, Sequence, Tuple

Cloud = Sequence[Sequence[float]]


def _dims(clouds: Sequence[Cloud]) -> int:
    for cloud in clouds:
        for pt in cloud:
            return len(pt)
    raise ValueError("dataset must contain at least one point")


def global_stats(clouds: Sequence[Cloud]) -> Tuple[List[float], float]:
    """Per-axis mean and a single scalar global std over *all* points.

    Returns ``(mean, std)`` where ``mean`` is one value per coordinate axis and
    ``std`` is a single scalar (the root-mean-square deviation pooled over every
    axis and every point), matching PointFlow's global normalisation. A dataset
    with zero variance yields ``std == 1.0`` so normalisation is a no-op shift.
    """
    dims = _dims(clouds)
    total = [0.0] * dims
    count = 0
    for cloud in clouds:
        for pt in cloud:
            for d in range(dims):
                total[d] += float(pt[d])
            count += 1
    if count == 0:
        raise ValueError("dataset must contain at least one point")
    mean = [total[d] / count for d in range(dims)]
    sq = 0.0
    for cloud in clouds:
        for pt in cloud:
            for d in range(dims):
                diff = float(pt[d]) - mean[d]
                sq += diff * diff
    var = sq / (count * dims)
    std = sqrt(var) if var > 0 else 1.0
    return mean, std


def global_normalize(
    clouds: Sequence[Cloud],
    stats: Tuple[Sequence[float], float] | None = None,
) -> List[List[Tuple[float, ...]]]:
    """Apply global (dataset-level) normalisation ``(x - mean) / std``.

    If ``stats`` is provided (e.g. computed on a training split) it is reused so
    validation/test shapes are normalised with the *same* statistics; otherwise
    the statistics are estimated from ``clouds`` itself.
    """
    if stats is None:
        mean, std = global_stats(clouds)
    else:
        mean, std = list(stats[0]), float(stats[1])
        if std == 0:
            std = 1.0
    dims = len(mean)
    out: List[List[Tuple[float, ...]]] = []
    for cloud in clouds:
        norm_cloud = [
            tuple((float(pt[d]) - mean[d]) / std for d in range(dims))
            for pt in cloud
        ]
        out.append(norm_cloud)
    return out


def per_shape_normalize_unit_range(cloud: Cloud) -> List[Tuple[float, ...]]:
    """Normalise a single shape *individually* into ``[-1, 1]`` (LION Table 2).

    Centres on the shape's bounding-box midpoint and scales by half the largest
    extent so the longest axis spans exactly ``[-1, 1]`` (aspect ratio kept). A
    degenerate zero-extent cloud is centred without scaling.
    """
    pts = [tuple(float(c) for c in p) for p in cloud]
    if not pts:
        raise ValueError("cloud must be non-empty")
    dims = len(pts[0])
    lo = [min(p[d] for p in pts) for d in range(dims)]
    hi = [max(p[d] for p in pts) for d in range(dims)]
    center = [(lo[d] + hi[d]) / 2.0 for d in range(dims)]
    half_extent = max(hi[d] - lo[d] for d in range(dims)) / 2.0
    scale = (1.0 / half_extent) if half_extent > 0 else 1.0
    return [tuple((p[d] - center[d]) * scale for d in range(dims)) for p in pts]


def bounding_box(clouds: Sequence[Cloud]) -> Tuple[List[float], List[float]]:
    """Axis-aligned dataset bounding box ``(lo, hi)`` over all points."""
    dims = _dims(clouds)
    lo = [float("inf")] * dims
    hi = [float("-inf")] * dims
    for cloud in clouds:
        for pt in cloud:
            for d in range(dims):
                v = float(pt[d])
                if v < lo[d]:
                    lo[d] = v
                if v > hi[d]:
                    hi[d] = v
    if lo[0] == float("inf"):
        raise ValueError("dataset must contain at least one point")
    return lo, hi
