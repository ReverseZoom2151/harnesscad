"""CAD-model distance protocol for non-categorical clustering (Cluster3D).

Xiang, Tseng, Wen et al., *Evaluating Deep Clustering Algorithms on
Non-Categorical 3D CAD Models*, Section 4.2 and Appendix A.5 / A.6.

To score clusters with an internal index (silhouette) the paper needs a distance
"between every two objects". It defines two distances between CAD models, each
computed on a **min-max normalised** point cloud (Appendix A.5: 4096 sampled
points, per-dimension min/max mapped to 0/1):

* Chamfer distance (Fan et al.) between the two point clouds.
* Jaccard distance ``1 - IoU`` on voxel grids built from the normalised cloud;
  Appendix A.6 keeps the IoU un-normalised so that an object "too thin to
  normalise" has IoU zero (hence Jaccard distance 1) against anything.

This module packages that exact distance protocol deterministically, feeding the
pairwise distance matrix that :mod:`bench.contrastcad_latent_metrics` silhouette
and :mod:`bench.deepclustering_internal_indices` consume. It reuses no learned
encoder -- point clouds are supplied by the caller. It is distinct from the
generic ``voxel_iou`` helpers elsewhere: here the emphasis is the normalisation +
thin-object rule + ``1 - IoU`` distance packaging the paper specifies.

Stdlib only, deterministic.
"""

from __future__ import annotations

import math
from typing import List, Sequence, Tuple

Vec3 = Sequence[float]


def min_max_normalize(points: Sequence[Vec3]) -> List[Tuple[float, float, float]]:
    """Per-dimension min-max normalise a point cloud into the unit cube.

    Each coordinate ``d`` is mapped so its min becomes 0 and max becomes 1
    (Appendix A.5). A dimension with zero extent (all equal) maps to 0 to avoid
    division by zero -- e.g. a planar sketch stays flat. Returns 3-tuples.
    """
    if not points:
        raise ValueError("point cloud is empty")
    dims = 3
    mins = [min(p[d] for p in points) for d in range(dims)]
    maxs = [max(p[d] for p in points) for d in range(dims)]
    spans = [maxs[d] - mins[d] for d in range(dims)]
    out: List[Tuple[float, float, float]] = []
    for p in points:
        coords = []
        for d in range(dims):
            if spans[d] == 0.0:
                coords.append(0.0)
            else:
                coords.append((p[d] - mins[d]) / spans[d])
        out.append((coords[0], coords[1], coords[2]))
    return out


def _euclid(a: Vec3, b: Vec3) -> float:
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))


def chamfer_distance(cloud_a: Sequence[Vec3], cloud_b: Sequence[Vec3],
                     *, normalize: bool = True) -> float:
    """Symmetric Chamfer distance between two point clouds.

    ``CD = mean_a min_b ||a - b||^2 + mean_b min_a ||b - a||^2`` (Fan et al.,
    the paper's Chamfer distance). When ``normalize`` is true (default) both
    clouds are min-max normalised first, matching Appendix A.5.
    """
    if not cloud_a or not cloud_b:
        raise ValueError("both point clouds must be non-empty")
    a = min_max_normalize(cloud_a) if normalize else [tuple(p) for p in cloud_a]
    b = min_max_normalize(cloud_b) if normalize else [tuple(p) for p in cloud_b]

    def one_way(src, dst) -> float:
        total = 0.0
        for p in src:
            best = min((_euclid(p, q) ** 2) for q in dst)
            total += best
        return total / len(src)

    return one_way(a, b) + one_way(b, a)


def voxelize(points: Sequence[Vec3], resolution: int,
             *, normalize: bool = True) -> set:
    """Occupied voxel indices of a point cloud at the given grid resolution.

    With ``normalize`` (default) the cloud is min-max normalised into the unit
    cube first, then each point is binned into an integer ``(x, y, z)`` cell in
    ``[0, resolution)``. Returns the set of occupied cells.
    """
    if resolution <= 0:
        raise ValueError("resolution must be positive")
    pts = min_max_normalize(points) if normalize else [tuple(p) for p in points]
    occupied = set()
    for p in pts:
        cell = tuple(min(resolution - 1, max(0, int(coord * resolution)))
                     for coord in p)
        occupied.add(cell)
    return occupied


def voxel_jaccard_distance(cloud_a: Sequence[Vec3], cloud_b: Sequence[Vec3],
                           resolution: int = 16) -> float:
    """Jaccard distance ``1 - IoU`` between voxelised CAD models (Appendix A.6).

    Both clouds are min-max normalised and voxelised; the IoU is
    ``|A ∩ B| / |A ∪ B|`` and the distance is ``1 - IoU`` (Kosub). Two empty
    grids are treated as maximally dissimilar (distance 1), reflecting the
    thin-object rule where an un-normalisable object has zero IoU with anything.
    Result is in ``[0, 1]``.
    """
    va = voxelize(cloud_a, resolution)
    vb = voxelize(cloud_b, resolution)
    union = va | vb
    if not union:
        return 1.0
    inter = va & vb
    iou = len(inter) / len(union)
    return 1.0 - iou


def pairwise_distance_matrix(clouds: Sequence[Sequence[Vec3]],
                             metric: str = "chamfer",
                             resolution: int = 16) -> List[List[float]]:
    """Symmetric pairwise distance matrix between CAD models (Fig. 5 / Fig. 8).

    ``metric`` is ``chamfer`` or ``jaccard``. The resulting matrix feeds the
    silhouette / internal-index rankings the paper reports. Diagonal is 0.
    """
    if metric not in ("chamfer", "jaccard"):
        raise ValueError("metric must be chamfer or jaccard")
    n = len(clouds)
    matrix = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            if metric == "chamfer":
                d = chamfer_distance(clouds[i], clouds[j])
            else:
                d = voxel_jaccard_distance(clouds[i], clouds[j], resolution)
            matrix[i][j] = d
            matrix[j][i] = d
    return matrix
