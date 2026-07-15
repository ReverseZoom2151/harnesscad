"""Deterministic PointBERT point-cloud patch tokeniser (FPS + kNN grouping).

Ported from BlendCLIP's PointBERT ``Group`` module
(``models/pointbert/dvae.py`` and ``models/pointbert/misc.py``).  PointBERT
turns a raw point cloud into a fixed set of *patch tokens* -- the point-cloud
analogue of ViT image patches -- with a two-stage, learning-free procedure:

1.  **Farthest-point sampling (FPS)** picks ``num_group`` patch centres that are
    spread as evenly as possible over the cloud.  Starting from a seed point, it
    repeatedly adds the point whose distance to the already-chosen set is
    largest (a greedy 2-approximation of the k-centre problem).  This is exactly
    ``misc.fps`` / ``pointnet2_utils.furthest_point_sample``.
2.  **k-nearest-neighbour grouping** collects, for each centre, the
    ``group_size`` closest points by squared Euclidean distance
    (``knn_point`` = ``topk(-dist)`` in the reference).
3.  **Local normalisation** subtracts the centre from each neighbour so every
    patch is expressed in its own local frame (``neighborhood - center``).

The result is a ``(num_group, group_size, C)`` tensor of normalised local
patches plus the ``(num_group, C)`` centres -- the deterministic tokenisation
that the (trained) transformer would then embed.  Everything the *learned*
model does after this point (linear patch encoder, attention) is dropped; what
remains is a genuine, reproducible geometry primitive.

The harness had FPS nowhere: ``geometry.mesh.sampling`` does area-weighted
random surface sampling, not farthest-point coverage, and there was no kNN
patch grouping at all.

Design notes / determinism
--------------------------
*   Points are sequences of floats of dimension ``>= 3``.  Only the first three
    coordinates are used for distances and centring; any extra channels (e.g.
    RGB) are carried through the gather but *not* normalised, mirroring the
    ``C > 3`` branch of ``Group.forward``.
*   FPS starts at index ``seed`` (default 0) instead of a random point, so the
    whole pipeline is byte-reproducible.  The reference uses ``randint``; index
    0 is the standard deterministic substitute.
*   All ties (equal distances in FPS or kNN) are broken by ascending point
    index, so output is a pure function of the input.

Pure stdlib, no numpy / torch.
"""

from __future__ import annotations

import math
from typing import List, Sequence, Tuple

__all__ = [
    "Point",
    "square_distance",
    "farthest_point_sampling",
    "knn_indices",
    "group_patches",
    "Patches",
]

Point = Sequence[float]


def _sq_dist3(a: Point, b: Point) -> float:
    """Squared Euclidean distance over the first three coordinates."""
    dx = a[0] - b[0]
    dy = a[1] - b[1]
    dz = a[2] - b[2]
    return dx * dx + dy * dy + dz * dz


def square_distance(src: Sequence[Point], dst: Sequence[Point]) -> List[List[float]]:
    """Pairwise squared distance matrix ``dist[i][j] = |src[i]-dst[j]|^2``.

    Mirrors ``dvae.square_distance`` (over the xyz coordinates).
    """
    return [[_sq_dist3(s, d) for d in dst] for s in src]


def farthest_point_sampling(points: Sequence[Point], number: int, *, seed: int = 0) -> List[int]:
    """Greedy farthest-point sampling; returns selected indices in pick order.

    Reproduces the commented reference ``fps`` in ``misc.py``:

        farthest = seed
        for i in range(number):
            centroids[i] = farthest
            distance = min(distance, dist_to(farthest))
            farthest = argmax(distance)

    If ``number`` exceeds the number of distinct farthest picks available, the
    argmax simply keeps returning already-saturated points; we still return
    ``number`` indices (with repeats only when ``number > len(points)``), which
    matches the fixed-size behaviour the tokeniser relies on.
    """
    n = len(points)
    if n == 0:
        raise ValueError("cannot sample an empty point cloud")
    if number <= 0:
        return []
    farthest = seed % n
    distance = [math.inf] * n
    centroids: List[int] = []
    for _ in range(number):
        centroids.append(farthest)
        cpt = points[farthest]
        for j in range(n):
            d = _sq_dist3(points[j], cpt)
            if d < distance[j]:
                distance[j] = d
        # argmax with ascending-index tie-break
        best_i = 0
        best_v = distance[0]
        for j in range(1, n):
            if distance[j] > best_v:
                best_v = distance[j]
                best_i = j
        farthest = best_i
    return centroids


def knn_indices(points: Sequence[Point], centers: Sequence[Point], nsample: int) -> List[List[int]]:
    """For each centre, the ``nsample`` nearest point indices (by sq. distance).

    Mirrors ``knn_point`` = ``topk(sqrdists, nsample, largest=False)``.  Ties are
    broken by ascending point index (``sorted`` key ``(dist, index)``), which the
    unordered CUDA topk does not guarantee but which makes this deterministic.
    If ``nsample`` exceeds the cloud size the neighbourhood is padded by repeating
    the single nearest point (constant padding, as PointBERT never groups more
    neighbours than points).
    """
    n = len(points)
    if n == 0:
        raise ValueError("cannot group an empty point cloud")
    out: List[List[int]] = []
    for c in centers:
        order = sorted(range(n), key=lambda j: (_sq_dist3(points[j], c), j))
        if nsample <= n:
            out.append(order[:nsample])
        else:
            padded = order[:] + [order[0]] * (nsample - n)
            out.append(padded)
    return out


class Patches:
    """Result of tokenising a point cloud into local patches.

    Attributes
    ----------
    neighborhoods
        ``num_group`` patches, each a list of ``group_size`` neighbour points;
        the xyz of every neighbour has had its centre subtracted (local frame),
        extra channels are carried unchanged.
    centers
        the ``num_group`` FPS centre points (full original coordinates).
    center_indices
        indices (into the input cloud) of the centres, in pick order.
    neighbor_indices
        for each patch, the indices (into the input cloud) of its neighbours.
    """

    __slots__ = ("neighborhoods", "centers", "center_indices", "neighbor_indices")

    def __init__(self, neighborhoods, centers, center_indices, neighbor_indices):
        self.neighborhoods = neighborhoods
        self.centers = centers
        self.center_indices = center_indices
        self.neighbor_indices = neighbor_indices

    @property
    def num_group(self) -> int:
        return len(self.centers)

    @property
    def group_size(self) -> int:
        return len(self.neighborhoods[0]) if self.neighborhoods else 0


def group_patches(
    points: Sequence[Point],
    num_group: int,
    group_size: int,
    *,
    seed: int = 0,
) -> Patches:
    """Full PointBERT patch tokeniser: FPS centres + kNN grouping + centring.

    This is the deterministic core of ``Group.forward``: FPS ``num_group``
    centres, kNN ``group_size`` neighbours per centre, then subtract each centre
    from its neighbours' xyz.  Returns a :class:`Patches`.
    """
    if num_group <= 0:
        raise ValueError("num_group must be positive")
    if group_size <= 0:
        raise ValueError("group_size must be positive")
    center_idx = farthest_point_sampling(points, num_group, seed=seed)
    centers = [list(points[i]) for i in center_idx]
    neigh_idx = knn_indices(points, centers, group_size)
    neighborhoods: List[List[List[float]]] = []
    for g, (c, idxs) in enumerate(zip(centers, neigh_idx)):
        patch: List[List[float]] = []
        for j in idxs:
            p = list(points[j])
            # normalise only the xyz coordinates
            p[0] -= c[0]
            p[1] -= c[1]
            p[2] -= c[2]
            patch.append(p)
        neighborhoods.append(patch)
    return Patches(neighborhoods, centers, list(center_idx), neigh_idx)
