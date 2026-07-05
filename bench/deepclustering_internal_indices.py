"""Internal clustering-validity indices for the Cluster3D benchmark.

Xiang, Tseng, Wen et al., *Evaluating Deep Clustering Algorithms on
Non-Categorical 3D CAD Models*. Section 4.2 uses internal evaluation indices
(the silhouette coefficient) to rank baselines without class labels. Silhouette
already lives in :mod:`bench.contrastcad_latent_metrics`; this module adds the
other classic internal indices that a non-categorical benchmark needs and that
require no ground-truth labels:

* :func:`davies_bouldin_index` -- lower is better; mean worst-case
  within/between-cluster scatter ratio.
* :func:`calinski_harabasz_index` -- higher is better; between/within dispersion
  ratio (variance-ratio criterion).
* :func:`dunn_index` -- higher is better; min inter-cluster gap over max cluster
  diameter.

Stdlib only, deterministic. Points are equal-length float vectors; labels align.
"""

from __future__ import annotations

import math
from typing import Dict, List, Sequence

Point = Sequence[float]


def _euclid(u: Point, v: Point) -> float:
    return math.sqrt(sum((a - b) ** 2 for a, b in zip(u, v)))


def _centroid(points: Sequence[Point]) -> List[float]:
    n = len(points)
    dim = len(points[0])
    return [sum(p[d] for p in points) / n for d in range(dim)]


def _by_label(points: Sequence[Point], labels: Sequence[int]) -> Dict[int, List[Point]]:
    if len(points) != len(labels):
        raise ValueError("points and labels must align")
    if len(points) == 0:
        raise ValueError("no points supplied")
    groups: Dict[int, List[Point]] = {}
    for point, label in zip(points, labels):
        groups.setdefault(label, []).append(point)
    if len(groups) < 2:
        raise ValueError("need at least two clusters")
    return groups


def davies_bouldin_index(points: Sequence[Point], labels: Sequence[int]) -> float:
    """Davies-Bouldin index (lower is better).

    For cluster ``k`` with centroid ``c_k``, ``S_k`` is the mean distance of its
    members to ``c_k``. For clusters ``i != j``,
    ``R_ij = (S_i + S_j) / d(c_i, c_j)``; the index is the mean over clusters of
    ``max_{j != i} R_ij``. Coincident centroids in distinct clusters raise
    ``ValueError`` (ratio undefined).
    """
    groups = _by_label(points, labels)
    keys = list(groups.keys())
    centroids = {k: _centroid(groups[k]) for k in keys}
    scatter = {k: sum(_euclid(p, centroids[k]) for p in groups[k]) / len(groups[k])
               for k in keys}
    total = 0.0
    for i in keys:
        worst = 0.0
        for j in keys:
            if i == j:
                continue
            d = _euclid(centroids[i], centroids[j])
            if d == 0.0:
                raise ValueError("distinct clusters share a centroid")
            r = (scatter[i] + scatter[j]) / d
            worst = max(worst, r)
        total += worst
    return total / len(keys)


def calinski_harabasz_index(points: Sequence[Point], labels: Sequence[int]) -> float:
    """Calinski-Harabasz variance-ratio criterion (higher is better).

    ``CH = (B / (k - 1)) / (W / (n - k))`` where ``B`` is the between-cluster
    dispersion (sum of ``n_k ||c_k - c||^2``) and ``W`` the within-cluster
    dispersion (sum of squared member-to-centroid distances). Returns ``inf``
    when clusters are perfectly separated (``W = 0``).
    """
    groups = _by_label(points, labels)
    n = len(points)
    k = len(groups)
    if n <= k:
        raise ValueError("need more points than clusters")
    overall = _centroid(points)
    between = 0.0
    within = 0.0
    for members in groups.values():
        c = _centroid(members)
        between += len(members) * (_euclid(c, overall) ** 2)
        within += sum(_euclid(p, c) ** 2 for p in members)
    if within == 0.0:
        return math.inf
    return (between / (k - 1)) / (within / (n - k))


def dunn_index(points: Sequence[Point], labels: Sequence[int]) -> float:
    """Dunn index (higher is better).

    Ratio of the minimum inter-cluster distance (closest pair of points in
    different clusters) to the maximum intra-cluster diameter (farthest pair of
    points in the same cluster). Returns ``inf`` when every cluster is a
    singleton (max diameter zero).
    """
    groups = _by_label(points, labels)
    keys = list(groups.keys())
    max_diameter = 0.0
    for members in groups.values():
        for i in range(len(members)):
            for j in range(i + 1, len(members)):
                max_diameter = max(max_diameter, _euclid(members[i], members[j]))
    min_sep = math.inf
    for a in range(len(keys)):
        for b in range(a + 1, len(keys)):
            for p in groups[keys[a]]:
                for q in groups[keys[b]]:
                    min_sep = min(min_sep, _euclid(p, q))
    if max_diameter == 0.0:
        return math.inf
    return min_sep / max_diameter
