"""Classic clustering-inference algorithms for the Cluster3D benchmark.

Xiang, Tseng, Wen et al., *Evaluating Deep Clustering Algorithms on
Non-Categorical 3D CAD Models*. The two-stage baselines apply a *classic*
clustering algorithm (K-means) on top of learned features, and the annotation
workflow oversegments with K-means. The learned encoders are research-heavy /
external, but the classic partitioners are fully deterministic. A shuffle-seeded
Lloyd K-means already exists in :mod:`bench.contrastcad_latent_metrics`; this
module adds the algorithms that are *not* present:

* :func:`kmeans_plus_plus` -- K-means with k-means++ (D^2-weighted) seeding, the
  standard init the baselines actually use, plus Lloyd iterations.
* :func:`agglomerative_clustering` -- bottom-up hierarchical clustering with
  single / complete / average linkage, working from points or a distance matrix.
* :func:`spectral_clustering` -- spectral-lite: normalised-Laplacian embedding
  (via a self-contained Jacobi eigensolver) followed by k-means, given an
  affinity matrix.

Stdlib only, deterministic: all randomness flows through ``random.Random(seed)``,
ties break to the lowest index.
"""

from __future__ import annotations

import math
import random
from typing import List, Sequence, Tuple

Point = Sequence[float]


def _euclid(u: Point, v: Point) -> float:
    return math.sqrt(sum((a - b) ** 2 for a, b in zip(u, v)))


def _centroid(points: Sequence[Point]) -> List[float]:
    n = len(points)
    dim = len(points[0])
    return [sum(p[d] for p in points) / n for d in range(dim)]


def kmeans_plus_plus(points: Sequence[Point], k: int, seed,
                     max_iters: int = 100) -> Tuple[List[int], List[List[float]]]:
    """K-means with k-means++ seeding; returns ``(labels, centroids)``.

    Centres are chosen one at a time with probability proportional to the squared
    distance to the nearest already-chosen centre (Arthur & Vassilvitskii), using
    ``random.Random(seed)`` so the run is reproducible. Lloyd iterations then run
    until assignments stabilise or ``max_iters`` is reached; nearest-centroid
    ties break to the lowest centroid index.
    """
    if k <= 0:
        raise ValueError("k must be positive")
    if k > len(points):
        raise ValueError("k cannot exceed the number of points")
    rng = random.Random(seed)
    n = len(points)
    first = rng.randrange(n)
    centroids: List[List[float]] = [list(points[first])]
    while len(centroids) < k:
        dist_sq = []
        for p in points:
            best = min(_euclid(p, c) ** 2 for c in centroids)
            dist_sq.append(best)
        total = sum(dist_sq)
        if total == 0.0:
            # All remaining points coincide with a centre; pad deterministically.
            for idx in range(n):
                if len(centroids) >= k:
                    break
                centroids.append(list(points[idx]))
            break
        target = rng.random() * total
        cumulative = 0.0
        chosen = n - 1
        for idx, d in enumerate(dist_sq):
            cumulative += d
            if cumulative >= target:
                chosen = idx
                break
        centroids.append(list(points[chosen]))

    labels = [0] * n
    for _ in range(max_iters):
        changed = False
        for idx, point in enumerate(points):
            best, best_d = 0, math.inf
            for c, centroid in enumerate(centroids):
                d = _euclid(point, centroid)
                if d < best_d:
                    best, best_d = c, d
            if labels[idx] != best:
                changed = True
            labels[idx] = best
        for c in range(k):
            members = [points[i] for i in range(n) if labels[i] == c]
            if members:
                centroids[c] = _centroid(members)
        if not changed:
            break
    return labels, centroids


def _distance_matrix(points: Sequence[Point]) -> List[List[float]]:
    n = len(points)
    matrix = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            d = _euclid(points[i], points[j])
            matrix[i][j] = d
            matrix[j][i] = d
    return matrix


def agglomerative_clustering(k: int, points: Sequence[Point] = None,
                             distances: Sequence[Sequence[float]] = None,
                             linkage: str = "average") -> List[int]:
    """Bottom-up hierarchical clustering into ``k`` clusters.

    Supply either ``points`` (Euclidean distances computed internally) or a
    precomputed symmetric ``distances`` matrix. ``linkage`` is ``single``,
    ``complete`` or ``average``. Merges the closest pair of clusters repeatedly;
    ties break to the lowest-index pair. Returns contiguous labels ``0..k-1``
    assigned in order of first appearance for determinism.
    """
    if (points is None) == (distances is None):
        raise ValueError("supply exactly one of points or distances")
    if points is not None:
        dist = _distance_matrix(points)
    else:
        dist = [list(row) for row in distances]
    n = len(dist)
    if k <= 0 or k > n:
        raise ValueError("k must be in 1..n")
    if linkage not in ("single", "complete", "average"):
        raise ValueError("linkage must be single, complete or average")

    clusters: List[List[int]] = [[i] for i in range(n)]

    def cluster_distance(a: List[int], b: List[int]) -> float:
        vals = [dist[i][j] for i in a for j in b]
        if linkage == "single":
            return min(vals)
        if linkage == "complete":
            return max(vals)
        return sum(vals) / len(vals)

    while len(clusters) > k:
        best_pair = (0, 1)
        best_d = math.inf
        for a in range(len(clusters)):
            for b in range(a + 1, len(clusters)):
                d = cluster_distance(clusters[a], clusters[b])
                if d < best_d:
                    best_d = d
                    best_pair = (a, b)
        a, b = best_pair
        clusters[a] = clusters[a] + clusters[b]
        del clusters[b]

    labels = [0] * n
    for cid, members in enumerate(clusters):
        for idx in members:
            labels[idx] = cid
    return labels


def jacobi_eigen(matrix: Sequence[Sequence[float]],
                 max_sweeps: int = 100,
                 tol: float = 1e-12) -> Tuple[List[float], List[List[float]]]:
    """Eigenvalues/eigenvectors of a real symmetric matrix (Jacobi rotations).

    Returns ``(eigenvalues, eigenvectors)`` where ``eigenvectors[i]`` is the
    ``i``-th eigenvector (column form: ``eigenvectors[row][i]``), sorted by
    ascending eigenvalue. Deterministic; no external libraries.
    """
    n = len(matrix)
    a = [[float(matrix[i][j]) for j in range(n)] for i in range(n)]
    v = [[1.0 if i == j else 0.0 for j in range(n)] for i in range(n)]
    for _ in range(max_sweeps):
        off = 0.0
        for p in range(n):
            for q in range(p + 1, n):
                off += a[p][q] ** 2
        if off <= tol:
            break
        for p in range(n):
            for q in range(p + 1, n):
                if abs(a[p][q]) <= tol:
                    continue
                theta = (a[q][q] - a[p][p]) / (2.0 * a[p][q])
                t = math.copysign(1.0, theta) / (abs(theta) + math.sqrt(theta * theta + 1.0))
                if theta == 0.0:
                    t = 1.0
                c = 1.0 / math.sqrt(t * t + 1.0)
                s = t * c
                for i in range(n):
                    aip, aiq = a[i][p], a[i][q]
                    a[i][p] = c * aip - s * aiq
                    a[i][q] = s * aip + c * aiq
                for i in range(n):
                    api, aqi = a[p][i], a[q][i]
                    a[p][i] = c * api - s * aqi
                    a[q][i] = s * api + c * aqi
                for i in range(n):
                    vip, viq = v[i][p], v[i][q]
                    v[i][p] = c * vip - s * viq
                    v[i][q] = s * vip + c * viq
    eigenvalues = [a[i][i] for i in range(n)]
    order = sorted(range(n), key=lambda i: eigenvalues[i])
    sorted_vals = [eigenvalues[i] for i in order]
    sorted_vecs = [[v[row][i] for i in order] for row in range(n)]
    return sorted_vals, sorted_vecs


def spectral_clustering(affinity: Sequence[Sequence[float]], k: int, seed,
                        max_iters: int = 100) -> List[int]:
    """Spectral-lite clustering given a symmetric non-negative affinity matrix.

    Builds the normalised Laplacian ``L_sym = I - D^{-1/2} A D^{-1/2}``, takes the
    ``k`` eigenvectors of smallest eigenvalue as a spectral embedding (via
    :func:`jacobi_eigen`), row-normalises them (Ng-Jordan-Weiss), and runs
    :func:`kmeans_plus_plus` on the embedding. Deterministic given ``seed``.
    """
    n = len(affinity)
    if k <= 0 or k > n:
        raise ValueError("k must be in 1..n")
    degrees = [sum(affinity[i]) for i in range(n)]
    d_inv_sqrt = [1.0 / math.sqrt(d) if d > 0 else 0.0 for d in degrees]
    lap = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            norm = d_inv_sqrt[i] * affinity[i][j] * d_inv_sqrt[j]
            lap[i][j] = (1.0 if i == j else 0.0) - norm
    _, vecs = jacobi_eigen(lap)
    embedding: List[List[float]] = []
    for row in range(n):
        vec = [vecs[row][i] for i in range(k)]
        norm = math.sqrt(sum(x * x for x in vec))
        if norm > 0:
            vec = [x / norm for x in vec]
        embedding.append(vec)
    labels, _ = kmeans_plus_plus(embedding, k, seed, max_iters=max_iters)
    return labels
