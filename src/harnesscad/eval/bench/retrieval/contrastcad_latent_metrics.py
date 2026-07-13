"""Latent-space representation-quality metrics for ContrastCAD.

Jung, Kim & Kim, *ContrastCAD* (2024), Section 5.4.

To show that contrastive learning clusters similar CAD models more tightly, the
paper evaluates the *unlabelled* latent space by (a) running K-means over the
latent vectors and scoring the resulting clusters with the silhouette coefficient
(SC, Eq. 9) and the within-cluster sum of squared error (SSE, Eq. 10), and (b)
measuring the Euclidean distance (ED, Eq. 11) between latent vectors of related
sequences -- e.g. the average ED from a query model to a hand-picked *similar set*
(0.67 for ContrastCAD vs 0.80 for DeepCAD, Section 5.4.5) and the SIM/ED between a
sequence and its permuted equivalent (Table 5).

Everything here is deterministic:

* :func:`euclidean_distance` -- ED (Eq. 11).
* :func:`silhouette_coefficient` -- mean silhouette over labelled points (Eq. 9).
* :func:`sse` -- within-cluster squared error to centroids (Eq. 10).
* :func:`kmeans` -- a seeded, fixed-iteration Lloyd's algorithm so cluster labels
  are reproducible (the paper uses K-means to assign labels before scoring).
* :func:`average_set_distance` / :func:`distance_matrix` -- the similar/dissimilar
  set analysis of Section 5.4.5 and the ED matrix of Fig. 8.

Stdlib only, no numpy. Learned encoders are out of scope; latent vectors are
supplied by the caller.
"""

from __future__ import annotations

import math
import random
from typing import List, Sequence, Tuple

Point = Sequence[float]


def euclidean_distance(u: Point, v: Point) -> float:
    """ED(u, v) = ||u - v||_2 (Eq. 11)."""
    if len(u) != len(v):
        raise ValueError("points must have equal dimension")
    return math.sqrt(sum((a - b) ** 2 for a, b in zip(u, v)))


def distance_matrix(points: Sequence[Point]) -> List[List[float]]:
    """Symmetric pairwise Euclidean-distance matrix (Fig. 8 similarity matrix)."""
    n = len(points)
    matrix = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            d = euclidean_distance(points[i], points[j])
            matrix[i][j] = d
            matrix[j][i] = d
    return matrix


def average_set_distance(query: Point, others: Sequence[Point]) -> float:
    """Mean ED from a query latent vector to a set of others (Section 5.4.5).

    This reproduces the paper's "average Euclidean distance in the latent space
    between a CAD model ... and other CAD models within the similar set".
    """
    if not others:
        raise ValueError("need at least one other point")
    return sum(euclidean_distance(query, o) for o in others) / len(others)


def _centroid(points: Sequence[Point]) -> List[float]:
    n = len(points)
    dim = len(points[0])
    return [sum(p[d] for p in points) / n for d in range(dim)]


def sse(points: Sequence[Point], labels: Sequence[int]) -> float:
    """Within-cluster sum of squared error to cluster centroids (Eq. 10).

    ``SSE = sum_i ||x_i - centroid(cluster(x_i))||^2``. Lower is better (tighter
    clusters). Empty clusters contribute nothing.
    """
    if len(points) != len(labels):
        raise ValueError("points and labels must align")
    clusters: dict = {}
    for point, label in zip(points, labels):
        clusters.setdefault(label, []).append(point)
    total = 0.0
    for members in clusters.values():
        c = _centroid(members)
        total += sum(euclidean_distance(p, c) ** 2 for p in members)
    return total


def silhouette_coefficient(points: Sequence[Point],
                           labels: Sequence[int]) -> float:
    """Mean silhouette coefficient over all points (Eq. 9).

    For point ``x_i``: ``a_i`` is its mean distance to other points in its own
    cluster and ``b_i`` the minimum over other clusters of the mean distance to
    that cluster; ``s_i = (b_i - a_i) / max(a_i, b_i)``. A singleton cluster point
    contributes 0. Returns the mean over all points, in ``[-1, 1]`` (higher is
    better). Requires at least two distinct clusters.
    """
    if len(points) != len(labels):
        raise ValueError("points and labels must align")
    n = len(points)
    if n == 0:
        raise ValueError("no points supplied")
    if len(set(labels)) < 2:
        raise ValueError("silhouette needs at least two clusters")

    by_label: dict = {}
    for idx, label in enumerate(labels):
        by_label.setdefault(label, []).append(idx)

    scores: List[float] = []
    for i in range(n):
        own = by_label[labels[i]]
        if len(own) == 1:
            scores.append(0.0)
            continue
        a_i = sum(euclidean_distance(points[i], points[j])
                  for j in own if j != i) / (len(own) - 1)
        b_i = math.inf
        for label, members in by_label.items():
            if label == labels[i]:
                continue
            mean = sum(euclidean_distance(points[i], points[j])
                       for j in members) / len(members)
            b_i = min(b_i, mean)
        denom = max(a_i, b_i)
        scores.append(0.0 if denom == 0.0 else (b_i - a_i) / denom)
    return sum(scores) / n


def kmeans(points: Sequence[Point], k: int, seed,
           max_iters: int = 100) -> Tuple[List[int], List[List[float]]]:
    """Deterministic Lloyd's K-means; returns ``(labels, centroids)``.

    Centroids are seeded from a shuffled copy of the points (k-means with a fixed
    seed), then Lloyd iterations run until assignments stabilise or ``max_iters``.
    Ties in nearest-centroid assignment break to the lowest index, so the labels
    are reproducible. This supplies the cluster labels the paper feeds to SC/SSE.
    """
    if k <= 0:
        raise ValueError("k must be positive")
    if k > len(points):
        raise ValueError("k cannot exceed the number of points")
    rng = random.Random(seed)
    order = list(range(len(points)))
    rng.shuffle(order)
    centroids = [list(points[order[i]]) for i in range(k)]

    labels = [0] * len(points)
    for _ in range(max_iters):
        changed = False
        for idx, point in enumerate(points):
            best, best_d = 0, math.inf
            for c, centroid in enumerate(centroids):
                d = euclidean_distance(point, centroid)
                if d < best_d:
                    best, best_d = c, d
            if labels[idx] != best:
                changed = True
            labels[idx] = best
        for c in range(k):
            members = [points[i] for i in range(len(points)) if labels[i] == c]
            if members:
                centroids[c] = _centroid(members)
        if not changed:
            break
    return labels, centroids
