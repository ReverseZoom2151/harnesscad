"""Earth Mover's Distance reconstruction metric.

Reconstruction fidelity is evaluated with three point-cloud metrics:
Chamfer Distance (CD), Hausdorff Distance (HD) and Earth Mover's Distance (EMD).
CD and HD are already implemented in ``geometry.dreamcad_metrics``; this module
adds the missing EMD, defined as the minimum-cost bijection between two
equal-cardinality point sets::

    EMD(A, B) = min_{phi: A -> B bijection}  sum_{a in A} || a - phi(a) ||

The optimal bijection is computed exactly and deterministically with the
Hungarian (Kuhn-Munkres) assignment algorithm over the pairwise Euclidean cost
matrix -- no learned model, no randomness, no wall clock. This is a self-contained
exact solver for modest clouds (O(n^3)); it is intended as an evaluation metric,
not a training-time optimal-transport tool.
"""

from __future__ import annotations

from math import sqrt
from typing import List, Sequence, Tuple

Point = Sequence[float]


def _euclidean(a: Point, b: Point) -> float:
    return sqrt(sum((float(a[d]) - float(b[d])) ** 2 for d in range(len(a))))


def cost_matrix(cloud_a: Sequence[Point], cloud_b: Sequence[Point]) -> List[List[float]]:
    """Pairwise Euclidean cost matrix ``C[i][j] = ||a_i - b_j||``."""
    return [[_euclidean(a, b) for b in cloud_b] for a in cloud_a]


def hungarian(cost: Sequence[Sequence[float]]) -> List[int]:
    """Solve the square assignment problem; return ``col`` for each ``row``.

    Exact O(n^3) Kuhn-Munkres using the potentials / augmenting-path formulation.
    ``assignment[i] = j`` means row ``i`` is matched to column ``j`` and the total
    ``sum(cost[i][assignment[i]])`` is minimised. Requires a square matrix.
    """
    n = len(cost)
    if any(len(row) != n for row in cost):
        raise ValueError("cost matrix must be square")
    if n == 0:
        return []
    INF = float("inf")
    # 1-indexed potentials; u over rows, v over columns.
    u = [0.0] * (n + 1)
    v = [0.0] * (n + 1)
    p = [0] * (n + 1)   # p[j] = row assigned to column j
    way = [0] * (n + 1)
    for i in range(1, n + 1):
        p[0] = i
        j0 = 0
        minv = [INF] * (n + 1)
        used = [False] * (n + 1)
        while True:
            used[j0] = True
            i0 = p[j0]
            delta = INF
            j1 = -1
            for j in range(1, n + 1):
                if not used[j]:
                    cur = cost[i0 - 1][j - 1] - u[i0] - v[j]
                    if cur < minv[j]:
                        minv[j] = cur
                        way[j] = j0
                    if minv[j] < delta:
                        delta = minv[j]
                        j1 = j
            for j in range(n + 1):
                if used[j]:
                    u[p[j]] += delta
                    v[j] -= delta
                else:
                    minv[j] -= delta
            j0 = j1
            if p[j0] == 0:
                break
        while j0:
            j1 = way[j0]
            p[j0] = p[j1]
            j0 = j1
    assignment = [0] * n
    for j in range(1, n + 1):
        if p[j] != 0:
            assignment[p[j] - 1] = j - 1
    return assignment


def earth_movers_distance(cloud_a: Sequence[Point],
                          cloud_b: Sequence[Point]) -> float:
    """Total EMD (Eq. 21): min-cost bijection sum over equal-size clouds.

    The two clouds must be non-empty and have the same number of points (EMD as
    it is defined as a bijection). Returns the summed transport cost.
    """
    if not cloud_a or not cloud_b:
        raise ValueError("both clouds must be non-empty")
    if len(cloud_a) != len(cloud_b):
        raise ValueError("EMD requires equal-cardinality clouds")
    c = cost_matrix(cloud_a, cloud_b)
    assignment = hungarian(c)
    return sum(c[i][assignment[i]] for i in range(len(cloud_a)))


def mean_emd(cloud_a: Sequence[Point], cloud_b: Sequence[Point]) -> float:
    """Per-point mean EMD (``earth_movers_distance / n``)."""
    return earth_movers_distance(cloud_a, cloud_b) / len(cloud_a)
