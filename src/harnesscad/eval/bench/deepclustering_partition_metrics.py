"""External clustering-comparison metrics for non-categorical CAD clustering.

Xiang, Tseng, Wen et al., *Evaluating Deep Clustering Algorithms on
Non-Categorical 3D CAD Models* (Cluster3D). The paper argues that the standard
"external evaluation principles ... based on class labels" oversimplify
inter/intra-cluster similarity, yet it still relies on external indices that
compare two partitions. This module implements the classic label-agnostic
partition-comparison indices that any such benchmark needs and that are *not*
already present in :mod:`bench.contrastcad_latent_metrics` (which only provides
silhouette / SSE / k-means):

* :func:`contingency_table` -- the |U| x |V| co-occurrence counts.
* :func:`entropy` / :func:`mutual_information` -- information-theoretic building
  blocks (natural log; base-invariant in the normalised forms).
* :func:`normalized_mutual_information` -- NMI with ``arithmetic``, ``geometric``,
  ``min`` or ``max`` denominator conventions.
* :func:`rand_index` / :func:`adjusted_rand_index` -- pair-counting agreement,
  ARI chance-corrected.
* :func:`clustering_accuracy` -- best label permutation via a self-contained
  Hungarian (Kuhn-Munkres) assignment, the ``ACC`` metric used across the deep
  clustering literature the paper benchmarks.
* :func:`purity` -- majority-label purity.

Stdlib only, deterministic (no randomness). Labels are arbitrary hashables.
"""

from __future__ import annotations

import math
from typing import Dict, Hashable, List, Sequence, Tuple

Label = Hashable


def _comb2(n: int) -> int:
    """Number of unordered pairs C(n, 2)."""
    return n * (n - 1) // 2


def _check(a: Sequence[Label], b: Sequence[Label]) -> None:
    if len(a) != len(b):
        raise ValueError("label sequences must have equal length")
    if not a:
        raise ValueError("label sequences must be non-empty")


def contingency_table(a: Sequence[Label],
                      b: Sequence[Label]) -> Tuple[List[List[int]], List[Label], List[Label]]:
    """Return ``(matrix, rows, cols)`` co-occurrence counts of two labellings.

    ``matrix[i][j]`` is the number of items with label ``rows[i]`` in ``a`` and
    ``cols[j]`` in ``b``. ``rows``/``cols`` are the sorted-by-first-appearance
    distinct labels so results are deterministic.
    """
    _check(a, b)
    rows: List[Label] = []
    cols: List[Label] = []
    row_idx: Dict[Label, int] = {}
    col_idx: Dict[Label, int] = {}
    for label in a:
        if label not in row_idx:
            row_idx[label] = len(rows)
            rows.append(label)
    for label in b:
        if label not in col_idx:
            col_idx[label] = len(cols)
            cols.append(label)
    matrix = [[0] * len(cols) for _ in range(len(rows))]
    for la, lb in zip(a, b):
        matrix[row_idx[la]][col_idx[lb]] += 1
    return matrix, rows, cols


def entropy(labels: Sequence[Label]) -> float:
    """Shannon entropy (natural log) of a labelling's cluster-size distribution."""
    _check(labels, labels)
    n = len(labels)
    counts: Dict[Label, int] = {}
    for label in labels:
        counts[label] = counts.get(label, 0) + 1
    h = 0.0
    for c in counts.values():
        p = c / n
        h -= p * math.log(p)
    return h


def mutual_information(a: Sequence[Label], b: Sequence[Label]) -> float:
    """Mutual information ``I(U; V)`` in nats between two labellings."""
    matrix, rows, cols = contingency_table(a, b)
    n = len(a)
    row_sums = [sum(r) for r in matrix]
    col_sums = [sum(matrix[i][j] for i in range(len(rows))) for j in range(len(cols))]
    mi = 0.0
    for i in range(len(rows)):
        for j in range(len(cols)):
            nij = matrix[i][j]
            if nij == 0:
                continue
            mi += (nij / n) * math.log((nij * n) / (row_sums[i] * col_sums[j]))
    return mi


def normalized_mutual_information(a: Sequence[Label], b: Sequence[Label],
                                  average: str = "arithmetic") -> float:
    """Normalised mutual information in ``[0, 1]`` (higher = more agreement).

    ``average`` picks the denominator: ``arithmetic`` (mean of the entropies),
    ``geometric`` (sqrt product), ``min`` or ``max``. When both labellings are a
    single cluster (both entropies zero) the partitions are trivially identical
    and NMI is defined as ``1.0``.
    """
    ha = entropy(a)
    hb = entropy(b)
    if ha == 0.0 and hb == 0.0:
        return 1.0
    mi = mutual_information(a, b)
    if average == "arithmetic":
        denom = (ha + hb) / 2.0
    elif average == "geometric":
        denom = math.sqrt(ha * hb)
    elif average == "min":
        denom = min(ha, hb)
    elif average == "max":
        denom = max(ha, hb)
    else:
        raise ValueError("average must be arithmetic, geometric, min or max")
    if denom == 0.0:
        return 0.0
    value = mi / denom
    # Clamp tiny floating error outside [0, 1].
    return max(0.0, min(1.0, value))


def rand_index(a: Sequence[Label], b: Sequence[Label]) -> float:
    """Rand index: fraction of item pairs that agree (same-or-different) in both."""
    matrix, rows, cols = contingency_table(a, b)
    n = len(a)
    sum_ij = sum(_comb2(matrix[i][j]) for i in range(len(rows)) for j in range(len(cols)))
    row_sums = [sum(r) for r in matrix]
    col_sums = [sum(matrix[i][j] for i in range(len(rows))) for j in range(len(cols))]
    sum_a = sum(_comb2(s) for s in row_sums)
    sum_b = sum(_comb2(s) for s in col_sums)
    total = _comb2(n)
    # agreements = same-same (sum_ij) + diff-diff
    diff_diff = total - sum_a - sum_b + sum_ij
    return (sum_ij + diff_diff) / total


def adjusted_rand_index(a: Sequence[Label], b: Sequence[Label]) -> float:
    """Adjusted Rand index: Rand index corrected for chance, in ``[-1, 1]``.

    ``1.0`` is identical clusterings; ``0.0`` is the expected value of a random
    labelling. When both partitions are trivial (one cluster each, or all
    singletons) ARI is defined as ``1.0``.
    """
    matrix, rows, cols = contingency_table(a, b)
    n = len(a)
    row_sums = [sum(r) for r in matrix]
    col_sums = [sum(matrix[i][j] for i in range(len(rows))) for j in range(len(cols))]
    sum_ij = sum(_comb2(matrix[i][j]) for i in range(len(rows)) for j in range(len(cols)))
    sum_a = sum(_comb2(s) for s in row_sums)
    sum_b = sum(_comb2(s) for s in col_sums)
    total = _comb2(n)
    if total == 0:
        return 1.0
    expected = (sum_a * sum_b) / total
    maximum = (sum_a + sum_b) / 2.0
    if maximum == expected:
        # Both partitions trivial; agreement is complete by convention.
        return 1.0
    return (sum_ij - expected) / (maximum - expected)


def _hungarian_min(cost: List[List[float]]) -> List[int]:
    """Kuhn-Munkres min-cost assignment on a square matrix; returns row->col."""
    n = len(cost)
    if n == 0:
        return []
    INF = float("inf")
    u = [0.0] * (n + 1)
    v = [0.0] * (n + 1)
    p = [0] * (n + 1)
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


def clustering_accuracy(predicted: Sequence[Label],
                        truth: Sequence[Label]) -> float:
    """Unsupervised clustering accuracy (ACC) via optimal label matching.

    Builds the predicted-vs-truth contingency table and finds the one-to-one
    label permutation that maximises the number of correctly matched items using
    a Hungarian assignment, returning ``matched / n`` in ``[0, 1]``. This is the
    standard ``ACC`` reported by the deep-clustering baselines the paper adapts.
    """
    _check(predicted, truth)
    matrix, rows, cols = contingency_table(predicted, truth)
    size = max(len(rows), len(cols))
    max_count = max((matrix[i][j] for i in range(len(rows)) for j in range(len(cols))),
                    default=0)
    # Pad to square and convert maximisation of counts into minimisation.
    cost = [[float(max_count)] * size for _ in range(size)]
    for i in range(len(rows)):
        for j in range(len(cols)):
            cost[i][j] = float(max_count - matrix[i][j])
    assignment = _hungarian_min(cost)
    matched = 0
    for i, j in enumerate(assignment):
        if i < len(rows) and j < len(cols):
            matched += matrix[i][j]
    return matched / len(predicted)


def purity(predicted: Sequence[Label], truth: Sequence[Label]) -> float:
    """Cluster purity: fraction of items in the majority truth-label of each
    predicted cluster (in ``[0, 1]``)."""
    matrix, rows, cols = contingency_table(predicted, truth)
    n = len(predicted)
    total = sum(max(matrix[i]) if matrix[i] else 0 for i in range(len(rows)))
    return total / n
