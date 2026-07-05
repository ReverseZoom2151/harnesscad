"""Pairwise-edge similarity evaluation protocol (Cluster3D external indices).

Xiang, Tseng, Wen et al., *Evaluating Deep Clustering Algorithms on
Non-Categorical 3D CAD Models*, Sections 3.1 / 4.2 / Appendix A.7.

The paper models the dataset as an undirected complete graph ``G(V, E)`` whose
edge labels ``e_ij in {+1, -1, 0}`` mean similar / dissimilar / unknown. A
clustering result is turned into a *dense* similarity matrix (``+1`` for a pair
inside the same cluster, ``-1`` across clusters), and evaluated against a
ground-truth / reference edge set over the *known* edges only. Two external
indices are defined:

* Edge accuracy (Appendix A.7): binary similarity-classification accuracy,
  the correlation-clustering metric, ``acc = 1 - sum|e_hat - e| / (2 n(E'))``.
* Balanced accuracy (Section 4.2, Brodersen et al.): mean of the true-positive
  and true-negative rates, robust to the heavy negative-edge imbalance of CAD
  similarity graphs.

This module builds those edges and indices deterministically:

* :func:`partition_to_edges` -- clustering labels -> ``{(i, j): +/-1}`` edge set.
* :func:`partition_to_similarity_matrix` -- the dense ``+/-1`` matrix (Fig. 11).
* :func:`known_edges` -- keep only ``+1``/``-1`` reference edges (drop ``0``).
* :func:`edge_accuracy` -- Appendix A.7 correlation-clustering accuracy.
* :func:`edge_confusion_matrix` -- TP/FP/FN/TN over known edges (positive =
  similar).
* :func:`balanced_accuracy` -- ``0.5 (TPR + TNR)``.

Stdlib only, deterministic. Edges are keyed by ordered index pairs ``(i, j)``
with ``i < j``; labels are ``+1`` / ``-1`` (and ``0`` = unknown in references).
"""

from __future__ import annotations

from typing import Dict, List, Sequence, Tuple

Edge = Tuple[int, int]
EdgeSet = Dict[Edge, int]


def partition_to_edges(labels: Sequence[int]) -> EdgeSet:
    """Convert a partition into a dense edge set ``{(i, j): +1/-1}`` for i < j.

    ``+1`` when items ``i`` and ``j`` share a cluster, ``-1`` otherwise -- the
    ``e_ij = +1 <=> v_i, v_j in C_k`` rule of Section 4.3.
    """
    n = len(labels)
    edges: EdgeSet = {}
    for i in range(n):
        for j in range(i + 1, n):
            edges[(i, j)] = 1 if labels[i] == labels[j] else -1
    return edges


def partition_to_similarity_matrix(labels: Sequence[int]) -> List[List[int]]:
    """Dense symmetric ``+1``/``-1`` similarity matrix (diagonal ``+1``)."""
    n = len(labels)
    matrix = [[1] * n for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            value = 1 if labels[i] == labels[j] else -1
            matrix[i][j] = value
            matrix[j][i] = value
    return matrix


def known_edges(reference: EdgeSet) -> EdgeSet:
    """Return only edges with a known label (``+1`` or ``-1``); drop ``0``."""
    return {edge: label for edge, label in reference.items() if label in (1, -1)}


def _normalise_key(edge: Edge) -> Edge:
    i, j = edge
    return (i, j) if i <= j else (j, i)


def edge_accuracy(predicted: EdgeSet, reference: EdgeSet) -> float:
    """Correlation-clustering edge accuracy over the reference's known edges.

    Evaluated only on edges whose reference label is ``+1``/``-1``. Since labels
    are in ``{+1, -1}``, ``|e_hat - e|`` is 0 when they agree and 2 otherwise, so
    ``acc = 1 - sum|e_hat - e| / (2 n(E'))`` reduces to the agreement fraction
    (Appendix A.7). Raises ``ValueError`` if there are no known reference edges
    or a needed prediction is missing.
    """
    known = known_edges(reference)
    if not known:
        raise ValueError("reference has no known (+/-1) edges")
    disagreement = 0
    for edge, ref_label in known.items():
        key = _normalise_key(edge)
        if key not in predicted:
            raise ValueError("prediction missing edge %r" % (key,))
        disagreement += abs(predicted[key] - ref_label)
    return 1.0 - disagreement / (2 * len(known))


def edge_confusion_matrix(predicted: EdgeSet, reference: EdgeSet) -> Dict[str, int]:
    """TP/FP/FN/TN counts over the reference's known edges.

    Positive class = *similar* (``+1``). ``tp``: both ``+1``; ``tn``: both
    ``-1``; ``fp``: predicted ``+1`` but reference ``-1``; ``fn``: predicted
    ``-1`` but reference ``+1``.
    """
    known = known_edges(reference)
    if not known:
        raise ValueError("reference has no known (+/-1) edges")
    tp = fp = fn = tn = 0
    for edge, ref_label in known.items():
        key = _normalise_key(edge)
        if key not in predicted:
            raise ValueError("prediction missing edge %r" % (key,))
        pred = predicted[key]
        if ref_label == 1 and pred == 1:
            tp += 1
        elif ref_label == -1 and pred == -1:
            tn += 1
        elif ref_label == -1 and pred == 1:
            fp += 1
        else:
            fn += 1
    return {"tp": tp, "fp": fp, "fn": fn, "tn": tn}


def balanced_accuracy(predicted: EdgeSet, reference: EdgeSet) -> float:
    """Balanced accuracy ``0.5 (TPR + TNR)`` over known edges (Section 4.2).

    Robust to the CAD graph's negative-edge imbalance. A class with no reference
    support contributes its rate as ``0`` (only the present class is averaged in
    when the other is absent). Requires at least one positive or negative edge.
    """
    cm = edge_confusion_matrix(predicted, reference)
    pos = cm["tp"] + cm["fn"]
    neg = cm["tn"] + cm["fp"]
    if pos == 0 and neg == 0:
        raise ValueError("no known edges to score")
    if pos == 0:
        return cm["tn"] / neg
    if neg == 0:
        return cm["tp"] / pos
    tpr = cm["tp"] / pos
    tnr = cm["tn"] / neg
    return 0.5 * (tpr + tnr)
