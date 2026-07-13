"""Cluster-initialisation / oversegmentation annotation protocol (Cluster3D).

Xiang, Tseng, Wen et al., *Evaluating Deep Clustering Algorithms on
Non-Categorical 3D CAD Models*, Section 3.1.1 and Section 3 (MVCNN / AtlasNet
based initialisation).

To make sparse human annotation tractable *and* balanced, the paper oversegments
the dataset into small initial clusters (at most ``T = 12`` models each) by
recursively splitting any cluster larger than ``T`` with K-means, then only
annotates edges *inside* the same initial cluster (all cross-cluster edges are
set to label ``0`` = unknown). This module reproduces that deterministic
sub-protocol (the learned features are supplied by the caller; only the
clustering/edge bookkeeping is here):

* :func:`oversegment` -- recursively K-means-split feature vectors until every
  cluster has ``<= max_size`` members; returns the list of index clusters.
* :func:`induced_known_edges` -- the edge subset the annotators would label:
  ``+1``/``-1`` inside initial clusters (from a ground-truth partition), ``0``
  across them. With no ground truth it just marks intra-cluster edges as known.
* :func:`annotation_budget_fraction` -- the fraction of the full
  ``|V|(|V|-1)/2`` edge matrix that ends up annotated (the paper's ~0.5%).

Stdlib only, deterministic (all randomness via ``random.Random(seed)``).
"""

from __future__ import annotations

from typing import Dict, List, Sequence, Tuple

from harnesscad.eval.bench.deepclustering_algorithms import kmeans_plus_plus

Point = Sequence[float]
Edge = Tuple[int, int]
EdgeSet = Dict[Edge, int]


def oversegment(points: Sequence[Point], max_size: int, seed,
                split_k: int = 2, max_iters: int = 100) -> List[List[int]]:
    """Recursively split points with K-means until each cluster has <= max_size.

    Starts from all indices as one cluster; any cluster with more than
    ``max_size`` members is re-clustered into ``split_k`` sub-clusters with
    :func:`kmeans_plus_plus` and the split recurses. Deterministic given
    ``seed``; the seed is perturbed per split so nested splits differ yet remain
    reproducible. Returns a list of index lists, each of size ``<= max_size``,
    partitioning ``range(len(points))``.
    """
    if max_size <= 0:
        raise ValueError("max_size must be positive")
    if split_k < 2:
        raise ValueError("split_k must be at least 2")
    result: List[List[int]] = []
    # Stack of (indices, depth) to process; deterministic order.
    stack: List[Tuple[List[int], int]] = [(list(range(len(points))), 0)]
    counter = 0
    while stack:
        indices, depth = stack.pop(0)
        if len(indices) <= max_size:
            if indices:
                result.append(indices)
            continue
        sub_points = [points[i] for i in indices]
        k = min(split_k, len(sub_points))
        sub_seed = "oversegment-%r-%d" % (seed, counter)
        labels, _ = kmeans_plus_plus(sub_points, k, sub_seed, max_iters=max_iters)
        counter += 1
        groups: Dict[int, List[int]] = {}
        for local, label in enumerate(labels):
            groups.setdefault(label, []).append(indices[local])
        non_empty = [g for g in groups.values() if g]
        if len(non_empty) <= 1:
            # Degenerate split (all points coincided): force a positional halve.
            mid = len(indices) // 2
            non_empty = [indices[:mid], indices[mid:]]
        for g in non_empty:
            stack.append((g, depth + 1))
    result.sort(key=lambda g: g[0])
    return result


def induced_known_edges(clusters: Sequence[Sequence[int]],
                        truth_labels: Sequence[int] = None) -> EdgeSet:
    """Edge set the annotators would label from the initial clusters.

    Only intra-cluster edges are "known". With ``truth_labels`` supplied, each
    intra-cluster edge gets ``+1`` when the two items share a truth label and
    ``-1`` otherwise (simulating expert similar/dissimilar decisions inside a
    cluster). Without truth, intra-cluster edges are marked ``+1`` (annotatable)
    and everything else stays absent (label ``0`` = unknown). Keys are ``i < j``.
    """
    edges: EdgeSet = {}
    for cluster in clusters:
        members = sorted(cluster)
        for a in range(len(members)):
            for b in range(a + 1, len(members)):
                i, j = members[a], members[b]
                if truth_labels is None:
                    edges[(i, j)] = 1
                else:
                    edges[(i, j)] = 1 if truth_labels[i] == truth_labels[j] else -1
    return edges


def annotation_budget_fraction(clusters: Sequence[Sequence[int]],
                               total_items: int) -> float:
    """Fraction of the full pairwise matrix that the intra-cluster edges cover.

    ``sum_k C(|C_k|, 2) / C(total_items, 2)`` -- the paper's ~0.5% annotation
    budget for ``T = 12`` clusters over ~23k models.
    """
    if total_items < 2:
        raise ValueError("need at least two items")
    annotated = sum(len(c) * (len(c) - 1) // 2 for c in clusters)
    full = total_items * (total_items - 1) // 2
    return annotated / full
