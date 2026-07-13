"""Ensemble-based clustering evaluation protocol (Cluster3D, Section 4.3).

Xiang, Tseng, Wen et al., *Evaluating Deep Clustering Algorithms on
Non-Categorical 3D CAD Models*.

Because a fully annotated ground-truth similarity matrix is intractable, the
paper's central contribution is a *relative, dynamic, democratic* evaluation:
combine many similarity matrices (from baseline clusterings and/or human
annotators) into a proxy ground truth by **majority voting**, then rank methods
by their balanced accuracy against that proxy.

This module implements that protocol deterministically:

* :func:`ensemble_by_majority_vote` -- combine ``N`` dense clustering edge sets
  into the ``Ensemble`` matrix; an edge is ``+1`` iff its positive count
  ``>= ceil((N + 1) / 2)``, otherwise ``-1`` (Section 4.3).
* :func:`human_ensemble` -- combine ``N`` (possibly sparse) human annotation edge
  sets; ties (equal ``+1``/``-1`` votes) become ``0`` = unknown, as with the
  four-vs-four annotator case.
* :func:`ensemble_human_balanced_accuracy` -- the ``EnsembleHuman`` value: the
  average of a method's balanced accuracy against ``Ensemble`` and against
  ``Human`` (Section 4.3).
* :func:`rank_methods` -- rank baselines by a score (descending).
* :func:`ranking_agreement` / :func:`kendall_tau` -- quantify how consistent two
  rankings are, the "ranking consistency" evidence of Section 5.

Stdlib only, deterministic. Edge sets are ``{(i, j): +1/-1/0}`` dicts keyed by
``i < j`` (see :mod:`bench.deepclustering_edge_protocol`).
"""

from __future__ import annotations

import math
from typing import Dict, List, Sequence, Tuple

from harnesscad.eval.bench.deepclustering_edge_protocol import balanced_accuracy

Edge = Tuple[int, int]
EdgeSet = Dict[Edge, int]


def _all_edges(edge_sets: Sequence[EdgeSet]) -> List[Edge]:
    keys = set()
    for es in edge_sets:
        keys.update(es.keys())
    return sorted(keys)


def ensemble_by_majority_vote(edge_sets: Sequence[EdgeSet]) -> EdgeSet:
    """Combine ``N`` dense clustering edge sets into the Ensemble matrix.

    For each edge, count the positive (``+1``) votes among the ``N`` matrices;
    the final label is ``+1`` iff that count ``>= ceil((N + 1) / 2)``, else
    ``-1`` (Section 4.3 majority voting). Any missing edge in a set counts as a
    non-positive vote. Requires at least one edge set.
    """
    n = len(edge_sets)
    if n == 0:
        raise ValueError("need at least one edge set")
    threshold = math.ceil((n + 1) / 2)
    result: EdgeSet = {}
    for edge in _all_edges(edge_sets):
        positives = sum(1 for es in edge_sets if es.get(edge, 0) == 1)
        result[edge] = 1 if positives >= threshold else -1
    return result


def human_ensemble(edge_sets: Sequence[EdgeSet]) -> EdgeSet:
    """Combine ``N`` human annotation edge sets; ties become ``0`` (unknown).

    Counts ``+1`` and ``-1`` votes per edge (``0``/absent contribute nothing).
    The final label is the sign of the majority; an exact tie (e.g. four ``+1``
    and four ``-1``) is ``0`` = unknown, matching the paper's handling of split
    annotators. Edges with no ``+/-1`` votes at all are ``0``.
    """
    if not edge_sets:
        raise ValueError("need at least one edge set")
    result: EdgeSet = {}
    for edge in _all_edges(edge_sets):
        pos = sum(1 for es in edge_sets if es.get(edge, 0) == 1)
        neg = sum(1 for es in edge_sets if es.get(edge, 0) == -1)
        if pos > neg:
            result[edge] = 1
        elif neg > pos:
            result[edge] = -1
        else:
            result[edge] = 0
    return result


def ensemble_human_balanced_accuracy(predicted: EdgeSet,
                                     ensemble: EdgeSet,
                                     human: EdgeSet) -> float:
    """EnsembleHuman score: mean of balanced accuracy vs Ensemble and vs Human.

    Section 4.3 averages the two balanced-accuracy values to form the
    ``EnsembleHuman`` reference score for a baseline method.
    """
    ba_ensemble = balanced_accuracy(predicted, ensemble)
    ba_human = balanced_accuracy(predicted, human)
    return 0.5 * (ba_ensemble + ba_human)


def rank_methods(scores: Dict[str, float]) -> List[str]:
    """Rank method names by score, highest first; ties break by name for
    determinism."""
    if not scores:
        raise ValueError("no scores to rank")
    return [name for name, _ in sorted(scores.items(), key=lambda kv: (-kv[1], kv[0]))]


def ranking_agreement(ranking_a: Sequence[str], ranking_b: Sequence[str]) -> float:
    """Fraction of method pairs ordered the same way in both rankings.

    This is the pairwise-concordance measure of ranking consistency the paper
    uses to argue its ensemble protocol yields a stable baseline order across
    references (Section 5). Both rankings must cover the same set of names.
    Returns ``1.0`` for identical orderings.
    """
    set_a = set(ranking_a)
    set_b = set(ranking_b)
    if set_a != set_b:
        raise ValueError("rankings must cover the same methods")
    names = list(ranking_a)
    pos_a = {name: i for i, name in enumerate(ranking_a)}
    pos_b = {name: i for i, name in enumerate(ranking_b)}
    total = 0
    concordant = 0
    for x in range(len(names)):
        for y in range(x + 1, len(names)):
            a, b = names[x], names[y]
            total += 1
            same = (pos_a[a] < pos_a[b]) == (pos_b[a] < pos_b[b])
            if same:
                concordant += 1
    if total == 0:
        return 1.0
    return concordant / total


def kendall_tau(ranking_a: Sequence[str], ranking_b: Sequence[str]) -> float:
    """Kendall's tau in ``[-1, 1]`` between two full rankings of the same set.

    ``tau = (concordant - discordant) / total_pairs``. ``+1`` identical order,
    ``-1`` fully reversed. Derived from :func:`ranking_agreement` (which is the
    concordant fraction).
    """
    agree = ranking_agreement(ranking_a, ranking_b)
    return 2.0 * agree - 1.0
