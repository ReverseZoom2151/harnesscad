"""Ranked-retrieval quality metrics: DCG/NDCG, MRR, top-k success rate, enrichment factor.

Domain-agnostic list-ranking primitives. Each function takes a ranking -- an ordered
sequence describing a single query's retrieved results from best to worst -- and returns a
deterministic scalar. They complement the recall@k / average-precision helpers already in
``bench`` (``tool_retrieval``, ``vision_metrics``) which do not cover graded-gain ranking
(DCG), reciprocal rank (MRR), success rate, or top-fraction enrichment.

Two relevance conventions are supported:

* Binary relevance -- pass an iterable of truthy/falsy flags (``relevances``), 1 = relevant.
* Graded relevance -- pass an iterable of non-negative gains for DCG/NDCG.

All functions are pure, stdlib-only, and deterministic. Ties are resolved by the caller's
ordering; these functions never re-sort a provided ranking (except ``ndcg`` which sorts a
*copy* of the gains to form the ideal ranking).
"""

from __future__ import annotations

import math


def _as_list(relevances):
    return [float(r) for r in relevances]


def dcg_at_k(gains, k=None):
    """Discounted Cumulative Gain over the first ``k`` ranked items.

    Uses the standard log2 discount: ``sum(gain_i / log2(i + 2))`` for 0-based rank ``i``.
    ``gains`` are graded relevance values in ranked order (best first). ``k=None`` uses all.
    """
    values = _as_list(gains)
    if k is not None:
        if k < 0:
            raise ValueError("k must be non-negative")
        values = values[:k]
    return sum(g / math.log2(i + 2) for i, g in enumerate(values))


def ndcg_at_k(gains, k=None):
    """Normalised DCG: ``dcg_at_k`` divided by the DCG of the ideal (sorted) ranking.

    Returns 0.0 when the ideal DCG is 0 (no positive gains present).
    """
    values = _as_list(gains)
    ideal = sorted(values, reverse=True)
    denom = dcg_at_k(ideal, k)
    if denom == 0.0:
        return 0.0
    return dcg_at_k(values, k) / denom


def reciprocal_rank(relevances):
    """Reciprocal of the 1-based rank of the first relevant item; 0.0 if none relevant."""
    for i, r in enumerate(relevances):
        if r:
            return 1.0 / (i + 1)
    return 0.0


def mean_reciprocal_rank(rankings):
    """Mean of ``reciprocal_rank`` over an iterable of per-query relevance rankings."""
    rows = [reciprocal_rank(r) for r in rankings]
    return sum(rows) / len(rows) if rows else 0.0


def success_at_k(relevances, k):
    """1.0 if at least one relevant item appears in the top ``k``, else 0.0.

    This is the per-query Top-k Success Rate (a.k.a. hit rate / recall@k>=1).
    """
    if k < 0:
        raise ValueError("k must be non-negative")
    return 1.0 if any(relevances[:k]) else 0.0


def success_rate_at_k(rankings, k):
    """Mean ``success_at_k`` over an iterable of per-query relevance rankings."""
    rows = [success_at_k(r, k) for r in rankings]
    return sum(rows) / len(rows) if rows else 0.0


def enrichment_factor(relevances, fraction, total_relevant=None):
    """Enrichment factor at a top ``fraction`` of the ranked list.

    EF = (hit_rate in the top fraction) / (hit_rate expected under a random ranking).
    Concretely ``(actives_in_top / n_top) / (total_relevant / n_total)``. An EF of 1.0
    means no better than random; higher means relevant items are concentrated near the top.

    ``fraction`` is in (0, 1]. ``total_relevant`` defaults to the number of relevant items
    in ``relevances`` (use this default when the ranking already contains every candidate).
    Returns 0.0 when there are no relevant items overall.
    """
    if not 0.0 < fraction <= 1.0:
        raise ValueError("fraction must be in (0, 1]")
    flags = [1.0 if r else 0.0 for r in relevances]
    n_total = len(flags)
    if n_total == 0:
        return 0.0
    total_rel = sum(flags) if total_relevant is None else float(total_relevant)
    if total_rel == 0.0:
        return 0.0
    n_top = max(1, math.ceil(fraction * n_total))
    actives_top = sum(flags[:n_top])
    return (actives_top / n_top) / (total_rel / n_total)
