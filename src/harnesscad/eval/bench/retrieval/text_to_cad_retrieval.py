"""Text-to-CAD retrieval benchmark protocol: Recall@K, Median Rank, Rsum.

Paper: *Text-to-CAD Retrieval -- a Strong Baseline* (Pan et al., IEEE T-II).
The paper formalizes *text-to-CAD retrieval*: given a natural-language query,
rank a database of CAD models and check where the single paired ground-truth
model lands. Its evaluation protocol (Sec. IV.A "Metrics") is entirely
deterministic and model-agnostic -- it only consumes rankings -- so it is the
buildable core of an otherwise-neural paper:

  * **Recall@K** -- fraction of queries whose ground-truth model appears within
    the top-K results (reported at K in {1, 2, 5, 10, 20});
  * **Median Rank (MedR)** -- median 1-indexed rank of the ground-truth across
    queries (lower is better);
  * **Rsum** -- the sum of the reported Recall@K values, a single aggregate.

This module scores rankings, not embeddings: it takes, per query, either the
1-indexed rank of the ground-truth model or a ranked list of candidate ids plus
the ground-truth id, and returns the report. No learning, no similarity model --
purely the metric definitions. It complements
:mod:`ranked_retrieval_metrics` (graded relevance / nDCG) and
:mod:`tiered_retrieval_metrics` (multi-relevant tiers): here every query has
exactly one correct answer, matching the paired-annotation setting.

Stdlib-only and deterministic.
"""

from __future__ import annotations

__all__ = [
    "DEFAULT_KS",
    "rank_of",
    "recall_at_k",
    "median_rank",
    "rsum",
    "retrieval_report",
]

DEFAULT_KS = (1, 2, 5, 10, 20)


def rank_of(ranked_ids, gt_id) -> int:
    """1-indexed position of ``gt_id`` in ``ranked_ids``; 0 if absent.

    A rank of 0 means the ground-truth never appears in the ranking (treated as
    a miss for every K and as ``len(ranked_ids)+1`` for the median).
    """
    for i, cand in enumerate(ranked_ids, start=1):
        if cand == gt_id:
            return i
    return 0


def _normalize_ranks(queries) -> list:
    """Coerce a list of queries into 1-indexed ground-truth ranks.

    Each query is either an ``int`` (already the rank) or a ``(ranked_ids,
    gt_id)`` pair. A non-positive/absent rank stays 0 (a miss).
    """
    ranks = []
    for q in queries:
        if isinstance(q, int):
            ranks.append(q if q > 0 else 0)
        else:
            ranked_ids, gt_id = q
            ranks.append(rank_of(ranked_ids, gt_id))
    return ranks


def recall_at_k(queries, k: int) -> float:
    """Percentage of queries whose ground-truth is within the top-``k``.

    Returned on a 0-100 scale to match the paper's tables (e.g. R1 = 9.71).
    """
    if k <= 0:
        raise ValueError("k must be positive")
    ranks = _normalize_ranks(queries)
    if not ranks:
        return 0.0
    hits = sum(1 for r in ranks if 0 < r <= k)
    return 100.0 * hits / len(ranks)


def median_rank(queries, missing_penalty=None) -> float:
    """Median 1-indexed rank of the ground-truth across queries (lower better).

    Misses (rank 0) are ranked at ``missing_penalty`` if given, else at
    ``max observed rank + 1`` so they push the median toward the worst case
    without an unbounded value.
    """
    ranks = _normalize_ranks(queries)
    if not ranks:
        return 0.0
    observed = [r for r in ranks if r > 0]
    penalty = missing_penalty
    if penalty is None:
        penalty = (max(observed) + 1) if observed else 1
    filled = sorted(r if r > 0 else penalty for r in ranks)
    n = len(filled)
    mid = n // 2
    if n % 2 == 1:
        return float(filled[mid])
    return (filled[mid - 1] + filled[mid]) / 2.0


def rsum(queries, ks=DEFAULT_KS) -> float:
    """Sum of Recall@K over ``ks`` -- the paper's single aggregate score."""
    return sum(recall_at_k(queries, k) for k in ks)


def retrieval_report(queries, ks=DEFAULT_KS) -> dict:
    """Full report: ``recall`` per K, ``medr``, ``rsum`` and query count.

    ``queries`` is a list of per-query items, each either a 1-indexed rank
    (``int``) or a ``(ranked_ids, gt_id)`` pair. Deterministic.
    """
    recalls = {k: recall_at_k(queries, k) for k in ks}
    return {
        "num_queries": len(list(queries)),
        "ks": tuple(ks),
        "recall": recalls,
        "medr": median_rank(queries),
        "rsum": sum(recalls.values()),
    }
