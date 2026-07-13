"""MI3DOR cross-domain 3D-model retrieval metrics (OSCAR, Table 1).

Pulli et al., *OSCAR: Open-Set CAD Retrieval from a Language Prompt and a Single
Image* (2024), Section 5.1 / Table 1. OSCAR is evaluated on the MI3DOR
open-set, cross-domain 3D-model retrieval benchmark with the six standard
shape-retrieval criteria (Shilane et al. Princeton Shape Benchmark family):

* **Nearest Neighbour (NN)** -- fraction of queries whose top-1 retrieved item
  shares the query class (higher is better).
* **First Tier (FT)** -- recall within the top ``C-1`` results, where ``C`` is
  the number of database items sharing the query class (relevant set size).
* **Second Tier (ST)** -- recall within the top ``2*(C-1)`` results.
* **F-measure** -- F1 of precision and recall at a fixed top-``k`` cut-off
  (paper's F column; ``k`` defaults to 20).
* **Discounted Cumulative Gain (DCG)** -- normalised graded gain over the full
  ranking (reused from :mod:`bench.ranked_retrieval_metrics`).
* **ANMRR** -- Average Normalised Modified Retrieval Rank; the only
  *lower-is-better* criterion, penalising relevant items ranked late, averaged
  and normalised so it lies in ``[0, 1]`` independent of class size.

These are the *class-labelled gallery-ranking* criteria specific to the shape
benchmark and are NOT in the repo: :mod:`bench.geomretr_eval` provides NN
accuracy / NN-F1 / a single NDCG@N, and :mod:`bench.ranked_retrieval_metrics`
provides DCG/MRR/success@k -- neither implements First/Second Tier, the tiered
F-measure, or ANMRR/MRR-rank normalisation.

Input convention: each query supplies a ranking of retrieved gallery items as an
ordered list of booleans (or 0/1) -- ``True`` where the retrieved item is
relevant (shares the query class) -- together with the total number of relevant
items ``num_relevant`` in the gallery for that query. Pure stdlib, deterministic.
"""

from __future__ import annotations

from typing import List, Sequence

from harnesscad.eval.bench.ranked_retrieval_metrics import ndcg_at_k

__all__ = [
    "nearest_neighbour",
    "first_tier",
    "second_tier",
    "f_measure_at_k",
    "dcg",
    "anmrr",
    "mi3dor_report",
]


def _rel(relevances: Sequence) -> List[int]:
    return [1 if r else 0 for r in relevances]


def nearest_neighbour(relevances: Sequence) -> float:
    """NN: 1.0 if the top-1 retrieved item is relevant, else 0.0.

    Averaging this over queries gives the benchmark's NN score. Empty ranking
    scores 0.0.
    """
    rels = _rel(relevances)
    return float(rels[0]) if rels else 0.0


def first_tier(relevances: Sequence, num_relevant: int) -> float:
    """FT: recall within the top ``C-1`` results (``C = num_relevant``).

    The query itself is excluded from its own relevant set, so the tier size is
    ``C - 1``. Returns 0.0 when ``num_relevant <= 1`` (no other relevant items).
    """
    rels = _rel(relevances)
    tier = num_relevant - 1
    if tier <= 0:
        return 0.0
    hits = sum(rels[:tier])
    return hits / tier


def second_tier(relevances: Sequence, num_relevant: int) -> float:
    """ST: recall within the top ``2*(C-1)`` results (``C = num_relevant``).

    Returns 0.0 when ``num_relevant <= 1``.
    """
    rels = _rel(relevances)
    tier = num_relevant - 1
    if tier <= 0:
        return 0.0
    hits = sum(rels[:2 * tier])
    return hits / tier if hits <= tier else 1.0


def f_measure_at_k(relevances: Sequence, num_relevant: int, k: int = 20) -> float:
    """F1 of precision@k and recall@k for a single query.

    Precision@k = hits / k, recall@k = hits / (C-1) where ``C = num_relevant``
    (the query is excluded from its own relevant set). Returns 0.0 when the
    relevant set is empty or no relevant items appear in the top ``k``.
    """
    if k <= 0:
        raise ValueError("k must be positive")
    rels = _rel(relevances)
    denom_rel = num_relevant - 1
    if denom_rel <= 0:
        return 0.0
    hits = sum(rels[:k])
    if hits == 0:
        return 0.0
    precision = hits / k
    recall = hits / denom_rel
    if recall > 1.0:
        recall = 1.0
    if precision + recall == 0.0:
        return 0.0
    return 2.0 * precision * recall / (precision + recall)


def dcg(relevances: Sequence, k: int = None) -> float:
    """Normalised DCG over the ranking (binary gains).

    Thin adapter over :func:`bench.ranked_retrieval_metrics.ndcg_at_k` so the
    DCG column matches the repo's canonical NDCG implementation.
    """
    return ndcg_at_k(_rel(relevances), k)


def anmrr(relevances: Sequence, num_relevant: int) -> float:
    """Average Normalised Modified Retrieval Rank for one query (lower better).

    Standard MPEG-7 / Princeton definition. With ``NG = num_relevant`` ground
    truth items and a window ``K = min(4*NG, 2*max_tier)`` (here ``K = 2*NG``,
    the common choice), each relevant item found at 1-based rank ``r <= K``
    contributes ``r``; each relevant item missed within the window contributes a
    penalty ``1.25*K``. The average rank is de-biased and normalised to ``[0,
    1]``:

        AVR   = mean(rank(i))
        MRR   = AVR - 0.5*(1 + NG)
        NMRR  = MRR / (1.25*K - 0.5*(1 + NG))

    Returns 0.0 when ``num_relevant <= 0`` (a perfect / undefined query maps to
    the best score).
    """
    ng = num_relevant
    if ng <= 0:
        return 0.0
    rels = _rel(relevances)
    k = 2 * ng  # retrieval window
    penalty = 1.25 * k
    positions = [i + 1 for i, r in enumerate(rels) if r]  # 1-based ranks of hits
    rank_sum = 0.0
    for i in range(ng):
        # rank of the (i+1)-th relevant item if within window, else penalty
        if i < len(positions) and positions[i] <= k:
            rank_sum += positions[i]
        else:
            rank_sum += penalty
    avr = rank_sum / ng
    mrr = avr - 0.5 * (1.0 + ng)
    denom = penalty - 0.5 * (1.0 + ng)
    if denom <= 0.0:
        return 0.0
    nmrr = mrr / denom
    return max(0.0, min(1.0, nmrr))


def mi3dor_report(queries: Sequence[dict], f_k: int = 20) -> dict:
    """Aggregate the six MI3DOR criteria over a batch of queries.

    Each element of ``queries`` is a dict ``{"relevances": [...],
    "num_relevant": C}``. Returns a dict with keys ``NN, FT, ST, F, DCG, ANMRR``
    holding the mean over queries. Empty batch yields all zeros.
    """
    n = len(queries)
    if n == 0:
        return {"NN": 0.0, "FT": 0.0, "ST": 0.0, "F": 0.0, "DCG": 0.0, "ANMRR": 0.0}
    nn = ft = st = fm = dg = am = 0.0
    for q in queries:
        rels = q["relevances"]
        c = q["num_relevant"]
        nn += nearest_neighbour(rels)
        ft += first_tier(rels, c)
        st += second_tier(rels, c)
        fm += f_measure_at_k(rels, c, f_k)
        dg += dcg(rels)
        am += anmrr(rels, c)
    return {
        "NN": nn / n,
        "FT": ft / n,
        "ST": st / n,
        "F": fm / n,
        "DCG": dg / n,
        "ANMRR": am / n,
    }
