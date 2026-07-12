"""GC-CAD retrieval evaluation: graded Recall@k and NDCG@k.

Quan et al., *Self-supervised GNN for Mechanical CAD Retrieval* (GC-CAD),
Section 4.2. After the GNN produces one embedding per CAD part, retrieval ranks
the database by **cosine similarity** to a query embedding, and the annotators
label each retrieved result as *similar*, *partially similar*, or *dissimilar*
(graded relevance). Performance is reported as ``Recall@5``, ``Recall@10``,
``NDCG@5`` and ``NDCG@10`` (Tables 1, 3).

This differs from the query/gallery *classification* protocol in
:mod:`bench.geomretr_eval` (1-NN accuracy / macro-F1 / mAP with a single class
label per item): here relevance is **graded** and per (query, candidate) pair,
and the headline numbers are graded Recall@k and NDCG@k. NDCG reuses
:func:`bench.ranked_retrieval_metrics.ndcg_at_k`; the cosine ranking mirrors the
FAISS vector search the paper uses at inference time.

Deterministic and stdlib-only; the learned encoder is external, so callers pass
precomputed embeddings (e.g. from
:mod:`reconstruction.ssgnn_graph_descriptors`).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Sequence

from bench.contrastcad_contrastive import cosine_similarity
from bench.ranked_retrieval_metrics import ndcg_at_k

Vector = Sequence[float]

# Graded relevance gains used by GC-CAD's human annotation.
GAIN_SIMILAR = 2.0
GAIN_PARTIAL = 1.0
GAIN_DISSIMILAR = 0.0


def _cos(u: Vector, v: Vector) -> float:
    # cosine_similarity raises on a zero vector; treat that as no similarity.
    nu = math.sqrt(sum(x * x for x in u))
    nv = math.sqrt(sum(x * x for x in v))
    if nu == 0.0 or nv == 0.0:
        return -1.0
    return cosine_similarity(u, v)


def rank_database(query: Vector, database: Sequence[Vector],
                  exclude: int = None) -> List[int]:
    """Database indices sorted by *descending* cosine similarity to ``query``.

    Ties are broken by ascending index for determinism. ``exclude`` drops one
    database index (the query's own entry) so that ``p_r != p_q`` (Section 3.1).
    """
    scored = [(-_cos(query, d), i) for i, d in enumerate(database) if i != exclude]
    scored.sort(key=lambda t: (t[0], t[1]))
    return [i for _, i in scored]


def retrieval_ranking(query_embeddings: Sequence[Vector],
                      database: Sequence[Vector],
                      exclude: Sequence[int] = None) -> List[List[int]]:
    """Ranked database-index list for every query (cosine-similarity search).

    ``exclude[q]`` optionally removes each query's own database index.
    """
    excl = list(exclude) if exclude is not None else [None] * len(query_embeddings)
    return [rank_database(q, database, excl[i]) for i, q in enumerate(query_embeddings)]


def recall_at_k(ranking: Sequence[int], relevant: Sequence[int], k: int) -> float:
    """Fraction of the relevant set retrieved within the top ``k``.

    ``Recall@k = |relevant ∩ top-k| / |relevant|``. Returns 0.0 when the query has
    no relevant items. ``relevant`` is the set of database indices judged similar.
    """
    if k < 0:
        raise ValueError("k must be non-negative")
    rel = set(relevant)
    if not rel:
        return 0.0
    hits = sum(1 for i in ranking[:k] if i in rel)
    return hits / len(rel)


def graded_gains(ranking: Sequence[int], gains: Dict[int, float]) -> List[float]:
    """Gain vector in ranked order (missing entries default to 0.0 / dissimilar)."""
    return [float(gains.get(i, 0.0)) for i in ranking]


def ndcg_graded_at_k(ranking: Sequence[int], gains: Dict[int, float],
                     k: int) -> float:
    """NDCG@k over graded relevance for one query (reuses ``ndcg_at_k``)."""
    return ndcg_at_k(graded_gains(ranking, gains), k)


@dataclass
class RetrievalReport:
    """Aggregate GC-CAD retrieval metrics over a query set."""

    n_queries: int
    recall: Dict[int, float]
    ndcg: Dict[int, float]

    def to_dict(self) -> dict:
        return {
            "n_queries": self.n_queries,
            "recall": {f"recall@{k}": round(v, 6) for k, v in sorted(self.recall.items())},
            "ndcg": {f"ndcg@{k}": round(v, 6) for k, v in sorted(self.ndcg.items())},
        }


def evaluate_retrieval(query_embeddings: Sequence[Vector],
                       database: Sequence[Vector],
                       relevant_sets: Sequence[Sequence[int]],
                       gain_maps: Sequence[Dict[int, float]] = None, *,
                       ks: Sequence[int] = (5, 10),
                       exclude: Sequence[int] = None) -> RetrievalReport:
    """End-to-end GC-CAD retrieval evaluation (mean Recall@k and NDCG@k).

    ``relevant_sets[q]`` is the set of database indices judged *similar* to query
    ``q`` (used for Recall). ``gain_maps[q]`` maps database index -> graded gain
    (2 = similar, 1 = partial, 0 = dissimilar) for NDCG; when omitted, the
    relevant set is used with a binary gain of 1. Returns mean metrics over all
    queries for each ``k``.
    """
    rankings = retrieval_ranking(query_embeddings, database, exclude)
    if gain_maps is None:
        gain_maps = [{i: 1.0 for i in rel} for rel in relevant_sets]
    n = len(query_embeddings)
    recall = {k: 0.0 for k in ks}
    ndcg = {k: 0.0 for k in ks}
    for q in range(n):
        for k in ks:
            recall[k] += recall_at_k(rankings[q], relevant_sets[q], k)
            ndcg[k] += ndcg_graded_at_k(rankings[q], gain_maps[q], k)
    if n:
        recall = {k: v / n for k, v in recall.items()}
        ndcg = {k: v / n for k, v in ndcg.items()}
    return RetrievalReport(n_queries=n, recall=recall, ndcg=ndcg)
