"""Geometric-object retrieval evaluation protocol (query/gallery, NN + ranking).

Van den Herrewegen et al., *Fine-Tuning 3D Foundation Models for Geometric
Object Retrieval* (2024), Section 4.1 evaluation protocol. The encoder embeds
the *test* split (queries) and the *train* split (gallery); for each query a
nearest-neighbour search under **cosine distance** yields a ranked list of
gallery items. Three metrics are reported:

* **NN accuracy** -- the label of the single closest gallery item is used as the
  prediction; accuracy is the fraction of queries predicted correctly.
* **NN F1** -- "the averaged harmonic mean of the accuracy and the precision per
  class" (macro-averaged per-class F1 of that same 1-NN classifier).
* **NDCG@N** -- retrieve the ``N`` closest gallery items, assign gain 1 to each
  gallery item sharing the query's class, and normalise by the ideal DCG. The
  paper uses ``N = 100``.

This is the query/gallery *ranking* protocol, distinct from the set-overlap part
retrieval in :mod:`bench.partretr_eval`. It **reuses** :func:`ndcg_at_k` from
:mod:`bench.ranked_retrieval_metrics` rather than reimplementing DCG, and adds a
per-category mean-average-precision (mAP) not present elsewhere.

Deterministic, stdlib-only. The learned encoder is external; callers pass
precomputed embeddings (e.g. from :mod:`reconstruction.geomretr_descriptors`).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Sequence

from harnesscad.eval.bench.retrieval.ranked_retrieval_metrics import ndcg_at_k

Vector = Sequence[float]


def _l2(v: Vector) -> List[float]:
    n = math.sqrt(sum(x * x for x in v))
    if n == 0.0:
        return [0.0 for _ in v]
    return [x / n for x in v]


def cosine_distance(u: Vector, v: Vector) -> float:
    """Cosine distance ``1 - cos_sim`` in ``[0, 2]``; 1.0 if either is a zero vector."""
    nu = math.sqrt(sum(x * x for x in u))
    nv = math.sqrt(sum(x * x for x in v))
    if nu == 0.0 or nv == 0.0:
        return 1.0
    dot = sum(a * b for a, b in zip(u, v))
    cos = max(-1.0, min(1.0, dot / (nu * nv)))
    return 1.0 - cos


def rank_gallery(query: Vector, gallery: Sequence[Vector]) -> List[int]:
    """Indices of ``gallery`` sorted by ascending cosine distance to ``query``.

    Ties are broken by gallery index for full determinism.
    """
    dists = [(cosine_distance(query, g), i) for i, g in enumerate(gallery)]
    dists.sort(key=lambda t: (t[0], t[1]))
    return [i for _, i in dists]


def retrieval_ranking(query_embeddings: Sequence[Vector],
                      gallery_embeddings: Sequence[Vector]) -> List[List[int]]:
    """Full ranked gallery-index list for every query (cosine-distance NN search)."""
    return [rank_gallery(q, gallery_embeddings) for q in query_embeddings]


def nn_accuracy(rankings: Sequence[Sequence[int]], query_labels: Sequence,
                gallery_labels: Sequence) -> float:
    """1-NN classification accuracy: closest gallery label == query label."""
    if not rankings:
        return 0.0
    correct = 0
    for order, qlab in zip(rankings, query_labels):
        if order and gallery_labels[order[0]] == qlab:
            correct += 1
    return correct / len(rankings)


def nn_macro_f1(rankings: Sequence[Sequence[int]], query_labels: Sequence,
                gallery_labels: Sequence) -> float:
    """Macro-averaged F1 of the 1-NN classifier over all query classes.

    Predictions are the closest gallery item's label. Per class we compute
    precision and recall, take the F1, then average over every class present in
    the query labels (macro / "averaged per class", Section 4.1).
    """
    if not rankings:
        return 0.0
    preds = []
    for order in rankings:
        preds.append(gallery_labels[order[0]] if order else None)
    classes = sorted(set(query_labels), key=lambda c: str(c))
    f1s = []
    for c in classes:
        tp = sum(1 for p, t in zip(preds, query_labels) if p == c and t == c)
        fp = sum(1 for p, t in zip(preds, query_labels) if p == c and t != c)
        fn = sum(1 for p, t in zip(preds, query_labels) if p != c and t == c)
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
        f1s.append(f1)
    return sum(f1s) / len(f1s) if f1s else 0.0


def ndcg_at_n(rankings: Sequence[Sequence[int]], query_labels: Sequence,
              gallery_labels: Sequence, n: int = 100) -> float:
    """Mean NDCG@``n`` over queries, gain 1 for same-class gallery hits.

    Reuses :func:`bench.ranked_retrieval_metrics.ndcg_at_k`: for each query, the
    top-``n`` retrieved gallery items become a gain vector (1 if same class as the
    query, else 0), and NDCG normalises against the ideal ordering. The paper uses
    ``n = 100``.
    """
    if not rankings:
        return 0.0
    total = 0.0
    for order, qlab in zip(rankings, query_labels):
        gains = [1.0 if gallery_labels[i] == qlab else 0.0 for i in order[:n]]
        total += ndcg_at_k(gains, n)
    return total / len(rankings)


def average_precision(order: Sequence[int], qlab, gallery_labels: Sequence) -> float:
    """Average precision of one ranked list for same-class relevance.

    ``AP = (1/R) sum_k precision@k * rel_k`` where ``R`` is the total number of
    same-class gallery items. Returns 0.0 if the class has no gallery items.
    """
    total_rel = sum(1 for g in gallery_labels if g == qlab)
    if total_rel == 0:
        return 0.0
    hits = 0
    ap = 0.0
    for rank, idx in enumerate(order, start=1):
        if gallery_labels[idx] == qlab:
            hits += 1
            ap += hits / rank
    return ap / total_rel


def per_category_map(rankings: Sequence[Sequence[int]], query_labels: Sequence,
                     gallery_labels: Sequence) -> Dict:
    """Per-category and overall mean average precision (mAP).

    Returns ``{"per_category": {class: mAP}, "macro_map": ..., "micro_map": ...}``
    where ``macro_map`` averages the per-category means and ``micro_map`` averages
    AP across all queries equally.
    """
    per_class: Dict = {}
    aps: List[float] = []
    for order, qlab in zip(rankings, query_labels):
        ap = average_precision(order, qlab, gallery_labels)
        aps.append(ap)
        per_class.setdefault(qlab, []).append(ap)
    per_category = {c: sum(v) / len(v) for c, v in per_class.items()}
    macro = sum(per_category.values()) / len(per_category) if per_category else 0.0
    micro = sum(aps) / len(aps) if aps else 0.0
    return {"per_category": per_category, "macro_map": macro, "micro_map": micro}


@dataclass
class RetrievalReport:
    """Aggregate geometric-object retrieval evaluation (Section 4.1 metrics)."""

    n_queries: int
    nn_accuracy: float
    nn_f1: float
    ndcg: float
    macro_map: float
    micro_map: float
    per_category_map: Dict

    def to_dict(self) -> dict:
        return {
            "n_queries": self.n_queries,
            "nn_accuracy": round(self.nn_accuracy, 6),
            "nn_f1": round(self.nn_f1, 6),
            "ndcg": round(self.ndcg, 6),
            "macro_map": round(self.macro_map, 6),
            "micro_map": round(self.micro_map, 6),
            "per_category_map": {str(k): round(v, 6)
                                 for k, v in self.per_category_map.items()},
        }


def evaluate_retrieval(query_embeddings: Sequence[Vector], query_labels: Sequence,
                       gallery_embeddings: Sequence[Vector], gallery_labels: Sequence,
                       *, ndcg_n: int = 100) -> RetrievalReport:
    """End-to-end retrieval evaluation for one query/gallery split.

    Ranks every query against the gallery under cosine distance, then computes the
    paper's NN accuracy, NN macro-F1 and NDCG@``ndcg_n`` plus per-category mAP.
    """
    rankings = retrieval_ranking(query_embeddings, gallery_embeddings)
    mp = per_category_map(rankings, query_labels, gallery_labels)
    return RetrievalReport(
        n_queries=len(query_embeddings),
        nn_accuracy=nn_accuracy(rankings, query_labels, gallery_labels),
        nn_f1=nn_macro_f1(rankings, query_labels, gallery_labels),
        ndcg=ndcg_at_n(rankings, query_labels, gallery_labels, ndcg_n),
        macro_map=mp["macro_map"],
        micro_map=mp["micro_map"],
        per_category_map=mp["per_category"],
    )
