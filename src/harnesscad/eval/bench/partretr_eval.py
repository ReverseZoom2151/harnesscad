"""partretr_eval — evaluation harness for CAD-assembly part retrieval.

Metrics for the paper "Error Notebook-Guided, Training-Free Part Retrieval in
3D CAD Assemblies via Vision-Language Models". Given predicted filename subsets
and ground-truth subsets over CAD assemblies, compute:

  - **accuracy** — exact-set match rate (the paper's headline "accuracy":
    prediction set == ground-truth set), optionally bucketed by assembly part
    count (<10, 10-20, 20-50, >50) as in Tables 1/2.
  - **recall / precision / F1** — set-overlap relevance (App. A.1, Eq. A.1-A.3):
    TP=|GT & Pred|, FP=|Pred - GT|, FN=|GT - Pred|; global-averaged.
  - **recall@k / MRR** — for *ranked* candidate lists, treating any ground-truth
    part as a relevant hit: recall@k = fraction of queries with a GT part in the
    top-k; MRR = mean reciprocal rank of the first GT-containing candidate.

Deterministic and stdlib-only. Set comparisons are order-insensitive.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Sequence, Set, Tuple


def _as_set(xs: Sequence[str]) -> Set[str]:
    return {str(x).strip() for x in xs if str(x).strip()}


# ---------------------------------------------------------------------------
# Per-instance relevance (App. A.1)
# ---------------------------------------------------------------------------
def relevance(pred: Sequence[str], gt: Sequence[str]) -> Dict[str, float]:
    """Recall/precision/F1 for one prediction vs ground truth (Eq. A.1-A.3)."""
    p, g = _as_set(pred), _as_set(gt)
    tp = len(p & g)
    fp = len(p - g)
    fn = len(g - p)
    recall = tp / (tp + fn) if (tp + fn) else 1.0
    precision = tp / (tp + fp) if (tp + fp) else 1.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    return {"tp": tp, "fp": fp, "fn": fn,
            "recall": recall, "precision": precision, "f1": f1}


def exact_match(pred: Sequence[str], gt: Sequence[str]) -> bool:
    """True iff the predicted set equals the ground-truth set (paper's accuracy)."""
    return _as_set(pred) == _as_set(gt)


# ---------------------------------------------------------------------------
# Ranked-list metrics
# ---------------------------------------------------------------------------
def _candidate_hits_gt(candidate: Sequence[str], gt: Set[str]) -> bool:
    """A ranked candidate counts as relevant if it shares any part with GT."""
    return bool(_as_set(candidate) & gt)


def recall_at_k(ranked: Sequence[Sequence[str]], gt: Sequence[str], k: int) -> float:
    """1.0 if any of the top-``k`` candidates shares a part with GT, else 0.0."""
    g = _as_set(gt)
    if not g:
        return 1.0
    for cand in list(ranked)[:k]:
        if _candidate_hits_gt(cand, g):
            return 1.0
    return 0.0


def reciprocal_rank(ranked: Sequence[Sequence[str]], gt: Sequence[str]) -> float:
    """1/rank of the first candidate that shares a part with GT (0 if none)."""
    g = _as_set(gt)
    if not g:
        return 1.0
    for i, cand in enumerate(ranked, start=1):
        if _candidate_hits_gt(cand, g):
            return 1.0 / i
    return 0.0


# ---------------------------------------------------------------------------
# Part-count bucketing (Tables 1/2 difficulty groups)
# ---------------------------------------------------------------------------
def part_bucket(n_parts: int) -> str:
    if n_parts < 10:
        return "<10"
    if n_parts <= 20:
        return "10-20"
    if n_parts <= 50:
        return "20-50"
    return ">50"


BUCKETS = ("<10", "10-20", "20-50", ">50")


@dataclass
class EvalReport:
    """Aggregate part-retrieval evaluation over a set of queries."""

    n: int
    accuracy: float
    recall: float
    precision: float
    f1: float
    mrr: float
    recall_at_k: Dict[int, float]
    per_bucket: Dict[str, Dict[str, float]]

    def to_dict(self) -> dict:
        return {
            "n": self.n,
            "accuracy": round(self.accuracy, 6),
            "recall": round(self.recall, 6),
            "precision": round(self.precision, 6),
            "f1": round(self.f1, 6),
            "mrr": round(self.mrr, 6),
            "recall_at_k": {k: round(v, 6) for k, v in self.recall_at_k.items()},
            "per_bucket": {
                b: {m: round(v, 6) for m, v in d.items()}
                for b, d in self.per_bucket.items()
            },
        }


def evaluate(
    queries: Sequence[dict],
    ks: Sequence[int] = (1, 3),
) -> EvalReport:
    """Evaluate a list of query records.

    Each query dict may contain:
      - ``"pred"``   : predicted filename subset  (for accuracy / relevance)
      - ``"gt"``     : ground-truth filename subset (required)
      - ``"ranked"`` : optional list of ranked candidate subsets (for recall@k /
                       MRR). If absent, ``[pred]`` is used as a length-1 ranking.
      - ``"n_parts"``: optional assembly part count (for bucketing).

    Returns an :class:`EvalReport` with global metrics plus per-bucket accuracy,
    recall and F1.
    """
    if not queries:
        return EvalReport(0, 0.0, 0.0, 0.0, 0.0, 0.0, {k: 0.0 for k in ks},
                          {b: {} for b in BUCKETS})

    n = len(queries)
    acc_sum = rec_sum = prec_sum = f1_sum = mrr_sum = 0.0
    rk_sum: Dict[int, float] = {k: 0.0 for k in ks}
    bucket_acc: Dict[str, List[float]] = {b: [] for b in BUCKETS}
    bucket_rec: Dict[str, List[float]] = {b: [] for b in BUCKETS}
    bucket_f1: Dict[str, List[float]] = {b: [] for b in BUCKETS}

    for q in queries:
        gt = q.get("gt", [])
        pred = q.get("pred", [])
        ranked = q.get("ranked") or [pred]

        em = 1.0 if exact_match(pred, gt) else 0.0
        rel = relevance(pred, gt)
        rr = reciprocal_rank(ranked, gt)

        acc_sum += em
        rec_sum += rel["recall"]
        prec_sum += rel["precision"]
        f1_sum += rel["f1"]
        mrr_sum += rr
        for k in ks:
            rk_sum[k] += recall_at_k(ranked, gt, k)

        if "n_parts" in q:
            b = part_bucket(int(q["n_parts"]))
            bucket_acc[b].append(em)
            bucket_rec[b].append(rel["recall"])
            bucket_f1[b].append(rel["f1"])

    def _avg(xs: List[float]) -> float:
        return sum(xs) / len(xs) if xs else 0.0

    per_bucket: Dict[str, Dict[str, float]] = {}
    for b in BUCKETS:
        if bucket_acc[b]:
            per_bucket[b] = {
                "n": len(bucket_acc[b]),
                "accuracy": _avg(bucket_acc[b]),
                "recall": _avg(bucket_rec[b]),
                "f1": _avg(bucket_f1[b]),
            }

    return EvalReport(
        n=n,
        accuracy=acc_sum / n,
        recall=rec_sum / n,
        precision=prec_sum / n,
        f1=f1_sum / n,
        mrr=mrr_sum / n,
        recall_at_k={k: rk_sum[k] / n for k in ks},
        per_bucket=per_bucket,
    )
