"""Grounding-accuracy metrics for text-based B-Rep grounding (FutureCAD).

Li et al., "Towards High-Fidelity CAD Generation via LLM-Driven Program
Generation and Text-Based B-Rep Primitive Grounding" (FutureCAD, 2026),
Sec. 6.1 / Table 2. BRepGround is evaluated with three metrics over the ranked
per-primitive scores it predicts for each query:

    Recall@k  whether the ground-truth primitives are retrieved within the
              top-k predictions (the paper reports Recall@{3,5,10}).
    mAP       mean Average Precision -- summarises ranking quality across
              queries.
    F1        balances precision and recall for the selected primitive set,
              after thresholding scores into a binary selection.

This module computes those three deterministically from a *ranked prediction*
(a list of predicted primitive ids, best first) and the *ground-truth id set*
for each query. Scores/logits are not needed -- only the induced ranking -- so
this works equally for the paper's BRepGround, the CLIP-DE / Late-Fusion
baselines, or the deterministic grounder in
:mod:`reconstruction.brepground_grounding` (whose ranked output plugs straight
in via ``[p.index for p in ground(...)]``).

All metrics are in [0, 1]; multiply by 100 for the paper's percentages.
Pure, deterministic, stdlib-only.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Sequence, Set


@dataclass(frozen=True)
class GroundingCase:
    """One grounding query for evaluation.

    ``ranked``     predicted primitive ids, best first (no duplicates).
    ``truth``      the set of ground-truth primitive ids for the query.
    ``selected``   the ids the method actually selected (for precision/recall);
                   when None, precision/recall/F1 fall back to ``ranked`` being
                   thresholded elsewhere -- see :func:`f1` for the convention.
    """

    ranked: Sequence[int]
    truth: Set[int]
    selected: Sequence[int] = ()

    def __post_init__(self) -> None:
        if len(set(self.ranked)) != len(self.ranked):
            raise ValueError("ranked contains duplicate ids")


def recall_at_k(ranked: Sequence[int], truth: Iterable[int], k: int) -> float:
    """Fraction of the ground-truth ids appearing in the top-``k`` predictions.

    With an empty ground-truth set recall is defined as 1.0 (nothing to miss).
    """
    truth_set = set(truth)
    if not truth_set:
        return 1.0
    if k <= 0:
        return 0.0
    topk = set(ranked[:k])
    return len(topk & truth_set) / len(truth_set)


def average_precision(ranked: Sequence[int], truth: Iterable[int]) -> float:
    """Average Precision for one ranked list against ``truth``.

    AP averages the precision evaluated at each rank where a relevant item is
    retrieved. Relevant items never retrieved contribute 0 precision (they are
    divided into by ``len(truth)``), matching the standard information-retrieval
    definition. Empty ground truth yields 1.0.
    """
    truth_set = set(truth)
    if not truth_set:
        return 1.0
    hits = 0
    precision_sum = 0.0
    for rank, pid in enumerate(ranked, start=1):
        if pid in truth_set:
            hits += 1
            precision_sum += hits / rank
    return precision_sum / len(truth_set)


def mean_average_precision(cases: Sequence[GroundingCase]) -> float:
    """mAP over a set of queries (mean of per-query AP)."""
    if not cases:
        return 0.0
    return sum(average_precision(c.ranked, c.truth) for c in cases) / len(cases)


def precision_recall_f1(
    selected: Iterable[int], truth: Iterable[int]
) -> tuple:
    """Return (precision, recall, F1) for a selected id set vs ground truth."""
    sel = set(selected)
    truth_set = set(truth)
    if not sel and not truth_set:
        return (1.0, 1.0, 1.0)
    tp = len(sel & truth_set)
    precision = tp / len(sel) if sel else 0.0
    recall = tp / len(truth_set) if truth_set else 0.0
    if precision + recall == 0.0:
        return (precision, recall, 0.0)
    f1_val = 2 * precision * recall / (precision + recall)
    return (precision, recall, f1_val)


def f1(case: GroundingCase) -> float:
    """F1 for one case.

    Uses ``selected`` when provided; otherwise treats the entire ``ranked`` list
    as the selection (the method returns exactly the primitives it grounds, as
    the deterministic grounder does).
    """
    selected = case.selected if case.selected else case.ranked
    return precision_recall_f1(selected, case.truth)[2]


def mean_f1(cases: Sequence[GroundingCase]) -> float:
    """Macro-averaged F1 over cases."""
    if not cases:
        return 0.0
    return sum(f1(c) for c in cases) / len(cases)


def evaluate(
    cases: Sequence[GroundingCase], ks: Sequence[int] = (3, 5, 10)
) -> Dict[str, float]:
    """Compute the full BRepGround report (Table 2 style).

    Returns a dict with ``recall@k`` for each ``k``, ``mAP`` and ``F1``. All
    values are in [0, 1].
    """
    report: Dict[str, float] = {}
    n = len(cases) or 1
    for k in ks:
        report["recall@%d" % k] = (
            sum(recall_at_k(c.ranked, c.truth, k) for c in cases) / n
        )
    report["mAP"] = mean_average_precision(cases)
    report["F1"] = mean_f1(cases)
    return report
