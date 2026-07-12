"""Autoconstrain evaluation metrics (SketchGraphs ``sketchgraphs_models/autoconstraint/eval.py``).

The autoconstrain task: given a sketch's *primitives* (the node ops) with the
constraints stripped, predict the constraints (the edge ops).  Scoring it is
where the subtlety lives, and the reference implementation's choices are load-
bearing but undocumented in the paper.  This module reimplements exactly those
choices, with no model attached -- it scores predicted vs ground-truth edge-op
lists, however they were produced.

Scoring conventions (faithful to the reference)
-----------------------------------------------
* **An edge is identified by ``(label, references[0], references[-1])``** -- its
  type, its first reference and its *last*.  Middle references are ignored, so a
  hyperedge is scored on its extremities only.
* **Edges are compared as a set, not a list.**  Constraints are unordered and
  duplicate-free: predicting the same constraint twice cannot inflate the score,
  and predicting them in a different order cannot deflate it.
* **``subnode`` edges are excluded from both sides.**  They are structural
  (a curve owns its endpoints), not predictions, so scoring them would inflate
  every model identically.
* **Empty-set edge cases are asymmetric, and deliberately so:**

  - predicting nothing gives ``precision = 0`` (a model that abstains earns no
    credit), but
  - a sketch with no ground-truth constraints gives ``recall = 1`` (there was
    nothing to find, so nothing was missed).

  This asymmetry is what stops a degenerate abstain-always model from scoring
  perfectly on the unconstrained sketches that dominate the dataset.
* **The corpus figure is the macro mean** -- per-sketch precision/recall averaged
  over sketches, so a 3-constraint sketch counts as much as a 300-constraint one.
  :func:`micro_scores` also provides the pooled (edge-weighted) alternative,
  which the macro figure can differ from substantially on a skewed corpus.

Public API
----------
``edge_key(op)`` / ``edge_key_set(ops)``  -- canonicalisation.
``sketch_scores(gt, pred)``               -- one sketch -> SketchScore.
``corpus_scores(pairs)`` (macro) / ``micro_scores(pairs)`` (pooled).
``per_type_scores(pairs)``                -- pooled breakdown by constraint type.
``f1(precision, recall)``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Sequence, Set, Tuple

from reconstruction.sgraphs2_dof_mask import EdgeOp

__all__ = [
    "STRUCTURAL_LABELS",
    "EdgeKey",
    "SketchScore",
    "CorpusScore",
    "edge_key",
    "edge_key_set",
    "f1",
    "sketch_scores",
    "corpus_scores",
    "micro_scores",
    "per_type_scores",
]

#: Labels that are structural rather than predicted, and so are scored on
#: neither side.
STRUCTURAL_LABELS = frozenset({"subnode"})

EdgeKey = Tuple[str, int, int]


@dataclass(frozen=True)
class SketchScore:
    """Precision/recall for one sketch, with the counts they derive from."""

    precision: float
    recall: float
    num_correct: int
    num_predicted: int
    num_ground_truth: int

    @property
    def f1(self) -> float:
        return f1(self.precision, self.recall)


@dataclass(frozen=True)
class CorpusScore:
    """Aggregate precision/recall over a corpus."""

    precision: float
    recall: float
    num_sketches: int

    @property
    def f1(self) -> float:
        return f1(self.precision, self.recall)


def edge_key(op: EdgeOp) -> EdgeKey:
    """Canonical identity of an edge op: ``(label, first ref, last ref)``.

    Middle references of a hyperedge are dropped -- the reference implementation
    scores extremities only.  Raises ``ValueError`` on an edge with no references.
    """
    if not op.references:
        raise ValueError("edge op has no references")
    return (op.label, op.references[0], op.references[-1])


def edge_key_set(ops: Iterable[EdgeOp]) -> Set[EdgeKey]:
    """Canonical key set of an edge-op list, with structural edges dropped.

    Set semantics mean duplicates collapse: a model cannot gain (or lose) by
    emitting the same constraint twice.
    """
    return {edge_key(op) for op in ops if op.label not in STRUCTURAL_LABELS}


def f1(precision: float, recall: float) -> float:
    """Harmonic mean of precision and recall; 0 when both are 0."""
    if precision + recall == 0.0:
        return 0.0
    return 2.0 * precision * recall / (precision + recall)


def sketch_scores(
    ground_truth: Sequence[EdgeOp], predicted: Sequence[EdgeOp]
) -> SketchScore:
    """Precision and recall for a single sketch.

    Empty-set handling follows the reference: no predictions -> precision 0;
    no ground truth -> recall 1.
    """
    gt_keys = edge_key_set(ground_truth)
    pred_keys = edge_key_set(predicted)
    correct = gt_keys & pred_keys

    precision = len(correct) / len(pred_keys) if pred_keys else 0.0
    recall = len(correct) / len(gt_keys) if gt_keys else 1.0

    return SketchScore(
        precision=precision,
        recall=recall,
        num_correct=len(correct),
        num_predicted=len(pred_keys),
        num_ground_truth=len(gt_keys),
    )


def corpus_scores(
    pairs: Iterable[Tuple[Sequence[EdgeOp], Sequence[EdgeOp]]]
) -> CorpusScore:
    """Macro-averaged precision/recall -- the headline figure.

    ``pairs`` yields ``(ground_truth_ops, predicted_ops)`` per sketch.  Every
    sketch is weighted equally regardless of its constraint count.  An empty
    corpus scores 0/0.
    """
    scores: List[SketchScore] = [sketch_scores(gt, pred) for gt, pred in pairs]
    if not scores:
        return CorpusScore(0.0, 0.0, 0)
    n = len(scores)
    return CorpusScore(
        precision=sum(s.precision for s in scores) / n,
        recall=sum(s.recall for s in scores) / n,
        num_sketches=n,
    )


def micro_scores(
    pairs: Iterable[Tuple[Sequence[EdgeOp], Sequence[EdgeOp]]]
) -> CorpusScore:
    """Pooled (edge-weighted) precision/recall over the corpus.

    Counts are summed across sketches before dividing, so a constraint-dense
    sketch carries proportionally more weight.  The empty-set conventions apply
    to the pooled totals: no predictions anywhere -> precision 0; no ground truth
    anywhere -> recall 1.
    """
    correct = predicted = ground_truth = 0
    count = 0
    for gt_ops, pred_ops in pairs:
        gt_keys = edge_key_set(gt_ops)
        pred_keys = edge_key_set(pred_ops)
        correct += len(gt_keys & pred_keys)
        predicted += len(pred_keys)
        ground_truth += len(gt_keys)
        count += 1

    return CorpusScore(
        precision=correct / predicted if predicted else 0.0,
        recall=correct / ground_truth if ground_truth else 1.0,
        num_sketches=count,
    )


def per_type_scores(
    pairs: Iterable[Tuple[Sequence[EdgeOp], Sequence[EdgeOp]]]
) -> Dict[str, SketchScore]:
    """Pooled precision/recall broken down by constraint label.

    A type is reported if it appears in the ground truth or the predictions of
    any sketch.  This is what exposes the common failure mode of an autoconstrain
    model: near-perfect on ``coincident`` (which dominates the corpus) and near
    zero on everything else, at a healthy overall score.
    """
    correct: Dict[str, int] = {}
    predicted: Dict[str, int] = {}
    ground_truth: Dict[str, int] = {}

    for gt_ops, pred_ops in pairs:
        gt_keys = edge_key_set(gt_ops)
        pred_keys = edge_key_set(pred_ops)
        hits = gt_keys & pred_keys

        for label, _, _ in gt_keys:
            ground_truth[label] = ground_truth.get(label, 0) + 1
        for label, _, _ in pred_keys:
            predicted[label] = predicted.get(label, 0) + 1
        for label, _, _ in hits:
            correct[label] = correct.get(label, 0) + 1

    out: Dict[str, SketchScore] = {}
    for label in sorted(set(ground_truth) | set(predicted)):
        n_correct = correct.get(label, 0)
        n_pred = predicted.get(label, 0)
        n_gt = ground_truth.get(label, 0)
        out[label] = SketchScore(
            precision=n_correct / n_pred if n_pred else 0.0,
            recall=n_correct / n_gt if n_gt else 1.0,
            num_correct=n_correct,
            num_predicted=n_pred,
            num_ground_truth=n_gt,
        )
    return out
