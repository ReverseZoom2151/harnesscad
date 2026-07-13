"""SymPoint panoptic symbol-spotting evaluation (ECCV 2024).

SymPoint scores symbol spotting with a panoptic protocol that differs from the
greedy one in :mod:`bench.cadtransformer_panoptic` in four concrete ways, all
reimplemented here (``svgnet/evaluation/point_wise_eval.py::InstanceEval``):

1. **Score gate.**  Predicted instances carry an objectness ``score``; those
   below ``MIN_OBJ_SCORE`` (0.1) are invisible to the metric.  A query-based
   model emits a fixed number of instances, so the gate -- not a matcher -- is
   what removes the junk queries.
2. **GT-driven scan, not greedy matching.**  Every ground-truth instance is
   scanned against *every* surviving prediction.  A prediction overlapping a GT
   above ``IOU_THRESHOLD`` with the *wrong* class contributes a false positive
   for its own class; the GT counts as a false negative only when *nothing*
   overlaps it.  A GT can therefore be a TP and simultaneously generate FPs --
   this is the reference behaviour and it is why SymPoint's numbers are not
   reproducible with a one-to-one Hungarian/greedy matcher.
3. **Rounded log-length weights.**  The point weights are
   ``round(log(1 + length), 3)`` -- the rounding is in the reference and is kept
   here (:func:`point_weights`) so scores match bit-for-bit.
4. **thing / stuff split.**  Countable symbols (doors, furniture, ...) are
   *things*; wall / railing / parking-spot style classes are *stuff*, and PQ is
   reported for each group as well as overall, with ``PQ = RQ * SQ`` computed on
   the *aggregated* counts (not by averaging per-class PQ).

An instance is ``(class_id, score, point_indices)``; ground truth omits the
score.  Everything is stdlib-only and deterministic.
"""

from __future__ import annotations

import math
from typing import Dict, Iterable, List, Mapping, Sequence, Set, Tuple

MIN_OBJ_SCORE = 0.1
IOU_THRESHOLD = 0.5
EPS = 1e-6

#: Reference SVG-dataset split: classes 0..29 are things, 30..34 are stuff.
DEFAULT_THING_CLASSES: Tuple[int, ...] = tuple(range(30))
DEFAULT_STUFF_CLASSES: Tuple[int, ...] = (30, 31, 32, 33, 34)
DEFAULT_IGNORE_LABEL = 35


def point_weights(lengths: Sequence[float], digits: int = 3) -> List[float]:
    """``round(log(1 + length), digits)`` per point (SymPoint weighting)."""
    out = []
    for length in lengths:
        if length < 0:
            raise ValueError("length must be non-negative")
        out.append(round(math.log(1.0 + float(length)), digits))
    return out


def weighted_mask_iou(pred: Iterable[int], gt: Iterable[int],
                      weights: Sequence[float]) -> float:
    """Length-weighted IoU of two point-index sets."""
    a: Set[int] = set(int(i) for i in pred)
    b: Set[int] = set(int(i) for i in gt)
    for idx in a | b:
        if not 0 <= idx < len(weights):
            raise IndexError("point index %d outside the weight vector" % idx)
    inter = sum(weights[i] for i in a & b)
    union = sum(weights[i] for i in a | b)
    return inter / (union + EPS)


def accumulate(predictions: Sequence[Tuple[int, float, Sequence[int]]],
               ground_truth: Sequence[Tuple[int, Sequence[int]]],
               weights: Sequence[float], num_classes: int = 35,
               ignore_label: int = DEFAULT_IGNORE_LABEL,
               min_obj_score: float = MIN_OBJ_SCORE,
               iou_threshold: float = IOU_THRESHOLD,
               counts: Dict[str, List[float]] | None = None) -> Dict[str, List[float]]:
    """Accumulate TP / TP-IoU / FP / FN per class over one drawing.

    ``counts`` may be an accumulator returned by a previous call, allowing the
    statistics to be summed over a whole dataset.
    """
    if counts is None:
        counts = {
            "tp": [0.0] * num_classes,
            "tp_iou": [0.0] * num_classes,
            "fp": [0.0] * num_classes,
            "fn": [0.0] * num_classes,
        }
    kept = [(int(c), float(s), p) for c, s, p in predictions
            if int(c) != ignore_label and float(s) >= min_obj_score]
    for gt_label, gt_points in ground_truth:
        gt_label = int(gt_label)
        if gt_label == ignore_label:
            continue
        matched = False
        for pred_label, _score, pred_points in kept:
            iou = weighted_mask_iou(pred_points, gt_points, weights)
            if iou < iou_threshold:
                continue
            matched = True
            if pred_label == gt_label:
                counts["tp"][gt_label] += 1.0
                counts["tp_iou"][gt_label] += iou
            else:
                counts["fp"][pred_label] += 1.0
        if not matched:
            counts["fn"][gt_label] += 1.0
    return counts


def _group_quality(counts: Mapping[str, Sequence[float]],
                   classes: Iterable[int]) -> Dict[str, float]:
    classes = list(classes)
    tp = sum(counts["tp"][c] for c in classes)
    tp_iou = sum(counts["tp_iou"][c] for c in classes)
    fp = sum(counts["fp"][c] for c in classes)
    fn = sum(counts["fn"][c] for c in classes)
    rq = tp / (tp + 0.5 * fp + 0.5 * fn + EPS)
    sq = tp_iou / (tp + EPS)
    return {"pq": rq * sq, "rq": rq, "sq": sq,
            "tp": tp, "fp": fp, "fn": fn}


def per_class_quality(counts: Mapping[str, Sequence[float]]) -> List[Dict[str, float]]:
    """PQ / RQ / SQ of each class."""
    return [_group_quality(counts, [c]) for c in range(len(counts["tp"]))]


def panoptic_report(counts: Mapping[str, Sequence[float]],
                    thing_classes: Sequence[int] = DEFAULT_THING_CLASSES,
                    stuff_classes: Sequence[int] = DEFAULT_STUFF_CLASSES) -> Dict[str, object]:
    """Overall / thing / stuff PQ-RQ-SQ plus the per-class table."""
    n = len(counts["tp"])
    things = [c for c in thing_classes if c < n]
    stuffs = [c for c in stuff_classes if c < n]
    return {
        "all": _group_quality(counts, range(n)),
        "thing": _group_quality(counts, things),
        "stuff": _group_quality(counts, stuffs),
        "per_class": per_class_quality(counts),
    }


def evaluate(predictions: Sequence[Tuple[int, float, Sequence[int]]],
             ground_truth: Sequence[Tuple[int, Sequence[int]]],
             lengths: Sequence[float], num_classes: int = 35,
             thing_classes: Sequence[int] = DEFAULT_THING_CLASSES,
             stuff_classes: Sequence[int] = DEFAULT_STUFF_CLASSES,
             **kwargs) -> Dict[str, object]:
    """One-shot evaluation of a single drawing from raw primitive lengths."""
    weights = point_weights(lengths)
    counts = accumulate(predictions, ground_truth, weights,
                        num_classes=num_classes, **kwargs)
    return panoptic_report(counts, thing_classes, stuff_classes)
