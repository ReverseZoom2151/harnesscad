"""Length-weighted primitive-instance metrics (CADTransformer, CVPR 2022).

Panel-symbol spotting evaluates *instances* made of vectorised primitives, not
pixel masks, so CADTransformer scores an instance by a **length-weighted IoU**
over the primitive indices two instances share (``utils_dataset.cal_instance_iou``):

    IoU(A, B) = sum_{i in A cap B} w_i / sum_{i in A cup B} w_i

where ``w_i = log(1 + length_i)`` down-weights long structural strokes so a
symbol is not dominated by the wall it sits on.  A predicted instance is a
true positive when it matches a ground-truth instance of the *same class* with
IoU above ``0.5`` (the ``IoU_thres`` constant).

This module is stdlib-only and mask-free -- distinct from
``bench.instance_segmentation`` (mask-IoU panoptic quality).  It provides:

* :func:`log_length_weight` -- the ``log(1 + length)`` weighting.
* :func:`weighted_instance_iou` -- length-weighted set IoU of two primitive
  index collections.
* :func:`match_instances` -- greedy IoU + same-class matching to TP/FP/FN.
* :func:`per_class_f1` -- precision / recall / F1 per class and micro-averaged.
* :func:`panoptic_quality` -- SQ / RQ / PQ over the matched pairs.
"""

from __future__ import annotations

import math
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple

IOU_THRESHOLD = 0.5

# An instance is (class_id, {primitive_index: length, ...}).
Instance = Tuple[int, Mapping[int, float]]


def log_length_weight(length: float) -> float:
    """CADTransformer per-primitive weight ``log(1 + length)`` (length >= 0)."""
    if length < 0:
        raise ValueError("length must be non-negative")
    return math.log(1.0 + length)


def weighted_instance_iou(a: Mapping[int, float], b: Mapping[int, float],
                          eps: float = 1e-6) -> float:
    """Length-weighted IoU over the primitive indices of two instances.

    ``a`` and ``b`` map primitive index -> primitive length.  Weights use
    :func:`log_length_weight`.  Shared indices must agree on length; when they
    differ (e.g. GT vs prediction rounding) the ``a`` value is used for the
    intersection and both contribute to the union bookkeeping, matching the
    reference's single ``lengths_dict``.
    """
    lengths: Dict[int, float] = {}
    for idx, ln in a.items():
        lengths[idx] = ln
    for idx, ln in b.items():
        lengths.setdefault(idx, ln)

    inter = set(a) & set(b)
    union = set(a) | set(b)
    w_inter = sum(log_length_weight(lengths[i]) for i in inter)
    w_union = sum(log_length_weight(lengths[i]) for i in union)
    return w_inter / (w_union + eps)


def match_instances(predicted: Sequence[Instance], ground_truth: Sequence[Instance],
                    threshold: float = IOU_THRESHOLD
                    ) -> Tuple[List[Tuple[int, int, float]], List[int], List[int]]:
    """Greedy same-class IoU matching.

    Returns ``(matches, unmatched_pred, unmatched_gt)`` where ``matches`` is a
    list of ``(pred_index, gt_index, iou)`` triples.  Each prediction and GT is
    used at most once; pairs are considered in descending IoU order (ties break
    by index) so the result is deterministic.  Only same-class pairs with IoU
    strictly greater than ``threshold`` can match (the reference's
    ``iou_max > IoU_thres and class_gt == class_pred`` rule).
    """
    scored: List[Tuple[float, int, int]] = []
    for pi, (pcls, pmap) in enumerate(predicted):
        for gi, (gcls, gmap) in enumerate(ground_truth):
            if pcls != gcls:
                continue
            iou = weighted_instance_iou(pmap, gmap)
            if iou > threshold:
                scored.append((iou, pi, gi))
    scored.sort(key=lambda t: (-t[0], t[1], t[2]))

    used_p: set = set()
    used_g: set = set()
    matches: List[Tuple[int, int, float]] = []
    for iou, pi, gi in scored:
        if pi in used_p or gi in used_g:
            continue
        used_p.add(pi)
        used_g.add(gi)
        matches.append((pi, gi, iou))
    unmatched_pred = [i for i in range(len(predicted)) if i not in used_p]
    unmatched_gt = [i for i in range(len(ground_truth)) if i not in used_g]
    return matches, unmatched_pred, unmatched_gt


def per_class_f1(predicted: Sequence[Instance], ground_truth: Sequence[Instance],
                 threshold: float = IOU_THRESHOLD) -> Dict[str, object]:
    """Per-class precision / recall / F1 and micro-averaged totals.

    A matched pair is a TP for its class; an unmatched prediction is a FP for
    its class; an unmatched GT is a FN for its class.
    """
    matches, un_p, un_g = match_instances(predicted, ground_truth, threshold)
    classes = sorted({c for c, _ in predicted} | {c for c, _ in ground_truth})
    tp = {c: 0 for c in classes}
    fp = {c: 0 for c in classes}
    fn = {c: 0 for c in classes}
    for pi, _gi, _iou in matches:
        tp[predicted[pi][0]] += 1
    for pi in un_p:
        fp[predicted[pi][0]] += 1
    for gi in un_g:
        fn[ground_truth[gi][0]] += 1

    per_class: Dict[int, Dict[str, float]] = {}
    for c in classes:
        prec = tp[c] / (tp[c] + fp[c]) if (tp[c] + fp[c]) else 0.0
        rec = tp[c] / (tp[c] + fn[c]) if (tp[c] + fn[c]) else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
        per_class[c] = {"tp": tp[c], "fp": fp[c], "fn": fn[c],
                        "precision": prec, "recall": rec, "f1": f1}

    total_tp = sum(tp.values())
    total_fp = sum(fp.values())
    total_fn = sum(fn.values())
    mprec = total_tp / (total_tp + total_fp) if (total_tp + total_fp) else 0.0
    mrec = total_tp / (total_tp + total_fn) if (total_tp + total_fn) else 0.0
    mf1 = 2 * mprec * mrec / (mprec + mrec) if (mprec + mrec) else 0.0
    return {
        "per_class": per_class,
        "micro": {"tp": total_tp, "fp": total_fp, "fn": total_fn,
                  "precision": mprec, "recall": mrec, "f1": mf1},
    }


def panoptic_quality(predicted: Sequence[Instance], ground_truth: Sequence[Instance],
                     threshold: float = IOU_THRESHOLD) -> Dict[str, float]:
    """Segmentation / recognition / panoptic quality over matched instances.

    ``SQ`` is the mean IoU of matched pairs, ``RQ = TP / (TP + 0.5 FP + 0.5 FN)``
    and ``PQ = SQ * RQ`` -- the standard panoptic decomposition applied to
    length-weighted primitive instances.
    """
    matches, un_p, un_g = match_instances(predicted, ground_truth, threshold)
    tp = len(matches)
    fp = len(un_p)
    fn = len(un_g)
    sq = sum(iou for _p, _g, iou in matches) / tp if tp else 0.0
    denom = tp + 0.5 * fp + 0.5 * fn
    rq = tp / denom if denom else 0.0
    return {"tp": tp, "fp": fp, "fn": fn, "sq": sq, "rq": rq, "pq": sq * rq}


def instance_from_lengths(class_id: int, indices: Iterable[int],
                          lengths: Iterable[float]) -> Instance:
    """Build an :data:`Instance` from parallel index / length sequences."""
    idx = list(indices)
    lns = list(lengths)
    if len(idx) != len(lns):
        raise ValueError("indices and lengths must have equal length")
    return (int(class_id), dict(zip(idx, (float(v) for v in lns))))
