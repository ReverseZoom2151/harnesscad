"""Deterministic part-segmentation IoU / mIoU metrics.

Wang et al. report per-class mean Intersection-over-Union (mIoU, Table I) as the
sole criterion. The metric itself is deterministic and independent of the
learned model, so it is implemented here for scoring any point-level part
segmentation: per-class IoU over point sets, class-averaged mIoU, and the
per-sample instance mIoU used when averaging over query point clouds.
"""

from __future__ import annotations


def _validate(pred, gt):
    if len(pred) != len(gt):
        raise ValueError("prediction and ground-truth length mismatch")


def confusion(pred, gt):
    """Per-class ``{label: (intersection, union)}`` point counts."""
    _validate(pred, gt)
    inter = {}
    total_p = {}
    total_g = {}
    for p, g in zip(pred, gt):
        total_p[p] = total_p.get(p, 0) + 1
        total_g[g] = total_g.get(g, 0) + 1
        if p == g:
            inter[p] = inter.get(p, 0) + 1
    out = {}
    for lab in set(total_p) | set(total_g):
        i = inter.get(lab, 0)
        u = total_p.get(lab, 0) + total_g.get(lab, 0) - i
        out[lab] = (i, u)
    return out


def per_class_iou(pred, gt, labels=None):
    """IoU for each label. A label absent from both pred and gt is IoU 1.0.

    If ``labels`` is given the result is restricted to (and includes all of)
    those labels, so mIoU can be computed over a fixed episode label space.
    """
    conf = confusion(pred, gt)
    if labels is None:
        labels = sorted(conf, key=repr)
    result = {}
    for lab in labels:
        i, u = conf.get(lab, (0, 0))
        result[lab] = 1.0 if u == 0 else i / u
    return result


def mean_iou(pred, gt, labels=None):
    """Class-averaged IoU (mIoU)."""
    ious = per_class_iou(pred, gt, labels)
    if not ious:
        return 1.0
    return sum(ious.values()) / len(ious)


def instance_miou(predictions, ground_truths, labels=None):
    """Mean over samples of each sample's mIoU (the paper's reporting mode).

    ``predictions`` and ``ground_truths`` are parallel sequences of per-sample
    label sequences.
    """
    if len(predictions) != len(ground_truths):
        raise ValueError("mismatched number of samples")
    if not predictions:
        return 1.0
    return sum(mean_iou(p, g, labels)
               for p, g in zip(predictions, ground_truths)) / len(predictions)


def accuracy(pred, gt):
    """Point-level overall accuracy."""
    _validate(pred, gt)
    if not pred:
        return 1.0
    return sum(1 for p, g in zip(pred, gt) if p == g) / len(pred)
