"""Nearest-neighbour label transfer for segmentation evaluation.

The joint SDF paper evaluates segmentation by *transferring* ground-truth labels
to the predicted mesh samples via nearest neighbour in 3D, then computing per-part
IoU / mIoU / accuracy.  The transfer is a deterministic geometric operation and is
implemented here; the resulting label arrays can be scored with the existing
``reconstruction.fewshot_partseg_metrics`` module.

We also provide a palette-invariant *optimal label matching* helper: predicted
part IDs need not coincide with ground-truth IDs (the network may use an
arbitrary palette / part count), so before IoU one may remap predicted labels to
GT labels by greedy majority overlap.  This is deterministic (ties broken by
label order) and is what makes mIoU meaningful across arbitrary palettes.
"""

from __future__ import annotations


def _sqdist(a, b):
    s = 0.0
    for x, y in zip(a, b):
        d = x - y
        s += d * d
    return s


def transfer_labels(query_points, source_points, source_labels):
    """Label for every ``query_points[i]`` = label of its nearest source point.

    Returns a list of transferred labels (length ``len(query_points)``).  Ties in
    distance are broken by ascending source index for determinism.
    """
    if len(source_points) != len(source_labels):
        raise ValueError("source points and labels length mismatch")
    if not source_points:
        raise ValueError("no source points to transfer from")
    out = []
    for q in query_points:
        best_j = 0
        best_d = _sqdist(q, source_points[0])
        for j in range(1, len(source_points)):
            d = _sqdist(q, source_points[j])
            if d < best_d:
                best_d = d
                best_j = j
        out.append(source_labels[best_j])
    return out


def overlap_counts(pred, gt):
    """``{(pred_label, gt_label): count}`` co-occurrence over aligned points."""
    if len(pred) != len(gt):
        raise ValueError("pred and gt length mismatch")
    counts = {}
    for p, g in zip(pred, gt):
        key = (p, g)
        counts[key] = counts.get(key, 0) + 1
    return counts


def match_labels(pred, gt):
    """Greedy palette-invariant remap of predicted labels onto GT labels.

    Each predicted label is assigned the GT label with which it co-occurs most
    (majority vote); ties break by ``repr`` order.  Returns the mapping
    ``{pred_label: gt_label}``.  This does not force a bijection: multiple
    over-segmented predicted parts may map to the same GT part, matching the
    paper's tolerance for over-segmentation.
    """
    counts = overlap_counts(pred, gt)
    by_pred = {}
    for (p, g), c in counts.items():
        by_pred.setdefault(p, []).append((c, g))
    mapping = {}
    for p, cand in by_pred.items():
        # maximise count, tie-break deterministically on the gt label repr
        best = max(cand, key=lambda cg: (cg[0], _neg_key(cg[1])))
        mapping[p] = best[1]
    return mapping


def _neg_key(label):
    # Sort helper: prefer the smaller label repr on ties (so it wins the max).
    return tuple(-ord(ch) for ch in repr(label))


def remap(pred, mapping):
    """Apply a ``{pred_label: gt_label}`` mapping to a predicted label list."""
    return [mapping.get(p, p) for p in pred]


def transferred_accuracy(query_points, source_points, source_labels, query_labels):
    """Convenience: point accuracy of ``query_labels`` vs NN-transferred GT.

    ``query_labels`` are the network's predicted labels at the query points;
    the GT is transferred from the source cloud.  Returns overall accuracy.
    """
    gt = transfer_labels(query_points, source_points, source_labels)
    if len(gt) != len(query_labels):
        raise ValueError("query labels length mismatch")
    if not gt:
        return 1.0
    correct = sum(1 for a, b in zip(query_labels, gt) if a == b)
    return correct / len(gt)
