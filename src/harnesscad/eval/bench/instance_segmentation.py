"""Prompt-conditioned instance F1 and Panoptic Quality."""

from __future__ import annotations

from harnesscad.domain.vision.instance_matching import mask_iou


def instance_metrics(predicted, expected, threshold=.5):
    pairs = sorted(((mask_iou(p, e), pi, ei)
                    for pi, p in enumerate(predicted)
                    for ei, e in enumerate(expected)), reverse=True)
    used_p, used_e, scores = set(), set(), []
    for iou, pi, ei in pairs:
        if iou < threshold or pi in used_p or ei in used_e:
            continue
        used_p.add(pi); used_e.add(ei); scores.append(iou)
    tp, fp, fn = len(scores), len(predicted)-len(scores), len(expected)-len(scores)
    denominator = tp + .5*fp + .5*fn
    rq = tp/denominator if denominator else 1.0
    sq = sum(scores)/tp if tp else (1.0 if not predicted and not expected else 0.0)
    f1 = 2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 1.0
    return {"tp": tp, "fp": fp, "fn": fn, "f1": f1, "sq": sq,
            "rq": rq, "pq": sq*rq}
