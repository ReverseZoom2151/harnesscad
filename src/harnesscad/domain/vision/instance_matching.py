"""Deterministic one-to-many mask matching and mask NMS."""

from __future__ import annotations


def mask_iou(left, right):
    a, b = set(left), set(right)
    union = a | b
    return len(a & b)/len(union) if union else 1.0


def one_to_many(predictions, ground_truth, *, maximum_per_gt=5, threshold=0.5):
    candidates = []
    for pi, prediction in enumerate(predictions):
        for gi, truth in enumerate(ground_truth):
            candidates.append((mask_iou(prediction["mask"], truth), prediction["score"],
                               -pi, -gi, pi, gi))
    candidates.sort(reverse=True)
    matched, gt_counts, used = [], {}, set()
    for iou, score, _, __, pi, gi in candidates:
        if iou < threshold or pi in used or gt_counts.get(gi, 0) >= maximum_per_gt:
            continue
        used.add(pi); gt_counts[gi] = gt_counts.get(gi, 0) + 1
        matched.append((pi, gi, iou))
    return tuple(sorted(matched)), tuple(1 if index in used else 0
                                         for index in range(len(predictions)))


def mask_nms(predictions, threshold=.5):
    ordered = sorted(enumerate(predictions),
                     key=lambda item: (-item[1]["score"], item[0]))
    kept = []
    for index, prediction in ordered:
        if all(mask_iou(prediction["mask"], prior["mask"]) <= threshold
               for _, prior in kept):
            kept.append((index, prediction))
    return tuple(index for index, _ in kept)
