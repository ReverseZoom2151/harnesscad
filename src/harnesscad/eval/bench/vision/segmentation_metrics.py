"""Stable-ID face segmentation confusion, accuracy and macro IoU."""

from __future__ import annotations


def face_segmentation_metrics(expected, predicted, *, labels=None):
    if set(expected) != set(predicted):
        return {"available": False, "error": "face-id-mismatch",
                "missing": tuple(sorted(set(expected) - set(predicted))),
                "extra": tuple(sorted(set(predicted) - set(expected)))}
    classes = tuple(sorted(set(labels or ()) | set(expected.values()) | set(predicted.values())))
    matrix = {actual: {guess: 0 for guess in classes} for actual in classes}
    for face_id in sorted(expected):
        matrix[expected[face_id]][predicted[face_id]] += 1
    total = len(expected)
    correct = sum(matrix[label][label] for label in classes)
    per_class = {}
    for label in classes:
        tp = matrix[label][label]
        fp = sum(matrix[other][label] for other in classes if other != label)
        fn = sum(matrix[label][other] for other in classes if other != label)
        union = tp + fp + fn
        support = sum(matrix[label].values())
        per_class[label] = {
            "support": support, "correct": tp,
            "accuracy": tp / support if support else None,
            "iou": tp / union if union else None,
        }
    ious = [item["iou"] for item in per_class.values() if item["iou"] is not None]
    return {"available": True, "accuracy": correct / total if total else 1.0,
            "macro_iou": sum(ious) / len(ious) if ious else 1.0,
            "classes": per_class, "confusion": matrix}
