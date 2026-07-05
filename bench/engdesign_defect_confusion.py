"""Post-manufacturing defect-inspection confusion metrics (Section 5.2, Tables 10-14).

A VLM inspects concrete images and predicts which of five defect classes (crack,
spallation, efflorescence, exposed bars, corrosion stain) are present. Each
class yields a binary confusion matrix; this module computes precision, recall
(sensitivity / true-positive rate), specificity (true-negative rate), F1 and
accuracy per class plus a macro average, and a perfect-prediction tally over
image queries (splitting defect-present vs defect-absent images).

VLM predictions are injected. No randomness.
"""

from __future__ import annotations

DEFECT_CLASSES = ("crack", "spallation", "efflorescence", "exposed bars",
                  "corrosion stain")


def confusion_metrics(tp, fp, fn, tn):
    """Precision, recall, specificity, F1 and accuracy from a binary matrix."""
    for name, v in (("tp", tp), ("fp", fp), ("fn", fn), ("tn", tn)):
        if v < 0:
            raise ValueError("%s must be non-negative" % name)
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    specificity = tn / (tn + fp) if (tn + fp) else 0.0
    f1 = (2 * precision * recall / (precision + recall)
          if (precision + recall) else 0.0)
    total = tp + fp + fn + tn
    accuracy = (tp + tn) / total if total else 0.0
    return {"precision": precision, "recall": recall,
            "specificity": specificity, "f1": f1, "accuracy": accuracy,
            "tp": tp, "fp": fp, "fn": fn, "tn": tn}


def f1_from_matrix(tp, fp, fn, tn):
    """Convenience: F1 score for a binary confusion matrix."""
    return confusion_metrics(tp, fp, fn, tn)["f1"]


def defect_scorecard(matrices):
    """Per-class and macro-averaged metrics.

    matrices: mapping class_name -> {"tp","fp","fn","tn"}. Returns per-class
    confusion_metrics and macro averages of precision/recall/specificity/f1.
    """
    per = {cls: confusion_metrics(m["tp"], m["fp"], m["fn"], m["tn"])
           for cls, m in matrices.items()}
    if not per:
        raise ValueError("no matrices")
    n = len(per)
    macro = {}
    for k in ("precision", "recall", "specificity", "f1", "accuracy"):
        macro[k] = sum(v[k] for v in per.values()) / n
    return {"per_class": per, "macro": macro, "n_classes": n}


def perfect_prediction_tally(queries):
    """Count image queries where the predicted defect set exactly matches truth.

    queries: iterable of dicts {"predicted": set-like, "truth": set-like}. An
    empty truth set marks a defect-absent (background) image. Returns overall
    perfect count and the split between defect-present and defect-absent images.
    """
    total = perfect = perfect_present = perfect_absent = 0
    present = absent = 0
    for q in queries:
        pred = {s.strip().lower() for s in q["predicted"]}
        truth = {s.strip().lower() for s in q["truth"]}
        total += 1
        is_absent = not truth
        if is_absent:
            absent += 1
        else:
            present += 1
        if pred == truth:
            perfect += 1
            if is_absent:
                perfect_absent += 1
            else:
                perfect_present += 1
    return {"total": total, "perfect": perfect,
            "perfect_present": perfect_present, "perfect_absent": perfect_absent,
            "present": present, "absent": absent}
