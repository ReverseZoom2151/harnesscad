"""Threshold calibration for compiler-judge distance labels."""

from __future__ import annotations


def calibrate_threshold(records, thresholds):
    rows = []
    for threshold in sorted(set(map(float, thresholds))):
        tp = fp = tn = fn = 0
        for record in records:
            predicted = record["distance"] <= threshold
            actual = bool(record["accepted"])
            tp += predicted and actual; fp += predicted and not actual
            tn += not predicted and not actual; fn += not predicted and actual
        precision = tp/(tp+fp) if tp+fp else 0.0
        recall = tp/(tp+fn) if tp+fn else 0.0
        rows.append({"threshold": threshold, "tp": tp, "fp": fp, "tn": tn, "fn": fn,
                     "precision": precision, "recall": recall,
                     "f1": 2*precision*recall/(precision+recall)
                     if precision+recall else 0.0,
                     "acceptance_rate": (tp+fp)/(tp+fp+tn+fn) if records else 0.0})
    return tuple(rows)


def select_threshold(calibration_rows):
    if not calibration_rows:
        raise ValueError("calibration rows required")
    return max(calibration_rows, key=lambda row: (row["f1"], row["precision"],
                                                   -row["threshold"]))
