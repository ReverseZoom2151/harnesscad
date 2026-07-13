"""Dependency-free classification, detection, and segmentation metrics."""

from __future__ import annotations
from collections import defaultdict
from typing import Callable, Iterable, Mapping, Sequence


def classification_metrics(expected, predicted) -> dict:
    a, b = tuple(expected), tuple(predicted)
    if len(a) != len(b): raise ValueError("length mismatch")
    labels = sorted(set(a) | set(b), key=repr)
    per = {}
    for label in labels:
        tp = sum(x == label and y == label for x, y in zip(a, b))
        fp = sum(x != label and y == label for x, y in zip(a, b))
        fn = sum(x == label and y != label for x, y in zip(a, b))
        p = tp / (tp + fp) if tp + fp else 0.0
        r = tp / (tp + fn) if tp + fn else 0.0
        per[label] = {"precision": p, "recall": r,
                      "f1": 2*p*r/(p+r) if p+r else 0.0}
    return {"accuracy": sum(x == y for x, y in zip(a, b))/len(a) if a else None,
            "per_class": per,
            "macro_f1": sum(v["f1"] for v in per.values())/len(per) if per else None}


def top_k_accuracy(expected, ranked: Iterable[Sequence[object]], k: int) -> float | None:
    pairs = tuple(zip(expected, ranked))
    return sum(y in tuple(r)[:k] for y, r in pairs)/len(pairs) if pairs else None


def box_iou(a: Sequence[float], b: Sequence[float]) -> float:
    x1, y1, x2, y2 = max(a[0],b[0]), max(a[1],b[1]), min(a[2],b[2]), min(a[3],b[3])
    inter = max(0,x2-x1)*max(0,y2-y1)
    aa=max(0,a[2]-a[0])*max(0,a[3]-a[1]); bb=max(0,b[2]-b[0])*max(0,b[3]-b[1])
    return inter/(aa+bb-inter) if aa+bb-inter else 0.0


def detection_at_threshold(expected, predicted, threshold=.5) -> dict:
    """Greedy score-ordered matching; tuples are (label, box, score)."""
    gold=list(expected); used=set(); tp=fp=0
    for label, box, score in sorted(predicted, key=lambda x: (-x[2], repr(x[0]), tuple(x[1]))):
        choices=[(box_iou(box,gbox),i) for i,(glab,gbox) in enumerate(gold)
                 if i not in used and glab==label]
        best=max(choices, default=(0,-1))
        if best[0] >= threshold: tp+=1; used.add(best[1])
        else: fp+=1
    fn=len(gold)-len(used)
    p=tp/(tp+fp) if tp+fp else 0.0; r=tp/(tp+fn) if tp+fn else 0.0
    return {"precision":p,"recall":r,"f1":2*p*r/(p+r) if p+r else 0.0,
            "tp":tp,"fp":fp,"fn":fn}


def average_precision(expected, predicted, threshold=.5) -> float:
    """Area under the score-ranked precision/recall curve (all-point envelope)."""
    gold = list(expected)
    if not gold:
        return 0.0
    used = set()
    points = []
    tp = fp = 0
    for label, box, score in sorted(
        predicted, key=lambda x: (-x[2], repr(x[0]), tuple(x[1]))
    ):
        choices = [(box_iou(box, gbox), i)
                   for i, (glab, gbox) in enumerate(gold)
                   if i not in used and glab == label]
        best = max(choices, default=(0.0, -1))
        if best[0] >= threshold:
            tp += 1
            used.add(best[1])
        else:
            fp += 1
        points.append((tp / len(gold), tp / (tp + fp)))
    ap = previous = 0.0
    for recall in sorted({0.0, 1.0, *(r for r, _ in points)}):
        precision = max((p for r, p in points if r >= recall), default=0.0)
        ap += (recall - previous) * precision
        previous = recall
    return ap


def mean_average_precision(expected, predicted, thresholds=None) -> float:
    values = tuple(thresholds if thresholds is not None
                   else (0.50 + 0.05 * i for i in range(10)))
    return (sum(average_precision(expected, predicted, value) for value in values)
            / len(values)) if values else 0.0


def mask_iou(expected: Iterable[object], predicted: Iterable[object]) -> float:
    a,b=set(expected),set(predicted); u=a|b
    return len(a&b)/len(u) if u else 1.0


def mean_iou(expected: Mapping[object,Iterable], predicted: Mapping[object,Iterable]) -> float|None:
    labels=set(expected)|set(predicted)
    return sum(mask_iou(expected.get(x,()),predicted.get(x,())) for x in labels)/len(labels) if labels else None


def slice_metric(records: Iterable[object], group: Callable[[object], object],
                 metric: Callable[[tuple], object]) -> dict:
    groups=defaultdict(list)
    for record in records: groups[group(record)].append(record)
    return {k:metric(tuple(groups[k])) for k in sorted(groups,key=repr)}
