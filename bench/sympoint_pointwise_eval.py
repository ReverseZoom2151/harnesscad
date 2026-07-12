"""Point-wise semantic evaluation for vectorised CAD (SymPoint, ECCV 2024).

The semantic half of SymPoint's benchmark (``PointWiseEval``) scores *every
primitive-point* of a floor plan with a confusion matrix and reports three
numbers the harness did not have:

* **mIoU** -- unweighted mean of the per-class IoUs (classes with no ground-truth
  support are excluded, not counted as zero).
* **fwIoU** -- *frequency-weighted* IoU: the per-class IoUs weighted by the
  ground-truth share of each class.  On floor plans the class distribution is
  wildly skewed (walls dominate), so mIoU and fwIoU tell very different stories
  and SymPoint reports both.
* **pACC** -- overall point accuracy, and **mAcc**, the mean per-class recall.

Points labelled ``ignore_label`` (background, id 35 in the reference dataset)
are dropped before the matrix is updated -- background is not a class you can
score.

This complements, and does not duplicate, ``bench.segmentation_metrics``
(face-id confusion, accuracy, macro IoU only): the ignore-label handling, the
frequency weighting and the streaming :class:`ConfusionMatrix` accumulator are
the SymPoint-specific pieces.  Stdlib-only and deterministic.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence

EPS = 1e-8

DEFAULT_NUM_CLASSES = 35
DEFAULT_IGNORE_LABEL = 35


class ConfusionMatrix:
    """Streaming ``num_classes x num_classes`` confusion accumulator.

    ``matrix[gt][pred]`` counts points of true class ``gt`` predicted ``pred``.
    Predictions equal to the ignore label are *kept* (a model may predict
    background where a real class exists, and that must cost recall), whereas
    ground-truth ignore points are dropped.
    """

    def __init__(self, num_classes: int = DEFAULT_NUM_CLASSES,
                 ignore_label: int = DEFAULT_IGNORE_LABEL) -> None:
        if num_classes <= 0:
            raise ValueError("num_classes must be positive")
        self.num_classes = num_classes
        self.ignore_label = ignore_label
        # one extra row/column absorbs a predicted ignore label
        self.matrix = [[0 for _ in range(num_classes + 1)] for _ in range(num_classes + 1)]

    def update(self, predicted: Sequence[int], ground_truth: Sequence[int]) -> "ConfusionMatrix":
        if len(predicted) != len(ground_truth):
            raise ValueError("predicted and ground_truth must be the same length")
        n = self.num_classes
        for pred, gt in zip(predicted, ground_truth):
            gt = int(gt)
            pred = int(pred)
            if gt == self.ignore_label:
                continue
            if not 0 <= gt < n:
                raise ValueError("ground-truth label out of range: %d" % gt)
            if pred == self.ignore_label or not 0 <= pred < n:
                pred = n  # absorb into the extra column
            self.matrix[gt][pred] += 1
        return self

    def support(self) -> List[int]:
        """Ground-truth point count per class."""
        return [sum(self.matrix[c][:self.num_classes]) + self.matrix[c][self.num_classes]
                for c in range(self.num_classes)]

    def predicted_count(self) -> List[int]:
        """Predicted point count per class (over non-ignored ground truth)."""
        return [sum(self.matrix[g][c] for g in range(self.num_classes))
                for c in range(self.num_classes)]

    def true_positives(self) -> List[int]:
        return [self.matrix[c][c] for c in range(self.num_classes)]


def per_class_scores(cm: ConfusionMatrix) -> List[Dict[str, Optional[float]]]:
    """Per-class support, accuracy (recall) and IoU; ``None`` where undefined."""
    tp = cm.true_positives()
    pos_gt = cm.support()
    pos_pred = cm.predicted_count()
    out: List[Dict[str, Optional[float]]] = []
    for c in range(cm.num_classes):
        union = pos_gt[c] + pos_pred[c] - tp[c]
        out.append({
            "support": float(pos_gt[c]),
            "accuracy": (tp[c] / (pos_gt[c] + EPS)) if pos_gt[c] > 0 else None,
            "iou": (tp[c] / (union + EPS)) if (pos_gt[c] > 0 and union > 0) else None,
        })
    return out


def evaluate(cm: ConfusionMatrix) -> Dict[str, object]:
    """mIoU / fwIoU / mAcc / pACC plus the per-class table."""
    scores = per_class_scores(cm)
    tp = cm.true_positives()
    pos_gt = cm.support()
    total_gt = sum(pos_gt)

    ious = [(c, s["iou"]) for c, s in enumerate(scores) if s["iou"] is not None]
    accs = [s["accuracy"] for s in scores if s["accuracy"] is not None]

    miou = sum(v for _, v in ious) / len(ious) if ious else 0.0
    macc = sum(accs) / len(accs) if accs else 0.0
    fwiou = sum(v * (pos_gt[c] / (total_gt + EPS)) for c, v in ious) if total_gt else 0.0
    pacc = sum(tp) / (total_gt + EPS) if total_gt else 0.0
    return {"miou": miou, "fwiou": fwiou, "macc": macc, "pacc": pacc,
            "per_class": scores}


def point_wise_eval(predicted: Sequence[int], ground_truth: Sequence[int],
                    num_classes: int = DEFAULT_NUM_CLASSES,
                    ignore_label: int = DEFAULT_IGNORE_LABEL) -> Dict[str, object]:
    """One-shot semantic evaluation of a single drawing."""
    cm = ConfusionMatrix(num_classes, ignore_label)
    cm.update(predicted, ground_truth)
    return evaluate(cm)
