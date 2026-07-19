"""Evaluation metrics for predicted operation maps.

The four regression networks are scored with three distinct,
non-obvious objectives, all of which double as evaluation metrics for any
predictor of the same maps (learned or hand-written).  They are implemented
here, stdlib-only and deterministic:

  * :func:`face_heatmap_error` -- plain MSE / MAE against the ground-truth
    stitching-face heat map (``fh_loss`` / ``real_fh_loss``).
  * :func:`masked_curve_error` -- the *stroke-masked* curve regression used by
    addSub / extrusion / sweep: the prediction is first multiplied by the stroke
    mask ``1 - user_stroke`` (only pixels away from the drawn stroke count), and
    the error is a weighted mean with the same mask as weights.  Predictions on
    stroke pixels are therefore free, which is exactly the training convention.
  * :func:`foreground_background_curve_error` — the bevel objective: the
    prediction is split into a foreground part (inside the ground-truth curve
    mask) and a background part (the rest of the stroke-masked region); the two
    squared/absolute sums are added and normalised by the number of stroke-mask
    pixels, not by the image area.  This keeps a thin curve from being drowned
    out by background.
  * :func:`curve_class_metrics` — per-class precision / recall / F1 / IoU over
    the derived base/offset/profile curve labels.
  * :func:`operation_report` — one dict combining the above for a sample.

Every function takes flat row-major float lists of equal length.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Sequence

from harnesscad.domain.drawings.block_codec import CURVE_CLASS_NAMES


class MetricError(ValueError):
    """Raised when maps of different sizes (or empty maps) are compared."""


def _pair(a: Sequence[float], b: Sequence[float]) -> int:
    if len(a) != len(b):
        raise MetricError("map size mismatch: {} vs {}".format(len(a), len(b)))
    if not a:
        raise MetricError("empty map")
    return len(a)


@dataclass(frozen=True)
class ErrorPair:
    """L2 (mse) and L1 ("real") error, the pair every such loss reports."""
    mse: float
    mae: float


def face_heatmap_error(
    predicted: Sequence[float], truth: Sequence[float]
) -> ErrorPair:
    n = _pair(predicted, truth)
    se = sum((float(p) - float(t)) ** 2 for p, t in zip(predicted, truth))
    ae = sum(abs(float(p) - float(t)) for p, t in zip(predicted, truth))
    return ErrorPair(mse=se / n, mae=ae / n)


def stroke_mask(user_stroke: Sequence[float]) -> List[float]:
    """``1 - user_stroke`` — the region the curve heads are scored on."""
    if not user_stroke:
        raise MetricError("empty map")
    return [1.0 - float(v) for v in user_stroke]


def masked_curve_error(
    predicted: Sequence[float],
    truth: Sequence[float],
    user_stroke: Sequence[float],
) -> ErrorPair:
    """Stroke-masked curve regression error (addSub / extrusion / sweep)."""
    _pair(predicted, truth)
    _pair(predicted, user_stroke)
    mask = stroke_mask(user_stroke)
    wsum = sum(mask)
    if wsum <= 0.0:
        raise MetricError("stroke mask is empty (every pixel is a stroke pixel)")
    se = 0.0
    ae = 0.0
    for p, t, m in zip(predicted, truth, mask):
        masked_pred = float(p) * m
        d = masked_pred - float(t)
        se += m * d * d
        ae += m * abs(d)
    return ErrorPair(mse=se / wsum, mae=ae / wsum)


def foreground_background_curve_error(
    predicted: Sequence[float],
    curve_mask: Sequence[float],
    user_stroke: Sequence[float],
) -> ErrorPair:
    """Bevel-style fg/bg curve error, normalised by the stroke-mask pixel count."""
    _pair(predicted, curve_mask)
    _pair(predicted, user_stroke)
    mask = stroke_mask(user_stroke)
    nb = sum(mask)
    if nb <= 0.0:
        raise MetricError("stroke mask is empty (every pixel is a stroke pixel)")
    se = 0.0
    ae = 0.0
    for p, c, m in zip(predicted, curve_mask, mask):
        cm = float(c)
        diff = m - cm  # background = stroke mask minus the curve
        fg = float(p) * cm
        bg = float(p) * diff
        se += (cm - fg) ** 2 + bg * bg
        ae += abs(cm - fg) + abs(bg)
    return ErrorPair(mse=se / nb, mae=ae / nb)


@dataclass(frozen=True)
class ClassScore:
    name: str
    tp: int
    fp: int
    fn: int

    @property
    def precision(self) -> float:
        d = self.tp + self.fp
        return self.tp / float(d) if d else 0.0

    @property
    def recall(self) -> float:
        d = self.tp + self.fn
        return self.tp / float(d) if d else 0.0

    @property
    def f1(self) -> float:
        p, r = self.precision, self.recall
        return 2.0 * p * r / (p + r) if (p + r) else 0.0

    @property
    def iou(self) -> float:
        d = self.tp + self.fp + self.fn
        return self.tp / float(d) if d else 0.0


def curve_class_metrics(
    predicted: Sequence[int],
    truth: Sequence[int],
    curve_mask: Sequence[float],
) -> Dict[str, ClassScore]:
    """Per-class scores over the pixels where a curve actually exists.

    ``predicted``/``truth`` are class indices (0 base, 1 offset, 2 profile);
    ``curve_mask`` selects the pixels that carry a curve at all, so background
    pixels (which default to class 0) cannot inflate the base-curve score.
    """
    _pair([float(v) for v in predicted], [float(v) for v in truth])
    _pair([float(v) for v in predicted], curve_mask)
    tp = [0] * len(CURVE_CLASS_NAMES)
    fp = [0] * len(CURVE_CLASS_NAMES)
    fn = [0] * len(CURVE_CLASS_NAMES)
    for p, t, m in zip(predicted, truth, curve_mask):
        if float(m) <= 0.0:
            continue
        pi, ti = int(p), int(t)
        if not (0 <= pi < len(tp)) or not (0 <= ti < len(tp)):
            raise MetricError("curve class index out of range")
        if pi == ti:
            tp[ti] += 1
        else:
            fp[pi] += 1
            fn[ti] += 1
    return {
        name: ClassScore(name, tp[i], fp[i], fn[i])
        for i, name in enumerate(CURVE_CLASS_NAMES)
    }


def mean_iou(scores: Dict[str, ClassScore]) -> float:
    """Mean IoU over the classes that appear in the ground truth."""
    present = [s for s in scores.values() if (s.tp + s.fn) > 0]
    if not present:
        return 0.0
    return sum(s.iou for s in present) / float(len(present))


def operation_report(
    face_pred: Sequence[float],
    face_truth: Sequence[float],
    curve_pred: Sequence[float],
    curve_truth: Sequence[float],
    user_stroke: Sequence[float],
    curve_head: str = "regression",
) -> Dict[str, float]:
    """Combined face + curve report for one predicted operation.

    ``curve_head`` picks the objective the branch is trained with:
    ``'regression'`` (addSub/extrusion/sweep) or ``'heatmap'`` (bevel).
    """
    if curve_head not in ("regression", "heatmap"):
        raise MetricError("unknown curve head: {}".format(curve_head))
    fh = face_heatmap_error(face_pred, face_truth)
    if curve_head == "regression":
        cv = masked_curve_error(curve_pred, curve_truth, user_stroke)
    else:
        cv = foreground_background_curve_error(curve_pred, curve_truth, user_stroke)
    return {
        "face_mse": fh.mse,
        "face_mae": fh.mae,
        "curve_mse": cv.mse,
        "curve_mae": cv.mae,
        "total_loss": fh.mse + cv.mse,
    }
