"""Depth-estimation metrics from E3D-Bench (Sec. 3.1 / 3.2).

E3D-Bench evaluates sparse-view and video depth with two standard metrics:

* Absolute Relative Error (AbsRel): mean over valid pixels of
  ``|d_pred - d_gt| / d_gt``.
* Inlier ratio ``delta < tau``: fraction of pixels whose ratio
  ``max(d_pred/d_gt, d_gt/d_pred)`` is below a threshold ``tau``.
  The paper uses a strict ``tau = 1.03`` for sparse-view depth and a looser
  ``tau = 1.25`` for video depth.

For normalized (scale-ambiguous) models the paper applies *median scaling*:
each prediction is multiplied by ``median(d_gt) / median(d_pred)`` over valid
pixels before scoring.  Metric-scale models are scored on raw output and,
optionally, additionally under median alignment "for fair comparison".

All functions are stdlib-only and deterministic.  Depths are flat sequences of
floats; a per-pixel validity mask (or a positivity default) selects the pixels
that contribute.
"""

from __future__ import annotations

from typing import List, Optional, Sequence

Depths = Sequence[float]
Mask = Optional[Sequence[bool]]

# threshold constants used by the paper
TAU_SPARSE = 1.03
TAU_VIDEO = 1.25
DELTA_THRESHOLDS = (1.25, 1.25 ** 2, 1.25 ** 3)


def _valid_indices(pred: Depths, gt: Depths, mask: Mask) -> List[int]:
    if len(pred) != len(gt):
        raise ValueError("pred and gt must have equal length")
    if mask is not None and len(mask) != len(gt):
        raise ValueError("mask must match depth length")
    idx = []
    for i in range(len(gt)):
        if mask is not None and not mask[i]:
            continue
        if gt[i] <= 0.0:
            continue  # invalid / no ground-truth depth
        idx.append(i)
    return idx


def _median(values: Sequence[float]) -> float:
    s = sorted(values)
    n = len(s)
    if n == 0:
        raise ValueError("median of empty sequence")
    mid = n // 2
    if n % 2 == 1:
        return s[mid]
    return 0.5 * (s[mid - 1] + s[mid])


def median_scale_factor(pred: Depths, gt: Depths, mask: Mask = None) -> float:
    """Scalar ``median(gt)/median(pred)`` over valid pixels (>0 predictions)."""
    idx = _valid_indices(pred, gt, mask)
    pv = [pred[i] for i in idx if pred[i] > 0.0]
    gv = [gt[i] for i in idx if pred[i] > 0.0]
    if not pv:
        return 1.0
    mp = _median(pv)
    if mp == 0.0:
        return 1.0
    return _median(gv) / mp


def apply_scale(pred: Depths, factor: float) -> List[float]:
    return [p * factor for p in pred]


def abs_rel(pred: Depths, gt: Depths, mask: Mask = None) -> float:
    """Mean absolute relative error over valid pixels."""
    idx = _valid_indices(pred, gt, mask)
    if not idx:
        return 0.0
    total = 0.0
    for i in idx:
        total += abs(pred[i] - gt[i]) / gt[i]
    return total / len(idx)


def sq_rel(pred: Depths, gt: Depths, mask: Mask = None) -> float:
    """Mean squared relative error ``(d_pred-d_gt)^2 / d_gt`` (auxiliary)."""
    idx = _valid_indices(pred, gt, mask)
    if not idx:
        return 0.0
    total = 0.0
    for i in idx:
        total += (pred[i] - gt[i]) ** 2 / gt[i]
    return total / len(idx)


def delta_inlier_ratio(pred: Depths, gt: Depths, tau: float = TAU_VIDEO,
                       mask: Mask = None) -> float:
    """Fraction of pixels with ``max(p/g, g/p) < tau`` (returned in [0, 1])."""
    if tau <= 1.0:
        raise ValueError("tau must be > 1")
    idx = _valid_indices(pred, gt, mask)
    if not idx:
        return 0.0
    inliers = 0
    for i in idx:
        p = pred[i]
        g = gt[i]
        if p <= 0.0:
            continue  # ratio undefined / infinite error -> not an inlier
        ratio = max(p / g, g / p)
        if ratio < tau:
            inliers += 1
    return inliers / len(idx)


def rmse(pred: Depths, gt: Depths, mask: Mask = None) -> float:
    """Root-mean-square depth error over valid pixels (auxiliary)."""
    import math
    idx = _valid_indices(pred, gt, mask)
    if not idx:
        return 0.0
    total = sum((pred[i] - gt[i]) ** 2 for i in idx)
    return math.sqrt(total / len(idx))


def evaluate_depth(pred: Depths, gt: Depths, mask: Mask = None,
                   tau: float = TAU_VIDEO, align_median: bool = False) -> dict:
    """Full depth report; optionally median-align first (normalized models).

    Returns a dict with ``abs_rel``, ``sq_rel``, ``rmse``, ``delta`` (inlier
    ratio at ``tau`` in percent, matching the paper's tables), the standard
    ``delta1/delta2/delta3`` at 1.25/1.25^2/1.25^3, ``scale`` applied, and the
    valid-pixel ``count``.
    """
    scale = 1.0
    p = list(pred)
    if align_median:
        scale = median_scale_factor(pred, gt, mask)
        p = apply_scale(pred, scale)
    idx = _valid_indices(p, gt, mask)
    return {
        "abs_rel": abs_rel(p, gt, mask),
        "sq_rel": sq_rel(p, gt, mask),
        "rmse": rmse(p, gt, mask),
        "delta": 100.0 * delta_inlier_ratio(p, gt, tau, mask),
        "delta1": delta_inlier_ratio(p, gt, DELTA_THRESHOLDS[0], mask),
        "delta2": delta_inlier_ratio(p, gt, DELTA_THRESHOLDS[1], mask),
        "delta3": delta_inlier_ratio(p, gt, DELTA_THRESHOLDS[2], mask),
        "scale": scale,
        "count": len(idx),
    }
