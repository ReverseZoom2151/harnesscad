"""6D pose error metrics for RAG-6DPose-style evaluation.

RAG-6DPose (Wang et al.) estimates the 6D pose ``p = (R, t)`` of a rigid object
whose CAD model ``M`` is known, then evaluates against ground truth on the BOP
benchmark.  This module provides the *deterministic* pose-error metrics used to
score such predictions -- independent of the learned feature-matching network:

  * ``add`` -- ADD, the average distance between the model points transformed by
    the predicted pose and by the ground-truth pose (Hinterstoisser et al.);
  * ``add_s`` -- ADD-S, the symmetric variant that matches each predicted model
    point to its *nearest* ground-truth model point (for symmetric objects such
    as the LM-O eggbox / glue);
  * ``rotation_angle_error`` -- geodesic rotation error
    ``arccos((tr(R_gt^T R_pred) - 1) / 2)`` in radians/degrees;
  * ``translation_error`` -- Euclidean distance ``||t_gt - t_pred||``;
  * ``pose_accuracy_5cm_5deg`` -- the classic 5 cm / 5 deg accuracy flag;
  * ``add_recall`` / ``add_s_recall`` -- fraction of a dataset whose ADD(-S) is
    below a fraction of the object diameter (the "ADD < 0.1 d" recall).

Everything is stdlib-only, no wall clock and no randomness.  Poses are expressed
as a 3x3 rotation ``R`` (row-major nested tuples/lists) and a length-3
translation ``t``; model points are length-3 sequences.  SE(3) basics are not
duplicated -- these are pure error functions layered on top of raw (R, t).
"""

from __future__ import annotations

import math
from typing import List, Sequence, Tuple

Vec3 = Sequence[float]
Mat3 = Sequence[Sequence[float]]


def _transform(R: Mat3, t: Vec3, p: Vec3) -> Tuple[float, float, float]:
    return (
        R[0][0] * p[0] + R[0][1] * p[1] + R[0][2] * p[2] + t[0],
        R[1][0] * p[0] + R[1][1] * p[1] + R[1][2] * p[2] + t[1],
        R[2][0] * p[0] + R[2][1] * p[1] + R[2][2] * p[2] + t[2],
    )


def transform_points(R: Mat3, t: Vec3, points: Sequence[Vec3]) -> List[Tuple[float, float, float]]:
    """Apply the rigid pose ``x -> R x + t`` to every model point."""
    return [_transform(R, t, p) for p in points]


def _dist(a: Vec3, b: Vec3) -> float:
    return math.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2 + (a[2] - b[2]) ** 2)


def add(R_pred: Mat3, t_pred: Vec3, R_gt: Mat3, t_gt: Vec3,
        model_points: Sequence[Vec3]) -> float:
    """ADD -- average distance of model points under the two poses.

    ``ADD = (1/n) sum_i || (R_pred p_i + t_pred) - (R_gt p_i + t_gt) ||``.
    Identical poses give exactly 0.  Raises for an empty model.
    """
    n = len(model_points)
    if n == 0:
        raise ValueError("model_points must be non-empty")
    total = 0.0
    for p in model_points:
        a = _transform(R_pred, t_pred, p)
        b = _transform(R_gt, t_gt, p)
        total += _dist(a, b)
    return total / n


def add_s(R_pred: Mat3, t_pred: Vec3, R_gt: Mat3, t_gt: Vec3,
          model_points: Sequence[Vec3]) -> float:
    """ADD-S -- symmetric average distance (nearest-point matching).

    For each predicted model point, the closest ground-truth model point is used
    instead of the identically-indexed one, making the metric invariant to the
    object's symmetries.  ``ADD_S <= ADD`` always; equal poses give 0.
    """
    n = len(model_points)
    if n == 0:
        raise ValueError("model_points must be non-empty")
    gt = [_transform(R_gt, t_gt, p) for p in model_points]
    total = 0.0
    for p in model_points:
        a = _transform(R_pred, t_pred, p)
        best = min(_dist(a, g) for g in gt)
        total += best
    return total / n


def rotation_angle_error(R_pred: Mat3, R_gt: Mat3) -> float:
    """Geodesic rotation error in radians.

    ``theta = arccos((tr(R_gt^T R_pred) - 1) / 2)`` -- the angle of the relative
    rotation ``R_gt^T R_pred``.  The cosine argument is clamped to [-1, 1] so
    floating-point drift never raises.  Result lies in ``[0, pi]``.
    """
    # trace(R_gt^T R_pred) = sum over i,k of R_gt[k][i] * R_pred[k][i]
    tr = 0.0
    for i in range(3):
        for k in range(3):
            tr += R_gt[k][i] * R_pred[k][i]
    cos_theta = (tr - 1.0) / 2.0
    if cos_theta > 1.0:
        cos_theta = 1.0
    elif cos_theta < -1.0:
        cos_theta = -1.0
    return math.acos(cos_theta)


def rotation_angle_error_deg(R_pred: Mat3, R_gt: Mat3) -> float:
    """Geodesic rotation error in degrees (see :func:`rotation_angle_error`)."""
    return math.degrees(rotation_angle_error(R_pred, R_gt))


def translation_error(t_pred: Vec3, t_gt: Vec3) -> float:
    """Euclidean translation error ``||t_gt - t_pred||``."""
    return _dist(t_pred, t_gt)


def pose_accuracy_5cm_5deg(R_pred: Mat3, t_pred: Vec3, R_gt: Mat3, t_gt: Vec3,
                           trans_thresh: float = 0.05,
                           rot_thresh_deg: float = 5.0) -> bool:
    """The 5 cm / 5 deg accuracy flag.

    Returns ``True`` iff the translation error is below ``trans_thresh`` (metres)
    *and* the rotation error is below ``rot_thresh_deg`` degrees.  Thresholds are
    configurable so the same primitive serves e.g. 2 cm / 2 deg protocols.
    """
    return (translation_error(t_pred, t_gt) < trans_thresh
            and rotation_angle_error_deg(R_pred, R_gt) < rot_thresh_deg)


def model_diameter(model_points: Sequence[Vec3]) -> float:
    """Largest pairwise distance between model points (the object diameter).

    Used as the reference length for ADD recall thresholds.  O(n^2); intended for
    the small sampled point sets used in evaluation.
    """
    n = len(model_points)
    if n == 0:
        raise ValueError("model_points must be non-empty")
    best = 0.0
    for i in range(n):
        for j in range(i + 1, n):
            d = _dist(model_points[i], model_points[j])
            if d > best:
                best = d
    return best


def add_recall(predictions: Sequence[Tuple[Mat3, Vec3, Mat3, Vec3]],
               model_points: Sequence[Vec3],
               diameter_fraction: float = 0.1,
               symmetric: bool = False) -> float:
    """Fraction of predictions whose ADD (or ADD-S) is below ``k * diameter``.

    ``predictions`` is a sequence of ``(R_pred, t_pred, R_gt, t_gt)`` tuples all
    scored against the same ``model_points``.  With ``symmetric=True`` the ADD-S
    metric is used.  This is the standard "ADD(-S) < 0.1 d" recall.
    """
    if not predictions:
        return 0.0
    d = model_diameter(model_points)
    thresh = diameter_fraction * d
    metric = add_s if symmetric else add
    hits = 0
    for R_pred, t_pred, R_gt, t_gt in predictions:
        if metric(R_pred, t_pred, R_gt, t_gt, model_points) <= thresh:
            hits += 1
    return hits / len(predictions)


def add_s_recall(predictions: Sequence[Tuple[Mat3, Vec3, Mat3, Vec3]],
                 model_points: Sequence[Vec3],
                 diameter_fraction: float = 0.1) -> float:
    """Convenience wrapper: ADD-S recall (see :func:`add_recall`)."""
    return add_recall(predictions, model_points, diameter_fraction, symmetric=True)
