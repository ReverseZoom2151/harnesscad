"""Multi-view relative camera-pose metrics (E3D-Bench, Sec. 3.3).

E3D-Bench reports three standard trajectory metrics, computed *after* a Sim(3)
Umeyama alignment between the predicted and ground-truth camera trajectories:

* Absolute Translation Error (ATE): RMS distance between aligned predicted and
  ground-truth camera centres.
* Relative Pose Error, translation (RPE-trans): error in the frame-to-frame
  translation increments.
* Relative Pose Error, rotation (RPE-rot): geodesic angle error in the
  frame-to-frame rotation increments (degrees).

A camera pose is given as ``(R, t)`` with ``R`` a 3x3 world-from-camera rotation
and ``t`` the camera centre (translation).  All functions are stdlib-only and
deterministic.
"""

from __future__ import annotations

import math
from typing import List, Sequence, Tuple

from harnesscad.domain.geometry.transforms.e3dbench_umeyama import (
    Sim3,
    matmul,
    transpose,
    umeyama_alignment,
)

Mat = List[List[float]]
Vec3 = Sequence[float]
Pose = Tuple[Sequence[Sequence[float]], Vec3]


def rotation_angle_error(Ra: Sequence[Sequence[float]],
                         Rb: Sequence[Sequence[float]],
                         degrees: bool = True) -> float:
    """Geodesic angle between two rotations, ``angle(Ra^T Rb)``."""
    rel = matmul(transpose([list(r) for r in Ra]), [list(r) for r in Rb])
    trace = rel[0][0] + rel[1][1] + rel[2][2]
    cos = (trace - 1.0) / 2.0
    cos = max(-1.0, min(1.0, cos))
    ang = math.acos(cos)
    return math.degrees(ang) if degrees else ang


def translation_error(ta: Vec3, tb: Vec3) -> float:
    """Euclidean distance between two translation vectors."""
    return math.sqrt(sum((ta[d] - tb[d]) ** 2 for d in range(3)))


def _centres(poses: Sequence[Pose]) -> List[List[float]]:
    return [list(p[1]) for p in poses]


def absolute_translation_error(pred: Sequence[Pose], gt: Sequence[Pose],
                               with_scale: bool = True) -> float:
    """ATE: RMS camera-centre distance after Sim(3) alignment of pred to gt."""
    if len(pred) != len(gt) or not pred:
        raise ValueError("pred and gt must be non-empty and equal length")
    src = _centres(pred)
    dst = _centres(gt)
    xform = umeyama_alignment(src, dst, with_scale=with_scale)
    total = 0.0
    for i in range(len(src)):
        p = xform.apply(src[i])
        total += sum((p[d] - dst[i][d]) ** 2 for d in range(3))
    return math.sqrt(total / len(src))


def _relative_motion(pa: Pose, pb: Pose) -> Tuple[Mat, List[float]]:
    """Relative pose from ``pa`` to ``pb`` in ``pa``'s frame: R_a^T R_b, R_a^T(t_b-t_a)."""
    Ra = [list(r) for r in pa[0]]
    Rb = [list(r) for r in pb[0]]
    Rrel = matmul(transpose(Ra), Rb)
    dt = [pb[1][d] - pa[1][d] for d in range(3)]
    RaT = transpose(Ra)
    trel = [RaT[r][0] * dt[0] + RaT[r][1] * dt[1] + RaT[r][2] * dt[2]
            for r in range(3)]
    return Rrel, trel


def relative_pose_error(pred: Sequence[Pose], gt: Sequence[Pose],
                        delta: int = 1) -> Tuple[float, float]:
    """RPE over frame gaps of ``delta``.

    Returns ``(rpe_trans, rpe_rot_degrees)``: RMS translation error and RMS
    geodesic rotation error of the relative motion between frames ``i`` and
    ``i+delta``, comparing predicted and ground-truth increments.
    """
    if len(pred) != len(gt):
        raise ValueError("pred and gt must be equal length")
    n = len(pred)
    if n <= delta:
        raise ValueError("not enough frames for the requested delta")
    trans_sq = 0.0
    rot_sq = 0.0
    count = 0
    for i in range(n - delta):
        Rp, tp = _relative_motion(pred[i], pred[i + delta])
        Rg, tg = _relative_motion(gt[i], gt[i + delta])
        trans_sq += sum((tp[d] - tg[d]) ** 2 for d in range(3))
        rot_sq += rotation_angle_error(Rp, Rg) ** 2
        count += 1
    rpe_trans = math.sqrt(trans_sq / count)
    rpe_rot = math.sqrt(rot_sq / count)
    return rpe_trans, rpe_rot


def evaluate_trajectory(pred: Sequence[Pose], gt: Sequence[Pose],
                        delta: int = 1, with_scale: bool = True) -> dict:
    """Full E3D-Bench pose report: ATE, RPE-trans, RPE-rot (degrees)."""
    ate = absolute_translation_error(pred, gt, with_scale=with_scale)
    rpe_trans, rpe_rot = relative_pose_error(pred, gt, delta=delta)
    return {
        "ate": ate,
        "rpe_trans": rpe_trans,
        "rpe_rot": rpe_rot,
        "frames": len(pred),
    }
