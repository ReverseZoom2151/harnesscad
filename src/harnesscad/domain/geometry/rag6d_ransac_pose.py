"""RANSAC rigid-pose selection over noisy 3D-3D correspondences (RAG-6DPose).

RAG-6DPose samples 2D-3D (or, with depth, 3D-3D) correspondences from a learned
similarity matrix and solves for the object pose with *PnP-RANSAC* (Sec. III-D,
"Deployment: Pose Estimation and Refinement"): a robust estimator that tolerates
the many wrong matches produced by the retrieval step, then keeps the hypothesis
with the most inliers and refines it on that consensus set.

This module implements the deterministic RANSAC wrapper for the 3D-3D case.  The
minimal rigid solver reuses the existing Umeyama SE(3) alignment
(``geometry.e3dbench_umeyama.umeyama_alignment`` with ``with_scale=False``) -- it
is *not* re-derived here.  What is genuinely new is the robust sampling loop:

  * draw minimal 3-point samples with a seeded ``random.Random`` (fully
    reproducible -- no wall clock);
  * fit a candidate ``(R, t)`` from each sample;
  * score it by the number of correspondences whose residual
    ``|| (R x_i + t) - y_i ||`` is below an inlier threshold;
  * keep the best-scoring hypothesis, then refit on all its inliers.

The result is an ``se3`` rotation/translation plus the inlier mask, ready to be
scored by ``bench.rag6d_pose_metrics``.
"""

from __future__ import annotations

import math
import random
from typing import List, Optional, Sequence, Tuple

from harnesscad.domain.geometry.e3dbench_umeyama import Sim3, umeyama_alignment

Vec3 = Sequence[float]
Mat3 = List[List[float]]


class PoseHypothesis:
    """A rigid pose ``y = R x + t`` with the inliers that support it."""

    __slots__ = ("R", "t", "inliers", "num_inliers", "residual")

    def __init__(self, R: Mat3, t: List[float], inliers: List[bool],
                 residual: float) -> None:
        self.R = [list(row) for row in R]
        self.t = list(t)
        self.inliers = list(inliers)
        self.num_inliers = sum(1 for f in inliers if f)
        self.residual = float(residual)

    def apply(self, x: Vec3) -> Tuple[float, float, float]:
        R, t = self.R, self.t
        return (
            R[0][0] * x[0] + R[0][1] * x[1] + R[0][2] * x[2] + t[0],
            R[1][0] * x[0] + R[1][1] * x[1] + R[1][2] * x[2] + t[1],
            R[2][0] * x[0] + R[2][1] * x[1] + R[2][2] * x[2] + t[2],
        )


def _dist(a: Vec3, b: Vec3) -> float:
    return math.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2 + (a[2] - b[2]) ** 2)


def _fit_rigid(src: Sequence[Vec3], dst: Sequence[Vec3]) -> Sim3:
    """Least-squares SE(3) fit (Umeyama, scale fixed to 1)."""
    return umeyama_alignment(src, dst, with_scale=False)


def _residuals(sim: Sim3, src: Sequence[Vec3], dst: Sequence[Vec3]) -> List[float]:
    return [_dist(sim.apply(src[i]), dst[i]) for i in range(len(src))]


def _inlier_mask(res: Sequence[float], thresh: float) -> List[bool]:
    return [r <= thresh for r in res]


def ransac_rigid_pose(src_points: Sequence[Vec3], dst_points: Sequence[Vec3],
                      inlier_thresh: float, iterations: int = 100,
                      seed: int = 0,
                      min_inliers: int = 3) -> Optional[PoseHypothesis]:
    """Robustly estimate a rigid pose ``dst ~= R src + t`` via RANSAC.

    ``src_points`` are 3D CAD-model points, ``dst_points`` the matched observed
    3D points (same length; index ``i`` is a correspondence).  A correspondence
    counts as an inlier when its residual is ``<= inlier_thresh``.

    Sampling is driven by ``random.Random(seed)`` so the outcome is deterministic.
    Returns the best :class:`PoseHypothesis` (refit on its consensus set), or
    ``None`` if fewer than three correspondences are supplied or no hypothesis
    reaches ``min_inliers``.
    """
    n = len(src_points)
    if n != len(dst_points):
        raise ValueError("src and dst must have equal length")
    if n < 3:
        return None

    rng = random.Random(seed)
    best: Optional[PoseHypothesis] = None

    for _ in range(iterations):
        idx = rng.sample(range(n), 3)
        sample_src = [src_points[i] for i in idx]
        sample_dst = [dst_points[i] for i in idx]
        try:
            sim = _fit_rigid(sample_src, sample_dst)
        except ValueError:
            continue
        res = _residuals(sim, src_points, dst_points)
        mask = _inlier_mask(res, inlier_thresh)
        n_in = sum(1 for f in mask if f)
        if n_in < min_inliers:
            continue
        # tie-break by total residual so ties are deterministic
        total_res = sum(res)
        if best is None or n_in > best.num_inliers or (
                n_in == best.num_inliers and total_res < best.residual):
            best = PoseHypothesis(sim.R, sim.t, mask, total_res)

    if best is None:
        return None

    # Refit on the full consensus set for a lower-variance final estimate.
    in_src = [src_points[i] for i in range(n) if best.inliers[i]]
    in_dst = [dst_points[i] for i in range(n) if best.inliers[i]]
    if len(in_src) >= 3:
        try:
            refit = _fit_rigid(in_src, in_dst)
            res = _residuals(refit, src_points, dst_points)
            mask = _inlier_mask(res, inlier_thresh)
            if sum(1 for f in mask if f) >= best.num_inliers:
                best = PoseHypothesis(refit.R, refit.t, mask, sum(res))
        except ValueError:
            pass
    return best
