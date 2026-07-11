"""Voxel-IoU and viewpoint-error metrics from Magic3DSketch (Zang et al., 2024).

Magic3DSketch reports two deterministic quantitative metrics against ShapeNet
ground truth:

* **Voxel IoU** (Tables 1, 3, 7): a reconstructed shape and the ground-truth
  shape are voxelised onto a common occupancy grid and scored by the volumetric
  Jaccard index |A AND B| / |A OR B|.  This is the headline reconstruction
  fidelity number ("VoxelIoU up").

* **Viewpoint error** (Table 2): the network's predicted camera pose (elevation
  and azimuth Euler angles) is compared with the ground-truth pose by mean
  absolute error (MAE, reported in the table) -- and, as used in the training
  loss L_v (paper Eq. 4), by mean squared error over the pose vector.  Azimuth
  is an angle on a circle, so this module offers a circular MAE that accounts
  for wrap-around (e.g. 350 deg vs 10 deg differ by 20 deg, not 340).

The paper also averages per-category scores into a single "mean" column; a
category-mean helper is provided.

Everything is stdlib-only and deterministic.  Occupancy grids are given as sets
of integer ``(i, j, k)`` voxel indices (sparse) so the metric never allocates a
dense volume.  The learned encoder-decoder, SoftRas renderer and CLIP guidance
that *produce* the shapes and poses are external.
"""

from __future__ import annotations

import math
from typing import Dict, Iterable, Mapping, Sequence, Tuple

Voxel = Tuple[int, int, int]


def voxel_iou(a: Iterable[Voxel], b: Iterable[Voxel]) -> float:
    """Volumetric IoU |A AND B| / |A OR B| of two sparse occupancy sets.

    Two empty grids are defined to have IoU 1.0 (they agree perfectly on
    "nothing is occupied").
    """
    sa = set(a)
    sb = set(b)
    union = len(sa | sb)
    if union == 0:
        return 1.0
    return len(sa & sb) / union


def voxelize_points(
    points: Sequence[Sequence[float]],
    *,
    origin: Sequence[float] = (0.0, 0.0, 0.0),
    spacing: float = 1.0,
) -> set:
    """Map continuous ``(x, y, z)`` points to a set of integer voxel indices.

    A point falls in voxel ``floor((p - origin) / spacing)`` per axis.  Useful to
    build occupancy sets from sampled surface / interior points before scoring.
    """
    if spacing <= 0.0:
        raise ValueError("spacing must be positive")
    ox, oy, oz = origin
    out = set()
    for p in points:
        x, y, z = p
        out.add(
            (
                int(math.floor((x - ox) / spacing)),
                int(math.floor((y - oy) / spacing)),
                int(math.floor((z - oz) / spacing)),
            )
        )
    return out


def pose_mse(pred: Sequence[float], gt: Sequence[float]) -> float:
    """Mean squared error over a pose vector (paper Eq. 4, L_v = ||gt - pred||^2).

    Returned as the mean of squared per-component differences.
    """
    if len(pred) != len(gt):
        raise ValueError("pose vectors must have equal length")
    if not pred:
        raise ValueError("pose vector must be non-empty")
    return sum((p - g) ** 2 for p, g in zip(pred, gt)) / len(pred)


def pose_mae(pred: Sequence[float], gt: Sequence[float]) -> float:
    """Mean absolute error over a pose vector (Table 2 metric)."""
    if len(pred) != len(gt):
        raise ValueError("pose vectors must have equal length")
    if not pred:
        raise ValueError("pose vector must be non-empty")
    return sum(abs(p - g) for p, g in zip(pred, gt)) / len(pred)


def circular_abs_error(pred: float, gt: float, *, period: float = 360.0) -> float:
    """Absolute angular difference on a circle of given period (degrees default).

    Returns the shortest wrap-around distance in ``[0, period/2]`` -- e.g.
    ``circular_abs_error(350, 10) == 20``.  Appropriate for azimuth angles.
    """
    if period <= 0.0:
        raise ValueError("period must be positive")
    d = abs(pred - gt) % period
    return min(d, period - d)


def azimuth_mae(
    pred: Sequence[float], gt: Sequence[float], *, period: float = 360.0
) -> float:
    """Mean circular absolute error between predicted and GT azimuth angles."""
    if len(pred) != len(gt):
        raise ValueError("azimuth sequences must have equal length")
    if not pred:
        raise ValueError("azimuth sequence must be non-empty")
    return sum(
        circular_abs_error(p, g, period=period) for p, g in zip(pred, gt)
    ) / len(pred)


def category_mean(per_category: Mapping[str, float]) -> float:
    """Unweighted mean over per-category scores (the paper's "mean" column)."""
    if not per_category:
        raise ValueError("need at least one category")
    return sum(per_category.values()) / len(per_category)
