"""Back-compat shim: voxel_iou moved into the geometry layer.

The voxel-IoU / viewpoint-error primitives are general geometry, not an
eval-only concern, so the module now lives at
``harnesscad.domain.geometry.volumes.voxel_iou`` (alongside tsdf /
marching_cubes).  This shim keeps the old ``harnesscad.eval.bench.geometry``
import path working so eval-side callers do not break.  It can be removed once
those callers are repointed at the new location.
"""

from harnesscad.domain.geometry.volumes.voxel_iou import (  # moved; kept as a shim
    azimuth_mae,
    category_mean,
    circular_abs_error,
    pose_mae,
    pose_mse,
    voxel_iou,
    voxelize_points,
)

__all__ = [
    "voxel_iou",
    "voxelize_points",
    "pose_mse",
    "pose_mae",
    "circular_abs_error",
    "azimuth_mae",
    "category_mean",
]
