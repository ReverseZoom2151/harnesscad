"""Camera-pose ID grid for Sketch2CAD (Yang, EPFL 2023).

Sketch2CAD renders each synthetic scene from many viewpoints and tokenises the
camera pose *directly* as a discrete ID (Sec. III-B-2): "since the 2D image is
rendered from a pre-defined position and angle, we keep a map of
``(ID_pose, (azimuth, elevation))`` and use the pose ID as the vocabulary directly."

The rendering protocol (Sec. III-A-2) uses the Horizontal Coordinate System with:

  * elevations ranging from -15 deg to 45 deg, every 15 deg  -> 5 values
  * azimuths ranging from -180 deg to 180 deg, every 30 deg -> 12 distinct
    directions (``+180`` coincides with ``-180``)

giving the paper's **60 images per scene** (5 x 12 = 60). This module builds that
deterministic ``ID <-> (azimuth, elevation)`` map and converts a pose to a unit
view direction, so the discrete camera-pose token of the scene descriptor can be
turned into an actual camera bearing. Pure ``math`` only.

This complements :mod:`reconstruction.sketch2cad_scene_descriptor` (which allocates
the ``n_cam_pose`` vocabulary block) and is distinct from
:mod:`vision.cvcad_pixel_calibration` (pixel/world calibration) -- here we only
enumerate discrete render viewpoints.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

# Paper's ranges (Sec. III-A-2).
ELEVATIONS: tuple[float, ...] = (-15.0, 0.0, 15.0, 30.0, 45.0)
# -180..180 every 30, dropping the duplicate +180 (== -180): 12 azimuths.
AZIMUTHS: tuple[float, ...] = tuple(float(a) for a in range(-180, 180, 30))

N_ELEVATION = len(ELEVATIONS)   # 5
N_AZIMUTH = len(AZIMUTHS)       # 12
N_POSES = N_ELEVATION * N_AZIMUTH  # 60


@dataclass(frozen=True)
class Pose:
    """A discrete render viewpoint: azimuth + elevation in degrees."""

    azimuth: float
    elevation: float


def _build_map():
    """Row-major over (elevation, azimuth): stable ``id -> Pose`` ordering."""
    poses = []
    for el in ELEVATIONS:
        for az in AZIMUTHS:
            poses.append(Pose(az, el))
    return tuple(poses)


_POSES: tuple[Pose, ...] = _build_map()
_INDEX: dict[tuple[float, float], int] = {
    (p.azimuth, p.elevation): i for i, p in enumerate(_POSES)
}


def num_poses() -> int:
    """Total number of discrete camera poses (60 for the paper's grid)."""
    return N_POSES


def id_to_pose(pose_id: int) -> Pose:
    """Look up the ``(azimuth, elevation)`` pose for a pose ID."""
    if not 0 <= pose_id < N_POSES:
        raise ValueError(f"pose_id out of range [0,{N_POSES}): {pose_id}")
    return _POSES[pose_id]


def pose_to_id(azimuth: float, elevation: float) -> int:
    """Inverse of :func:`id_to_pose`; azimuth is wrapped into the grid's range."""
    az = _wrap_azimuth(azimuth)
    key = (az, float(elevation))
    if key not in _INDEX:
        raise ValueError(f"pose ({azimuth}, {elevation}) is not on the grid")
    return _INDEX[key]


def _wrap_azimuth(az: float) -> float:
    """Wrap an azimuth into [-180, 180); +180 folds to -180 to match the grid."""
    a = (az + 180.0) % 360.0 - 180.0
    # ((-180)+180)%360-180 == -180 already; guard float +180 exactly:
    if a == 180.0:
        a = -180.0
    return a


def view_direction(pose_id: int) -> tuple[float, float, float]:
    """Unit vector *from the camera toward the scene* for a pose.

    Uses the Horizontal Coordinate System convention: azimuth measured in the
    ground (x-y) plane, elevation above it. Azimuth 0 looks along +x.
    """
    p = id_to_pose(pose_id)
    az = math.radians(p.azimuth)
    el = math.radians(p.elevation)
    ce = math.cos(el)
    # camera sits up-and-out; it looks back down toward the origin, so the view
    # direction is the negation of the outward bearing.
    outward = (ce * math.cos(az), ce * math.sin(az), math.sin(el))
    return (-outward[0], -outward[1], -outward[2])


def all_poses():
    """Iterate ``(pose_id, Pose)`` over the whole grid in ID order."""
    return tuple(enumerate(_POSES))
