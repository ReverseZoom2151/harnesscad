"""CADSmith three-view render specification (CADSmith sec. III-E).

The Validator's Judge receives a three-view rendered image of the generated
part. The paper fixes the exact cameras used, rendered with Phong shading at
2400x800 (three 800x800 panels side by side):

  * Isometric      — elevation 35 deg, azimuth 45 deg  (overall 3D shape)
  * High-angle rear— elevation 65 deg, azimuth 220 deg (top-face features:
                     holes, bores, cavities)
  * Front profile  — elevation 10 deg, azimuth 0 deg   (vertical profile, wall
                     heights, layered features)

This module encodes that spec deterministically: the named views, and the exact
conversion from (elevation, azimuth) to a unit camera *direction* vector (the
direction from the target toward the camera) and the derived camera position for
a given target and distance. This is the pure-geometry half of the render stage
— no VTK, no rasteriser — reusable for any renderer the pipeline plugs in.

Convention: Z-up, azimuth measured in the XY plane from the +X axis toward +Y
(right-handed), elevation measured up from the XY plane toward +Z. Matches the
benchmark's XY-base-plane / Z-up convention.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Tuple

Vec3 = Tuple[float, float, float]


# --------------------------------------------------------------------------- #
# View spec
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class ViewSpec:
    name: str
    elevation_deg: float
    azimuth_deg: float
    purpose: str

    def direction(self) -> Vec3:
        """Unit vector pointing from the scene target toward the camera.

        x = cos(el) cos(az), y = cos(el) sin(az), z = sin(el).
        """
        el = math.radians(self.elevation_deg)
        az = math.radians(self.azimuth_deg)
        ce = math.cos(el)
        return (ce * math.cos(az), ce * math.sin(az), math.sin(el))

    def camera_position(self, target: Vec3 = (0.0, 0.0, 0.0),
                        distance: float = 1.0) -> Vec3:
        """Camera position = target + distance * direction."""
        if distance <= 0:
            raise ValueError("distance must be positive")
        d = self.direction()
        return (target[0] + distance * d[0],
                target[1] + distance * d[1],
                target[2] + distance * d[2])


# --------------------------------------------------------------------------- #
# The three canonical CADSmith views
# --------------------------------------------------------------------------- #
ISOMETRIC = ViewSpec("isometric", 35.0, 45.0, "overall 3D shape")
HIGH_ANGLE_REAR = ViewSpec("high_angle_rear", 65.0, 220.0,
                           "top-face features: holes, bores, cavities")
FRONT_PROFILE = ViewSpec("front_profile", 10.0, 0.0,
                         "vertical profile, wall heights, layered features")

THREE_VIEWS: Tuple[ViewSpec, ...] = (ISOMETRIC, HIGH_ANGLE_REAR, FRONT_PROFILE)

# Render layout: three square panels side by side.
PANEL_SIZE = 800
RENDER_WIDTH = PANEL_SIZE * len(THREE_VIEWS)   # 2400
RENDER_HEIGHT = PANEL_SIZE                      # 800


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def view_by_name(name: str) -> ViewSpec:
    for v in THREE_VIEWS:
        if v.name == name:
            return v
    raise KeyError(name)


def all_directions() -> Tuple[Tuple[str, Vec3], ...]:
    """(name, unit direction) for each canonical view — deterministic."""
    return tuple((v.name, v.direction()) for v in THREE_VIEWS)


def render_resolution() -> Tuple[int, int]:
    """(width, height) of the composed three-view image = (2400, 800)."""
    return (RENDER_WIDTH, RENDER_HEIGHT)


def fit_distance(bbox_mm: Vec3, *, margin: float = 1.5) -> float:
    """A deterministic camera distance that frames a part of the given bbox.

    Uses the bounding sphere radius (half the diagonal) times ``margin`` so the
    whole part stays in frame for any of the three views.
    """
    if margin <= 0:
        raise ValueError("margin must be positive")
    radius = 0.5 * math.sqrt(sum(v * v for v in bbox_mm))
    return radius * margin
