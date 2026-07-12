"""Gear-pair meshing geometry and assembly placement (deterministic, stdlib).

This captures the *spatial-position encoding* CAD-GPT uses to place a meshing
gear pair, extracted from ``paper_code/prompts/scripting_prompt.py`` and the
worked parameter files (``common/*/gears_parameters.json``). CAD-GPT's LLM only
emits a JSON parameter block; turning that block into an OpenSCAD
``translate(...) rotate(...)`` placement is a set of deterministic rules:

  * two meshing gears sit a *centre distance* ``a = m*(z1+z2)/2`` apart
    (verified by the repo's spur example: m=2.5, z1=20, z2=60 -> a=100, which is
    exactly the x-offset of gear 2);
  * the driven gear is rotated by a *meshing phase offset* ``gamma = 360/(2*z)``
    degrees so a tooth of one gear falls into a gap of the other;
  * a non-spur (helix / herringbone) driven gear takes the *inverse* helix angle;
  * a bevel driven gear is flipped: ``gamma = 0`` and ``beta = -90``.

Also included: the helical-extrude twist and bevel-extrude taper scale that the
OpenSCAD library derives inside ``gear_helix`` / ``gear_bevel``.

Angles are DEGREES throughout (OpenSCAD convention).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Tuple

__all__ = [
    "gear_ratio",
    "center_distance",
    "meshing_phase_offset",
    "inverse_helix_angle",
    "helix_twist",
    "bevel_scale",
    "GearPlacement",
    "place_driven_gear",
    "place_driving_gear",
]


def gear_ratio(teeth_driving: int, teeth_driven: int) -> float:
    """Speed-reduction gear ratio ``i = z_driven / z_driving``."""
    if teeth_driving <= 0 or teeth_driven <= 0:
        raise ValueError("teeth counts must be positive")
    return teeth_driven / teeth_driving


def center_distance(module: float, teeth_a: int, teeth_b: int,
                    shift_a: float = 0.0, shift_b: float = 0.0) -> float:
    """Standard centre distance ``a = m*(z1+z2)/2 + (x1+x2)`` between two gears."""
    if module <= 0:
        raise ValueError("module must be positive")
    return module * (teeth_a + teeth_b) / 2.0 + (shift_a + shift_b)


def meshing_phase_offset(teeth: int) -> float:
    """Half-tooth mesh rotation ``gamma = 360/(2*z)`` degrees for the driven gear."""
    if teeth <= 0:
        raise ValueError("teeth must be positive")
    return 360.0 / (teeth * 2.0)


def inverse_helix_angle(helix_angle: float) -> float:
    """The driven gear of a helical/herringbone pair takes the negated helix."""
    return -helix_angle


def helix_twist(height: float, helix_angle: float, pitch_radius: float) -> float:
    """Twist angle (degrees) for a helical gear's ``linear_extrude``.

    Port of the OpenSCAD ``gear_helix`` expression::

        tw = h / tan(90 - w_helix) / PI * 180 / r_wk

    where ``tan`` takes degrees. Zero helix angle yields zero twist.
    """
    if pitch_radius == 0:
        raise ValueError("pitch_radius must be non-zero")
    denom = math.tan(math.radians(90.0 - helix_angle))
    return height / denom / math.pi * 180.0 / pitch_radius


def bevel_scale(pitch_radius: float, bevel_angle: float, height: float) -> float:
    """Taper scale for a bevel gear's ``linear_extrude`` (``gear_bevel``)::

        sc = (r_wk - tan(w_bevel)*h) / r_wk
    """
    if pitch_radius == 0:
        raise ValueError("pitch_radius must be non-zero")
    return (pitch_radius - math.tan(math.radians(bevel_angle)) * height) / pitch_radius


@dataclass(frozen=True)
class GearPlacement:
    """A gear's rigid placement: translation ``(x,y,z)`` and Euler rotation.

    ``rotation`` is ``(alpha, beta, gamma)`` in degrees, matching the OpenSCAD
    ``rotate([alpha, beta, gamma])`` applied after ``translate([x,y,z])``.
    """

    translation: Tuple[float, float, float]
    rotation: Tuple[float, float, float]
    helix_angle: float

    def to_dict(self) -> dict:
        return {
            "translation": list(self.translation),
            "rotation": list(self.rotation),
            "helix_angle": self.helix_angle,
        }


def place_driving_gear(coordinate: Tuple[float, float, float] = (0.0, 0.0, 0.0),
                       helix_angle: float = 0.0) -> GearPlacement:
    """The driving gear sits at its coordinate with no phase offset."""
    return GearPlacement(
        translation=tuple(float(c) for c in coordinate),
        rotation=(0.0, 0.0, 0.0),
        helix_angle=float(helix_angle),
    )


def place_driven_gear(teeth: int,
                      coordinate: Tuple[float, float, float],
                      gear_type: str = "spur",
                      driving_helix_angle: float = 0.0) -> GearPlacement:
    """Compute the driven gear's placement from the CAD-GPT scripting rules.

    Rules (``scripting_prompt.py``):
      * ``gamma = 360/(2*z)`` (meshing phase) for the driven gear;
      * non-spur gears take the inverse helix angle of the driving gear;
      * a bevel driven gear overrides ``gamma = 0`` and ``beta = -90``.
    """
    gtype = gear_type.strip().lower()
    x, y, z = (float(c) for c in coordinate)

    if gtype in ("helix", "helical", "herringbone"):
        helix = inverse_helix_angle(driving_helix_angle)
    else:
        helix = 0.0

    alpha = 0.0
    beta = 0.0
    gamma = meshing_phase_offset(teeth)

    if gtype == "bevel":
        gamma = 0.0
        beta = -90.0

    return GearPlacement(
        translation=(x, y, z),
        rotation=(alpha, beta, gamma),
        helix_angle=helix,
    )
