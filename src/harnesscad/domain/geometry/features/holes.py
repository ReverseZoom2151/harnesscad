"""Deterministic geometry of CadQuery's hole / counterbore / countersink features.

The ``cadquery-contrib`` examples lean heavily on ``hole``, ``cboreHole`` and
``cskHole`` (parametric enclosure lids, remote-control cases, connector
panels).  CadQuery implements these by revolving an axial profile and cutting
it; the *profile* itself is pure, closed-form geometry and is reimplemented
here without any CAD kernel.

Conventions: the hole axis is -Z, drilled from a face at ``z = 0`` downwards.
A profile is a list of ``(radius, z)`` points from the top of the feature to
its bottom, describing the swept cut as a stack of cylinders / frusta.  The
removed volume is the exact solid of revolution of that profile.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Sequence, Tuple

__all__ = [
    "HoleError",
    "HoleFeature",
    "simple_hole",
    "counterbore_hole",
    "countersink_hole",
    "countersink_depth",
    "profile_volume",
    "profile_points",
    "hole_breaks_wall",
]


class HoleError(ValueError):
    """Raised when hole parameters are geometrically inconsistent."""


@dataclass(frozen=True)
class HoleFeature:
    """A drilled feature: ``sections`` are ``(r_top, r_bottom, height)`` stacks."""

    kind: str
    diameter: float
    depth: float
    sections: Tuple[Tuple[float, float, float], ...]

    @property
    def volume(self) -> float:
        return sum(_frustum_volume(rt, rb, h) for rt, rb, h in self.sections)

    @property
    def max_radius(self) -> float:
        return max(max(rt, rb) for rt, rb, _ in self.sections)

    def profile(self) -> List[Tuple[float, float]]:
        """``(radius, z)`` polyline from z=0 downward along the -Z axis."""
        return profile_points(self.sections)


def _frustum_volume(r_top: float, r_bot: float, h: float) -> float:
    return math.pi * h * (r_top * r_top + r_top * r_bot + r_bot * r_bot) / 3.0


def profile_points(sections: Sequence[Tuple[float, float, float]]
                   ) -> List[Tuple[float, float]]:
    """Convert stacked sections into a ``(radius, z)`` polyline (z <= 0)."""
    pts: List[Tuple[float, float]] = []
    z = 0.0
    for r_top, r_bot, h in sections:
        pts.append((r_top, z))
        z -= h
        pts.append((r_bot, z))
    return pts


def profile_volume(sections: Sequence[Tuple[float, float, float]]) -> float:
    """Exact solid-of-revolution volume of a stack of cylinders / frusta."""
    return sum(_frustum_volume(rt, rb, h) for rt, rb, h in sections)


def _check_common(diameter: float, depth: float) -> None:
    if diameter <= 0.0:
        raise HoleError("diameter must be positive")
    if depth <= 0.0:
        raise HoleError("depth must be positive")


def simple_hole(diameter: float, depth: float) -> HoleFeature:
    """A plain drilled hole (``Workplane.hole``)."""
    _check_common(diameter, depth)
    r = diameter / 2.0
    return HoleFeature("hole", diameter, depth, ((r, r, depth),))


def counterbore_hole(diameter: float, cbore_diameter: float,
                     cbore_depth: float, depth: float) -> HoleFeature:
    """``Workplane.cboreHole``: a flat-bottomed enlarged pocket over a hole."""
    _check_common(diameter, depth)
    if cbore_diameter <= diameter:
        raise HoleError("counterbore diameter must exceed hole diameter")
    if cbore_depth <= 0.0:
        raise HoleError("counterbore depth must be positive")
    if cbore_depth >= depth:
        raise HoleError("counterbore depth must be less than total depth")
    r = diameter / 2.0
    rb = cbore_diameter / 2.0
    sections = ((rb, rb, cbore_depth), (r, r, depth - cbore_depth))
    return HoleFeature("cbore", diameter, depth, sections)


def countersink_depth(diameter: float, csk_diameter: float,
                      csk_angle: float) -> float:
    """Axial depth of the conical countersink for the included ``csk_angle``."""
    if not (0.0 < csk_angle < 180.0):
        raise HoleError("countersink angle must be in (0, 180) degrees")
    if csk_diameter <= diameter:
        raise HoleError("countersink diameter must exceed hole diameter")
    half = math.radians(csk_angle / 2.0)
    return (csk_diameter - diameter) / 2.0 / math.tan(half)


def countersink_hole(diameter: float, csk_diameter: float, csk_angle: float,
                     depth: float) -> HoleFeature:
    """``Workplane.cskHole``: a conical seat above a drilled hole."""
    _check_common(diameter, depth)
    h_csk = countersink_depth(diameter, csk_diameter, csk_angle)
    if h_csk >= depth:
        raise HoleError("countersink cone is deeper than the hole")
    r = diameter / 2.0
    rc = csk_diameter / 2.0
    sections = ((rc, r, h_csk), (r, r, depth - h_csk))
    return HoleFeature("csk", diameter, depth, sections)


def hole_breaks_wall(feature: HoleFeature, wall_thickness: float) -> bool:
    """True when the feature drills through a wall of ``wall_thickness``."""
    if wall_thickness <= 0.0:
        raise HoleError("wall thickness must be positive")
    return feature.depth >= wall_thickness
