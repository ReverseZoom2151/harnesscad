"""Analytic mass properties for assemblies of parametric solid primitives.

Motivation
----------
In "How Can Large Language Models Help Humans in Design and Manufacturing"
(Makatura et al., sections 7.1 and 8.1, Figures 52-53) GPT-4 repeatedly failed
to compute the center of gravity / static stability of an assembly of
primitive solids, defaulting to guesses such as "the center of the seat". The
paper supplies the exact deterministic formulas for a radially-symmetric table
modeled as parametric cylinders. This module implements the correct,
fully deterministic analytic mass-properties calculator that GPT-4 could not,
for assemblies of parametric boxes and cylinders.

Scope (what this module is and is NOT)
--------------------------------------
This computes ANALYTIC volume, mass, and center of mass (centroid) of
parametric SOLID primitives and their assemblies. It is deliberately distinct
from:

  * ``verifiers/standability.py`` -- point-based support-polygon / tipping check
    that consumes a center-of-mass plus contact points. This module produces a
    center of mass that can FEED that check; it does not reimplement it.
  * ``quality/mesh_stability.py`` -- mesh-based base-flatness metrics.
  * ``geometry/dreamcad_primitives.py`` -- parameterizes primitive SURFACES
    (points/normals), not volume or mass.

Conventions
-----------
Coordinates are a right-handed (x, y, z) system with +z "up".

``Box(cx, cy, cz, w, h, d)``
    An axis-aligned box CENTERED at (cx, cy, cz) with full extents
    w (along x), h (along y), d (along z).
    volume = w * h * d ; centroid = (cx, cy, cz).

``Cylinder(cx, cy, cz, radius, height, axis='z')``
    An axis-aligned cylinder. The two coordinates orthogonal to ``axis`` give
    the axis line (e.g. for axis='z', (cx, cy) is the axis in the xy-plane).
    The cylinder spans from the coordinate along ``axis`` (the BASE) to that
    coordinate + ``height``. In other words the named coordinate is the base
    plane, NOT the mid-plane. Therefore the centroid along the axis is
    (base + height/2). For axis='z' the centroid is (cx, cy, cz + height/2).
    volume = pi * radius**2 * height.

Each primitive carries a ``density`` (default 1.0, uniform) so that
mass = density * volume. The centroid is independent of density.

Table stacking convention (Figure 53)
--------------------------------------
For the radially-symmetric table we stack vertically as:

    legs:  z in [0, h]        (4 leg cylinders, radius r, height h)
    top:   z in [h, h + H]    (1 tabletop cylinder, radius R, height H)

so the leg centroids sit at z = h/2 and the tabletop centroid at z = h + H/2.
With uniform density rho the closed form for the vertical center of mass is

    z_cm = ( rho*pi*R^2*H*(H/2 + h) + 4*rho*pi*r^2*h*(h/2) )
           / ( rho*pi*R^2*H + 4*rho*pi*r^2*h )

which is exactly the paper's generalized z_cm = (1/M)(M_top*(h + H/2)
+ 4*M_leg*(h/2)). By symmetry x_cm = y_cm = 0 when the legs are placed
symmetrically about the axis.

Static stability follows the paper's conclusion: a LOWER center of mass is more
stable, so z_cm is MINIMIZED by minimizing the tabletop height H and leg
height h within their allowed bounds.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from math import pi
from typing import Sequence

Point3 = tuple[float, float, float]

_AXES = ("x", "y", "z")


@dataclass(frozen=True)
class Box:
    """Axis-aligned box centered at (cx, cy, cz) with full extents w, h, d."""

    cx: float
    cy: float
    cz: float
    w: float
    h: float
    d: float
    density: float = 1.0

    def __post_init__(self) -> None:
        for name, value in (("w", self.w), ("h", self.h), ("d", self.d)):
            if value <= 0:
                raise ValueError(f"Box dimension {name} must be positive, got {value}")
        if self.density <= 0:
            raise ValueError(f"Box density must be positive, got {self.density}")

    @property
    def volume(self) -> float:
        return self.w * self.h * self.d

    @property
    def mass(self) -> float:
        return self.density * self.volume

    @property
    def centroid(self) -> Point3:
        return (self.cx, self.cy, self.cz)

    @property
    def aabb(self) -> tuple[Point3, Point3]:
        """Axis-aligned bounding box as (min_corner, max_corner)."""
        return (
            (self.cx - self.w / 2, self.cy - self.h / 2, self.cz - self.d / 2),
            (self.cx + self.w / 2, self.cy + self.h / 2, self.cz + self.d / 2),
        )


@dataclass(frozen=True)
class Cylinder:
    """Axis-aligned cylinder whose base plane is the named-axis coordinate.

    Spans from the axis coordinate (base) to base + height along ``axis``.
    """

    cx: float
    cy: float
    cz: float
    radius: float
    height: float
    axis: str = "z"
    density: float = 1.0

    def __post_init__(self) -> None:
        if self.radius <= 0:
            raise ValueError(f"Cylinder radius must be positive, got {self.radius}")
        if self.height <= 0:
            raise ValueError(f"Cylinder height must be positive, got {self.height}")
        if self.density <= 0:
            raise ValueError(f"Cylinder density must be positive, got {self.density}")
        if self.axis not in _AXES:
            raise ValueError(f"Cylinder axis must be one of {_AXES}, got {self.axis!r}")

    @property
    def volume(self) -> float:
        return pi * self.radius * self.radius * self.height

    @property
    def mass(self) -> float:
        return self.density * self.volume

    @property
    def centroid(self) -> Point3:
        # The base center is (cx, cy, cz); the centroid advances height/2 along axis.
        cx, cy, cz = self.cx, self.cy, self.cz
        if self.axis == "x":
            return (cx + self.height / 2, cy, cz)
        if self.axis == "y":
            return (cx, cy + self.height / 2, cz)
        return (cx, cy, cz + self.height / 2)

    @property
    def aabb(self) -> tuple[Point3, Point3]:
        r, hgt = self.radius, self.height
        cx, cy, cz = self.cx, self.cy, self.cz
        if self.axis == "x":
            return ((cx, cy - r, cz - r), (cx + hgt, cy + r, cz + r))
        if self.axis == "y":
            return ((cx - r, cy, cz - r), (cx + r, cy + hgt, cz + r))
        return ((cx - r, cy - r, cz), (cx + r, cy + r, cz + hgt))


Primitive = Box | Cylinder


@dataclass(frozen=True)
class MassProperties:
    """Aggregate analytic mass properties of an assembly."""

    total_volume: float
    total_mass: float
    center_of_mass: Point3
    aabb: tuple[Point3, Point3]
    footprint_centers: tuple[tuple[float, float], ...]


@dataclass
class Assembly:
    """An assembly of parametric solid primitives."""

    primitives: list[Primitive] = field(default_factory=list)

    def add(self, primitive: Primitive) -> "Assembly":
        self.primitives.append(primitive)
        return self

    @property
    def total_volume(self) -> float:
        return sum(p.volume for p in self.primitives)

    @property
    def total_mass(self) -> float:
        return sum(p.mass for p in self.primitives)

    @property
    def center_of_mass(self) -> Point3:
        if not self.primitives:
            raise ValueError("center_of_mass undefined for an empty assembly")
        total_mass = self.total_mass
        # total_mass > 0 is guaranteed: every primitive has positive mass.
        sx = sy = sz = 0.0
        for p in self.primitives:
            m = p.mass
            cx, cy, cz = p.centroid
            sx += m * cx
            sy += m * cy
            sz += m * cz
        return (sx / total_mass, sy / total_mass, sz / total_mass)

    @property
    def aabb(self) -> tuple[Point3, Point3]:
        if not self.primitives:
            raise ValueError("aabb undefined for an empty assembly")
        mins = [float("inf")] * 3
        maxs = [float("-inf")] * 3
        for p in self.primitives:
            lo, hi = p.aabb
            for i in range(3):
                mins[i] = min(mins[i], lo[i])
                maxs[i] = max(maxs[i], hi[i])
        return (tuple(mins), tuple(maxs))

    @property
    def footprint_centers(self) -> tuple[tuple[float, float], ...]:
        """Lightweight support footprint: the xy centroids of the primitives.

        The rigorous support-polygon / tipping test lives in
        ``verifiers/standability.py``; this is only a convenience so callers can
        sanity-check the center of mass against the primitive footprint.
        """
        return tuple((p.centroid[0], p.centroid[1]) for p in self.primitives)

    def mass_properties(self) -> MassProperties:
        if not self.primitives:
            raise ValueError("mass_properties undefined for an empty assembly")
        return MassProperties(
            total_volume=self.total_volume,
            total_mass=self.total_mass,
            center_of_mass=self.center_of_mass,
            aabb=self.aabb,
            footprint_centers=self.footprint_centers,
        )


def radially_symmetric_table(
    R: float, r: float, H: float, h: float, rho: float = 1.0
) -> Assembly:
    """Build the paper's Figure-53 radially-symmetric table.

    Legs occupy z in [0, h] (4 cylinders, radius r), tabletop occupies
    z in [h, h + H] (1 cylinder, radius R). Legs are placed symmetrically at
    (+/-R/2, +/-R/2) so x_cm = y_cm = 0.
    """
    if R <= 0 or r <= 0 or H <= 0 or h <= 0:
        raise ValueError("R, r, H, h must all be positive")
    if rho <= 0:
        raise ValueError("rho must be positive")
    offset = R / 2
    legs = [
        Cylinder(sx * offset, sy * offset, 0.0, r, h, axis="z", density=rho)
        for sx in (-1.0, 1.0)
        for sy in (-1.0, 1.0)
    ]
    top = Cylinder(0.0, 0.0, h, R, H, axis="z", density=rho)
    return Assembly(primitives=[*legs, top])


def table_static_stability_zcm(
    R: float, r: float, H: float, h: float, rho: float = 1.0
) -> float:
    """Return the vertical center of mass z_cm of the Figure-53 table.

    Closed form (see module docstring):
        z_cm = ( rho*pi*R^2*H*(H/2 + h) + 4*rho*pi*r^2*h*(h/2) )
               / ( rho*pi*R^2*H + 4*rho*pi*r^2*h )

    Static stability is maximized by MINIMIZING z_cm: a lower center of mass is
    more stable. Because z_cm increases with both the tabletop height H and the
    leg height h, minimizing H and h within their bounds lowers the CoM and
    maximizes stability -- exactly the paper's conclusion.
    """
    if R <= 0 or r <= 0 or H <= 0 or h <= 0:
        raise ValueError("R, r, H, h must all be positive")
    if rho <= 0:
        raise ValueError("rho must be positive")
    m_top = rho * pi * R * R * H
    m_legs_total = 4 * rho * pi * r * r * h
    numerator = m_top * (h + H / 2) + m_legs_total * (h / 2)
    denominator = m_top + m_legs_total
    return numerator / denominator
