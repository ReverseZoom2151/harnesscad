"""Volumetric (region) selectors: keep shapes whose centre lies in a solid.

Source rule: the ``more_selectors`` plugin of the CadQuery community plugin
collection.  Where CadQuery's built-in string selectors pick faces / edges /
vertices by *direction* (``>Z``), *orientation* (``|Z``) or *type*
(``%CIRCLE``), this plugin adds a different family: **region selectors** that
keep every shape whose centre of mass falls inside a chosen solid region --
an (in)finite cylinder, a hollow cylinder, a sphere, or a hollow sphere.
The kernel merely supplies each shape's centre point; the accept/reject test
is pure analytic geometry, reproduced here in stdlib arithmetic.

The deterministic rule for the cylindrical family is a change of frame onto
the cylinder axis.  Given an origin ``O`` and a (unit) axis ``w``, a point
``p`` decomposes into

* an axial coordinate ``h = dot(p - O, w)`` (distance measured along the
  axis, ``0`` at the origin), and
* a radial coordinate ``rho = |(p - O) - h * w|`` (perpendicular distance
  from the axis).

The plugin builds a local ``cq.Plane`` and reads ``rho = sqrt(x^2 + y^2)``
and ``h = z`` from the projected point; the axis-projection form above is the
rotation-invariant equivalent and needs no orthonormal frame.  The regions
are then, matching the plugin's strict inequalities:

* infinite cylinder      -- ``rho < radius``
* infinite hollow cyl.   -- ``inner < rho < outer``
* finite cylinder        -- ``rho < radius`` and ``0 < h < height``
* finite hollow cylinder -- ``inner < rho < outer`` and ``0 < h < height``
* sphere                 -- ``|p - O| < radius``
* hollow sphere          -- ``inner < |p - O| < outer``

Relation to the rest of the harness: ``geometry/cqcontrib_selector_dsl.py``
parses and evaluates the *string* selector mini-language (direction / axis /
type set algebra) and ``geometry/cascade_entity_selector.py`` selects by
geometric intent (parallel / extreme / by size).  Neither has a volumetric
point-in-solid test -- that is the new capability ``more_selectors`` adds and
what this module supplies.

Axes may be given as a named string (``"X"``, ``"-Z"``, ...) or as any
non-zero 3-tuple.  Everything is stdlib-only and deterministic; filters
preserve input order.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, Iterable, List, Sequence, Tuple

Point = Tuple[float, float, float]

__all__ = [
    "RegionError",
    "axis_vector",
    "orthogonal_vector",
    "InfiniteCylinderRegion",
    "HollowInfiniteCylinderRegion",
    "CylinderRegion",
    "HollowCylinderRegion",
    "SphereRegion",
    "HollowSphereRegion",
    "select",
]

TOL = 1e-9

_NAMED_AXES = {
    "X": (1.0, 0.0, 0.0),
    "-X": (-1.0, 0.0, 0.0),
    "Y": (0.0, 1.0, 0.0),
    "-Y": (0.0, -1.0, 0.0),
    "Z": (0.0, 0.0, 1.0),
    "-Z": (0.0, 0.0, -1.0),
}


class RegionError(ValueError):
    """Raised for a degenerate region (zero axis, bad radii, ...)."""


# --------------------------------------------------------------------------
# vector helpers
# --------------------------------------------------------------------------

def _sub(a: Sequence[float], b: Sequence[float]) -> Point:
    return (a[0] - b[0], a[1] - b[1], a[2] - b[2])


def _dot(a: Sequence[float], b: Sequence[float]) -> float:
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


def _norm(a: Sequence[float]) -> float:
    return math.sqrt(_dot(a, a))


def axis_vector(axis) -> Point:
    """Resolve a named axis (``"X"``, ``"-Z"``, ...) or 3-tuple to a unit vector.

    Matches the plugin's ``get_axis``: named vectors for the six signed axes,
    otherwise any non-zero vector is accepted and normalised.
    """
    if isinstance(axis, str):
        try:
            v = _NAMED_AXES[axis]
        except KeyError:
            raise RegionError(
                "unknown axis %r (named axes: %s)"
                % (axis, ", ".join(sorted(_NAMED_AXES))))
    else:
        v = (float(axis[0]), float(axis[1]), float(axis[2]))
    n = _norm(v)
    if n <= TOL:
        raise RegionError("axis vector must be non-zero")
    return (v[0] / n, v[1] / n, v[2] / n)


def orthogonal_vector(axis) -> Point:
    """A unit vector orthogonal to ``axis`` (for building a local frame).

    Deterministic: crosses the axis with whichever world axis it is least
    parallel to, so the result is always well conditioned.
    """
    w = axis_vector(axis)
    ax, ay, az = abs(w[0]), abs(w[1]), abs(w[2])
    if ax <= ay and ax <= az:
        other = (1.0, 0.0, 0.0)
    elif ay <= az:
        other = (0.0, 1.0, 0.0)
    else:
        other = (0.0, 0.0, 1.0)
    cx = w[1] * other[2] - w[2] * other[1]
    cy = w[2] * other[0] - w[0] * other[2]
    cz = w[0] * other[1] - w[1] * other[0]
    n = _norm((cx, cy, cz))
    if n <= TOL:
        raise RegionError("could not build an orthogonal vector")
    return (cx / n, cy / n, cz / n)


def _axial_radial(point: Sequence[float], origin: Point,
                  axis_hat: Point) -> Tuple[float, float]:
    """Return ``(h, rho)``: axial coordinate and perpendicular distance."""
    d = _sub(point, origin)
    h = _dot(d, axis_hat)
    perp = (d[0] - h * axis_hat[0],
            d[1] - h * axis_hat[1],
            d[2] - h * axis_hat[2])
    rho = _norm(perp)
    return h, rho


# --------------------------------------------------------------------------
# region base
# --------------------------------------------------------------------------

class _Region:
    """Base class: a solid region with a ``contains`` point-membership test."""

    def contains(self, point: Sequence[float]) -> bool:  # pragma: no cover
        raise NotImplementedError

    def filter(self, points: Iterable[Sequence[float]]) -> List[Point]:
        """Keep the points inside the region, preserving order."""
        out: List[Point] = []
        for p in points:
            if self.contains(p):
                out.append((float(p[0]), float(p[1]), float(p[2])))
        return out


@dataclass(frozen=True)
class InfiniteCylinderRegion(_Region):
    """All points within ``radius`` of an infinite line through ``origin``."""

    origin: Point
    axis: Point
    radius: float

    def __post_init__(self) -> None:
        if self.radius <= 0.0:
            raise RegionError("radius must be positive")
        object.__setattr__(self, "_hat", axis_vector(self.axis))

    def contains(self, point: Sequence[float]) -> bool:
        _h, rho = _axial_radial(point, self.origin, self._hat)  # type: ignore[attr-defined]
        return rho < self.radius


@dataclass(frozen=True)
class HollowInfiniteCylinderRegion(_Region):
    """Points between an inner and outer radius of an infinite axis."""

    origin: Point
    axis: Point
    outer_radius: float
    inner_radius: float

    def __post_init__(self) -> None:
        if self.inner_radius < 0.0:
            raise RegionError("inner_radius must not be negative")
        if self.outer_radius <= self.inner_radius:
            raise RegionError("outer_radius must exceed inner_radius")
        object.__setattr__(self, "_hat", axis_vector(self.axis))

    def contains(self, point: Sequence[float]) -> bool:
        _h, rho = _axial_radial(point, self.origin, self._hat)  # type: ignore[attr-defined]
        return self.inner_radius < rho < self.outer_radius


@dataclass(frozen=True)
class CylinderRegion(_Region):
    """A finite cylinder: within ``radius`` and ``0 < h < height`` along axis."""

    origin: Point
    axis: Point
    height: float
    radius: float

    def __post_init__(self) -> None:
        if self.radius <= 0.0:
            raise RegionError("radius must be positive")
        if self.height <= 0.0:
            raise RegionError("height must be positive")
        object.__setattr__(self, "_hat", axis_vector(self.axis))

    def contains(self, point: Sequence[float]) -> bool:
        h, rho = _axial_radial(point, self.origin, self._hat)  # type: ignore[attr-defined]
        return rho < self.radius and 0.0 < h < self.height


@dataclass(frozen=True)
class HollowCylinderRegion(_Region):
    """A finite hollow cylinder (tube) of given height and inner/outer radii."""

    origin: Point
    axis: Point
    height: float
    outer_radius: float
    inner_radius: float

    def __post_init__(self) -> None:
        if self.inner_radius < 0.0:
            raise RegionError("inner_radius must not be negative")
        if self.outer_radius <= self.inner_radius:
            raise RegionError("outer_radius must exceed inner_radius")
        if self.height <= 0.0:
            raise RegionError("height must be positive")
        object.__setattr__(self, "_hat", axis_vector(self.axis))

    def contains(self, point: Sequence[float]) -> bool:
        h, rho = _axial_radial(point, self.origin, self._hat)  # type: ignore[attr-defined]
        return (self.inner_radius < rho < self.outer_radius
                and 0.0 < h < self.height)


@dataclass(frozen=True)
class SphereRegion(_Region):
    """All points within ``radius`` of ``origin``."""

    origin: Point
    radius: float

    def __post_init__(self) -> None:
        if self.radius <= 0.0:
            raise RegionError("radius must be positive")

    def contains(self, point: Sequence[float]) -> bool:
        return _norm(_sub(point, self.origin)) < self.radius


@dataclass(frozen=True)
class HollowSphereRegion(_Region):
    """A spherical shell between an inner and outer radius about ``origin``."""

    origin: Point
    outer_radius: float
    inner_radius: float

    def __post_init__(self) -> None:
        if self.inner_radius < 0.0:
            raise RegionError("inner_radius must not be negative")
        if self.outer_radius <= self.inner_radius:
            raise RegionError("outer_radius must exceed inner_radius")

    def contains(self, point: Sequence[float]) -> bool:
        d = _norm(_sub(point, self.origin))
        return self.inner_radius < d < self.outer_radius


def select(region: _Region, entities: Iterable,
           center: Callable[[object], Sequence[float]] = None) -> List:
    """Keep ``entities`` whose centre lies inside ``region`` (order preserved).

    ``center`` maps an entity to its centre point; it defaults to the identity
    (the entities are themselves points).  This mirrors the plugin's
    ``filter``, which reads ``o.Center()`` from each candidate shape.
    """
    if center is None:
        center = lambda e: e  # noqa: E731
    return [e for e in entities if region.contains(center(e))]
