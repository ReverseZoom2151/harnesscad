"""CadQuery ``Plane`` named-preset frame algebra (OCCT-free, pure Python).

CadQuery's ``cadquery/occ_impl/geom.py`` defines ``Plane`` -- a 2D coordinate
system embedded in 3D space -- via a table of *named presets*
(``XY``/``YZ``/``ZX``/``XZ``/``YX``/``ZY`` plus the view names
``front``/``back``/``left``/``right``/``top``/``bottom``) and the pair of
transforms ``toWorldCoords`` / ``toLocalCoords`` that move points between the
plane's local frame and global space.  The geometry is delegated to OCCT's
``gp_Ax3.SetTransformation`` there, but the maths is a plain orthonormal rigid
frame and is reproduced here deterministically.

The harness already has :mod:`geometry.codetocad_transform_stack` (a general
4x4 matrix lib: translate/rotate/scale/mirror/Rodrigues) but NOT this named
CAD-plane preset system, nor the exact world/local convention CadQuery uses.
This module adds only the plane-frame layer:

* :meth:`Plane.named` and the twelve preset classmethods, reproducing the exact
  ``(origin, xDir, normal)`` table from the reference.
* :meth:`Plane.toWorldCoords` / :meth:`Plane.toLocalCoords` -- ``world =
  origin + x*xDir + y*yDir + z*zDir`` and its inverse; a 2-tuple local point is
  treated as ``z = 0`` exactly as CadQuery does.
* :meth:`Plane.rotated` -- rotate the frame's direction vectors about its own
  x, y, z axes (in that order), origin unchanged, via Rodrigues rotation.
* :meth:`Plane.setOrigin2d`, ``__eq__`` with the reference's tolerances.

``yDir`` is always ``zDir x xDir`` (right-handed), matching ``_setPlaneDir``.
"""

from __future__ import annotations

import math
from typing import Dict, Sequence, Tuple

__all__ = ["Vec3", "Plane", "PlaneError"]

Vec = Tuple[float, float, float]


class PlaneError(ValueError):
    """Raised for invalid plane construction (null direction, unknown name)."""


# --------------------------------------------------------------------------
# minimal 3-vector helpers
# --------------------------------------------------------------------------

def _add(a: Vec, b: Vec) -> Vec:
    return (a[0] + b[0], a[1] + b[1], a[2] + b[2])


def _sub(a: Vec, b: Vec) -> Vec:
    return (a[0] - b[0], a[1] - b[1], a[2] - b[2])


def _scale(a: Vec, s: float) -> Vec:
    return (a[0] * s, a[1] * s, a[2] * s)


def _dot(a: Vec, b: Vec) -> float:
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


def _cross(a: Vec, b: Vec) -> Vec:
    return (
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    )


def _norm(a: Vec) -> float:
    return math.sqrt(_dot(a, a))


def _normalize(a: Vec) -> Vec:
    n = _norm(a)
    if n == 0.0:
        raise PlaneError("cannot normalize a null vector")
    return (a[0] / n, a[1] / n, a[2] / n)


def _rodrigues(v: Vec, axis: Vec, angle: float) -> Vec:
    """Rotate ``v`` about the unit ``axis`` through origin by ``angle`` radians."""
    k = _normalize(axis)
    c = math.cos(angle)
    s = math.sin(angle)
    # v*cos + (k x v)*sin + k*(k.v)*(1-cos)
    return _add(
        _add(_scale(v, c), _scale(_cross(k, v), s)),
        _scale(k, _dot(k, v) * (1.0 - c)),
    )


def Vec3(x: float, y: float, z: float) -> Vec:
    """Construct a 3-vector tuple (helper for callers)."""
    return (float(x), float(y), float(z))


# origin-independent (xDir, normal) part of CadQuery's namedPlanes table
_NAMED: Dict[str, Tuple[Vec, Vec]] = {
    "XY": ((1, 0, 0), (0, 0, 1)),
    "YZ": ((0, 1, 0), (1, 0, 0)),
    "ZX": ((0, 0, 1), (0, 1, 0)),
    "XZ": ((1, 0, 0), (0, -1, 0)),
    "YX": ((0, 1, 0), (0, 0, -1)),
    "ZY": ((0, 0, 1), (-1, 0, 0)),
    "front": ((1, 0, 0), (0, 0, 1)),
    "back": ((-1, 0, 0), (0, 0, -1)),
    "left": ((0, 0, 1), (-1, 0, 0)),
    "right": ((0, 0, -1), (1, 0, 0)),
    "top": ((1, 0, 0), (0, 1, 0)),
    "bottom": ((1, 0, 0), (0, -1, 0)),
}

# per-name default xDir used by the classmethod constructors
_DEFAULT_XDIR: Dict[str, Vec] = {name: v[0] for name, v in _NAMED.items()}


class Plane(object):
    """A 2D coordinate system embedded in 3D space.

    ``origin`` is in global coordinates; ``xDir``/``yDir``/``zDir`` form a
    right-handed orthonormal basis with ``yDir = zDir x xDir``.
    """

    _eq_tolerance_origin = 1e-6
    _eq_tolerance_dot = 1e-6

    def __init__(
        self,
        origin: Sequence[float] = (0, 0, 0),
        xDir: Sequence[float] = None,
        normal: Sequence[float] = (0, 0, 1),
    ):
        z = (float(normal[0]), float(normal[1]), float(normal[2]))
        if _norm(z) == 0.0:
            raise PlaneError("normal should be non null")
        self.zDir: Vec = _normalize(z)

        if xDir is None:
            # pick an arbitrary in-plane x direction perpendicular to z
            trial = (1.0, 0.0, 0.0)
            if abs(_dot(trial, self.zDir)) > 0.9:
                trial = (0.0, 1.0, 0.0)
            xd = _sub(trial, _scale(self.zDir, _dot(trial, self.zDir)))
        else:
            xd = (float(xDir[0]), float(xDir[1]), float(xDir[2]))
            if _norm(xd) == 0.0:
                raise PlaneError("xDir should be non null")
        self._set_plane_dir(xd)
        self.origin: Vec = (float(origin[0]), float(origin[1]), float(origin[2]))

    # ---- named presets -------------------------------------------------
    @classmethod
    def named(cls, stdName: str, origin: Sequence[float] = (0, 0, 0)) -> "Plane":
        if stdName not in _NAMED:
            raise PlaneError(f"Supported names are {list(_NAMED.keys())}")
        xDir, normal = _NAMED[stdName]
        return cls(origin, xDir, normal)

    @classmethod
    def _preset(cls, name: str, origin, xDir):
        p = cls.named(name, origin)
        if xDir is not None:
            p._set_plane_dir(xDir)
        return p

    @classmethod
    def XY(cls, origin=(0, 0, 0), xDir=None):
        return cls._preset("XY", origin, xDir)

    @classmethod
    def YZ(cls, origin=(0, 0, 0), xDir=None):
        return cls._preset("YZ", origin, xDir)

    @classmethod
    def ZX(cls, origin=(0, 0, 0), xDir=None):
        return cls._preset("ZX", origin, xDir)

    @classmethod
    def XZ(cls, origin=(0, 0, 0), xDir=None):
        return cls._preset("XZ", origin, xDir)

    @classmethod
    def YX(cls, origin=(0, 0, 0), xDir=None):
        return cls._preset("YX", origin, xDir)

    @classmethod
    def ZY(cls, origin=(0, 0, 0), xDir=None):
        return cls._preset("ZY", origin, xDir)

    @classmethod
    def front(cls, origin=(0, 0, 0), xDir=None):
        return cls._preset("front", origin, xDir)

    @classmethod
    def back(cls, origin=(0, 0, 0), xDir=None):
        return cls._preset("back", origin, xDir)

    @classmethod
    def left(cls, origin=(0, 0, 0), xDir=None):
        return cls._preset("left", origin, xDir)

    @classmethod
    def right(cls, origin=(0, 0, 0), xDir=None):
        return cls._preset("right", origin, xDir)

    @classmethod
    def top(cls, origin=(0, 0, 0), xDir=None):
        return cls._preset("top", origin, xDir)

    @classmethod
    def bottom(cls, origin=(0, 0, 0), xDir=None):
        return cls._preset("bottom", origin, xDir)

    # ---- frame maths ---------------------------------------------------
    def _set_plane_dir(self, xDir: Sequence[float]) -> None:
        xd = _normalize((float(xDir[0]), float(xDir[1]), float(xDir[2])))
        self.xDir: Vec = xd
        self.yDir: Vec = _normalize(_cross(self.zDir, xd))

    def toWorldCoords(self, point: Sequence[float]) -> Vec:
        """Local -> global.  A 2-tuple is taken as ``z = 0``."""
        if len(point) == 2:
            x, y, z = float(point[0]), float(point[1]), 0.0
        else:
            x, y, z = float(point[0]), float(point[1]), float(point[2])
        return _add(
            self.origin,
            _add(_scale(self.xDir, x), _add(_scale(self.yDir, y), _scale(self.zDir, z))),
        )

    def toLocalCoords(self, point: Sequence[float]) -> Vec:
        """Global -> local (project onto this plane's frame)."""
        p = (float(point[0]), float(point[1]), float(point[2]))
        rel = _sub(p, self.origin)
        return (_dot(rel, self.xDir), _dot(rel, self.yDir), _dot(rel, self.zDir))

    def setOrigin2d(self, x: float, y: float) -> None:
        """Move the origin within the plane by local ``(x, y)``."""
        self.origin = self.toWorldCoords((x, y))

    def rotated(self, rotate: Sequence[float] = (0, 0, 0)) -> "Plane":
        """Return a copy rotated about the frame's x, y, z axes (degrees, in order)."""
        rx, ry, rz = (math.radians(float(r)) for r in rotate)
        new_x = self.xDir
        new_z = self.zDir
        for axis, ang in ((self.xDir, rx), (self.yDir, ry), (self.zDir, rz)):
            if ang == 0.0:
                continue
            new_x = _rodrigues(new_x, axis, ang)
            new_z = _rodrigues(new_z, axis, ang)
        return Plane(self.origin, new_x, new_z)

    # ---- equality / repr ----------------------------------------------
    def __eq__(self, other) -> bool:
        if not isinstance(other, Plane):
            return NotImplemented
        if _norm(_sub(self.origin, other.origin)) >= self._eq_tolerance_origin:
            return False
        if abs(_dot(self.zDir, other.zDir) - 1) >= self._eq_tolerance_dot:
            return False
        if abs(_dot(self.xDir, other.xDir) - 1) >= self._eq_tolerance_dot:
            return False
        return True

    def __ne__(self, other) -> bool:
        result = self.__eq__(other)
        if result is NotImplemented:
            return result
        return not result

    def __repr__(self) -> str:
        return f"Plane(origin={self.origin}, xDir={self.xDir}, normal={self.zDir})"
