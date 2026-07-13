"""Distance-field and domain transforms, ported from Curv.

Two families of operators (``lib/curv/std.curv``):

**Distance-field (level-set) transforms** act on the *value* of a field:

* ``offset(d, field) = field - d``            -- inflate (``d>0``) / deflate.
* ``shell(t, field) = |field| - t/2``         -- hollow shell of thickness ``t``.
* ``round(r, field) = field - r``             -- round convex features (exact
  fields only); identical maths to ``offset`` but named for intent.
* ``morph(t, a, b) = lerp(a, b, t)``          -- linear field interpolation.

**Domain transforms** act on the *point* before it is fed to the field.  A
distance field ``f`` is a *metric* object, so any domain map ``g`` must be
distance-preserving (an isometry) for ``f(g(p))`` to stay a valid SDF; when
``g`` scales space by ``s`` the field value must be multiplied back by ``s``
(the *scale-compensation* factor) to preserve the Eikonal property:

* ``translate`` / ``rotate`` / ``mirror`` -- isometries, no compensation.
* ``scale(s, ...)``   -- isotropic: ``f(p/s) * s``.
* ``stretch(v, ...)`` -- anisotropic: ``f(p/v) * min(v)`` (approximate; ``min``
  keeps it a safe 1-Lipschitz lower bound).
* ``repeat_*``        -- mod-space tiling: infinite (``mod``) and finite
  (``round`` + ``clamp`` to a cell count) translational lattices, plus mirror
  symmetry.

Each transform here is a small pure function operating on point tuples and/or a
field callable ``f(point) -> float``.  stdlib-only, deterministic.
"""

from __future__ import annotations

from math import cos, floor, sin
from typing import Callable, Sequence


def _clamp(v: float, lo: float, hi: float) -> float:
    return lo if v < lo else hi if v > hi else v


def _mod(a: float, b: float) -> float:
    """Curv/GLSL ``mod``: result has the sign of ``b`` (floored division)."""
    return a - b * floor(a / b)


# --------------------------------------------------------------------------- #
# distance-field (level-set) transforms                                        #
# --------------------------------------------------------------------------- #
def offset(dist: float, d: float) -> float:
    """Inflate/deflate a field value: ``dist - d``.

    ``d > 0`` inflates (rounded offset for exact fields, Minkowski sum with a
    ball of radius ``d``); ``d < 0`` deflates.
    """
    return dist - d


def shell(dist: float, thickness: float) -> float:
    """Hollow shell of ``thickness`` centred on the boundary: ``|dist| - t/2``."""
    return abs(dist) - thickness / 2.0


def round_field(dist: float, r: float) -> float:
    """Round convex features by radius ``r`` (exact fields): ``dist - r``."""
    return dist - r


def morph(a: float, b: float, t: float) -> float:
    """Linearly interpolate two field values: ``lerp(a, b, t)``.

    ``t = 0`` yields ``a``, ``t = 1`` yields ``b``.
    """
    return a + (b - a) * t


# --------------------------------------------------------------------------- #
# domain transforms (act on the point, then call the field)                    #
# --------------------------------------------------------------------------- #
def translate(f: Callable, delta: Sequence[float]):
    """Return a field translated by ``delta``: ``p -> f(p - delta)``."""
    d = tuple(float(c) for c in delta)

    def g(p):
        return f(tuple(p[i] - d[i] for i in range(len(p))))

    return g


def scale(f: Callable, s: float):
    """Isotropic scale by ``s`` with distance compensation: ``f(p/s) * s``."""
    if s == 0.0:
        raise ValueError("scale factor must be non-zero")

    def g(p):
        return f(tuple(c / s for c in p)) * s

    return g


def stretch(f: Callable, v: Sequence[float]):
    """Anisotropic scale by per-axis ``v`` (approximate): ``f(p/v) * min(v)``.

    Multiplying by the smallest factor keeps the result a 1-Lipschitz lower
    bound on the true distance (Curv's ``stretch``); it is *not* exact.
    """
    vv = tuple(float(c) for c in v)
    mn = min(vv)

    def g(p):
        return f(tuple(p[i] / vv[i] for i in range(len(p)))) * mn

    return g


def rotate_z(f: Callable, angle: float):
    """Rotate a 2D/3D field about the Z axis by ``angle`` (radians).

    Applies the inverse rotation to the point (an isometry: no compensation).
    """
    ca, sa = cos(-angle), sin(-angle)

    def g(p):
        x = p[0] * ca - p[1] * sa
        y = p[0] * sa + p[1] * ca
        return f((x, y) + tuple(p[2:]))

    return g


def mirror_x(f: Callable):
    """Mirror symmetry across the YZ plane: ``f(|x|, y, z, ...)``.

    Reflects the +X half-space into the -X half-space (Curv ``repeat_mirror_x``).
    """
    def g(p):
        return f((abs(p[0]),) + tuple(p[1:]))

    return g


def reflect_x(f: Callable):
    """Reflect the field across the YZ plane: ``f(-x, y, z, ...)``."""
    def g(p):
        return f((-p[0],) + tuple(p[1:]))

    return g


# --------------------------------------------------------------------------- #
# repetition (mod-space tiling)                                                #
# --------------------------------------------------------------------------- #
def repeat_x(f: Callable, d: float):
    """Infinite translational repetition along X with cell width ``d``."""
    r = d / 2.0

    def g(p):
        return f((_mod(p[0] + r, d) - r,) + tuple(p[1:]))

    return g


def repeat_xyz(f: Callable, cell: Sequence[float]):
    """Infinite 3D lattice with cell dimensions ``cell = [dx, dy, dz]``."""
    dx, dy, dz = cell
    rx, ry, rz = dx / 2.0, dy / 2.0, dz / 2.0

    def g(p):
        return f((
            _mod(p[0] + rx, dx) - rx,
            _mod(p[1] + ry, dy) - ry,
            _mod(p[2] + rz, dz) - rz,
        ))

    return g


def repeat_finite(f: Callable, cell: Sequence[float], counts: Sequence[float]):
    """Finite 3D lattice: ``counts = [lx, ly, lz]`` copies from the origin.

    Curv's ``repeat_finite``: fold via ``x - dx*clamp(round(x/dx), 0, lx-1)``.
    Cell 0 sits at the origin and copies extend toward +axis.
    """
    dx, dy, dz = cell
    lx, ly, lz = counts

    def g(p):
        x = p[0] - dx * _clamp(round(p[0] / dx), 0, lx - 1)
        y = p[1] - dy * _clamp(round(p[1] / dy), 0, ly - 1)
        z = p[2] - dz * _clamp(round(p[2] / dz), 0, lz - 1)
        return f((x, y, z))

    return g
