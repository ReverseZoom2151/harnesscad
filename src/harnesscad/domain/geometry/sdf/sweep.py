"""Twisted / scaled linear extrusion of a 2D field, from ImplicitCAD ``ExtrudeM``.

The OpenSCAD-family languages (OpenSCAD, RapCAD, OpenJSCAD, replicad) all expose
a ``linear_extrude(height, twist=, scale=)`` that lifts a 2D profile into a 3D
solid while optionally *twisting* it about the extrusion axis and *scaling* the
cross-section from bottom to top.  ImplicitCAD implements exactly this on signed
distance fields (``Graphics/Implicit/ObjectUtil/GetImplicit3.hs``, the
``ExtrudeM`` case): at height ``z`` it rotates the query point back by the twist
accumulated over ``z`` and divides it by the interpolated scale before sampling
the 2D field, then intersects with the ``[0, h]`` slab.

The harness's :mod:`primitives` already has a plain :func:`primitives.extrude`
(straight prism); this module adds the *twisted* and *tapered* variants those
CAD languages provide, as a domain transform over any 2D field callable
``shape2d(x, y) -> float``.

Conventions (matching OpenSCAD):

* The solid occupies ``0 <= z <= height`` (base at ``z = 0``).
* ``twist`` is the **total** rotation in degrees over the full height; the
  cross-section at height ``z`` is rotated by ``twist * z / height`` degrees
  counter-clockwise (OpenSCAD's positive-twist convention).
* ``scale`` is the per-axis cross-section factor reached at the *top*; the factor
  at height ``z`` interpolates linearly from ``(1, 1)`` at the base to ``scale``
  at the top.  A uniform ``scale`` tapers the profile like a frustum.

Distance-field class: like ImplicitCAD's ``ExtrudeM`` (and OpenSCAD's mesh), the
twisted/tapered field is **approximate** -- twisting and non-uniform scaling do
not preserve the Eikonal property -- but it is sign-correct (negative strictly
inside the twisted solid).  The scale compensation multiplies by the smallest
axis factor to keep the field a conservative lower bound (never an overestimate),
so it stays safe for sphere tracing.

Pure stdlib, deterministic.
"""

from __future__ import annotations

from math import cos, radians, sin
from typing import Callable, Sequence

__all__ = ["twist_extrude", "taper_extrude", "linear_extrude"]


def _extrude_at(
    shape2d: Callable[[float, float], float],
    p: Sequence[float],
    height: float,
    twist_deg: float,
    scale: Sequence[float],
) -> float:
    if height <= 0.0:
        raise ValueError("extrude height must be positive")
    x, y, z = float(p[0]), float(p[1]), float(p[2])
    # fraction up the extrusion (clamped so the caps sample the end profiles).
    t = z / height
    if t < 0.0:
        t = 0.0
    elif t > 1.0:
        t = 1.0

    # undo the twist accumulated at this height (rotate the point back).
    theta = radians(twist_deg * t)
    if theta != 0.0:
        ca, sa = cos(theta), sin(theta)
        rx = x * ca + y * sa
        ry = -x * sa + y * ca
        x, y = rx, ry

    # undo the interpolated cross-section scale.
    sx = 1.0 + (float(scale[0]) - 1.0) * t
    sy = 1.0 + (float(scale[1]) - 1.0) * t
    if sx == 0.0 or sy == 0.0:
        raise ValueError("interpolated scale reached zero")
    x /= sx
    y /= sy

    d2 = shape2d(x, y)
    # keep a conservative (non-overestimating) field under the scale.
    comp = min(abs(sx), abs(sy))
    d2 *= comp

    # intersect with the [0, height] slab: cap distance about the mid-plane.
    cap = abs(z - height / 2.0) - height / 2.0
    inside = min(max(d2, cap), 0.0)
    outside = ((max(d2, 0.0)) ** 2 + (max(cap, 0.0)) ** 2) ** 0.5
    return inside + outside


def twist_extrude(
    shape2d: Callable[[float, float], float],
    p: Sequence[float],
    height: float,
    twist_deg: float,
) -> float:
    """Linear extrude of ``shape2d`` to ``height`` with a total ``twist_deg`` twist.

    ``linear_extrude(height, twist=twist_deg)``.  Approximate but sign-correct
    SDF; negative strictly inside the twisted prism.
    """
    return _extrude_at(shape2d, p, height, twist_deg, (1.0, 1.0))


def taper_extrude(
    shape2d: Callable[[float, float], float],
    p: Sequence[float],
    height: float,
    scale: Sequence[float],
) -> float:
    """Linear extrude of ``shape2d`` to ``height`` tapering to ``scale`` at the top.

    ``linear_extrude(height, scale=scale)`` with ``scale = (sx, sy)`` the top
    cross-section factor (base factor is ``(1, 1)``).
    """
    return _extrude_at(shape2d, p, height, 0.0, scale)


def linear_extrude(
    shape2d: Callable[[float, float], float],
    p: Sequence[float],
    height: float,
    twist_deg: float = 0.0,
    scale: Sequence[float] = (1.0, 1.0),
) -> float:
    """Full OpenSCAD-style ``linear_extrude(height, twist=, scale=)``.

    Combines :func:`twist_extrude` and :func:`taper_extrude`.  With the defaults
    (no twist, unit scale) it reduces to a straight prism over ``0 <= z <=
    height``.
    """
    return _extrude_at(shape2d, p, height, twist_deg, scale)
