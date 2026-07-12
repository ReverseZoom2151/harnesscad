"""Additional analytic signed-distance primitives from ``sdf-csg``.

Ported from wwwtyro/sdf-csg (``src/primitives.ts``), which is in turn a direct
transcription of Inigo Quilez's 3D SDF article.  These are the primitives that
the harness's :mod:`geometry.curv_sdf_primitives` does **not** already provide.

Already covered elsewhere (NOT re-implemented here): sphere, box (mitred and
exact), rounded box, cylinder, cone, capped cone, capsule, torus, ellipsoid,
plane -- see :mod:`geometry.curv_sdf_primitives`.

Genuine additions in this module:

* :func:`box_frame`         -- hollow wireframe of a box (12 struts)
* :func:`capped_torus`      -- an arc of a torus subtended by an angle
* :func:`link`              -- a torus stretched into a chain link
* :func:`hexagonal_prism`   -- regular hexagon extruded along z
* :func:`triangular_prism`  -- equilateral triangle extruded along z
* :func:`solid_angle`       -- a spherical wedge / ice-cream-cone cap

Every function takes a point ``p`` (a length-3 sequence ``(x, y, z)``) and
returns the signed distance: negative strictly inside, positive strictly
outside, zero on the boundary.  All are pure, deterministic, stdlib-only.
"""

from __future__ import annotations

import math
from typing import Sequence


def _len2(x: float, y: float) -> float:
    return math.sqrt(x * x + y * y)


def _len3(x: float, y: float, z: float) -> float:
    return math.sqrt(x * x + y * y + z * z)


def _clamp(v: float, lo: float, hi: float) -> float:
    return lo if v < lo else hi if v > hi else v


def _sign(v: float) -> float:
    return (v > 0.0) - (v < 0.0)


def box_frame(p: Sequence[float], b: Sequence[float], e: float) -> float:
    """Signed distance to the wireframe (12 edges) of a box.

    ``b`` are the half-extents of the outer box, ``e`` the strut thickness.
    """
    px = abs(p[0]) - b[0]
    py = abs(p[1]) - b[1]
    pz = abs(p[2]) - b[2]
    qx = abs(px + e) - e
    qy = abs(py + e) - e
    qz = abs(pz + e) - e

    def strut(ax: float, ay: float, az: float) -> float:
        return (
            _len3(max(ax, 0.0), max(ay, 0.0), max(az, 0.0))
            + min(max(ax, max(ay, az)), 0.0)
        )

    return min(
        min(strut(px, qy, qz), strut(qx, py, qz)),
        strut(qx, qy, pz),
    )


def capped_torus(p: Sequence[float], angle: float, major: float, minor: float) -> float:
    """Signed distance to an arc of a torus subtending ``2*angle`` radians.

    ``major`` is the ring radius, ``minor`` the tube radius.  The arc opens
    around +y; ``angle == pi`` recovers a full torus.
    """
    sx = math.sin(angle)
    sy = math.cos(angle)
    x = abs(p[0])
    y = p[1]
    z = p[2]
    if sy * x > sx * y:
        k = x * sx + y * sy
    else:
        k = _len2(x, y)
    return math.sqrt(x * x + y * y + z * z + major * major - 2.0 * major * k) - minor


def link(p: Sequence[float], length: float, major: float, minor: float) -> float:
    """Signed distance to a chain link (torus stretched by ``length`` along y)."""
    qx = p[0]
    qy = max(abs(p[1]) - length, 0.0)
    qz = p[2]
    return _len2(_len2(qx, qy) - major, qz) - minor


def hexagonal_prism(p: Sequence[float], radius: float, length: float) -> float:
    """Signed distance to a regular hexagonal prism (axis along z).

    ``radius`` is the apothem (centre-to-edge) of the hexagon, ``length`` the
    half-height along z.
    """
    kx, ky, kz = -0.8660254, 0.5, 0.57735
    px = abs(p[0])
    py = abs(p[1])
    pz = abs(p[2])
    d = 2.0 * min(kx * px + ky * py, 0.0)
    px -= d * kx
    py -= d * ky
    dx = _len2(px - _clamp(px, -kz * radius, kz * radius), py - radius) * _sign(py - radius)
    dy = pz - length
    return min(max(dx, dy), 0.0) + _len2(max(dx, 0.0), max(dy, 0.0))


def triangular_prism(p: Sequence[float], side: float, length: float) -> float:
    """Signed distance to an equilateral triangular prism (axis along z).

    ``side`` scales the triangle cross-section, ``length`` is the half-height.
    """
    qx = abs(p[0])
    qy = abs(p[1])
    qz = abs(p[2])
    return max(qz - length, max(qx * 0.866025 + p[1] * 0.5, -p[1]) - side * 0.5)


def solid_angle(p: Sequence[float], angle: float, radius: float) -> float:
    """Signed distance to a spherical wedge (cap of a sphere within a cone).

    ``angle`` is the half-aperture of the cone (about +y), ``radius`` the sphere
    radius.
    """
    cx = math.sin(angle)
    cy = math.cos(angle)
    qx = _len2(p[0], p[2])
    qy = p[1]
    l = _len2(qx, qy) - radius
    t = _clamp(qx * cx + qy * cy, 0.0, radius)
    m = _len2(qx - cx * t, qy - cy * t)
    return max(l, m * _sign(cy * qx - cx * qy))
