"""Sketch-plane orientation + extrusion decode.

The extrusion command carries the 3D pose of the sketch plane so that the
2D curve parameters (which live in the plane's local frame) can be lifted into world
space. The plane orientation is defined by a rotational matrix, determined by
``(theta, phi, gamma)``, to align the world frame of reference to the plane's local
frame of reference, and to align the z-axis to the plane's normal direction. The
extrusion command also carries the plane origin ``(px, py, pz)``, a profile scale
``s``, the two extrude distances ``(e1, e2)`` and an extrude *type* (one-sided /
symmetric / two-sided).

This module makes that decode explicit and deterministic. The three angles are
interpreted as a **ZYZ Euler parameterization** -- the standard, invertible way to
name a rotation by two spherical angles for the z-axis (plane normal) plus an
in-plane roll:

    R(theta, phi, gamma) = Rz(phi) . Ry(theta) . Rz(gamma)

so the plane normal (local +z mapped to world) is the spherical direction
``n = (sin theta cos phi, sin theta sin phi, cos theta)`` (theta = polar angle from
world +z, phi = azimuth) and ``gamma`` rolls the in-plane x/y axes about that
normal. This matches the requirement that the matrix aligns the z-axis to
the plane normal, and it is exactly invertible (ZYZ extraction) so a decoded pose
round-trips to the same angles.

Pure stdlib (``math`` only), no numpy, no CAD kernel. Vectors are length-3 tuples;
rotation matrices are row-major 3x3 tuples-of-tuples.
"""

from __future__ import annotations

import math

Vec3 = tuple[float, float, float]
Vec2 = tuple[float, float]
Mat3 = tuple[tuple[float, float, float], ...]

# Extrude-type codes (one-sided / symmetric / two-sided).
ONE_SIDED = 0
SYMMETRIC = 1
TWO_SIDED = 2


def _rz(a: float) -> Mat3:
    c, s = math.cos(a), math.sin(a)
    return ((c, -s, 0.0), (s, c, 0.0), (0.0, 0.0, 1.0))


def _ry(a: float) -> Mat3:
    c, s = math.cos(a), math.sin(a)
    return ((c, 0.0, s), (0.0, 1.0, 0.0), (-s, 0.0, c))


def _matmul(a: Mat3, b: Mat3) -> Mat3:
    return tuple(
        tuple(sum(a[i][k] * b[k][j] for k in range(3)) for j in range(3))
        for i in range(3)
    )


def _matvec(m: Mat3, v: Vec3) -> Vec3:
    return tuple(sum(m[i][j] * v[j] for j in range(3)) for i in range(3))


def rotation_matrix(theta: float, phi: float, gamma: float) -> Mat3:
    """Sketch-plane rotation ``R = Rz(phi) . Ry(theta) . Rz(gamma)`` (ZYZ).

    Columns of ``R`` are the world-space directions of the plane's local x, y and z
    (normal) axes. Angles are in radians.
    """
    return _matmul(_matmul(_rz(phi), _ry(theta)), _rz(gamma))


def plane_axes(theta: float, phi: float, gamma: float) -> tuple[Vec3, Vec3, Vec3]:
    """Return the world-space ``(x_axis, y_axis, normal)`` of the sketch plane."""
    m = rotation_matrix(theta, phi, gamma)
    x_axis = (m[0][0], m[1][0], m[2][0])
    y_axis = (m[0][1], m[1][1], m[2][1])
    normal = (m[0][2], m[1][2], m[2][2])
    return x_axis, y_axis, normal


def plane_normal(theta: float, phi: float) -> Vec3:
    """The plane normal (local +z in world) -- spherical direction of ``(theta, phi)``."""
    st = math.sin(theta)
    return (st * math.cos(phi), st * math.sin(phi), math.cos(theta))


def euler_from_matrix(m: Mat3) -> tuple[float, float, float]:
    """Inverse of :func:`rotation_matrix`: recover ``(theta, phi, gamma)`` (ZYZ).

    ``theta in [0, pi]``. At the gimbal poles (``sin theta ~ 0``) only ``phi+gamma``
    is determined; ``gamma`` is pinned to 0 by convention so the map is single-valued.
    """
    r22 = min(1.0, max(-1.0, m[2][2]))
    theta = math.acos(r22)
    sin_theta = math.sin(theta)
    if abs(sin_theta) < 1e-9:
        # Gimbal: normal along +/-z. Fold the two z-rotations into phi.
        gamma = 0.0
        phi = math.atan2(m[1][0], m[0][0])
    else:
        phi = math.atan2(m[1][2], m[0][2])
        gamma = math.atan2(m[2][1], -m[2][0])
    return theta, phi, gamma


def local_to_world(point2d: Vec2, theta: float, phi: float, gamma: float,
                   origin: Vec3 = (0.0, 0.0, 0.0), scale: float = 1.0) -> Vec3:
    """Lift a 2D sketch point ``(u, v)`` into world space.

    ``world = origin + scale * (u * x_axis + v * y_axis)`` where the plane axes come
    from :func:`plane_axes`. This is the decode of a curve's in-plane
    coordinates using the extrusion command's plane pose, origin and profile scale.
    """
    x_axis, y_axis, _ = plane_axes(theta, phi, gamma)
    u, v = point2d
    return tuple(
        origin[i] + scale * (u * x_axis[i] + v * y_axis[i]) for i in range(3)
    )


def world_to_local(point3d: Vec3, theta: float, phi: float, gamma: float,
                   origin: Vec3 = (0.0, 0.0, 0.0), scale: float = 1.0) -> Vec3:
    """Project a world point into plane-local ``(u, v, w)`` coordinates.

    ``w`` is the signed distance along the plane normal (0 for in-plane points).
    Inverse of :func:`local_to_world` for ``w = 0``. Requires ``scale != 0``.
    """
    if scale == 0.0:
        raise ValueError("scale must be non-zero")
    x_axis, y_axis, normal = plane_axes(theta, phi, gamma)
    d = tuple(point3d[i] - origin[i] for i in range(3))
    u = sum(d[i] * x_axis[i] for i in range(3)) / scale
    v = sum(d[i] * y_axis[i] for i in range(3)) / scale
    w = sum(d[i] * normal[i] for i in range(3)) / scale
    return (u, v, w)


def extrusion_extents(e1: float, e2: float, extrude_type: int) -> tuple[float, float]:
    """Signed extents along the plane normal for the three extrude types.

    Returns ``(near, far)`` offsets (near <= far) measured along the normal:

      * one-sided  -> ``(0, e1)``            (extrude one way by ``e1``);
      * symmetric  -> ``(-e1, +e1)``         (equal both sides);
      * two-sided  -> ``(-e2, +e1)``         (``e1`` one way, ``e2`` the other).
    """
    if extrude_type == ONE_SIDED:
        lo, hi = 0.0, e1
    elif extrude_type == SYMMETRIC:
        lo, hi = -e1, e1
    elif extrude_type == TWO_SIDED:
        lo, hi = -e2, e1
    else:
        raise ValueError(f"unknown extrude type: {extrude_type!r}")
    return (lo, hi) if lo <= hi else (hi, lo)


def extrude_point(point2d: Vec2, offset: float, theta: float, phi: float,
                  gamma: float, origin: Vec3 = (0.0, 0.0, 0.0),
                  scale: float = 1.0) -> Vec3:
    """Lift a 2D profile point and translate it ``offset`` along the plane normal."""
    base = local_to_world(point2d, theta, phi, gamma, origin, scale)
    _, _, normal = plane_axes(theta, phi, gamma)
    return tuple(base[i] + offset * normal[i] for i in range(3))
