"""Deterministic camera math for GaussianCAD's Camera-Pose-Localization stage.

Section 3.1 / 3.3 of GaussianCAD manually configures the cameras for the three
orthographic views instead of estimating them (SfM/DUSt3R fail on line-only CAD
sketches). Because the relative view geometry is fixed, the poses are computed by
closed-form geometry:

  * Euler angles -> rotation matrix in **ZYX** order  R = Rz(a) Ry(b) Rx(c)
    (Eqs. 2-5) -- distinct from the ZYZ parameterisation in
    ``reconstruction.deepcad_sketch_plane``;
  * a pinhole **intrinsic** matrix K (Eq. 7 / Eq. 8);
  * a 4x4 **extrinsic** matrix [R | t] (Eq. 6);
  * the fixed **front / left / bottom** orthographic poses (Eqs. 9-14) in
    Blender's convention and their conversion to COLMAP's convention (which
    GaussianObject consumes);
  * pinhole projection of a world point to a pixel via K [R | t].

This is the reconstruction-camera geometry GaussianCAD relies on; it is separate
from ``drawings.creft_view_consistency`` (which only *checks* view agreement) and
``drawings.cad2program_view_lifting`` (which lifts prismatic boxes). No learned
model, no wall clock, no randomness.
"""

from __future__ import annotations

from math import cos, radians, sin
from typing import Sequence, Tuple

Vec3 = Tuple[float, float, float]
Mat3 = Tuple[Tuple[float, float, float], ...]
Mat4 = Tuple[Tuple[float, float, float, float], ...]


def _mul3(a: Mat3, b: Mat3) -> Mat3:
    return tuple(
        tuple(sum(a[i][k] * b[k][j] for k in range(3)) for j in range(3))
        for i in range(3)
    )


def rot_x(gamma_deg: float) -> Mat3:
    """Rotation about the x-axis by ``gamma`` degrees (Eq. 5)."""
    c, s = cos(radians(gamma_deg)), sin(radians(gamma_deg))
    return ((1.0, 0.0, 0.0), (0.0, c, -s), (0.0, s, c))


def rot_y(beta_deg: float) -> Mat3:
    """Rotation about the y-axis by ``beta`` degrees (Eq. 4)."""
    c, s = cos(radians(beta_deg)), sin(radians(beta_deg))
    return ((c, 0.0, s), (0.0, 1.0, 0.0), (-s, 0.0, c))


def rot_z(alpha_deg: float) -> Mat3:
    """Rotation about the z-axis by ``alpha`` degrees (Eq. 3)."""
    c, s = cos(radians(alpha_deg)), sin(radians(alpha_deg))
    return ((c, -s, 0.0), (s, c, 0.0), (0.0, 0.0, 1.0))


def euler_zyx_to_matrix(alpha_deg: float, beta_deg: float, gamma_deg: float) -> Mat3:
    """Combined rotation ``R = Rz(alpha) Ry(beta) Rx(gamma)`` (Eq. 2), degrees."""
    return _mul3(_mul3(rot_z(alpha_deg), rot_y(beta_deg)), rot_x(gamma_deg))


def intrinsic_matrix(fx: float, fy: float, cx: float, cy: float) -> Mat3:
    """Pinhole intrinsic matrix K (Eq. 7)."""
    return ((float(fx), 0.0, float(cx)), (0.0, float(fy), float(cy)), (0.0, 0.0, 1.0))


# The paper's intrinsics: 1920x1080, fx=2480, fy=2080, principal point (960, 540).
GAUSSIANCAD_INTRINSIC: Mat3 = intrinsic_matrix(2480.0, 2080.0, 960.0, 540.0)


def extrinsic_matrix(rotation: Mat3, translation: Sequence[float]) -> Mat4:
    """Assemble a 4x4 extrinsic ``[R | t]`` with homogeneous bottom row (Eq. 6)."""
    t = [float(v) for v in translation]
    if len(t) != 3:
        raise ValueError("translation must have three components")
    return (
        (rotation[0][0], rotation[0][1], rotation[0][2], t[0]),
        (rotation[1][0], rotation[1][1], rotation[1][2], t[1]),
        (rotation[2][0], rotation[2][1], rotation[2][2], t[2]),
        (0.0, 0.0, 0.0, 1.0),
    )


# --------------------------------------------------------------------------- #
# Fixed three-orthographic-view poses (Blender convention, Eqs. 9-14)
# --------------------------------------------------------------------------- #
# (view -> (euler ZYX degrees, camera position)). Blender: x right, y down,
# z toward the scene. Front/left/bottom as stated in Sec. 3.3.
THREE_VIEW_EULER = {
    "front": (0.0, 90.0, 90.0),
    "left": (90.0, 0.0, 0.0),
    "bottom": (0.0, 180.0, 90.0),
}
THREE_VIEW_POSITION = {
    "front": (0.0, 0.0, 5.0),
    "left": (0.0, -5.0, 0.0),
    "bottom": (0.0, 0.0, -5.0),
}
THREE_VIEW_NAMES: Tuple[str, ...] = ("front", "left", "bottom")


def three_view_rotation(view: str) -> Mat3:
    """Rotation matrix for one of the fixed orthographic views (Eqs. 9-11)."""
    if view not in THREE_VIEW_EULER:
        raise ValueError("unknown view %r" % (view,))
    a, b, g = THREE_VIEW_EULER[view]
    return euler_zyx_to_matrix(a, b, g)


def three_view_extrinsic(view: str) -> Mat4:
    """Extrinsic ``[R | t]`` for a fixed orthographic view (Eqs. 12-14).

    Follows the paper's construction: the rotation from the ZYX Euler angles and
    the translation set to the documented camera position.
    """
    return extrinsic_matrix(three_view_rotation(view), THREE_VIEW_POSITION[view])


# --------------------------------------------------------------------------- #
# Blender <-> COLMAP convention conversion
# --------------------------------------------------------------------------- #
# GaussianObject works in COLMAP's convention. Blender's camera looks down -Z
# with +Y up; COLMAP's looks down +Z with +Y down. The change of basis flips the
# y and z axes: M = diag(1, -1, -1). It is its own inverse.
_BLENDER_TO_COLMAP: Mat3 = ((1.0, 0.0, 0.0), (0.0, -1.0, 0.0), (0.0, 0.0, -1.0))


def blender_to_colmap(rotation: Mat3, translation: Sequence[float]) -> Tuple[Mat3, Vec3]:
    """Convert a Blender-convention (R, t) pose to COLMAP convention.

    Applies ``M = diag(1, -1, -1)`` on the left: ``R' = M R`` and ``t' = M t``.
    Involutive: applying it twice returns the original pose.
    """
    m = _BLENDER_TO_COLMAP
    r2 = _mul3(m, rotation)
    t = [float(v) for v in translation]
    t2 = tuple(sum(m[i][k] * t[k] for k in range(3)) for i in range(3))
    return r2, t2


# --------------------------------------------------------------------------- #
# Pinhole projection
# --------------------------------------------------------------------------- #
def project_point(point: Sequence[float], intrinsic: Mat3, rotation: Mat3,
                  translation: Sequence[float]) -> Tuple[float, float]:
    """Project a world point to a pixel via ``K [R | t]`` (pinhole).

    Camera-space point is ``Xc = R X + t``; the pixel is ``K Xc`` de-homogenised
    by its third component. Raises if the point is on the camera plane (Xc_z = 0).
    """
    x = [float(v) for v in point]
    t = [float(v) for v in translation]
    xc = tuple(sum(rotation[i][k] * x[k] for k in range(3)) + t[i] for i in range(3))
    proj = tuple(sum(intrinsic[i][k] * xc[k] for k in range(3)) for i in range(3))
    if abs(proj[2]) < 1e-18:
        raise ValueError("point projects onto the camera plane (w=0)")
    return (proj[0] / proj[2], proj[1] / proj[2])
