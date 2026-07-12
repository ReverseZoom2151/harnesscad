"""Graph-CAD orientation and rotation directives -> rotation matrices.

Graph-CAD creates every primitive in its *native pose*, where the local +Z axis
is "straight up" (a cylinder's axis, a cuboid's top-face normal). The node's
``orientation=`` field then says where that local +Z must point after rotation,
and an optional ``rotation=`` field adds an explicit tilt about a *current
local* axis. The format's own rule is: apply ``orientation=`` first, then
``rotation=``.

Supported directives (from the format specification)::

    orientation = axis:+X | axis:-Z | ...        map local +Z to a world axis
    orientation = axis:radial_from <obj>         local +Z points away from obj
    orientation = axis:tangent_to <obj>          local +Z tangent to obj's surface
    orientation = normal:<obj>                   cutter normal faces the target
    rotation    = axis:X,30                      tilt about the current local X
    rotation    = tilt_then_spin(tilt=X,15, spin=Z,120)

The ``axis:+A`` mapping uses the minimal (geodesic) rotation carrying +Z onto
the requested direction, with the degenerate -Z case resolved as a half turn
about +X so results stay deterministic. ``tilt_then_spin`` -- the tripod-leg
idiom -- tilts away from vertical and then spins the tilted part around the
spin axis, i.e. ``R = R_spin * R_tilt``.

Matrices are plain 3x3 row-major tuples; no third-party linear algebra.
"""

from __future__ import annotations

import math
import re
from typing import Sequence, Tuple

__all__ = [
    "Matrix3",
    "Vec3",
    "IDENTITY",
    "normalize",
    "axis_direction",
    "rotation_about",
    "align_z_to",
    "compose",
    "apply",
    "is_rotation",
    "parse_orientation",
    "parse_rotation",
    "resolve_orientation",
    "resolve_rotation",
    "node_rotation",
    "tilt_then_spin",
    "radial_from",
]

Vec3 = Tuple[float, float, float]
Matrix3 = Tuple[Vec3, Vec3, Vec3]

IDENTITY: Matrix3 = ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0))

_WORLD_AXES = {
    "+X": (1.0, 0.0, 0.0),
    "-X": (-1.0, 0.0, 0.0),
    "+Y": (0.0, 1.0, 0.0),
    "-Y": (0.0, -1.0, 0.0),
    "+Z": (0.0, 0.0, 1.0),
    "-Z": (0.0, 0.0, -1.0),
}

_AXIS_DIRECTIVE = re.compile(
    r"axis\s*:\s*(?P<sign>[+\-–]?)\s*(?P<axis>[XYZxyz])\s*$"
)
_AXIS_RELATION = re.compile(
    r"axis\s*:\s*(?P<relation>radial_from|tangent_to)\s+(?P<obj>[A-Za-z_][\w]*)",
    re.IGNORECASE,
)
_NORMAL = re.compile(r"normal\s*:\s*(?P<obj>[A-Za-z_][\w]*)", re.IGNORECASE)
_ROTATION = re.compile(
    r"axis\s*:\s*(?P<axis>[XYZxyz])\s*,\s*(?P<angle>[-+0-9.eE]+)\s*(?:deg|°)?"
)
_TILT_SPIN = re.compile(
    r"tilt_then_spin\s*\(\s*tilt\s*=\s*(?P<taxis>[XYZxyz])\s*,\s*"
    r"(?P<tangle>[-+0-9.eE]+)\s*(?:deg|°)?\s*,\s*spin\s*=\s*"
    r"(?P<saxis>[XYZxyz])\s*,\s*(?P<sangle>[-+0-9.eE]+)\s*(?:deg|°)?\s*\)",
    re.IGNORECASE,
)


def normalize(vector: Sequence[float]) -> Vec3:
    length = math.sqrt(sum(component * component for component in vector))
    if length == 0.0:
        raise ValueError("cannot normalize the zero vector")
    return tuple(component / length for component in vector)  # type: ignore[return-value]


def axis_direction(token: str) -> Vec3:
    """Resolve an ``+X`` / ``-Z`` style token (the en dash used in the spec too)."""
    cleaned = token.strip().replace("–", "-").upper()
    if len(cleaned) == 1:
        cleaned = "+" + cleaned
    if cleaned not in _WORLD_AXES:
        raise ValueError(f"unknown axis token: {token!r}")
    return _WORLD_AXES[cleaned]


def rotation_about(axis: Sequence[float], degrees: float) -> Matrix3:
    """Right-handed rotation of ``degrees`` about ``axis`` (Rodrigues' formula)."""
    x, y, z = normalize(axis)
    angle = math.radians(degrees)
    cos = math.cos(angle)
    sin = math.sin(angle)
    one = 1.0 - cos
    return (
        (cos + x * x * one, x * y * one - z * sin, x * z * one + y * sin),
        (y * x * one + z * sin, cos + y * y * one, y * z * one - x * sin),
        (z * x * one - y * sin, z * y * one + x * sin, cos + z * z * one),
    )


def align_z_to(direction: Sequence[float]) -> Matrix3:
    """Minimal rotation carrying local +Z onto ``direction``.

    The antipodal case (+Z -> -Z) has no unique minimal rotation, so it is
    fixed deterministically as a half turn about +X.
    """
    target = normalize(direction)
    dot = target[2]
    if math.isclose(dot, 1.0, abs_tol=1e-12):
        return IDENTITY
    if math.isclose(dot, -1.0, abs_tol=1e-12):
        return rotation_about((1.0, 0.0, 0.0), 180.0)
    axis = (-target[1], target[0], 0.0)  # cross((0,0,1), target)
    angle = math.degrees(math.acos(max(-1.0, min(1.0, dot))))
    return rotation_about(axis, angle)


def compose(first: Matrix3, second: Matrix3) -> Matrix3:
    """Matrix product ``first * second`` (``second`` is applied to a vector first)."""
    return tuple(  # type: ignore[return-value]
        tuple(
            sum(first[row][k] * second[k][col] for k in range(3)) for col in range(3)
        )
        for row in range(3)
    )


def apply(matrix: Matrix3, vector: Sequence[float]) -> Vec3:
    return tuple(  # type: ignore[return-value]
        sum(matrix[row][col] * vector[col] for col in range(3)) for row in range(3)
    )


def is_rotation(matrix: Matrix3, tolerance: float = 1e-9) -> bool:
    """True if the matrix is orthonormal with determinant +1."""
    for i in range(3):
        for j in range(3):
            dot = sum(matrix[k][i] * matrix[k][j] for k in range(3))
            expected = 1.0 if i == j else 0.0
            if abs(dot - expected) > tolerance:
                return False
    determinant = (
        matrix[0][0] * (matrix[1][1] * matrix[2][2] - matrix[1][2] * matrix[2][1])
        - matrix[0][1] * (matrix[1][0] * matrix[2][2] - matrix[1][2] * matrix[2][0])
        + matrix[0][2] * (matrix[1][0] * matrix[2][1] - matrix[1][1] * matrix[2][0])
    )
    return abs(determinant - 1.0) <= tolerance


def radial_from(center: Sequence[float], other: Sequence[float]) -> Vec3:
    """Direction pointing from ``other``'s origin towards ``center``."""
    delta = tuple(center[axis] - other[axis] for axis in range(3))
    return normalize(delta)


def parse_orientation(text: str) -> Tuple[str, object]:
    """Classify an ``orientation=`` directive.

    Returns ``("axis", direction)`` for a fixed world axis, or
    ``("radial_from" | "tangent_to" | "normal", reference_id)`` for the
    reference-relative forms, which need the scene to be resolved.
    """
    body = text.split("=", 1)[1].strip() if "=" in text else text.strip()

    relation = _AXIS_RELATION.search(body)
    if relation:
        return relation.group("relation").lower(), relation.group("obj")

    normal = _NORMAL.search(body)
    if normal:
        return "normal", normal.group("obj")

    axis = _AXIS_DIRECTIVE.search(body.replace(" ", ""))
    if axis:
        sign = axis.group("sign").replace("–", "-") or "+"
        return "axis", axis_direction(sign + axis.group("axis"))

    raise ValueError(f"unsupported orientation directive: {text!r}")


def parse_rotation(text: str) -> Tuple[str, object]:
    """Classify a ``rotation=`` directive into ``axis_angle`` or ``tilt_then_spin``."""
    body = text.split("=", 1)[1].strip() if "=" in text else text.strip()

    tilt = _TILT_SPIN.search(body)
    if tilt:
        return "tilt_then_spin", (
            tilt.group("taxis").upper(),
            float(tilt.group("tangle")),
            tilt.group("saxis").upper(),
            float(tilt.group("sangle")),
        )

    simple = _ROTATION.search(body)
    if simple:
        return "axis_angle", (simple.group("axis").upper(), float(simple.group("angle")))

    raise ValueError(f"unsupported rotation directive: {text!r}")


def tilt_then_spin(
    tilt_axis: str,
    tilt_degrees: float,
    spin_axis: str,
    spin_degrees: float,
) -> Matrix3:
    """``R = R_spin * R_tilt``: tilt away from vertical, then spin about the spin axis."""
    tilt = rotation_about(axis_direction(tilt_axis), tilt_degrees)
    spin = rotation_about(axis_direction(spin_axis), spin_degrees)
    return compose(spin, tilt)


def resolve_orientation(
    text: str,
    center: Sequence[float] | None = None,
    reference_center: Sequence[float] | None = None,
) -> Matrix3:
    """Resolve an ``orientation=`` directive to a rotation matrix.

    The reference-relative forms need the node's own centre and the referenced
    object's centre. ``radial_from`` points local +Z away from the reference;
    ``normal:`` points it *at* the reference (a cutter faces its target);
    ``tangent_to`` points it along the tangent of the reference's surface at
    that bearing, i.e. 90 degrees around from the radial direction in XY.
    """
    kind, payload = parse_orientation(text)
    if kind == "axis":
        return align_z_to(payload)  # type: ignore[arg-type]
    if center is None or reference_center is None:
        raise ValueError(f"directive {kind!r} needs center and reference_center")

    radial = radial_from(center, reference_center)
    if kind == "radial_from":
        return align_z_to(radial)
    if kind == "normal":
        return align_z_to(tuple(-component for component in radial))
    if kind == "tangent_to":
        tangent = (-radial[1], radial[0], 0.0)
        if math.isclose(tangent[0], 0.0, abs_tol=1e-12) and math.isclose(
            tangent[1], 0.0, abs_tol=1e-12
        ):
            raise ValueError("tangent is undefined directly above the reference")
        return align_z_to(tangent)
    raise ValueError(f"unsupported orientation kind: {kind!r}")


def resolve_rotation(text: str) -> Matrix3:
    """Resolve a ``rotation=`` directive to a rotation matrix."""
    kind, payload = parse_rotation(text)
    if kind == "axis_angle":
        axis, degrees = payload  # type: ignore[misc]
        return rotation_about(axis_direction(axis), degrees)
    tilt_axis, tilt_degrees, spin_axis, spin_degrees = payload  # type: ignore[misc]
    return tilt_then_spin(tilt_axis, tilt_degrees, spin_axis, spin_degrees)


def node_rotation(
    orientation: str | None = None,
    rotation: str | None = None,
    center: Sequence[float] | None = None,
    reference_center: Sequence[float] | None = None,
) -> Matrix3:
    """Full node pose: orientation first, then the local rotation.

    ``rotation=`` is measured about the object's *current local* axis, so it is
    post-multiplied: ``R = R_orientation * R_rotation``. A node with neither
    field keeps its native pose.
    """
    base = (
        resolve_orientation(orientation, center, reference_center)
        if orientation
        else IDENTITY
    )
    if not rotation:
        return base
    return compose(base, resolve_rotation(rotation))
