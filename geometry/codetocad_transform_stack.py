"""Backend-free transform stack: translate / rotate / scale / mirror as 4x4 matrices.

CodeToCAD exposes ``translate``, ``rotate``, ``rotate_around_axis``, ``scale``,
``scale_uniform`` and ``mirror`` as interface functions whose implementation is
delegated to Blender / build123d.  The *maths* behind them is backend-free, so it
is reimplemented here as a small, dependency-free 4x4 column-vector matrix library
that speaks the same unit-expression language as the rest of the CodeToCAD layer
("10mm", "45deg").

Conventions
-----------
* Row-major nested tuples, column-vector convention: ``p' = M @ p``.
* ``rotation_euler`` applies X, then Y, then Z (i.e. ``M = Rz @ Ry @ Rx``), matching
  CodeToCAD's documented order.
* ``rotation_around_axis`` uses Rodrigues' formula around an arbitrary line, so it
  can rotate about an edge that does not pass through the origin.
* ``mirror`` reflects across the XY / XZ / YZ planes (optionally through a point).
* :func:`compose` multiplies left-to-right in *application* order:
  ``compose(A, B)`` applies ``A`` first, then ``B``.

Everything is deterministic and exact up to IEEE-754 rounding.
"""

from __future__ import annotations

import math

from geometry.codetocad_cardinal_landmark import BoundaryBox
from numeric.codetocad_length_expression import parse_angle, parse_length

__all__ = [
    "IDENTITY",
    "TransformError",
    "translation",
    "scaling",
    "rotation_x",
    "rotation_y",
    "rotation_z",
    "rotation_euler",
    "rotation_around_axis",
    "mirror",
    "matmul",
    "compose",
    "apply_point",
    "apply_direction",
    "apply_points",
    "transform_box",
    "invert_rigid",
    "is_close",
]

IDENTITY = (
    (1.0, 0.0, 0.0, 0.0),
    (0.0, 1.0, 0.0, 0.0),
    (0.0, 0.0, 1.0, 0.0),
    (0.0, 0.0, 0.0, 1.0),
)

PLANES = ("xy", "xz", "yz")


class TransformError(ValueError):
    """Raised for degenerate axes, unknown planes or non-invertible transforms."""


def translation(x=0, y=0, z=0):
    """Translation matrix; components may be unit expressions."""
    dx, dy, dz = parse_length(x), parse_length(y), parse_length(z)
    return (
        (1.0, 0.0, 0.0, dx),
        (0.0, 1.0, 0.0, dy),
        (0.0, 0.0, 1.0, dz),
        (0.0, 0.0, 0.0, 1.0),
    )


def scaling(x=1.0, y=1.0, z=1.0):
    """Non-uniform scale matrix (plain factors, not lengths)."""
    sx, sy, sz = float(x), float(y), float(z)
    return (
        (sx, 0.0, 0.0, 0.0),
        (0.0, sy, 0.0, 0.0),
        (0.0, 0.0, sz, 0.0),
        (0.0, 0.0, 0.0, 1.0),
    )


def rotation_x(angle):
    a = parse_angle(angle)
    c, s = math.cos(a), math.sin(a)
    return (
        (1.0, 0.0, 0.0, 0.0),
        (0.0, c, -s, 0.0),
        (0.0, s, c, 0.0),
        (0.0, 0.0, 0.0, 1.0),
    )


def rotation_y(angle):
    a = parse_angle(angle)
    c, s = math.cos(a), math.sin(a)
    return (
        (c, 0.0, s, 0.0),
        (0.0, 1.0, 0.0, 0.0),
        (-s, 0.0, c, 0.0),
        (0.0, 0.0, 0.0, 1.0),
    )


def rotation_z(angle):
    a = parse_angle(angle)
    c, s = math.cos(a), math.sin(a)
    return (
        (c, -s, 0.0, 0.0),
        (s, c, 0.0, 0.0),
        (0.0, 0.0, 1.0, 0.0),
        (0.0, 0.0, 0.0, 1.0),
    )


def rotation_euler(x=0, y=0, z=0):
    """Rotate about X, then Y, then Z (CodeToCAD's documented order)."""
    return matmul(rotation_z(z), matmul(rotation_y(y), rotation_x(x)))


def rotation_around_axis(point, direction, angle):
    """Rodrigues rotation about the line through ``point`` along ``direction``."""
    ax, ay, az = (parse_length(component) for component in point)
    dx, dy, dz = (float(component) for component in direction)
    norm = math.sqrt(dx * dx + dy * dy + dz * dz)
    if norm == 0.0:
        raise TransformError("rotation axis direction must be non-zero")
    ux, uy, uz = dx / norm, dy / norm, dz / norm
    a = parse_angle(angle)
    c, s = math.cos(a), math.sin(a)
    t = 1.0 - c
    rotation = (
        (t * ux * ux + c, t * ux * uy - s * uz, t * ux * uz + s * uy, 0.0),
        (t * ux * uy + s * uz, t * uy * uy + c, t * uy * uz - s * ux, 0.0),
        (t * ux * uz - s * uy, t * uy * uz + s * ux, t * uz * uz + c, 0.0),
        (0.0, 0.0, 0.0, 1.0),
    )
    to_origin = translation(-ax, -ay, -az)
    back = translation(ax, ay, az)
    return matmul(back, matmul(rotation, to_origin))


def mirror(plane: str, through=(0.0, 0.0, 0.0)):
    """Reflect across ``"xy"`` / ``"xz"`` / ``"yz"``, optionally through a point."""
    key = str(plane).lower()
    if key not in PLANES:
        raise TransformError("unknown plane: " + str(plane))
    factors = {
        "xy": (1.0, 1.0, -1.0),
        "xz": (1.0, -1.0, 1.0),
        "yz": (-1.0, 1.0, 1.0),
    }[key]
    px, py, pz = (parse_length(component) for component in through)
    return matmul(
        translation(px, py, pz),
        matmul(scaling(*factors), translation(-px, -py, -pz)),
    )


def matmul(a, b):
    """Matrix product ``a @ b`` (apply ``b`` first, then ``a``)."""
    return tuple(
        tuple(sum(a[r][k] * b[k][col] for k in range(4)) for col in range(4))
        for r in range(4)
    )


def compose(*matrices):
    """Compose in application order: ``compose(A, B)`` applies A, then B."""
    result = IDENTITY
    for matrix in matrices:
        result = matmul(matrix, result)
    return result


def apply_point(matrix, point):
    """Apply to a position (translation included)."""
    x, y, z = (float(component) for component in point)
    out = []
    for row in matrix[:3]:
        out.append(row[0] * x + row[1] * y + row[2] * z + row[3])
    w = matrix[3][0] * x + matrix[3][1] * y + matrix[3][2] * z + matrix[3][3]
    if w == 0.0:
        raise TransformError("degenerate homogeneous coordinate")
    return tuple(component / w for component in out)


def apply_direction(matrix, vector):
    """Apply to a direction (translation ignored)."""
    x, y, z = (float(component) for component in vector)
    return tuple(
        row[0] * x + row[1] * y + row[2] * z for row in matrix[:3]
    )


def apply_points(matrix, points):
    return [apply_point(matrix, point) for point in points]


def transform_box(matrix, box: BoundaryBox) -> BoundaryBox:
    """Transform a box's 8 corners and re-fit an axis-aligned bounding box."""
    corners = [
        (x, y, z)
        for x in (box.x.min, box.x.max)
        for y in (box.y.min, box.y.max)
        for z in (box.z.min, box.z.max)
    ]
    return BoundaryBox.from_points(apply_points(matrix, corners))


def invert_rigid(matrix):
    """Invert a rigid transform (rotation + translation only)."""
    rotation = [row[:3] for row in matrix[:3]]
    # Orthonormality check: R @ R^T must be the identity.
    for i in range(3):
        for j in range(3):
            dot = sum(rotation[i][k] * rotation[j][k] for k in range(3))
            expected = 1.0 if i == j else 0.0
            if abs(dot - expected) > 1e-9:
                raise TransformError("matrix is not a rigid transform")
    t = [matrix[i][3] for i in range(3)]
    inverse_translation = [
        -sum(rotation[k][i] * t[k] for k in range(3)) for i in range(3)
    ]
    return tuple(
        tuple(
            [rotation[0][i], rotation[1][i], rotation[2][i], inverse_translation[i]]
        )
        for i in range(3)
    ) + ((0.0, 0.0, 0.0, 1.0),)


def is_close(a, b, tolerance: float = 1e-9) -> bool:
    return all(
        abs(a[r][c] - b[r][c]) <= tolerance for r in range(4) for c in range(4)
    )
