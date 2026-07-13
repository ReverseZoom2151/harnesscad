"""Joint pose transforms for B-Rep joints (JoinABLe, CVPR 2022).

Given the joint axis of an entity on body one and the joint axis of an entity
on body two, the joint transform is the 4x4 affine matrix that moves body one
so that its axis becomes colinear with body two's axis.  The pose is then
parameterised by three user-visible values, exactly as in the Fusion 360
joint model:

* ``rotation``  -- radians about the joint axis;
* ``offset``    -- translation along the joint axis;
* ``flip``      -- reflect body one through the plane normal to the axis at the
  joint origin (a 180-degree "other side" toggle).

The full transform composes as ``offset_flip @ rotation @ alignment``.  All
matrices are row-major tuples of 4 rows of 4 floats.  Stdlib only.
"""

import math

from harnesscad.domain.geometry.kinematics.joint_axis import (
    as_vec3,
    cross,
    dot,
    norm,
    normalize,
    vec_add,
    vec_scale,
    vec_sub,
)

__all__ = [
    "identity_matrix",
    "matmul",
    "transform_point",
    "transform_vector",
    "align_vectors",
    "rotation_matrix_about_axis",
    "joint_alignment_matrix",
    "rotation_parameter_matrix",
    "offset_parameter_matrix",
    "joint_transform_from_parameters",
    "apply_joint_transform_to_axis",
]


def identity_matrix():
    return (
        (1.0, 0.0, 0.0, 0.0),
        (0.0, 1.0, 0.0, 0.0),
        (0.0, 0.0, 1.0, 0.0),
        (0.0, 0.0, 0.0, 1.0),
    )


def matmul(a, b):
    """4x4 * 4x4 matrix product."""
    return tuple(
        tuple(sum(a[i][k] * b[k][j] for k in range(4)) for j in range(4))
        for i in range(4)
    )


def transform_point(matrix, point):
    """Apply a 4x4 affine matrix to a point (w = 1)."""
    p = as_vec3(point)
    return tuple(
        matrix[i][0] * p[0] + matrix[i][1] * p[1] + matrix[i][2] * p[2]
        + matrix[i][3]
        for i in range(3)
    )


def transform_vector(matrix, vector):
    """Apply the rotational part of a 4x4 matrix to a direction (w = 0)."""
    v = as_vec3(vector)
    return tuple(
        matrix[i][0] * v[0] + matrix[i][1] * v[1] + matrix[i][2] * v[2]
        for i in range(3)
    )


def _perpendicular(v):
    """Any unit vector perpendicular to ``v`` (deterministic choice)."""
    if abs(v[0]) <= abs(v[1]) and abs(v[0]) <= abs(v[2]):
        other = (1.0, 0.0, 0.0)
    elif abs(v[1]) <= abs(v[2]):
        other = (0.0, 1.0, 0.0)
    else:
        other = (0.0, 0.0, 1.0)
    return normalize(cross(v, other))


def rotation_matrix_about_axis(angle_radians, axis):
    """3x3 rotation (Rodrigues) of ``angle_radians`` about a unit ``axis``."""
    x, y, z = normalize(as_vec3(axis))
    c = math.cos(angle_radians)
    s = math.sin(angle_radians)
    t = 1.0 - c
    return (
        (t * x * x + c, t * x * y - s * z, t * x * z + s * y),
        (t * x * y + s * z, t * y * y + c, t * y * z - s * x),
        (t * x * z - s * y, t * y * z + s * x, t * z * z + c),
    )


def align_vectors(a, b):
    """3x3 rotation matrix ``R`` with ``R @ a`` parallel to ``b``.

    Rotates about the common perpendicular by the angle between the two
    vectors; the antiparallel case rotates 180 degrees about an arbitrary but
    deterministic perpendicular axis.
    """
    a = normalize(as_vec3(a))
    b = normalize(as_vec3(b))
    if norm(a) == 0.0 or norm(b) == 0.0:
        raise ValueError("cannot align a zero-length vector")
    axis = cross(a, b)
    axis_len = norm(axis)
    c = max(-1.0, min(1.0, dot(a, b)))
    if axis_len < 1e-12:
        if c > 0.0:
            return (
                (1.0, 0.0, 0.0),
                (0.0, 1.0, 0.0),
                (0.0, 0.0, 1.0),
            )
        return rotation_matrix_about_axis(math.pi, _perpendicular(a))
    return rotation_matrix_about_axis(math.acos(c), normalize(axis))


def _affine(rot3, translation):
    return (
        (rot3[0][0], rot3[0][1], rot3[0][2], translation[0]),
        (rot3[1][0], rot3[1][1], rot3[1][2], translation[1]),
        (rot3[2][0], rot3[2][1], rot3[2][2], translation[2]),
        (0.0, 0.0, 0.0, 1.0),
    )


def _apply3(rot3, v):
    return tuple(
        rot3[i][0] * v[0] + rot3[i][1] * v[1] + rot3[i][2] * v[2]
        for i in range(3)
    )


def joint_alignment_matrix(origin1, direction1, origin2, direction2):
    """4x4 matrix aligning body one's joint axis onto body two's joint axis.

    Rotates ``direction1`` onto ``direction2`` about ``origin1`` and then
    translates ``origin1`` onto ``origin2``.
    """
    origin1 = as_vec3(origin1)
    origin2 = as_vec3(origin2)
    rot3 = align_vectors(direction1, direction2)
    # Rotate about origin1, then move origin1 to origin2.
    translation = vec_add(vec_sub(origin1, _apply3(rot3, origin1)),
                          vec_sub(origin2, origin1))
    return _affine(rot3, translation)


def rotation_parameter_matrix(rotation_radians, origin, direction):
    """4x4 rotation of ``rotation_radians`` about the joint axis."""
    origin = as_vec3(origin)
    rot3 = rotation_matrix_about_axis(rotation_radians, direction)
    translation = vec_sub(origin, _apply3(rot3, origin))
    return _affine(rot3, translation)


def offset_parameter_matrix(offset, origin, direction, flip=False):
    """4x4 matrix applying the ``offset`` along the axis and optional ``flip``.

    The flip is a reflection through the plane that passes through ``origin``
    with normal ``direction``.  With ``flip=False`` the rotational part is the
    identity and the matrix is a pure translation.
    """
    origin = as_vec3(origin)
    normal = normalize(as_vec3(direction))
    translation = vec_scale(normal, float(offset))
    if not flip:
        return _affine(((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)),
                       translation)
    # Householder reflection I - 2 n n^T
    rot3 = tuple(
        tuple((1.0 if i == j else 0.0) - 2.0 * normal[i] * normal[j]
              for j in range(3))
        for i in range(3)
    )
    translation = vec_add(translation,
                          vec_scale(normal, 2.0 * dot(origin, normal)))
    return _affine(rot3, translation)


def joint_transform_from_parameters(origin1, direction1, origin2, direction2,
                                    offset=0.0, rotation=0.0, flip=False):
    """Full joint transform for body one: offset/flip @ rotation @ alignment."""
    align = joint_alignment_matrix(origin1, direction1, origin2, direction2)
    rot = rotation_parameter_matrix(rotation, origin2, direction2)
    off = offset_parameter_matrix(offset, origin2, direction2, flip)
    return matmul(off, matmul(rot, align))


def apply_joint_transform_to_axis(matrix, axis_line):
    """Transform an ``(origin, direction)`` axis line by a 4x4 matrix."""
    origin, direction = axis_line
    return (transform_point(matrix, origin),
            normalize(transform_vector(matrix, direction)))
