"""Deterministic URDF forward kinematics on row-major 4x4 rigid transforms.

Ported from the ``cadjs`` package of the ``text-to-cad`` (CAD Skills) repository,
whose ``src/lib/urdf/kinematics.js`` solves world transforms for every link of a
robot description so the viewer can pose an articulated assembly.  The harness
had no URDF/robot-description kinematics of any kind, so the whole chain is new.

The model reproduced here is the URDF joint convention:

* every joint carries a static ``origin`` transform (parent link frame -> joint
  frame), expressed as a translation composed with a fixed-axis roll/pitch/yaw
  rotation, in that order: ``T = translate(xyz) @ rotate_rpy(rpy)``;
* the articulated motion is applied *after* the origin, about/along the joint
  ``axis`` expressed in the joint frame: a Rodrigues axis-angle rotation for
  ``revolute``/``continuous`` joints and a translation along the axis for
  ``prismatic`` joints;
* ``fixed`` joints contribute only the static transform;
* joint values are carried in *degrees* for angular joints and in native linear
  units for prismatic joints (the convention the source viewer uses for its
  slider UI), and are clamped to the joint limits -- except ``continuous``
  joints, which are unbounded;
* ``mimic`` joints derive their value from a master joint as
  ``multiplier * master + offset``, evaluated in native units (radians for
  angular joints) and converted back, with cycle protection.

World transforms are obtained by a depth-first walk from the single root link.
Everything is pure arithmetic on tuples of floats: no numpy, no scene graph, no
randomness, and identical inputs always produce identical outputs.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, Iterable, Mapping, Optional, Sequence, Tuple

Transform = Tuple[float, ...]
Vector3 = Tuple[float, float, float]

IDENTITY_TRANSFORM: Transform = (
    1.0, 0.0, 0.0, 0.0,
    0.0, 1.0, 0.0, 0.0,
    0.0, 0.0, 1.0, 0.0,
    0.0, 0.0, 0.0, 1.0,
)

ANGULAR_JOINT_TYPES = frozenset({"revolute", "continuous"})
JOINT_TYPES = frozenset({"fixed", "revolute", "continuous", "prismatic"})

_AXIS_EPSILON = 1e-9


class UrdfKinematicsError(ValueError):
    """Raised when a kinematic query is structurally impossible."""


@dataclass(frozen=True)
class JointMimic:
    """A URDF ``<mimic>`` declaration."""

    joint: str
    multiplier: float = 1.0
    offset: float = 0.0


@dataclass(frozen=True)
class Joint:
    """A single URDF joint in the form required by the solver."""

    name: str
    type: str
    parent_link: str
    child_link: str
    origin_transform: Transform = IDENTITY_TRANSFORM
    axis: Vector3 = (1.0, 0.0, 0.0)
    default_value_deg: float = 0.0
    min_value_deg: float = 0.0
    max_value_deg: float = 0.0
    mimic: Optional[JointMimic] = None

    def is_angular(self) -> bool:
        return self.type in ANGULAR_JOINT_TYPES


@dataclass(frozen=True)
class RobotModel:
    """A rooted joint tree ready for forward kinematics."""

    root_link: str
    joints: Tuple[Joint, ...] = ()
    link_names: Tuple[str, ...] = ()
    root_world_transform: Transform = IDENTITY_TRANSFORM
    joints_by_name: Dict[str, Joint] = field(default_factory=dict, compare=False)

    def __post_init__(self) -> None:
        if not self.joints_by_name:
            object.__setattr__(
                self, "joints_by_name", {joint.name: joint for joint in self.joints}
            )

    def movable_joints(self) -> Tuple[Joint, ...]:
        return tuple(
            joint
            for joint in self.joints
            if joint.type != "fixed" and joint.mimic is None
        )


def _as_float(value: object, fallback: float = 0.0) -> float:
    try:
        number = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return fallback
    if math.isnan(number) or math.isinf(number):
        return fallback
    return number


def normalize_vector(vector: Sequence[float], fallback: Vector3 = (0.0, 0.0, 1.0)) -> Vector3:
    """Return the unit vector of ``vector``; degenerate input yields ``fallback``."""
    values = tuple(_as_float(component) for component in tuple(vector)[:3])
    if len(values) < 3:
        return fallback
    length = math.sqrt(values[0] ** 2 + values[1] ** 2 + values[2] ** 2)
    if length <= _AXIS_EPSILON:
        return fallback
    return (values[0] / length, values[1] / length, values[2] / length)


def multiply_transforms(left: Sequence[float], right: Sequence[float]) -> Transform:
    """Row-major 4x4 matrix product ``left @ right``."""
    a = tuple(_as_float(value) for value in left)
    b = tuple(_as_float(value) for value in right)
    if len(a) != 16 or len(b) != 16:
        raise UrdfKinematicsError("transforms must hold exactly 16 components")
    product = [0.0] * 16
    for row in range(4):
        for column in range(4):
            total = 0.0
            for offset in range(4):
                total += a[row * 4 + offset] * b[offset * 4 + column]
            product[row * 4 + column] = total
    return tuple(product)


def translation_transform(x: float, y: float, z: float) -> Transform:
    return (
        1.0, 0.0, 0.0, _as_float(x),
        0.0, 1.0, 0.0, _as_float(y),
        0.0, 0.0, 1.0, _as_float(z),
        0.0, 0.0, 0.0, 1.0,
    )


def rotation_transform_from_rpy(roll: float, pitch: float, yaw: float) -> Transform:
    """Fixed-axis XYZ (roll-pitch-yaw) rotation, in radians -- the URDF convention."""
    sr, cr = math.sin(_as_float(roll)), math.cos(_as_float(roll))
    sp, cp = math.sin(_as_float(pitch)), math.cos(_as_float(pitch))
    sy, cy = math.sin(_as_float(yaw)), math.cos(_as_float(yaw))
    return (
        cy * cp, (cy * sp * sr) - (sy * cr), (cy * sp * cr) + (sy * sr), 0.0,
        sy * cp, (sy * sp * sr) + (cy * cr), (sy * sp * cr) - (cy * sr), 0.0,
        -sp, cp * sr, cp * cr, 0.0,
        0.0, 0.0, 0.0, 1.0,
    )


def pose_transform_from_xyz_rpy(values: Sequence[float]) -> Transform:
    """Build ``translate(xyz) @ rotate_rpy(rpy)`` from a 6-tuple."""
    pose = list(values) + [0.0] * 6
    return multiply_transforms(
        translation_transform(pose[0], pose[1], pose[2]),
        rotation_transform_from_rpy(pose[3], pose[4], pose[5]),
    )


def axis_angle_transform(axis: Sequence[float], angle_rad: float) -> Transform:
    """Rodrigues rotation of ``angle_rad`` about the (normalised) ``axis``."""
    x, y, z = normalize_vector(axis)
    angle = _as_float(angle_rad)
    c = math.cos(angle)
    s = math.sin(angle)
    k = 1.0 - c
    return (
        c + x * x * k, (x * y * k) - (z * s), (x * z * k) + (y * s), 0.0,
        (y * x * k) + (z * s), c + y * y * k, (y * z * k) - (x * s), 0.0,
        (z * x * k) - (y * s), (z * y * k) + (x * s), c + z * z * k, 0.0,
        0.0, 0.0, 0.0, 1.0,
    )


def translation_along_axis_transform(axis: Sequence[float], distance: float) -> Transform:
    x, y, z = normalize_vector(axis)
    d = _as_float(distance)
    return translation_transform(x * d, y * d, z * d)


def transform_point(transform: Sequence[float], point: Sequence[float]) -> Vector3:
    m = tuple(_as_float(value) for value in transform)
    if len(m) != 16:
        raise UrdfKinematicsError("transforms must hold exactly 16 components")
    coords = tuple(_as_float(value) for value in tuple(point)[:3])
    if len(coords) < 3:
        raise UrdfKinematicsError("points must hold three components")
    x, y, z = coords
    return (
        m[0] * x + m[1] * y + m[2] * z + m[3],
        m[4] * x + m[5] * y + m[6] * z + m[7],
        m[8] * x + m[9] * y + m[10] * z + m[11],
    )


def invert_rigid_transform(transform: Sequence[float]) -> Transform:
    """Inverse of a rotation+translation matrix: ``[R^T | -R^T t]``."""
    m = tuple(_as_float(value) for value in transform)
    if len(m) != 16:
        raise UrdfKinematicsError("transforms must hold exactly 16 components")
    return (
        m[0], m[4], m[8], -(m[0] * m[3] + m[4] * m[7] + m[8] * m[11]),
        m[1], m[5], m[9], -(m[1] * m[3] + m[5] * m[7] + m[9] * m[11]),
        m[2], m[6], m[10], -(m[2] * m[3] + m[6] * m[7] + m[10] * m[11]),
        0.0, 0.0, 0.0, 1.0,
    )


def transform_bounds(
    bounds: Tuple[Vector3, Vector3], transform: Sequence[float]
) -> Tuple[Vector3, Vector3]:
    """Axis-aligned bounds of the eight transformed corners of ``bounds``."""
    minimum, maximum = bounds
    corners = [
        (minimum[0], minimum[1], minimum[2]),
        (minimum[0], minimum[1], maximum[2]),
        (minimum[0], maximum[1], minimum[2]),
        (minimum[0], maximum[1], maximum[2]),
        (maximum[0], minimum[1], minimum[2]),
        (maximum[0], minimum[1], maximum[2]),
        (maximum[0], maximum[1], minimum[2]),
        (maximum[0], maximum[1], maximum[2]),
    ]
    moved = [transform_point(transform, corner) for corner in corners]
    return (
        (
            min(point[0] for point in moved),
            min(point[1] for point in moved),
            min(point[2] for point in moved),
        ),
        (
            max(point[0] for point in moved),
            max(point[1] for point in moved),
            max(point[2] for point in moved),
        ),
    )


def merge_bounds(
    bounds_list: Iterable[Optional[Tuple[Vector3, Vector3]]]
) -> Tuple[Vector3, Vector3]:
    """Union of a list of axis-aligned bounds; empty input yields a zero box."""
    entries = [entry for entry in bounds_list if entry is not None]
    if not entries:
        return ((0.0, 0.0, 0.0), (0.0, 0.0, 0.0))
    return (
        (
            min(entry[0][0] for entry in entries),
            min(entry[0][1] for entry in entries),
            min(entry[0][2] for entry in entries),
        ),
        (
            max(entry[1][0] for entry in entries),
            max(entry[1][1] for entry in entries),
            max(entry[1][2] for entry in entries),
        ),
    )


def clamp_joint_value_deg(joint: Joint, value_deg: Optional[float]) -> float:
    """Clamp a joint value to its limits.

    ``fixed`` joints ignore the request entirely; ``continuous`` joints are
    unbounded; every other type is clamped into ``[min, max]`` (with the bounds
    themselves ordered defensively).
    """
    if joint.type == "fixed":
        return _as_float(joint.default_value_deg)
    numeric = _as_float(value_deg, _as_float(joint.default_value_deg))
    if joint.type == "continuous":
        return numeric
    minimum = _as_float(joint.min_value_deg, numeric)
    maximum = _as_float(joint.max_value_deg, numeric)
    return min(max(numeric, minimum), max(minimum, maximum))


def _joint_value_to_native(joint: Joint, value_deg: float) -> float:
    clamped = clamp_joint_value_deg(joint, value_deg)
    return math.radians(clamped) if joint.is_angular() else clamped


def _native_to_joint_value(joint: Joint, native: float) -> float:
    numeric = _as_float(native)
    return math.degrees(numeric) if joint.is_angular() else numeric


def build_default_joint_values(model: RobotModel) -> Dict[str, float]:
    """Default value map covering only the joints a user may actually drive."""
    return {
        joint.name: _as_float(joint.default_value_deg)
        for joint in model.movable_joints()
    }


def resolve_joint_value(
    model: RobotModel,
    joint: Joint,
    joint_values: Mapping[str, float],
    _resolving: Optional[set] = None,
) -> float:
    """Resolve a joint's effective value, following ``mimic`` chains.

    A mimic cycle falls back to the joint's default value rather than recursing
    forever, matching the source implementation.
    """
    if joint.mimic is None:
        return clamp_joint_value_deg(joint, joint_values.get(joint.name))
    resolving = set() if _resolving is None else _resolving
    if joint.name in resolving:
        return clamp_joint_value_deg(joint, joint.default_value_deg)
    resolving.add(joint.name)
    master = model.joints_by_name.get(joint.mimic.joint)
    if master is None:
        master_native = _as_float(joint_values.get(joint.mimic.joint))
    else:
        master_value = resolve_joint_value(model, master, joint_values, resolving)
        master_native = _joint_value_to_native(master, master_value)
    resolving.discard(joint.name)
    native = _as_float(joint.mimic.multiplier, 1.0) * master_native + _as_float(
        joint.mimic.offset
    )
    return clamp_joint_value_deg(joint, _native_to_joint_value(joint, native))


def posed_joint_local_transform(joint: Joint, value_deg: float) -> Transform:
    """Parent-link -> child-link transform for a joint at the given value."""
    origin = tuple(_as_float(value) for value in joint.origin_transform)
    if joint.type == "fixed":
        return origin
    clamped = clamp_joint_value_deg(joint, value_deg)
    if joint.type == "prismatic":
        motion = translation_along_axis_transform(joint.axis, clamped)
    else:
        motion = axis_angle_transform(joint.axis, math.radians(clamped))
    return multiply_transforms(origin, motion)


def solve_link_world_transforms(
    model: RobotModel, joint_values: Optional[Mapping[str, float]] = None
) -> Dict[str, Transform]:
    """Forward kinematics: world transform of every reachable link."""
    values: Mapping[str, float] = joint_values or {}
    transforms: Dict[str, Transform] = {}
    if not model.root_link:
        return transforms
    joints_by_parent: Dict[str, list] = {}
    for joint in model.joints:
        if not joint.parent_link:
            continue
        joints_by_parent.setdefault(joint.parent_link, []).append(joint)

    stack = [(model.root_link, tuple(_as_float(v) for v in model.root_world_transform))]
    while stack:
        link_name, world = stack.pop()
        if link_name in transforms:
            continue
        transforms[link_name] = world
        for joint in reversed(joints_by_parent.get(link_name, [])):
            if not joint.child_link:
                continue
            value = resolve_joint_value(model, joint, values)
            child_world = multiply_transforms(
                world, posed_joint_local_transform(joint, value)
            )
            stack.append((joint.child_link, child_world))
    return transforms


def link_origin_in_frame(
    model: RobotModel,
    joint_values: Optional[Mapping[str, float]],
    link_name: str,
    frame_link_name: str,
) -> Optional[Vector3]:
    """Origin of ``link_name`` expressed in the frame of ``frame_link_name``."""
    if not link_name.strip() or not frame_link_name.strip():
        return None
    transforms = solve_link_world_transforms(model, joint_values)
    link_world = transforms.get(link_name.strip())
    frame_world = transforms.get(frame_link_name.strip())
    if link_world is None or frame_world is None:
        return None
    return transform_point(
        invert_rigid_transform(frame_world), transform_point(link_world, (0.0, 0.0, 0.0))
    )


def root_point_in_frame(
    model: RobotModel,
    joint_values: Optional[Mapping[str, float]],
    point: Sequence[float],
    frame_link_name: str,
) -> Optional[Vector3]:
    """Map a point given in the root/world frame into a link's local frame."""
    if not frame_link_name.strip():
        return None
    transforms = solve_link_world_transforms(model, joint_values)
    frame_world = transforms.get(frame_link_name.strip())
    if frame_world is None:
        return None
    return transform_point(invert_rigid_transform(frame_world), point)
