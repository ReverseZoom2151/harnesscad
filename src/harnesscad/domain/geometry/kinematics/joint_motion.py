"""Joint types, their degrees of freedom, and the motion they permit.

JoinABLe / Fusion 360 model a joint between two B-Rep entities as a joint axis
plus a joint *type*.  The type decides which pose parameters are free:

============  ====  ==========================================================
type          DOF   free parameters
============  ====  ==========================================================
rigid          0    none
revolute       1    ``rotation`` about the axis
slider         1    ``offset`` along the axis
cylindrical    2    ``rotation`` + ``offset``
pin_slot       2    ``rotation`` about the axis + ``slide_u`` across it
planar         3    ``slide_u`` + ``slide_v`` in the plane + ``rotation``
                    about its normal
ball           3    rotation about all three axes (not parameterised here)
============  ====  ==========================================================

``flip`` (reflecting the moved body through the plane normal to the axis at the
joint origin) is a *discrete* pose choice available to every joint type; it is
not a degree of freedom.

The module gives the vocabulary (DOF counts, free-parameter sets), a pose type
that can be projected onto the motion a joint type allows, deterministic
sampling of the reachable motion, and conversion of a pose to a 4x4 transform
built on :mod:`geometry.joinable_joint_transform`.  Stdlib only.
"""

import math

from harnesscad.domain.geometry.kinematics.joint_axis import (
    as_vec3,
    cross,
    normalize,
)
from harnesscad.domain.geometry.kinematics.joint_transform import (
    joint_alignment_matrix,
    matmul,
    offset_parameter_matrix,
    rotation_parameter_matrix,
)

__all__ = [
    "JOINT_TYPES",
    "JOINT_FREE_PARAMETERS",
    "JOINT_FREE_DOF",
    "POSE_PARAMETERS",
    "UnknownJointType",
    "normalize_joint_type",
    "joint_free_dof",
    "joint_constrained_dof",
    "joint_free_parameters",
    "is_pose_parameter_free",
    "JointPose",
    "project_pose",
    "sample_joint_motion",
    "axis_plane_basis",
    "pose_matrix",
    "joint_pose_transform",
]

#: Continuous pose parameters, in the order they are applied.
POSE_PARAMETERS = ("rotation", "offset", "slide_u", "slide_v")

#: Free (continuous) parameters allowed by each joint type.
JOINT_FREE_PARAMETERS = {
    "rigid": (),
    "revolute": ("rotation",),
    "slider": ("offset",),
    "cylindrical": ("rotation", "offset"),
    "pin_slot": ("rotation", "slide_u"),
    "planar": ("rotation", "slide_u", "slide_v"),
    "ball": ("rotation",),
}

#: Free DOF of each joint type (a ball joint has 3 rotational DOF, of which
#: only the axis rotation is parameterised here).
JOINT_FREE_DOF = {
    "rigid": 0,
    "revolute": 1,
    "slider": 1,
    "cylindrical": 2,
    "pin_slot": 2,
    "planar": 3,
    "ball": 3,
}

JOINT_TYPES = tuple(sorted(JOINT_FREE_DOF))

#: A free rigid body has six degrees of freedom.
BODY_DOF = 6

_ALIASES = {
    "fixed": "rigid",
    "rigidjoint": "rigid",
    "rigidjointtype": "rigid",
    "hinge": "revolute",
    "pin": "revolute",
    "revolutejointtype": "revolute",
    "prismatic": "slider",
    "sliderjointtype": "slider",
    "translational": "slider",
    "cylindricaljointtype": "cylindrical",
    "pinslot": "pin_slot",
    "pin-slot": "pin_slot",
    "pinslotjointtype": "pin_slot",
    "planarjointtype": "planar",
    "balljointtype": "ball",
    "spherical": "ball",
}


class UnknownJointType(ValueError):
    """Raised for a joint type outside the Fusion joint vocabulary."""


def normalize_joint_type(name):
    """Canonical joint-type name; raises :class:`UnknownJointType` if unknown."""
    key = str(name).strip().lower().replace(" ", "_")
    if key in JOINT_FREE_DOF:
        return key
    flat = key.replace("_", "").replace("-", "")
    if flat in JOINT_FREE_DOF:
        return flat
    if flat in _ALIASES:
        return _ALIASES[flat]
    if key in _ALIASES:
        return _ALIASES[key]
    raise UnknownJointType(f"unknown joint type: {name!r}")


def joint_free_dof(name):
    """Degrees of freedom left free by a joint of this type."""
    return JOINT_FREE_DOF[normalize_joint_type(name)]


def joint_constrained_dof(name):
    """Degrees of freedom removed by a joint of this type (``6 - free``)."""
    return BODY_DOF - joint_free_dof(name)


def joint_free_parameters(name):
    """The continuous pose parameters a joint of this type may vary."""
    return JOINT_FREE_PARAMETERS[normalize_joint_type(name)]


def is_pose_parameter_free(name, parameter):
    """True when ``parameter`` may vary for a joint of type ``name``."""
    if parameter not in POSE_PARAMETERS:
        raise ValueError(f"unknown pose parameter: {parameter!r}")
    return parameter in joint_free_parameters(name)


class JointPose(object):
    """A pose of the moved body relative to the joint axis.

    ``rotation`` is in radians about the axis, ``offset`` in model units along
    the axis, ``slide_u`` / ``slide_v`` in model units across the axis (in the
    plane normal to it), and ``flip`` is the discrete reflection toggle.
    """

    __slots__ = ("rotation", "offset", "slide_u", "slide_v", "flip")

    def __init__(self, rotation=0.0, offset=0.0, slide_u=0.0, slide_v=0.0,
                 flip=False):
        self.rotation = float(rotation)
        self.offset = float(offset)
        self.slide_u = float(slide_u)
        self.slide_v = float(slide_v)
        self.flip = bool(flip)

    def as_dict(self):
        return {
            "rotation": self.rotation,
            "offset": self.offset,
            "slide_u": self.slide_u,
            "slide_v": self.slide_v,
            "flip": self.flip,
        }

    def __eq__(self, other):
        if not isinstance(other, JointPose):
            return NotImplemented
        return self.as_dict() == other.as_dict()

    def __hash__(self):
        d = self.as_dict()
        return hash(tuple(sorted(d.items())))

    def __repr__(self):
        return (f"JointPose(rotation={self.rotation!r}, offset={self.offset!r},"
                f" slide_u={self.slide_u!r}, slide_v={self.slide_v!r},"
                f" flip={self.flip!r})")


def project_pose(joint_type, pose):
    """Zero out every pose parameter the joint type does not permit.

    ``flip`` is preserved, being a discrete choice rather than a DOF.
    """
    free = set(joint_free_parameters(joint_type))
    return JointPose(
        rotation=pose.rotation if "rotation" in free else 0.0,
        offset=pose.offset if "offset" in free else 0.0,
        slide_u=pose.slide_u if "slide_u" in free else 0.0,
        slide_v=pose.slide_v if "slide_v" in free else 0.0,
        flip=pose.flip,
    )


def sample_joint_motion(joint_type, steps=4, offset_range=1.0,
                        slide_range=1.0, include_flip=False):
    """Deterministic sweep of the motion a joint type allows.

    ``rotation`` sweeps ``[0, 2*pi)`` in ``steps`` samples; ``offset`` and the
    slides sweep symmetric ranges in ``steps`` samples.  The returned list is
    the cartesian product over the joint's free parameters (times the flip
    states when ``include_flip``), always starting with the neutral pose.
    """
    joint_type = normalize_joint_type(joint_type)
    if steps < 1:
        raise ValueError("steps must be >= 1")
    free = joint_free_parameters(joint_type)

    def linspace(span):
        if steps == 1:
            return [0.0]
        return [-span + 2.0 * span * i / (steps - 1) for i in range(steps)]

    values = {
        "rotation": ([0.0] if steps == 1
                     else [2.0 * math.pi * i / steps for i in range(steps)]),
        "offset": linspace(float(offset_range)),
        "slide_u": linspace(float(slide_range)),
        "slide_v": linspace(float(slide_range)),
    }

    poses = [{}]
    for parameter in POSE_PARAMETERS:
        if parameter not in free:
            continue
        expanded = []
        for base in poses:
            for value in values[parameter]:
                combo = dict(base)
                combo[parameter] = value
                expanded.append(combo)
        poses = expanded

    flips = (False, True) if include_flip else (False,)
    result = []
    for flip in flips:
        for combo in poses:
            result.append(JointPose(flip=flip, **combo))
    # Keep the neutral pose first for stable, greedy consumers.
    neutral = JointPose()
    result.sort(key=lambda p: (p != neutral,))
    return result


def axis_plane_basis(direction):
    """Deterministic orthonormal basis ``(u, v)`` of the plane normal to an axis."""
    axis = normalize(as_vec3(direction))
    if axis == (0.0, 0.0, 0.0):
        raise ValueError("cannot build a basis around a zero-length axis")
    if abs(axis[0]) <= abs(axis[1]) and abs(axis[0]) <= abs(axis[2]):
        seed = (1.0, 0.0, 0.0)
    elif abs(axis[1]) <= abs(axis[2]):
        seed = (0.0, 1.0, 0.0)
    else:
        seed = (0.0, 0.0, 1.0)
    u = normalize(cross(axis, seed))
    v = normalize(cross(axis, u))
    return u, v


def pose_matrix(pose, origin, direction):
    """4x4 matrix of a pose expressed about the joint axis at ``origin``."""
    origin = as_vec3(origin)
    axis = normalize(as_vec3(direction))
    mat = rotation_parameter_matrix(pose.rotation, origin, axis)
    mat = matmul(offset_parameter_matrix(pose.offset, origin, axis, pose.flip),
                 mat)
    if pose.slide_u or pose.slide_v:
        u, v = axis_plane_basis(axis)
        slide = (u[0] * pose.slide_u + v[0] * pose.slide_v,
                 u[1] * pose.slide_u + v[1] * pose.slide_v,
                 u[2] * pose.slide_u + v[2] * pose.slide_v)
        translate = (
            (1.0, 0.0, 0.0, slide[0]),
            (0.0, 1.0, 0.0, slide[1]),
            (0.0, 0.0, 1.0, slide[2]),
            (0.0, 0.0, 0.0, 1.0),
        )
        mat = matmul(translate, mat)
    return mat


def joint_pose_transform(joint_type, pose, axis_line1, axis_line2):
    """Transform placing body one on body two's joint axis with a legal pose.

    The pose is first projected onto the joint type's free parameters, so a
    rigid joint always yields the plain alignment transform.
    """
    pose = project_pose(joint_type, pose)
    origin1, direction1 = axis_line1
    origin2, direction2 = axis_line2
    align = joint_alignment_matrix(origin1, direction1, origin2, direction2)
    return matmul(pose_matrix(pose, origin2, direction2), align)
