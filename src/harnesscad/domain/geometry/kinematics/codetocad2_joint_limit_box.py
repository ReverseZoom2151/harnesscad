"""Per-axis 6-DOF limit box for assembly joints (CodeToCAD constraint model).

CodeToCAD expresses a joint not as a *type* but as a pair of per-axis limit
constraints -- ``apply_limit_location_constraint(x=[min,max], y=..., z=...)``
and ``apply_limit_rotation_constraint(x_radians=[min,max], ...)``.  A joint is
therefore a *box in 6-DOF pose space*: for every one of the six DOF (three
translations, three rotations) the joint either

* leaves it **free**       -- no limit given (``None``),
* **locks** it             -- ``min == max`` (usually ``[0, 0]``), or
* **limits** it to a range -- ``min < max``.

The joint *type* is then a consequence of that pattern (a single free/limited
rotation about one axis with everything else locked *is* a revolute joint), and
the joint's job at solve time is to **clamp** a proposed pose into the allowed
box.

The harness already has :mod:`geometry.joinable_joint_motion`, which zeroes the
pose parameters a named joint type disallows -- but it has no notion of a
``[min, max]`` range, so it can neither express "this hinge swings between -30
and +90 degrees" nor clamp a pose into that range.  This module supplies the
missing limit box:

* :class:`AxisLimit`     -- one DOF: free / locked / ranged, with clamping.
* :class:`JointLimitBox` -- the six of them, built from ``limit_location_xyz``
  and ``limit_rotation_xyz`` exactly as the upstream adapter takes them.
* :meth:`JointLimitBox.clamp` -- project a proposed 6-vector pose into the box.
* :meth:`JointLimitBox.classify` -- name the joint implied by the pattern
  (rigid / revolute / prismatic / cylindrical / planar / ball / free / ...).
* constructors :func:`revolute`, :func:`prismatic`, :func:`ball`, :func:`rigid`
  mirroring ``codetocad.core.cad.constraint``.
* :meth:`JointLimitBox.intersect` -- the box of two joints applied at once.

Rotations are radians.  Angles are wrapped onto the branch nearest the allowed
interval before clamping, so a proposed 350 degree rotation against a
``[-30, +90]`` degree hinge clamps to -10 degrees, not to +90.  Stdlib only,
deterministic.
"""

from __future__ import annotations

import math

__all__ = [
    "LimitError",
    "AxisLimit",
    "JointLimitBox",
    "DOF_NAMES",
    "FREE",
    "LOCKED",
    "RANGED",
    "revolute",
    "prismatic",
    "cylindrical",
    "planar",
    "ball",
    "rigid",
    "free_joint",
    "wrap_angle",
]

FREE = "free"
LOCKED = "locked"
RANGED = "ranged"

DOF_NAMES = ("x", "y", "z", "rx", "ry", "rz")

_TRANSLATION = (0, 1, 2)
_ROTATION = (3, 4, 5)

_TOL = 1e-12


class LimitError(ValueError):
    """Raised for a malformed limit specification."""


def wrap_angle(angle: float) -> float:
    """Wrap ``angle`` (radians) into ``[-pi, pi)``."""
    wrapped = math.fmod(angle + math.pi, 2.0 * math.pi)
    if wrapped < 0.0:
        wrapped += 2.0 * math.pi
    return wrapped - math.pi


class AxisLimit(object):
    """A single degree of freedom's allowed interval.

    ``minimum``/``maximum`` are ``None`` for an unbounded side.  Both ``None``
    means the DOF is entirely free.  ``minimum == maximum`` locks it.
    """

    __slots__ = ("minimum", "maximum", "angular")

    def __init__(self, minimum=None, maximum=None, angular: bool = False):
        lo = None if minimum is None else float(minimum)
        hi = None if maximum is None else float(maximum)
        if lo is not None and hi is not None and lo > hi + _TOL:
            raise LimitError(f"limit minimum {lo} exceeds maximum {hi}")
        self.minimum = lo
        self.maximum = hi
        self.angular = bool(angular)

    # -- construction ----------------------------------------------------
    @classmethod
    def free(cls, angular: bool = False) -> "AxisLimit":
        return cls(None, None, angular)

    @classmethod
    def locked(cls, value: float = 0.0, angular: bool = False) -> "AxisLimit":
        return cls(value, value, angular)

    @classmethod
    def from_pair(cls, pair, angular: bool = False) -> "AxisLimit":
        """Build from ``None`` or a ``[min, max]`` pair (either entry nullable)."""
        if pair is None:
            return cls.free(angular)
        if isinstance(pair, AxisLimit):
            return cls(pair.minimum, pair.maximum, angular)
        try:
            lo, hi = pair
        except (TypeError, ValueError):
            raise LimitError(f"limit must be None or a [min, max] pair: {pair!r}")
        return cls(lo, hi, angular)

    # -- queries ---------------------------------------------------------
    @property
    def kind(self) -> str:
        if self.minimum is None and self.maximum is None:
            return FREE
        if (
            self.minimum is not None
            and self.maximum is not None
            and abs(self.maximum - self.minimum) <= _TOL
        ):
            return LOCKED
        return RANGED

    @property
    def is_free(self) -> bool:
        return self.kind == FREE

    @property
    def is_locked(self) -> bool:
        return self.kind == LOCKED

    @property
    def is_ranged(self) -> bool:
        return self.kind == RANGED

    @property
    def dof(self) -> int:
        """1 when the DOF can move at all, 0 when locked."""
        return 0 if self.is_locked else 1

    @property
    def span(self):
        """Width of the interval; ``None`` when unbounded on either side."""
        if self.minimum is None or self.maximum is None:
            return None
        return self.maximum - self.minimum

    def contains(self, value: float, tolerance: float = 1e-9) -> bool:
        return abs(self.clamp(value) - self._representative(value)) <= tolerance

    # -- clamping --------------------------------------------------------
    def _representative(self, value: float) -> float:
        """The branch of ``value`` that clamping is measured against."""
        value = float(value)
        if not self.angular or self.is_free:
            return value
        centre = self._centre()
        if centre is None:
            return value
        return centre + wrap_angle(value - centre)

    def _centre(self):
        if self.minimum is not None and self.maximum is not None:
            return 0.5 * (self.minimum + self.maximum)
        if self.minimum is not None:
            return self.minimum
        if self.maximum is not None:
            return self.maximum
        return None

    def clamp(self, value: float) -> float:
        """Nearest allowed value to ``value``."""
        v = self._representative(value)
        if self.minimum is not None and v < self.minimum:
            v = self.minimum
        if self.maximum is not None and v > self.maximum:
            v = self.maximum
        return v

    def intersect(self, other: "AxisLimit") -> "AxisLimit":
        """The limit satisfying both (raises when the intersection is empty)."""
        lo = _max_opt(self.minimum, other.minimum)
        hi = _min_opt(self.maximum, other.maximum)
        if lo is not None and hi is not None and lo > hi + _TOL:
            raise LimitError("limits do not intersect")
        return AxisLimit(lo, hi, self.angular or other.angular)

    def as_pair(self):
        return (self.minimum, self.maximum)

    def __eq__(self, other):
        if not isinstance(other, AxisLimit):
            return NotImplemented
        return (
            self.minimum == other.minimum
            and self.maximum == other.maximum
            and self.angular == other.angular
        )

    def __hash__(self):
        return hash((self.minimum, self.maximum, self.angular))

    def __repr__(self):
        return (
            f"AxisLimit(minimum={self.minimum!r}, maximum={self.maximum!r},"
            f" angular={self.angular!r})"
        )


def _max_opt(a, b):
    if a is None:
        return b
    if b is None:
        return a
    return a if a > b else b


def _min_opt(a, b):
    if a is None:
        return b
    if b is None:
        return a
    return a if a < b else b


class JointLimitBox(object):
    """Six :class:`AxisLimit` -- a box in ``(x, y, z, rx, ry, rz)`` pose space."""

    __slots__ = ("limits",)

    def __init__(self, limits):
        limits = tuple(limits)
        if len(limits) != 6:
            raise LimitError("a limit box needs exactly 6 axis limits")
        self.limits = limits

    # -- construction ----------------------------------------------------
    @classmethod
    def from_xyz(
        cls,
        limit_location_xyz=None,
        limit_rotation_xyz=None,
    ) -> "JointLimitBox":
        """Build from the upstream adapter's two triples of ``[min, max]`` pairs.

        ``None`` for a triple leaves those three DOF free; ``None`` for an entry
        leaves that DOF free.
        """
        loc = _triple(limit_location_xyz)
        rot = _triple(limit_rotation_xyz)
        limits = [AxisLimit.from_pair(p, angular=False) for p in loc]
        limits += [AxisLimit.from_pair(p, angular=True) for p in rot]
        return cls(limits)

    @classmethod
    def locked_box(cls) -> "JointLimitBox":
        return cls(
            [AxisLimit.locked(0.0, False) for _ in _TRANSLATION]
            + [AxisLimit.locked(0.0, True) for _ in _ROTATION]
        )

    @classmethod
    def free_box(cls) -> "JointLimitBox":
        return cls(
            [AxisLimit.free(False) for _ in _TRANSLATION]
            + [AxisLimit.free(True) for _ in _ROTATION]
        )

    def with_limit(self, name: str, minimum=None, maximum=None) -> "JointLimitBox":
        """A copy with DOF ``name`` replaced by ``[minimum, maximum]``."""
        index = self.index_of(name)
        limits = list(self.limits)
        limits[index] = AxisLimit(minimum, maximum, angular=index in _ROTATION)
        return JointLimitBox(limits)

    def with_free(self, name: str) -> "JointLimitBox":
        index = self.index_of(name)
        limits = list(self.limits)
        limits[index] = AxisLimit.free(angular=index in _ROTATION)
        return JointLimitBox(limits)

    @staticmethod
    def index_of(name: str) -> int:
        try:
            return DOF_NAMES.index(name)
        except ValueError:
            raise LimitError(f"unknown degree of freedom: {name!r}")

    def limit(self, name: str) -> AxisLimit:
        return self.limits[self.index_of(name)]

    # -- queries ---------------------------------------------------------
    @property
    def location_limits(self):
        return self.limits[:3]

    @property
    def rotation_limits(self):
        return self.limits[3:]

    @property
    def dof(self) -> int:
        """Number of DOF the box leaves movable (0..6)."""
        return sum(limit.dof for limit in self.limits)

    @property
    def translational_dof(self) -> int:
        return sum(self.limits[i].dof for i in _TRANSLATION)

    @property
    def rotational_dof(self) -> int:
        return sum(self.limits[i].dof for i in _ROTATION)

    def movable(self):
        """Names of the DOF that are not locked, in canonical order."""
        return tuple(
            name
            for name, limit in zip(DOF_NAMES, self.limits)
            if not limit.is_locked
        )

    def is_bounded(self) -> bool:
        """True when every movable DOF has both a min and a max."""
        for limit in self.limits:
            if limit.is_locked:
                continue
            if limit.minimum is None or limit.maximum is None:
                return False
        return True

    # -- the core operation ----------------------------------------------
    def clamp(self, pose):
        """Project a proposed pose (6 numbers) into the allowed box."""
        pose = tuple(pose)
        if len(pose) != 6:
            raise LimitError("a pose is 6 numbers: x, y, z, rx, ry, rz")
        return tuple(
            limit.clamp(value) for limit, value in zip(self.limits, pose)
        )

    def contains(self, pose, tolerance: float = 1e-9) -> bool:
        pose = tuple(pose)
        if len(pose) != 6:
            raise LimitError("a pose is 6 numbers: x, y, z, rx, ry, rz")
        return all(
            limit.contains(value, tolerance)
            for limit, value in zip(self.limits, pose)
        )

    def intersect(self, other: "JointLimitBox") -> "JointLimitBox":
        return JointLimitBox(
            [a.intersect(b) for a, b in zip(self.limits, other.limits)]
        )

    # -- classification --------------------------------------------------
    def classify(self) -> str:
        """Name the joint implied by the limit pattern.

        ``rigid``, ``revolute``, ``prismatic``, ``cylindrical``, ``planar``,
        ``ball``, ``free``, or ``generic`` when the pattern matches no textbook
        joint.
        """
        t_free = [i for i in _TRANSLATION if not self.limits[i].is_locked]
        r_free = [i for i in _ROTATION if not self.limits[i].is_locked]
        nt, nr = len(t_free), len(r_free)

        if nt == 0 and nr == 0:
            return "rigid"
        if nt == 0 and nr == 1:
            return "revolute"
        if nt == 1 and nr == 0:
            return "prismatic"
        if nt == 0 and nr == 3:
            return "ball"
        if nt == 1 and nr == 1 and (t_free[0] + 3) == r_free[0]:
            return "cylindrical"
        if nt == 2 and nr == 1:
            missing = (set(_TRANSLATION) - set(t_free)).pop()
            if (missing + 3) == r_free[0]:
                return "planar"
        if nt == 3 and nr == 3:
            return "free"
        return "generic"

    def axis_of(self):
        """The single movable axis name (``x``/``y``/``z``) or ``None``.

        Defined for revolute, prismatic and cylindrical joints.
        """
        kind = self.classify()
        if kind == "revolute":
            index = [i for i in _ROTATION if not self.limits[i].is_locked][0] - 3
        elif kind == "prismatic":
            index = [i for i in _TRANSLATION if not self.limits[i].is_locked][0]
        elif kind == "cylindrical":
            index = [i for i in _TRANSLATION if not self.limits[i].is_locked][0]
        else:
            return None
        return DOF_NAMES[index]

    def as_dict(self):
        return {
            name: limit.as_pair() for name, limit in zip(DOF_NAMES, self.limits)
        }

    def __eq__(self, other):
        if not isinstance(other, JointLimitBox):
            return NotImplemented
        return self.limits == other.limits

    def __hash__(self):
        return hash(self.limits)

    def __repr__(self):
        return f"JointLimitBox({list(self.limits)!r})"


def _triple(value):
    if value is None:
        return (None, None, None)
    triple = tuple(value)
    if len(triple) != 3:
        raise LimitError("a limit triple needs exactly 3 entries (x, y, z)")
    return triple


# -- joint constructors (mirroring codetocad.core.cad.constraint) ----------

_AXIS_INDEX = {"x": 0, "y": 1, "z": 2}


def _axis_index(axis: str) -> int:
    try:
        return _AXIS_INDEX[str(axis).lower()]
    except KeyError:
        raise LimitError(f"unknown axis: {axis!r}")


def rigid() -> JointLimitBox:
    """Fixed joint: no freedom of movement."""
    return JointLimitBox.locked_box()


def free_joint() -> JointLimitBox:
    """Unconstrained: all six DOF free."""
    return JointLimitBox.free_box()


def revolute(axis: str = "z", limit_min=None, limit_max=None) -> JointLimitBox:
    """Hinge about ``axis``, optionally limited to ``[limit_min, limit_max]`` rad."""
    index = _axis_index(axis)
    box = JointLimitBox.locked_box()
    return box.with_limit(DOF_NAMES[3 + index], limit_min, limit_max)


def prismatic(axis: str = "z", limit_min=None, limit_max=None) -> JointLimitBox:
    """Slider along ``axis``, optionally limited to ``[limit_min, limit_max]``."""
    index = _axis_index(axis)
    box = JointLimitBox.locked_box()
    return box.with_limit(DOF_NAMES[index], limit_min, limit_max)


def cylindrical(
    axis: str = "z",
    slide_min=None,
    slide_max=None,
    angle_min=None,
    angle_max=None,
) -> JointLimitBox:
    """Rotation *and* translation about/along the same axis."""
    index = _axis_index(axis)
    box = JointLimitBox.locked_box()
    box = box.with_limit(DOF_NAMES[index], slide_min, slide_max)
    return box.with_limit(DOF_NAMES[3 + index], angle_min, angle_max)


def planar(normal: str = "z") -> JointLimitBox:
    """Slide in the plane normal to ``normal`` and spin about that normal."""
    index = _axis_index(normal)
    box = JointLimitBox.locked_box()
    for i in _TRANSLATION:
        if i != index:
            box = box.with_free(DOF_NAMES[i])
    return box.with_free(DOF_NAMES[3 + index])


def ball(
    angular_range_x=None,
    angular_range_y=None,
    angular_range_z=None,
) -> JointLimitBox:
    """Gimbal: rotation about all three axes, each optionally range-limited."""
    return JointLimitBox.from_xyz(
        limit_location_xyz=[(0.0, 0.0), (0.0, 0.0), (0.0, 0.0)],
        limit_rotation_xyz=[angular_range_x, angular_range_y, angular_range_z],
    )
