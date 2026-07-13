"""Graph-CAD assembly placement: Align / offset / polar resolution over AABBs.

The Graph-CAD decomposition graph places every non-root part *relatively*. Its
``pos`` / ``align`` fields use a small placement language::

    Align(XY) shelf.left_face   to side_L.right_face
    Align(Z)  shelf.bottom_face to side_L.bottom_face
    offset(0, 0, 0.010)

Each ``Align(<axes>)`` clause states which feature of the moving part must
coincide with which feature of a reference, and *locks only the listed axes* --
the remaining axes stay free for later clauses. Targets come in four forms: a
single feature ``B.top_face``, an indexed pattern instance ``B[2].top_face``,
an aggregate over every instance of a pattern ``B[*].top_face`` (the instances
are unioned into one bounding volume first, then the face is taken from that),
and ``Avg(T1, T2, ...)`` which averages the listed feature centres. ``offset``
slides the already-aligned part, and ``polar(theta; dr=...)`` places a part on a
curved anchor at an engine-computed radius that makes the two parts touch.

All of that is pure arithmetic, so it is reimplemented here exactly: axis-
aligned boxes, six named faces plus ``center`` and ``side_at(theta)``, clause
chaining that respects axis locking, and the touching-radius rule for polar
placement. The clause *source* (a language model) is external; resolving it is
not.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Dict, Iterable, Mapping, Sequence, Tuple

__all__ = [
    "Box",
    "AlignClause",
    "FACE_NAMES",
    "feature_point",
    "aggregate_box",
    "parse_align",
    "parse_align_chain",
    "parse_offset",
    "parse_polar",
    "resolve_target",
    "apply_align",
    "resolve_placement",
    "polar_position",
]

Vec3 = Tuple[float, float, float]

#: Face name -> (axis index, sign). ``left``/``right`` are -X/+X, ``front``/
#: ``back`` are -Y/+Y and ``bottom``/``top`` are -Z/+Z.
FACE_NAMES: Dict[str, Tuple[int, int]] = {
    "left_face": (0, -1),
    "right_face": (0, 1),
    "front_face": (1, -1),
    "back_face": (1, 1),
    "bottom_face": (2, -1),
    "top_face": (2, 1),
    # The spec also uses the short forms in ``Align tip.bottom to body.top``.
    "left": (0, -1),
    "right": (0, 1),
    "front": (1, -1),
    "back": (1, 1),
    "bottom": (2, -1),
    "top": (2, 1),
}

_AXES = {"X": 0, "Y": 1, "Z": 2}

_ALIGN = re.compile(
    r"Align\s*\(\s*(?P<axes>[XYZxyz]+)\s*\)\s*"
    r"(?P<this>[A-Za-z_][\w]*)\s*\.\s*(?P<feature>[\w]+(?:\([^)]*\))?)\s*"
    r"to\s+(?P<target>.+?)\s*$",
    re.IGNORECASE,
)
_OFFSET = re.compile(
    r"offset\s*\(\s*(?P<x>[-+0-9.eE]+)\s*,\s*(?P<y>[-+0-9.eE]+)\s*,"
    r"\s*(?P<z>[-+0-9.eE]+)\s*\)"
)
_POLAR = re.compile(
    r"polar\s*\(\s*(?P<theta>[-+0-9.eE]+)\s*(?:deg|°)?\s*"
    r"(?:;\s*dr\s*=\s*(?P<dr>[-+0-9.eE]+)\s*)?\)",
    re.IGNORECASE,
)
_SIDE_AT = re.compile(r"side_at\s*\(\s*(?P<theta>[-+0-9.eE]+)\s*(?:deg|°)?\s*\)",
                      re.IGNORECASE)
_TOKEN = re.compile(
    r"^\s*(?P<obj>[A-Za-z_][\w]*)"
    r"(?:\[\s*(?P<index>\*|[0-9]+)\s*\])?"
    r"\s*\.\s*(?P<feature>[\w]+(?:\([^)]*\))?)\s*$"
)
_AVG = re.compile(r"^\s*Avg\s*\((?P<args>.*)\)\s*$", re.IGNORECASE)


@dataclass(frozen=True)
class Box:
    """An axis-aligned box: centre plus full side lengths."""

    center: Vec3
    size: Vec3

    def __post_init__(self) -> None:
        if len(self.center) != 3 or len(self.size) != 3:
            raise ValueError("center and size must be 3-vectors")
        if any(value < 0 for value in self.size):
            raise ValueError("size components must be non-negative")

    def minimum(self) -> Vec3:
        return tuple(  # type: ignore[return-value]
            self.center[axis] - self.size[axis] / 2.0 for axis in range(3)
        )

    def maximum(self) -> Vec3:
        return tuple(  # type: ignore[return-value]
            self.center[axis] + self.size[axis] / 2.0 for axis in range(3)
        )

    def moved_to(self, center: Vec3) -> "Box":
        return Box(tuple(float(value) for value in center), self.size)  # type: ignore[arg-type]

    def translated(self, delta: Vec3) -> "Box":
        return self.moved_to(
            tuple(self.center[axis] + delta[axis] for axis in range(3))  # type: ignore[arg-type]
        )


@dataclass(frozen=True)
class AlignClause:
    """``Align(<axes>) <this>.<feature> to <target>``."""

    axes: Tuple[int, ...]
    this_id: str
    this_feature: str
    target: str


def aggregate_box(boxes: Sequence[Box]) -> Box:
    """Union a pattern's instances into the single bounding volume the spec uses."""
    if not boxes:
        raise ValueError("cannot aggregate an empty pattern")
    lows = [min(box.minimum()[axis] for box in boxes) for axis in range(3)]
    highs = [max(box.maximum()[axis] for box in boxes) for axis in range(3)]
    center = tuple((lows[axis] + highs[axis]) / 2.0 for axis in range(3))
    size = tuple(highs[axis] - lows[axis] for axis in range(3))
    return Box(center, size)  # type: ignore[arg-type]


def feature_point(box: Box, feature: str) -> Vec3:
    """Point of a named feature: a face centre, the centre, or ``side_at(theta)``.

    ``side_at(theta)`` is the point on the box's circumscribed side surface at
    angle ``theta`` measured counter-clockwise from +X in the XY plane, which is
    how the format anchors children to cylindrical parents.
    """
    name = feature.strip()
    if name in FACE_NAMES:
        axis, sign = FACE_NAMES[name]
        point = list(box.center)
        point[axis] += sign * box.size[axis] / 2.0
        return tuple(point)  # type: ignore[return-value]
    if name in {"center", "centre", "origin"}:
        return box.center
    match = _SIDE_AT.match(name)
    if match:
        theta = math.radians(float(match.group("theta")))
        return (
            box.center[0] + (box.size[0] / 2.0) * math.cos(theta),
            box.center[1] + (box.size[1] / 2.0) * math.sin(theta),
            box.center[2],
        )
    raise ValueError(f"unknown feature: {feature!r}")


def parse_align(text: str) -> AlignClause:
    """Parse one ``Align(...)`` clause."""
    match = _ALIGN.match(text.strip())
    if not match:
        raise ValueError(f"not an Align clause: {text!r}")
    axes = tuple(
        sorted({_AXES[char] for char in match.group("axes").upper()})
    )
    return AlignClause(
        axes=axes,
        this_id=match.group("this"),
        this_feature=match.group("feature"),
        target=match.group("target").strip().rstrip(";"),
    )


def parse_align_chain(text: str) -> Tuple[AlignClause, ...]:
    """Parse every ``Align`` clause in a field, keeping application order.

    Clauses may be separated by ``;`` or newlines and may be interleaved with
    other directives (``offset``, ``orientation``), which are ignored here.
    """
    clauses = []
    for chunk in re.split(r"[;\n]", text):
        stripped = chunk.strip()
        if not stripped:
            continue
        if _ALIGN.match(stripped):
            clauses.append(parse_align(stripped))
    return tuple(clauses)


def parse_offset(text: str) -> Vec3:
    """Parse ``offset(dx, dy, dz)``; a missing clause yields the zero vector."""
    match = _OFFSET.search(text)
    if not match:
        return (0.0, 0.0, 0.0)
    return (
        float(match.group("x")),
        float(match.group("y")),
        float(match.group("z")),
    )


def parse_polar(text: str) -> Tuple[float, float]:
    """Parse ``polar(theta)`` / ``polar(theta; dr=delta)`` into ``(theta, dr)``."""
    match = _POLAR.search(text)
    if not match:
        raise ValueError(f"not a polar clause: {text!r}")
    dr = match.group("dr")
    return float(match.group("theta")), float(dr) if dr is not None else 0.0


def _instances(boxes: Mapping[str, Box], obj: str) -> Tuple[Box, ...]:
    """All boxes belonging to pattern ``obj`` (``obj_0``, ``obj_1_2``, ...)."""
    prefix = f"{obj}_"
    keys = sorted(key for key in boxes if key.startswith(prefix))
    return tuple(boxes[key] for key in keys)


def _indexed(boxes: Mapping[str, Box], obj: str, index: int) -> Box:
    instances = _instances(boxes, obj)
    if not instances:
        raise KeyError(f"no instances of pattern {obj!r}")
    if not 0 <= index < len(instances):
        raise IndexError(f"{obj}[{index}] is out of range ({len(instances)} instances)")
    return instances[index]


def resolve_target(target: str, boxes: Mapping[str, Box]) -> Vec3:
    """Resolve a target expression to a world point.

    Handles ``B.feature``, ``B[k].feature``, ``B[*].feature`` (aggregate over
    the pattern's bounding volume) and ``Avg(T1, T2, ...)``.
    """
    text = target.strip()
    average = _AVG.match(text)
    if average:
        args = [item.strip() for item in _split_args(average.group("args")) if item.strip()]
        if not args:
            raise ValueError("Avg() needs at least one argument")
        points = [resolve_target(arg, boxes) for arg in args]
        return tuple(  # type: ignore[return-value]
            sum(point[axis] for point in points) / len(points) for axis in range(3)
        )

    match = _TOKEN.match(text)
    if not match:
        raise ValueError(f"unparseable target: {target!r}")
    obj = match.group("obj")
    index = match.group("index")
    feature = match.group("feature")

    if index is None:
        if obj not in boxes:
            raise KeyError(f"unknown reference: {obj!r}")
        box = boxes[obj]
    elif index == "*":
        instances = _instances(boxes, obj)
        if not instances and obj in boxes:
            instances = (boxes[obj],)
        box = aggregate_box(instances)
    else:
        box = _indexed(boxes, obj, int(index))
    return feature_point(box, feature)


def _split_args(text: str) -> Iterable[str]:
    depth = 0
    current: list = []
    for char in text:
        if char in "([":
            depth += 1
        elif char in ")]":
            depth -= 1
        if char == "," and depth == 0:
            yield "".join(current)
            current = []
        else:
            current.append(char)
    yield "".join(current)


def apply_align(box: Box, clause: AlignClause, boxes: Mapping[str, Box]) -> Box:
    """Move ``box`` so its feature meets the target -- on the clause's axes only."""
    target = resolve_target(clause.target, boxes)
    source = feature_point(box, clause.this_feature)
    delta = [0.0, 0.0, 0.0]
    for axis in clause.axes:
        delta[axis] = target[axis] - source[axis]
    return box.translated(tuple(delta))  # type: ignore[arg-type]


def resolve_placement(
    box: Box,
    clauses: Sequence[AlignClause],
    boxes: Mapping[str, Box],
    offset: Vec3 = (0.0, 0.0, 0.0),
) -> Box:
    """Apply a chain of Align clauses in order, then slide by ``offset``.

    Later clauses never disturb axes an earlier clause did not lock, so the
    chain behaves like the incremental axis-locking the format describes.
    """
    current = box
    for clause in clauses:
        current = apply_align(current, clause, boxes)
    if any(offset):
        current = current.translated(offset)
    return current


def polar_position(
    box: Box,
    reference: Box,
    theta_degrees: float,
    dr: float = 0.0,
) -> Box:
    """Place ``box`` on ``reference``'s curved side at ``theta``, touching it.

    The engine computes the radius so the child just touches the parent: the
    parent's XY radius plus the child's own XY half-extent along the radial
    direction, plus the optional radial shift ``dr``. Z is left untouched.
    """
    theta = math.radians(theta_degrees)
    parent_radius = max(reference.size[0], reference.size[1]) / 2.0
    child_radius = max(box.size[0], box.size[1]) / 2.0
    radius = parent_radius + child_radius + dr
    center = (
        reference.center[0] + radius * math.cos(theta),
        reference.center[1] + radius * math.sin(theta),
        box.center[2],
    )
    return box.moved_to(center)
