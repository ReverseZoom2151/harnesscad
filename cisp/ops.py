"""CISP op set (v0) — the typed, agent-facing CAD operations.

These are the *mutating* operations the agent emits (measure/export are queries,
handled by the backend's query()/export(), not the op log). Every op is a frozen
dataclass with a stable ``OP`` tag for JSON (de)serialisation, so an op stream is
deterministic and hashable — the substrate for the ops-DAG (see state/opdag.py).

Sketch + constraint ops come first by design: the wedge is sketch/constraint/layout
assist, not one-shot solids (see HARNESS_BLUEPRINT.md sec.18 sequencing).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import ClassVar, Optional


@dataclass(frozen=True)
class Op:
    """Base class for all CISP operations."""

    OP: ClassVar[str] = "op"

    def to_dict(self) -> dict:
        d = {"op": self.OP}
        for k, v in self.__dict__.items():
            d[k] = list(v) if isinstance(v, tuple) else v
        return d


# --- sketch primitives -----------------------------------------------------
@dataclass(frozen=True)
class NewSketch(Op):
    OP: ClassVar[str] = "new_sketch"
    plane: str = "XY"


@dataclass(frozen=True)
class AddPoint(Op):
    OP: ClassVar[str] = "add_point"
    sketch: str = ""
    x: float = 0.0
    y: float = 0.0


@dataclass(frozen=True)
class AddLine(Op):
    OP: ClassVar[str] = "add_line"
    sketch: str = ""
    x1: float = 0.0
    y1: float = 0.0
    x2: float = 0.0
    y2: float = 0.0


@dataclass(frozen=True)
class AddCircle(Op):
    OP: ClassVar[str] = "add_circle"
    sketch: str = ""
    cx: float = 0.0
    cy: float = 0.0
    r: float = 1.0


@dataclass(frozen=True)
class AddRectangle(Op):
    OP: ClassVar[str] = "add_rectangle"
    sketch: str = ""
    x: float = 0.0
    y: float = 0.0
    w: float = 1.0
    h: float = 1.0


@dataclass(frozen=True)
class Constrain(Op):
    """A geometric/dimensional constraint. `kind` is one of the CONSTRAINT_DOF keys.

    `a`/`b` reference sketch entities; `value` is required for dimensional
    constraints (distance, radius). The stub solver reduces the sketch's DOF by
    CONSTRAINT_DOF[kind]; a real solver (planegcs/SolveSpace) replaces it later.
    """

    OP: ClassVar[str] = "constrain"
    kind: str = "coincident"
    a: str = ""
    b: Optional[str] = None
    value: Optional[float] = None


# --- features --------------------------------------------------------------
@dataclass(frozen=True)
class Extrude(Op):
    OP: ClassVar[str] = "extrude"
    sketch: str = ""
    distance: float = 1.0


@dataclass(frozen=True)
class Fillet(Op):
    OP: ClassVar[str] = "fillet"
    edges: tuple = ()
    radius: float = 1.0


@dataclass(frozen=True)
class Boolean(Op):
    OP: ClassVar[str] = "boolean"
    kind: str = "union"  # union | cut | intersect
    target: str = ""
    tool: str = ""


# --- extended mechanical features -----------------------------------------
@dataclass(frozen=True)
class Revolve(Op):
    """Revolve a sketch profile about an axis (OCCT BRepPrimAPI_MakeRevol).

    ``axis`` is a 6-tuple (ax, ay, az, bx, by, bz) giving two points that define
    the revolution axis in the sketch plane's local frame; ``angle`` is in
    degrees (360 = full solid of revolution).
    """

    OP: ClassVar[str] = "revolve"
    sketch: str = ""
    axis: tuple = (0.0, 0.0, 0.0, 0.0, 1.0, 0.0)
    angle: float = 360.0


@dataclass(frozen=True)
class Chamfer(Op):
    """A straight chamfer on solid edges (BRepFilletAPI_MakeChamfer).

    Distinct from :class:`Fillet` (which rounds); ``distance`` is the setback.
    """

    OP: ClassVar[str] = "chamfer"
    edges: tuple = ()
    distance: float = 1.0


@dataclass(frozen=True)
class Hole(Op):
    """A semantic, DFM-legible hole: cut a cylinder into a face/sketch datum.

    ``kind`` is a manufacturing intent tag ("simple" | "counterbore" |
    "countersink"); only "simple" is realised geometrically today. ``through``
    selects a through-all cut; otherwise ``depth`` bounds a blind hole.
    """

    OP: ClassVar[str] = "hole"
    face_or_sketch: str = ""
    x: float = 0.0
    y: float = 0.0
    diameter: float = 1.0
    depth: Optional[float] = None
    through: bool = True
    kind: str = "simple"


@dataclass(frozen=True)
class Shell(Op):
    """Hollow a solid to a wall ``thickness`` (OCCT MakeThickSolid / cq.shell).

    ``faces`` names the faces to remove (open); empty removes a default face.
    """

    OP: ClassVar[str] = "shell"
    faces: tuple = ()
    thickness: float = 1.0


@dataclass(frozen=True)
class Draft(Op):
    """Apply a draft (taper) angle to faces relative to a neutral plane.

    Real drafting is not yet wired on the current CadQuery/OCCT build, so the
    cadquery backend returns a typed 'not-yet-supported' diagnostic rather than
    fabricating geometry; the stub tracks it as a first-class feature.
    """

    OP: ClassVar[str] = "draft"
    faces: tuple = ()
    angle: float = 0.0
    neutral_plane: str = ""


@dataclass(frozen=True)
class Loft(Op):
    """Loft a solid through an ordered list of sketch profiles."""

    OP: ClassVar[str] = "loft"
    sketches: tuple = ()
    ruled: bool = False


@dataclass(frozen=True)
class Sweep(Op):
    """Sweep a sketch profile along a path sketch."""

    OP: ClassVar[str] = "sweep"
    sketch: str = ""
    path: str = ""


@dataclass(frozen=True)
class LinearPattern(Op):
    """Replicate a feature ``count`` times along ``direction`` at ``spacing``."""

    OP: ClassVar[str] = "linear_pattern"
    feature: str = ""
    direction: tuple = (1.0, 0.0, 0.0)
    count: int = 2
    spacing: float = 1.0


@dataclass(frozen=True)
class CircularPattern(Op):
    """Replicate a feature ``count`` times about ``axis`` spanning ``angle``."""

    OP: ClassVar[str] = "circular_pattern"
    feature: str = ""
    axis: tuple = (0.0, 0.0, 0.0, 0.0, 0.0, 1.0)
    count: int = 4
    angle: float = 360.0


@dataclass(frozen=True)
class Mirror(Op):
    """Mirror a feature/body across a named plane (XY | XZ | YZ)."""

    OP: ClassVar[str] = "mirror"
    feature_or_body: str = ""
    plane: str = "XZ"


# DOF removed per constraint kind (placeholder for a real constraint solver).
CONSTRAINT_DOF = {
    "coincident": 2,
    "horizontal": 1,
    "vertical": 1,
    "parallel": 1,
    "perpendicular": 1,
    "distance": 1,
    "radius": 1,
    "equal": 1,
}

# DOF contributed by each sketch primitive.
PRIMITIVE_DOF = {"point": 2, "line": 4, "circle": 3, "rectangle": 4}

_REGISTRY = {
    c.OP: c
    for c in (
        NewSketch, AddPoint, AddLine, AddCircle, AddRectangle,
        Constrain, Extrude, Fillet, Boolean,
        Revolve, Chamfer, Hole, Shell, Draft,
        Loft, Sweep, LinearPattern, CircularPattern, Mirror,
    )
}


def parse_op(d: dict) -> Op:
    """Reconstruct an Op from its dict form (the inverse of Op.to_dict).

    JSON has no tuples, so any dataclass field whose default is a tuple (edges,
    axis, faces, sketches, direction, ...) is re-tupled from its list form to
    round-trip cleanly and keep ops hashable.
    """
    import dataclasses

    d = dict(d)
    tag = d.pop("op")
    cls = _REGISTRY[tag]
    tuple_fields = {
        f.name for f in dataclasses.fields(cls)
        if isinstance(f.default, tuple)
    }
    for k in tuple_fields:
        if k in d and isinstance(d[k], list):
            d[k] = tuple(d[k])
    return cls(**d)


def canonical_json(op: Op) -> str:
    """Deterministic serialisation of an op (sorted keys) for content hashing."""
    return json.dumps(op.to_dict(), sort_keys=True, separators=(",", ":"))
