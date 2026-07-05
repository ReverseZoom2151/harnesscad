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
    )
}


def parse_op(d: dict) -> Op:
    """Reconstruct an Op from its dict form (the inverse of Op.to_dict)."""
    d = dict(d)
    tag = d.pop("op")
    cls = _REGISTRY[tag]
    if "edges" in d and isinstance(d["edges"], list):
        d["edges"] = tuple(d["edges"])
    return cls(**d)


def canonical_json(op: Op) -> str:
    """Deterministic serialisation of an op (sorted keys) for content hashing."""
    return json.dumps(op.to_dict(), sort_keys=True, separators=(",", ":"))
