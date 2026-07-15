"""CISP op set (v0) — the typed, agent-facing CAD operations.

These are the *mutating* operations the agent emits (measure/export are queries,
handled by the backend's query()/export(), not the op log). Every op is a frozen
dataclass with a stable ``OP`` tag for JSON (de)serialisation, so an op stream is
deterministic and hashable — the substrate for the ops-DAG (see state/opdag.py).

Sketch + constraint ops come first by design: the wedge is sketch/constraint/layout
assist, not one-shot solids (see docs/blueprint.md sec.18 sequencing).
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
class AddArc(Op):
    """A circular arc entity: centre (cx, cy), radius r, from ``start`` to ``end``
    degrees (CCW). An arc is a boundary curve, not a closed region on its own — it
    contributes its sampled points to the sketch's polyline chain exactly as a run
    of ``AddLine``s would, so a profile of lines+arcs closes into one region. A
    backend whose profile system cannot chain a curved segment refuses it.
    """

    OP: ClassVar[str] = "add_arc"
    sketch: str = ""
    cx: float = 0.0
    cy: float = 0.0
    r: float = 1.0
    start: float = 0.0
    end: float = 90.0


@dataclass(frozen=True)
class AddEllipse(Op):
    """A closed ellipse entity centred at (cx, cy), semi-axes rx/ry, rotated
    ``rotation`` degrees CCW about its centre. Self-closing (its own region), like
    a circle."""

    OP: ClassVar[str] = "add_ellipse"
    sketch: str = ""
    cx: float = 0.0
    cy: float = 0.0
    rx: float = 1.0
    ry: float = 0.5
    rotation: float = 0.0


@dataclass(frozen=True)
class AddPolygon(Op):
    """A closed polygon entity. ``points`` is a FLAT tuple of coordinates
    (x0, y0, x1, y1, ...) — kept flat (not a tuple of pairs) so the op stays
    hashable and round-trips through JSON without a nested-list snag. Three or
    more vertices form a closed region (the closing edge is implied)."""

    OP: ClassVar[str] = "add_polygon"
    sketch: str = ""
    points: tuple = ()


@dataclass(frozen=True)
class AddSpline(Op):
    """A spline entity through control/interpolation ``points`` (a FLAT tuple
    (x0, y0, x1, y1, ...)). ``closed`` makes it a closed region; otherwise it is a
    boundary curve that contributes its sampled points to the sketch's polyline
    chain (like :class:`AddArc`). Backends that cannot express a freeform curve in
    their profile system refuse it."""

    OP: ClassVar[str] = "add_spline"
    sketch: str = ""
    points: tuple = ()
    closed: bool = False


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
    """Round solid edges.

    ``edges`` is a tuple of CadQuery selector strings naming the edges to round
    (``("|Z",)`` = the four vertical edges, ``(">Z",)`` = the top face's edges,
    ``("|Z and >Y",)``, ...). An EMPTY tuple means "every edge" — the historical
    behaviour, kept so existing op streams are unchanged. See
    :mod:`harnesscad.domain.geometry.topology.selector_dsl` for the grammar.
    """

    OP: ClassVar[str] = "fillet"
    edges: tuple = ()
    radius: float = 1.0


@dataclass(frozen=True)
class Boolean(Op):
    OP: ClassVar[str] = "boolean"
    kind: str = "union"  # union | cut | intersect
    target: str = ""
    tool: str = ""


@dataclass(frozen=True)
class Primitive(Op):
    """A parametric solid primitive placed at the origin (before any transform).

    ``shape`` selects the family; only the fields that shape uses are read::

        box       -> dx, dy, dz          (axis-aligned box, corner at origin)
        sphere    -> r
        cylinder  -> r, h                 (axis along +Z)
        cone      -> r, r2, h             (r at base, r2 at top; r2=0 = a point)
        torus     -> r, r2                (r = major radius, r2 = minor/tube)
        wedge     -> dx, dy, dz           (right-triangular prism)

    Unused fields are ignored. Every geometry kernel already carries these
    builders internally (holes use a cylinder, countersinks a cone), so exposing
    them as one op closes the "cannot make a sphere from its own ops" gap. A
    backend that cannot build ``shape`` refuses with a typed ``unsupported-op`` —
    the same discipline Shell / Draft already use.
    """

    OP: ClassVar[str] = "primitive"
    shape: str = "box"        # box | sphere | cylinder | cone | torus | wedge
    dx: float = 1.0
    dy: float = 1.0
    dz: float = 1.0
    r: float = 1.0
    r2: float = 0.0
    h: float = 1.0


@dataclass(frozen=True)
class Split(Op):
    """Section the current solid by an infinite plane; keep one or both halves.

    ``plane`` is a named datum (XY | XZ | YZ) offset by ``offset`` along its
    normal. ``keep`` is 'positive' (the +normal side), 'negative', or 'both'. It
    lowers to a boolean cut against a half-space, so every solid backend with a
    boolean path can express it.
    """

    OP: ClassVar[str] = "split"
    plane: str = "XY"
    offset: float = 0.0
    keep: str = "positive"   # positive | negative | both


@dataclass(frozen=True)
class Thicken(Op):
    """Grow / shrink a solid by a wall ``thickness`` (an offset-solid).

    ``faces`` optionally selects the surfaces to thicken (empty = the whole outer
    surface). ``thickness`` may be negative (inward). ``both`` thickens
    symmetrically about the surface. Exact in an SDF kernel (a field offset) and
    in OCCT (``MakeThickSolid`` / ``makeOffsetShape``); a mesh-boolean kernel with
    a 2D-only offset refuses it honestly.
    """

    OP: ClassVar[str] = "thicken"
    faces: tuple = ()
    thickness: float = 1.0
    both: bool = False


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
    ``distance2`` optionally makes the chamfer asymmetric — it maps onto
    CadQuery's ``Workplane.chamfer(length, length2=None)`` second setback.
    """

    OP: ClassVar[str] = "chamfer"
    edges: tuple = ()
    distance: float = 1.0
    distance2: Optional[float] = None


@dataclass(frozen=True)
class Hole(Op):
    """A semantic, DFM-legible hole: cut a cylinder into a face/sketch datum.

    ``kind`` is a manufacturing intent tag ("simple" | "counterbore" |
    "countersink"). ``through`` selects a through-all cut; otherwise ``depth``
    bounds a blind hole.

    Counterbore/countersink carry the extra stepped-profile dimensions that
    CadQuery's ``Workplane.cboreHole(diameter, cboreDiameter, cboreDepth)`` and
    ``Workplane.cskHole(diameter, cskDiameter, cskAngle)`` require. When they are
    left as ``None`` a backend may fall back to a conventional ratio, but callers
    that care about the exact stepped profile should set them explicitly.
    """

    OP: ClassVar[str] = "hole"
    face_or_sketch: str = ""
    x: float = 0.0
    y: float = 0.0
    diameter: float = 1.0
    depth: Optional[float] = None
    through: bool = True
    kind: str = "simple"
    cbore_diameter: Optional[float] = None
    cbore_depth: Optional[float] = None
    csk_diameter: Optional[float] = None
    csk_angle: float = 82.0


@dataclass(frozen=True)
class Shell(Op):
    """Hollow a solid to a wall ``thickness`` (OCCT MakeThickSolid / cq.shell).

    ``faces`` names the faces to remove (open) as CadQuery selector strings (e.g.
    ``(">Z",)``, ``(">Z or <X",)``); empty defaults to removing the top face
    (``">Z"``).

    ``thickness`` is always a POSITIVE wall thickness and always hollows INWARD:
    CadQuery's ``Workplane.shell`` documents "Negative values shell inwards,
    positive values shell outwards", so a backend must pass ``-thickness``. A
    shell must never grow the part's outer bounding box.

    ``kind`` is CadQuery's join kind, ``"arc"`` or ``"intersection"``.
    """

    OP: ClassVar[str] = "shell"
    faces: tuple = ()
    thickness: float = 1.0
    kind: str = "arc"


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
    """Loft a solid through an ordered list of sketch profiles.

    ``offsets`` gives each profile's offset along its own sketch-plane normal (so
    two profiles on the same plane can still be lofted, which is the usual case).
    When shorter than ``sketches`` the missing entries are 0. Profiles that all
    end up coincident give a zero-volume loft and are rejected as degenerate.
    """

    OP: ClassVar[str] = "loft"
    sketches: tuple = ()
    ruled: bool = False
    offsets: tuple = ()


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


# --- assembly --------------------------------------------------------------
@dataclass(frozen=True)
class AddInstance(Op):
    """Place a part instance into the assembly with a rigid transform.

    ``part`` names a prior body / sub-part (a feature id, a previously placed
    instance id, or the ``solid``/``body``/``last`` alias for the current
    combined solid). The instance is positioned by a translation
    (``x``/``y``/``z``, model units) and an intrinsic X-Y-Z rotation
    (``rx``/``ry``/``rz``, degrees). It becomes a *part* in ``query('assembly')``.
    """

    OP: ClassVar[str] = "add_instance"
    part: str = ""
    x: float = 0.0
    y: float = 0.0
    z: float = 0.0
    rx: float = 0.0
    ry: float = 0.0
    rz: float = 0.0


@dataclass(frozen=True)
class Mate(Op):
    """A joint (mate) coupling two placed instances / refs.

    ``kind`` is one of the :data:`checks_assembly.MATE_DOF` names
    (rigid/revolute/slider/cylindrical/planar and their aliases); it fixes how
    many rigid-body DOF the mate removes. ``a``/``b`` reference the two
    instances (or bodies) being coupled; ``value`` is an optional mate parameter
    (e.g. a target angle / offset), unused by the pure DOF count.
    """

    OP: ClassVar[str] = "mate"
    kind: str = "rigid"
    a: str = ""
    b: str = ""
    value: Optional[float] = None


@dataclass(frozen=True)
class SetParam(Op):
    """Editability-first primitive: mutate a prior op's parameter, then rebuild.

    ``target`` is the 0-based index of a previously applied op (in application
    order); ``param`` is the name of one of that op's dataclass fields; ``value``
    is the replacement (int / float / str). Applying a ``SetParam`` rewrites the
    referenced op in the recorded model and deterministically replays the whole
    op stream so every downstream feature is regenerated from the edited value.
    """

    OP: ClassVar[str] = "set_param"
    target: int = 0
    param: str = ""
    value: Optional[object] = None


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

# DOF contributed by each sketch primitive. The curve entities (arc/ellipse/
# spline/polygon) carry a nominal count — PRIMITIVE_DOF is an explicit placeholder
# for a real constraint solver, and these are not pinned by any oracle.
PRIMITIVE_DOF = {
    "point": 2, "line": 4, "circle": 3, "rectangle": 4,
    "arc": 5, "ellipse": 5, "polygon": 4, "spline": 4,
}

_REGISTRY = {
    c.OP: c
    for c in (
        NewSketch, AddPoint, AddLine, AddCircle, AddRectangle,
        AddArc, AddEllipse, AddPolygon, AddSpline,
        Constrain, Extrude, Fillet, Boolean,
        Primitive, Split, Thicken,
        Revolve, Chamfer, Hole, Shell, Draft,
        Loft, Sweep, LinearPattern, CircularPattern, Mirror,
        AddInstance, Mate, SetParam,
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


def edit_oplog(oplog, op: "SetParam"):
    """Apply a :class:`SetParam` edit to a recorded op log (pure, no side effects).

    ``oplog`` is the list of previously applied (mutating) ops. Returns
    ``(new_log, None)`` — a *copy* of ``oplog`` with the targeted op's ``param``
    replaced by ``value`` — or ``(None, (code, msg, where))`` describing why the
    edit is invalid (unknown target index or unknown param), so the caller can
    block-and-correct. Tuple-valued fields re-tuple a list value to stay hashable.
    """
    import dataclasses

    try:
        idx = int(op.target)
    except (TypeError, ValueError):
        return None, ("bad-ref", f"invalid SetParam target {op.target!r}", None)
    if idx < 0 or idx >= len(oplog):
        return None, ("bad-ref",
                      f"SetParam target index {idx} out of range "
                      f"(0..{len(oplog) - 1})", None)
    old = oplog[idx]
    fields = {f.name: f for f in dataclasses.fields(old)}
    if op.param not in fields:
        return None, ("bad-param",
                      f"op '{old.OP}' has no parameter '{op.param}'", op.param)
    value = op.value
    if isinstance(fields[op.param].default, tuple) and isinstance(value, list):
        value = tuple(value)
    new_log = list(oplog)
    new_log[idx] = dataclasses.replace(old, **{op.param: value})
    return new_log, None
