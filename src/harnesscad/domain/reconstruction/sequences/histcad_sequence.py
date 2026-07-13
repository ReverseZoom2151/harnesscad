"""HistCAD flat, constraint-aware modeling-sequence schema.

This implements the deterministic *representation* introduced by HistCAD
("Geometrically Constrained Parametric History-based CAD Dataset"): a flat,
non-hierarchical sketch-and-extrude sequence that (unlike DeepCAD / Text2CAD)
stores sketch primitives as an *unordered set* and encodes ten explicit
geometric constraints, with loops/faces inferred from connectivity rather than
declared by a face-loop hierarchy.

Only the deterministic schema and its structural operations live here (no
learned model). A :class:`ModelingSequence` bundles, in procedural order:

  * a :class:`Sketch` = a :class:`SketchPlane` (translation + Euler angles),
    an unordered tuple of primitives (:class:`Line`, :class:`Circle`,
    :class:`Arc`), and a tuple of :class:`Constraint` records;
  * an :class:`Extrusion` (direction + length), optionally *rotated* (rotation
    axis position/orientation + start/end angle) as in HistCAD-Industrial;
  * a :class:`BooleanOp` (create / join / subtract / intersect).

The module also implements the paper's **symmetric-difference primitive
deduplication** (``P = triangle_{i} triangle_{L in df_i} L``): flatten the
loops of every face into one collection and keep only edges that occur an ODD
number of times, which removes shared duplicate boundaries while preserving
outer contours and internal voids.

Everything is stdlib-only and deterministic (no wall clock, no RNG); dict
(de)serialisation round-trips exactly.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Sequence, Tuple

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
#: The ten geometric constraint types HistCAD encodes explicitly.
CONSTRAINT_TYPES: Tuple[str, ...] = (
    "coincident", "parallel", "perpendicular", "horizontal", "vertical",
    "tangent", "equal", "concentric", "fix", "normal",
)

#: The four Boolean operations HistCAD supports.
BOOLEAN_OPS: Tuple[str, ...] = ("create", "join", "subtract", "intersect")

#: Quantisation used for canonical edge keys (endpoint matching / dedup).
_QUANT = 1_000_000  # 1e-6 tolerance


def _q(v: float) -> int:
    return int(round(float(v) * _QUANT))


# ---------------------------------------------------------------------------
# Primitives (compact parametric form)
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class Line:
    """A line segment given by its two endpoints."""

    x1: float
    y1: float
    x2: float
    y2: float

    kind = "line"

    def endpoints(self) -> Tuple[Tuple[float, float], Tuple[float, float]]:
        return ((self.x1, self.y1), (self.x2, self.y2))

    def canonical_key(self) -> Tuple:
        a = (_q(self.x1), _q(self.y1))
        b = (_q(self.x2), _q(self.y2))
        return ("line",) + tuple(sorted((a, b)))

    def to_dict(self) -> Dict:
        return {"kind": "line", "x1": self.x1, "y1": self.y1,
                "x2": self.x2, "y2": self.y2}


@dataclass(frozen=True)
class Circle:
    """A full circle given by centre and radius (self-closing loop)."""

    cx: float
    cy: float
    r: float

    kind = "circle"

    def endpoints(self):  # circles have no free endpoints
        return ()

    def canonical_key(self) -> Tuple:
        return ("circle", _q(self.cx), _q(self.cy), _q(abs(self.r)))

    def to_dict(self) -> Dict:
        return {"kind": "circle", "cx": self.cx, "cy": self.cy, "r": self.r}


@dataclass(frozen=True)
class Arc:
    """An arc given by start, mid and end points."""

    xs: float
    ys: float
    xm: float
    ym: float
    xe: float
    ye: float

    kind = "arc"

    def endpoints(self):
        return ((self.xs, self.ys), (self.xe, self.ye))

    def canonical_key(self) -> Tuple:
        a = (_q(self.xs), _q(self.ys))
        e = (_q(self.xe), _q(self.ye))
        m = (_q(self.xm), _q(self.ym))
        # mid stays attached; endpoints order-normalised
        ends = tuple(sorted((a, e)))
        return ("arc",) + ends + (m,)

    def to_dict(self) -> Dict:
        return {"kind": "arc", "xs": self.xs, "ys": self.ys, "xm": self.xm,
                "ym": self.ym, "xe": self.xe, "ye": self.ye}


_PRIM_BUILDERS = {
    "line": lambda d: Line(d["x1"], d["y1"], d["x2"], d["y2"]),
    "circle": lambda d: Circle(d["cx"], d["cy"], d["r"]),
    "arc": lambda d: Arc(d["xs"], d["ys"], d["xm"], d["ym"], d["xe"], d["ye"]),
}


def primitive_from_dict(d: Dict):
    return _PRIM_BUILDERS[d["kind"]](d)


# ---------------------------------------------------------------------------
# Sketch plane, constraints, sketch
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class SketchPlane:
    """Locates/orients a sketch: translation vector + ZYX Euler angles (rad)."""

    tx: float = 0.0
    ty: float = 0.0
    tz: float = 0.0
    rx: float = 0.0
    ry: float = 0.0
    rz: float = 0.0

    def to_dict(self) -> Dict:
        return {"tx": self.tx, "ty": self.ty, "tz": self.tz,
                "rx": self.rx, "ry": self.ry, "rz": self.rz}

    @staticmethod
    def from_dict(d: Dict) -> "SketchPlane":
        return SketchPlane(d.get("tx", 0.0), d.get("ty", 0.0), d.get("tz", 0.0),
                           d.get("rx", 0.0), d.get("ry", 0.0), d.get("rz", 0.0))


@dataclass(frozen=True)
class Constraint:
    """An explicit geometric constraint over primitive indices.

    ``refs`` are indices into the owning sketch's ``primitives`` tuple.
    """

    ctype: str
    refs: Tuple[int, ...]

    def __post_init__(self):
        if self.ctype not in CONSTRAINT_TYPES:
            raise ValueError(f"unknown constraint type: {self.ctype!r}")

    def to_dict(self) -> Dict:
        return {"ctype": self.ctype, "refs": list(self.refs)}

    @staticmethod
    def from_dict(d: Dict) -> "Constraint":
        return Constraint(d["ctype"], tuple(d["refs"]))


@dataclass(frozen=True)
class Sketch:
    """A flat sketch: plane + unordered primitives + explicit constraints."""

    plane: SketchPlane
    primitives: Tuple = ()
    constraints: Tuple[Constraint, ...] = ()

    def to_dict(self) -> Dict:
        return {
            "plane": self.plane.to_dict(),
            "primitives": [p.to_dict() for p in self.primitives],
            "constraints": [c.to_dict() for c in self.constraints],
        }

    @staticmethod
    def from_dict(d: Dict) -> "Sketch":
        return Sketch(
            SketchPlane.from_dict(d["plane"]),
            tuple(primitive_from_dict(p) for p in d.get("primitives", ())),
            tuple(Constraint.from_dict(c) for c in d.get("constraints", ())),
        )


# ---------------------------------------------------------------------------
# Extrusion + boolean
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class Extrusion:
    """Extrusion by direction + length; optionally a rotated extrusion.

    A *rotated* extrusion (HistCAD-Industrial) is defined by a rotation axis
    (position + orientation) and a start/end angle.
    """

    dx: float
    dy: float
    dz: float
    length: float
    rotated: bool = False
    axis_pos: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    axis_dir: Tuple[float, float, float] = (0.0, 0.0, 1.0)
    angle_start: float = 0.0
    angle_end: float = 0.0

    def to_dict(self) -> Dict:
        return {
            "dx": self.dx, "dy": self.dy, "dz": self.dz, "length": self.length,
            "rotated": self.rotated,
            "axis_pos": list(self.axis_pos), "axis_dir": list(self.axis_dir),
            "angle_start": self.angle_start, "angle_end": self.angle_end,
        }

    @staticmethod
    def from_dict(d: Dict) -> "Extrusion":
        return Extrusion(
            d["dx"], d["dy"], d["dz"], d["length"],
            d.get("rotated", False),
            tuple(d.get("axis_pos", (0.0, 0.0, 0.0))),
            tuple(d.get("axis_dir", (0.0, 0.0, 1.0))),
            d.get("angle_start", 0.0), d.get("angle_end", 0.0),
        )


@dataclass(frozen=True)
class Feature:
    """One procedural step: sketch + extrusion + boolean combination."""

    sketch: Sketch
    extrusion: Extrusion
    boolean: str = "create"

    def __post_init__(self):
        if self.boolean not in BOOLEAN_OPS:
            raise ValueError(f"unknown boolean op: {self.boolean!r}")

    def to_dict(self) -> Dict:
        return {"sketch": self.sketch.to_dict(),
                "extrusion": self.extrusion.to_dict(),
                "boolean": self.boolean}

    @staticmethod
    def from_dict(d: Dict) -> "Feature":
        return Feature(Sketch.from_dict(d["sketch"]),
                       Extrusion.from_dict(d["extrusion"]),
                       d.get("boolean", "create"))


@dataclass(frozen=True)
class ModelingSequence:
    """An ordered list of features = one HistCAD modeling sequence."""

    features: Tuple[Feature, ...] = ()

    def to_dict(self) -> Dict:
        return {"features": [f.to_dict() for f in self.features]}

    @staticmethod
    def from_dict(d: Dict) -> "ModelingSequence":
        return ModelingSequence(tuple(Feature.from_dict(f)
                                      for f in d.get("features", ())))


# ---------------------------------------------------------------------------
# Symmetric-difference primitive deduplication
# ---------------------------------------------------------------------------
def symmetric_difference(loops: Sequence[Sequence]) -> Tuple:
    """Return primitives that occur an ODD number of times across ``loops``.

    Implements HistCAD's flattening operator: shared boundary edges (which
    appear an even number of times when adjacent loops are combined) are
    removed, while unique outer contours and internal voids are kept. First
    occurrence order is preserved for determinism.
    """
    counts: Dict[Tuple, int] = {}
    first: Dict[Tuple, object] = {}
    order: List[Tuple] = []
    for loop in loops:
        for prim in loop:
            key = prim.canonical_key()
            if key not in counts:
                counts[key] = 0
                first[key] = prim
                order.append(key)
            counts[key] += 1
    return tuple(first[k] for k in order if counts[k] % 2 == 1)


def flatten_faces(faces: Sequence[Sequence[Sequence]]) -> Tuple:
    """Flatten a face-loop hierarchy into one deduplicated primitive set.

    ``faces`` is a sequence of faces, each a sequence of loops, each a
    sequence of primitives. Applies :func:`symmetric_difference` across every
    loop of every face (the outer aggregation of the paper's operator).
    """
    all_loops: List[Sequence] = []
    for face in faces:
        for loop in face:
            all_loops.append(loop)
    return symmetric_difference(all_loops)


# ---------------------------------------------------------------------------
# Compactness (token) estimate
# ---------------------------------------------------------------------------
#: parameter counts per primitive kind, for a deterministic token estimate.
_PARAM_TOKENS = {"line": 4, "circle": 3, "arc": 6}


def token_estimate(seq: ModelingSequence, include_constraints: bool = True) -> int:
    """A deterministic token-count proxy for a modeling sequence.

    Counts one token per structural keyword plus one per numeric parameter,
    mirroring the paper's observation that constraints add overhead
    (HistCAD w/c is longer than w/o c). Not a real tokenizer, but monotone in
    the same quantities, so it is a valid relative compactness metric.
    """
    total = 0
    for feat in seq.features:
        total += 1  # sketch keyword
        total += 6  # plane params
        for p in feat.sketch.primitives:
            total += 1 + _PARAM_TOKENS[p.kind]  # kind token + params
        if include_constraints:
            for c in feat.sketch.constraints:
                total += 1 + len(c.refs)
        total += 1 + 4  # extrusion keyword + dir/length
        if feat.extrusion.rotated:
            total += 8  # axis pos/dir + start/end angle
        total += 1  # boolean token
    return total
