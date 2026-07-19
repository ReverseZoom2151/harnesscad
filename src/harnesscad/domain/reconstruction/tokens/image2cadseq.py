"""Simplified gallery ("Sim-Gallery") DSL and its vectorized CAD-sequence
representation.

The pipeline reverse-engineers a **CAD sequence** from a product image. The
deterministic, network-agnostic core is a compact operation-based
representation that is distinct from the flat 16-slot and 19-slot schemes:

* **Seven operation types** ``t in {0..6}`` (variable ``t``)::

      0  add_sketch(I)              choose a canonical sketch plane
      1  add_line(x, y)             line to an endpoint
      2  add_arc(x, y, alpha)       arc to an endpoint, sweep angle alpha
      3  add_circle(x, y, r)        circle at centre (x, y), radius r
      4  add_extrude(d)             extrude the current profile depth d
      5  SOP                        start-of-program marker (non-CAD)
      6  EOP                        end-of-program / padding marker (non-CAD)

  This differs from the six-type scheme (SOL/Line/Arc/Circle/Ext/EOS) both in
  vocabulary (explicit ``add_sketch`` plane op, SOP/EOP rather than SOL/EOS) and
  in the extrude op (a single signed depth + boolean, not an 11-slot plane
  transform).

* **A 7-dimensional operation vector** ``[t, I, x, y, alpha, r, d]`` (the
  reduced form; the three auxiliary variables ``[I]`` profile-index, ``O``
  boolean, ``s`` scale are held at defaults 0, 3, 10). The whole program is a
  fixed ``Nc x 7`` **feature matrix**, padded with ``EOP`` to a maximum length
  (``Nc = 10``).

* **Start-point elision** -- the start point of a ``Line``/``Arc`` is *not*
  stored; it is inherited from the endpoint of the preceding curve (origin
  ``(0, 0)`` for the first curve). Only the endpoint is kept, making the
  representation more compact.

* **Arc-centre reconstruction** -- the ``add_arc`` DSL call needs a
  *centre* point, but the vector stores the *endpoint* + sweep angle. Parsing
  recovers the centre from (start, end, sweep) by circle geometry.

* **Quantisation** -- continuous values are confined to a subset of ``[-1, 1]``
  and divided into 256 equal segments (8-bit ``0..255``); the sweep angle is
  multiplied by 180 on interpretation; unused slots hold the sentinel ``-1``.

Pure and deterministic; the learned image encoder and autoencoder are out of scope.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

# --- operation vocabulary (variable t) -------------------------------------
ADD_SKETCH = 0
ADD_LINE = 1
ADD_ARC = 2
ADD_CIRCLE = 3
ADD_EXTRUDE = 4
SOP = 5   # start of program
EOP = 6   # end of program / padding

OP_NAMES: dict[int, str] = {
    ADD_SKETCH: "add_sketch",
    ADD_LINE: "add_line",
    ADD_ARC: "add_arc",
    ADD_CIRCLE: "add_circle",
    ADD_EXTRUDE: "add_extrude",
    SOP: "SOP",
    EOP: "EOP",
}
N_OP_TYPES = 7

# The three canonical sketch planes (variable I in {0, 1, 2}).
PLANES: tuple[str, ...] = ("XY", "XZ", "YZ")

# Boolean operations (variable O in {0, 1, 2, 3}).
BOOLEANS: tuple[str, ...] = ("join", "cut", "intersect", "add")

# 7-dimensional operation-vector slot layout.
SLOTS: tuple[str, ...] = ("t", "I", "x", "y", "alpha", "r", "d")
SLOT_INDEX: dict[str, int] = {name: i for i, name in enumerate(SLOTS)}
VEC_LEN = len(SLOTS)  # == 7

UNUSED = -1.0

# Fixed maximum CAD-program length (Nc = 10).
NC_DEFAULT = 10

# Quantisation: 256 equal segments -> 8-bit integers 0..255.
N_QUANT_LEVELS = 256
UNUSED_LEVEL = -1  # integer sentinel for unused slots after quantisation

# Default auxiliary variables: [I]=0, O=3 ("add"), s=10.
DEFAULT_PROFILE_INDEX = 0
DEFAULT_BOOLEAN = 3
DEFAULT_SCALE = 10


# --- symbolic operations ---------------------------------------------------
@dataclass(frozen=True)
class GalleryOp:
    """A single Sim-Gallery operation.

    Only the fields relevant to ``type`` are meaningful; the rest hold
    :data:`UNUSED`. ``plane`` is the canonical-plane index for ``add_sketch``;
    ``boolean`` is the boolean-op index for ``add_extrude``.
    """

    type: int
    x: float = UNUSED
    y: float = UNUSED
    alpha: float = UNUSED
    r: float = UNUSED
    d: float = UNUSED
    plane: int = UNUSED
    boolean: int = DEFAULT_BOOLEAN

    def __post_init__(self):
        if self.type not in OP_NAMES:
            raise ValueError(f"unknown operation type: {self.type!r}")


def add_sketch(plane: int = 0) -> GalleryOp:
    if plane not in (0, 1, 2):
        raise ValueError("plane index must be one of 0 (XY), 1 (XZ), 2 (YZ)")
    return GalleryOp(ADD_SKETCH, plane=plane)


def add_line(x: float, y: float) -> GalleryOp:
    return GalleryOp(ADD_LINE, x=float(x), y=float(y))


def add_arc(x: float, y: float, alpha: float) -> GalleryOp:
    """Arc to endpoint ``(x, y)`` with sweep ``alpha`` (stored in [-1, 1]; the
    physical sweep angle in degrees is ``alpha * 180``)."""
    return GalleryOp(ADD_ARC, x=float(x), y=float(y), alpha=float(alpha))


def add_circle(x: float, y: float, r: float) -> GalleryOp:
    return GalleryOp(ADD_CIRCLE, x=float(x), y=float(y), r=float(r))


def add_extrude(d: float, boolean: int = DEFAULT_BOOLEAN) -> GalleryOp:
    if boolean not in (0, 1, 2, 3):
        raise ValueError("boolean op must be one of 0..3")
    return GalleryOp(ADD_EXTRUDE, d=float(d), boolean=boolean)


# --- raw (unquantised) 7-vector <-> op -------------------------------------
def to_vector7(op: GalleryOp) -> tuple[float, ...]:
    """Pack an op into ``[t, I, x, y, alpha, r, d]`` (unused slots = -1)."""
    vec = [UNUSED] * VEC_LEN
    vec[SLOT_INDEX["t"]] = float(op.type)
    if op.type == ADD_SKETCH:
        vec[SLOT_INDEX["I"]] = float(op.plane)
    elif op.type == ADD_LINE:
        vec[SLOT_INDEX["x"]] = op.x
        vec[SLOT_INDEX["y"]] = op.y
    elif op.type == ADD_ARC:
        vec[SLOT_INDEX["x"]] = op.x
        vec[SLOT_INDEX["y"]] = op.y
        vec[SLOT_INDEX["alpha"]] = op.alpha
    elif op.type == ADD_CIRCLE:
        vec[SLOT_INDEX["x"]] = op.x
        vec[SLOT_INDEX["y"]] = op.y
        vec[SLOT_INDEX["r"]] = op.r
    elif op.type == ADD_EXTRUDE:
        vec[SLOT_INDEX["d"]] = op.d
    return tuple(vec)


# --- quantisation ----------------------------------------------------------
def quantize(value: float, low: float, high: float,
             levels: int = N_QUANT_LEVELS) -> int:
    """Quantise a continuous value in ``[low, high]`` to ``0..levels-1``."""
    if high <= low:
        raise ValueError("high must exceed low")
    clamped = min(high, max(low, value))
    return int(round((clamped - low) / (high - low) * (levels - 1)))


def dequantize(level: int, low: float, high: float,
               levels: int = N_QUANT_LEVELS) -> float:
    """Inverse of :func:`quantize`."""
    return low + level * (high - low) / (levels - 1)


# Per-slot continuous ranges.
_SLOT_RANGE: dict[str, tuple[float, float]] = {
    "x": (-1.0, 1.0),
    "y": (-1.0, 1.0),
    "alpha": (-1.0, 1.0),
    "r": (0.0, 1.0),
    "d": (-1.0, 1.0),
}


def quantize_vector7(vec: tuple[float, ...]) -> tuple[int, ...]:
    """Quantise a raw 7-vector: discrete ``t``/``I`` stay integral, continuous
    slots become ``0..255``, unused slots become the sentinel ``-1``."""
    out = [UNUSED_LEVEL] * VEC_LEN
    out[SLOT_INDEX["t"]] = int(round(vec[SLOT_INDEX["t"]]))
    iv = vec[SLOT_INDEX["I"]]
    out[SLOT_INDEX["I"]] = UNUSED_LEVEL if iv == UNUSED else int(round(iv))
    for name, (low, high) in _SLOT_RANGE.items():
        j = SLOT_INDEX[name]
        out[j] = UNUSED_LEVEL if vec[j] == UNUSED else quantize(vec[j], low, high)
    return tuple(out)


def dequantize_vector7(levels: tuple[int, ...]) -> tuple[float, ...]:
    """Inverse of :func:`quantize_vector7`."""
    out = [UNUSED] * VEC_LEN
    out[SLOT_INDEX["t"]] = float(levels[SLOT_INDEX["t"]])
    iv = levels[SLOT_INDEX["I"]]
    out[SLOT_INDEX["I"]] = UNUSED if iv == UNUSED_LEVEL else float(iv)
    for name, (low, high) in _SLOT_RANGE.items():
        j = SLOT_INDEX[name]
        out[j] = UNUSED if levels[j] == UNUSED_LEVEL else dequantize(levels[j], low, high)
    return tuple(out)


def sweep_degrees(alpha: float) -> float:
    """Physical sweep angle in degrees: ``alpha * 180``."""
    return alpha * 180.0


# --- feature matrix (Nc x 7) -----------------------------------------------
def build_feature_matrix(ops: list[GalleryOp], nc: int = NC_DEFAULT) -> tuple[tuple[int, ...], ...]:
    """Build the quantised ``Nc x 7`` feature matrix ``P``.

    The sequence is wrapped as ``[SOP, ops..., EOP]`` and padded with additional
    ``EOP`` rows to length ``nc`` (the maximum-program-length treatment).
    """
    rows: list[GalleryOp] = [GalleryOp(SOP)] + list(ops) + [GalleryOp(EOP)]
    if len(rows) > nc:
        raise ValueError(f"program length {len(rows)} exceeds Nc={nc}")
    rows += [GalleryOp(EOP)] * (nc - len(rows))
    return tuple(quantize_vector7(to_vector7(op)) for op in rows)


def op_type_sequence(matrix) -> tuple[int, ...]:
    """The ``P[:, 0]`` column: the sequence of operation types."""
    return tuple(int(row[SLOT_INDEX["t"]]) for row in matrix)


# --- arc-centre geometry ---------------------------------------------------
def arc_center(start: tuple[float, float], end: tuple[float, float],
               sweep_deg: float) -> tuple[float, float]:
    """Centre of the arc through ``start`` and ``end`` subtending ``sweep_deg``.

    The centre lies on the perpendicular bisector of the chord, at distance
    ``(|chord|/2) / tan(sweep/2)`` from its midpoint; the sign of the sweep
    angle selects the side.
    """
    theta = math.radians(sweep_deg)
    if abs(math.sin(theta / 2.0)) < 1e-12:
        raise ValueError("degenerate arc: sweep angle multiple of 360 degrees")
    sx, sy = start
    ex, ey = end
    mx, my = (sx + ex) / 2.0, (sy + ey) / 2.0
    dx, dy = ex - sx, ey - sy
    chord = math.hypot(dx, dy)
    if chord < 1e-12:
        raise ValueError("degenerate arc: start and end coincide")
    half = chord / 2.0
    h = half / math.tan(theta / 2.0)
    # Unit perpendicular = chord direction rotated +90 degrees.
    px, py = -dy / chord, dx / chord
    return (mx + h * px, my + h * py)


# --- parsing: feature matrix -> resolved operations ------------------------
@dataclass
class ParsedOp:
    """A parsed operation with resolved geometry.

    ``start`` is the inherited start point for lines/arcs; ``end`` its endpoint;
    ``center`` the reconstructed arc/circle centre; ``sweep_deg`` the arc sweep.
    """

    type: int
    plane: int = UNUSED
    start: tuple[float, float] | None = None
    end: tuple[float, float] | None = None
    center: tuple[float, float] | None = None
    radius: float = UNUSED
    sweep_deg: float = UNUSED
    depth: float = UNUSED
    boolean: int = DEFAULT_BOOLEAN
    extras: dict = field(default_factory=dict)


def parse_feature_matrix(matrix) -> list[ParsedOp]:
    """Parse a quantised feature matrix into resolved operations.

    Reconstructs each curve's start point (inherited from the previous curve's
    endpoint; origin for the first curve of a sketch) and each arc's centre.
    ``SOP``/``EOP`` marker rows are dropped.
    """
    parsed: list[ParsedOp] = []
    cursor = (0.0, 0.0)  # current pen position; reset at each add_sketch
    for row in matrix:
        t = int(row[SLOT_INDEX["t"]])
        if t in (SOP, EOP):
            continue
        vec = dequantize_vector7(row)
        if t == ADD_SKETCH:
            cursor = (0.0, 0.0)
            parsed.append(ParsedOp(ADD_SKETCH, plane=int(vec[SLOT_INDEX["I"]])))
        elif t == ADD_LINE:
            end = (vec[SLOT_INDEX["x"]], vec[SLOT_INDEX["y"]])
            parsed.append(ParsedOp(ADD_LINE, start=cursor, end=end))
            cursor = end
        elif t == ADD_ARC:
            end = (vec[SLOT_INDEX["x"]], vec[SLOT_INDEX["y"]])
            sweep = sweep_degrees(vec[SLOT_INDEX["alpha"]])
            center = arc_center(cursor, end, sweep)
            parsed.append(ParsedOp(ADD_ARC, start=cursor, end=end,
                                   center=center, sweep_deg=sweep))
            cursor = end
        elif t == ADD_CIRCLE:
            center = (vec[SLOT_INDEX["x"]], vec[SLOT_INDEX["y"]])
            parsed.append(ParsedOp(ADD_CIRCLE, center=center,
                                   radius=vec[SLOT_INDEX["r"]]))
        elif t == ADD_EXTRUDE:
            parsed.append(ParsedOp(ADD_EXTRUDE, depth=vec[SLOT_INDEX["d"]]))
    return parsed


# --- Sim-Gallery DSL rendering ---------------------------------------------
def _fmt(value: float) -> str:
    return f"{value:.6g}"


def op_to_dsl(op: GalleryOp) -> str:
    """Render a single symbolic op as a Sim-Gallery DSL call."""
    if op.type == ADD_SKETCH:
        return f'add_sketch("{PLANES[op.plane]}")'
    if op.type == ADD_LINE:
        return f"add_line({_fmt(op.x)}, {_fmt(op.y)})"
    if op.type == ADD_ARC:
        return f"add_arc({_fmt(op.x)}, {_fmt(op.y)}, {_fmt(op.alpha)})"
    if op.type == ADD_CIRCLE:
        return f"add_circle({_fmt(op.x)}, {_fmt(op.y)}, {_fmt(op.r)})"
    if op.type == ADD_EXTRUDE:
        return f'add_extrude({_fmt(op.d)}, "{BOOLEANS[op.boolean]}")'
    return OP_NAMES[op.type]


def program_to_dsl(ops: list[GalleryOp]) -> str:
    """Render a whole CAD program as newline-separated Sim-Gallery DSL."""
    return "\n".join(op_to_dsl(op) for op in ops)
