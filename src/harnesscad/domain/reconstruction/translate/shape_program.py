"""Text-based parametric shape-program IR + codec.

This representation departs from fixed-slot command templates by
representing a 3D parametric model as a *script in a general-purpose language*
(Python or YAML).  A model is an assembly of primitive instances:

    Z = {Z_i},   Z_i = (M_i, B_i, P_i)

where ``M_i`` is a primitive model ID, ``B_i = (p_i, s_i, r_i)`` is a 3D box of
common parameters -- center position ``p in R^3``, size ``s in R^3`` and a single
rotation angle ``r`` about the vertical (z) axis -- and ``P_i`` is a list of
model-specific ``key=value`` parameters (empty when the primitive has none).

The Python serialisation is::

    bbox_0 = Bbox(507, 185, 805, 1014, 370, 50, 0)
    model_0 = <model_57761062>()
    bbox_2 = Bbox(532, 195, 390, 964, 350, 780, 0)
    model_2 = <model_115813862>(N=1, NKA=928, DBXX=1, BT=18)

with ``Bbox(position_x, position_y, position_z, scale_x, scale_y, scale_z,
angle_z)``.  The YAML serialisation is a list of ``- id:`` records with
``position_*``/``scale_*``/``angle_z``/``model_id`` and any model-specific keys).

The central deterministic property is that this text representation is
*lossless* for continuous values (no domain tokenizer quantization) and can be
round-tripped; that round-trip -- parse -> IR -> serialise -> parse -- is exactly
what this module implements, for both proxy languages.  The VLM that *predicts*
the script from a raster drawing is the learned, out-of-scope part.

Pure stdlib, deterministic (no wall clock, no randomness).
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Dict, List, Sequence, Tuple, Union

Number = Union[int, float]
ParamValue = Union[int, float, str]

# Field order of the Bbox reference dataclass (Figure 10).
BBOX_FIELDS: Tuple[str, ...] = (
    "position_x", "position_y", "position_z",
    "scale_x", "scale_y", "scale_z", "angle_z",
)


def _fmt_number(value: Number) -> str:
    """Format a number the way the listings do: no trailing ``.0``."""
    if isinstance(value, bool):  # guard: bool is an int subclass
        return str(int(value))
    if isinstance(value, int):
        return str(value)
    if value == int(value):
        return str(int(value))
    # Strip a trailing zero fraction like 407.50 -> 407.5 while staying exact.
    text = repr(float(value))
    return text


def _parse_number(text: str) -> Number:
    text = text.strip()
    try:
        return int(text)
    except ValueError:
        return float(text)


@dataclass(frozen=True)
class Bbox:
    """Common parameters of a primitive: center position, size, z-rotation."""

    position_x: Number
    position_y: Number
    position_z: Number
    scale_x: Number
    scale_y: Number
    scale_z: Number
    angle_z: Number = 0

    def as_tuple(self) -> Tuple[Number, ...]:
        return tuple(getattr(self, f) for f in BBOX_FIELDS)

    @property
    def position(self) -> Tuple[Number, Number, Number]:
        return (self.position_x, self.position_y, self.position_z)

    @property
    def size(self) -> Tuple[Number, Number, Number]:
        return (self.scale_x, self.scale_y, self.scale_z)

    def min_corner(self) -> Tuple[float, float, float]:
        """Corner with the smallest coordinate (center - size/2)."""
        return (self.position_x - self.scale_x / 2.0,
                self.position_y - self.scale_y / 2.0,
                self.position_z - self.scale_z / 2.0)

    def max_corner(self) -> Tuple[float, float, float]:
        return (self.position_x + self.scale_x / 2.0,
                self.position_y + self.scale_y / 2.0,
                self.position_z + self.scale_z / 2.0)


@dataclass(frozen=True)
class PrimitiveInstance:
    """A single primitive: model ID, common box, model-specific params (P_i)."""

    model_id: str
    bbox: Bbox
    params: Tuple[Tuple[str, ParamValue], ...] = ()

    def param_dict(self) -> Dict[str, ParamValue]:
        return dict(self.params)


@dataclass
class ShapeProgram:
    """An ordered assembly Z = {Z_i} of primitive instances."""

    instances: List[PrimitiveInstance] = field(default_factory=list)

    def __len__(self) -> int:
        return len(self.instances)

    def __iter__(self):
        return iter(self.instances)


# --------------------------------------------------------------------------- #
# Construction helpers
# --------------------------------------------------------------------------- #

def make_instance(model_id: str, box: Sequence[Number],
                  params: Union[Dict[str, ParamValue],
                                Sequence[Tuple[str, ParamValue]], None] = None
                  ) -> PrimitiveInstance:
    """Build a :class:`PrimitiveInstance` from a 6- or 7-tuple box.

    ``model_id`` may be given bare (``"model_57761062"`` or ``"57761062"``) or in
    the angle-bracket form ``"<model_57761062>"``; it is normalized to the bare
    ``model_<n>`` token used by the IR.
    """
    values = list(box)
    if len(values) == 6:
        values.append(0)
    if len(values) != 7:
        raise ValueError("box must have 6 or 7 numbers")
    bbox = Bbox(*values)
    if params is None:
        items: Tuple[Tuple[str, ParamValue], ...] = ()
    elif isinstance(params, dict):
        items = tuple(params.items())
    else:
        items = tuple(params)
    return PrimitiveInstance(normalize_model_id(model_id), bbox, items)


def normalize_model_id(model_id: str) -> str:
    """Return the bare ``model_<n>`` token for any accepted spelling."""
    token = model_id.strip()
    m = re.fullmatch(r"<\s*(.*?)\s*>", token)
    if m:
        token = m.group(1).strip()
    if not token.startswith("model_"):
        token = "model_" + token
    return token


def model_id_bracketed(model_id: str) -> str:
    """The ``<model_...>`` spelling used inside the scripts."""
    return "<" + normalize_model_id(model_id) + ">"


# --------------------------------------------------------------------------- #
# Python proxy language
# --------------------------------------------------------------------------- #

def serialize_python(program: ShapeProgram) -> str:
    lines: List[str] = []
    for i, inst in enumerate(program.instances):
        box = ", ".join(_fmt_number(v) for v in inst.bbox.as_tuple())
        lines.append(f"bbox_{i} = Bbox({box})")
        args = ", ".join(f"{k}={_fmt_param(v)}" for k, v in inst.params)
        lines.append(f"model_{i} = {model_id_bracketed(inst.model_id)}({args})")
    return "\n".join(lines)


def _fmt_param(value: ParamValue) -> str:
    if isinstance(value, str):
        return value
    return _fmt_number(value)


_BBOX_RE = re.compile(r"bbox_(\d+)\s*=\s*Bbox\(([^)]*)\)")
_MODEL_RE = re.compile(r"model_(\d+)\s*=\s*(<[^>]+>)\(([^)]*)\)")


def parse_python(text: str) -> ShapeProgram:
    """Parse a Python shape program into a program.

    Robust to arbitrary whitespace inside the listings (the extracted PDFs pad
    numbers with spaces).  Lines are matched by their ``bbox_<i>`` / ``model_<i>``
    index so the two halves of each primitive are paired even if interleaved.
    """
    boxes: Dict[int, Bbox] = {}
    models: Dict[int, Tuple[str, Tuple[Tuple[str, ParamValue], ...]]] = {}
    for m in _BBOX_RE.finditer(text):
        idx = int(m.group(1))
        nums = [_parse_number(p) for p in m.group(2).split(",") if p.strip()]
        if len(nums) == 6:
            nums.append(0)
        boxes[idx] = Bbox(*nums)
    for m in _MODEL_RE.finditer(text):
        idx = int(m.group(1))
        models[idx] = (m.group(2), _parse_param_list(m.group(3)))
    program = ShapeProgram()
    for idx in sorted(boxes):
        if idx not in models:
            raise ValueError(f"bbox_{idx} has no matching model_{idx}")
        model_id, params = models[idx]
        program.instances.append(
            PrimitiveInstance(normalize_model_id(model_id), boxes[idx], params))
    return program


def _parse_param_list(text: str) -> Tuple[Tuple[str, ParamValue], ...]:
    text = text.strip()
    if not text:
        return ()
    out: List[Tuple[str, ParamValue]] = []
    for part in text.split(","):
        if not part.strip():
            continue
        key, _, val = part.partition("=")
        out.append((key.strip(), _coerce_value(val.strip())))
    return tuple(out)


def _coerce_value(text: str) -> ParamValue:
    try:
        return _parse_number(text)
    except ValueError:
        return text


# --------------------------------------------------------------------------- #
# YAML proxy language (Figure 11)
# --------------------------------------------------------------------------- #

def serialize_yaml(program: ShapeProgram) -> str:
    lines: List[str] = []
    for i, inst in enumerate(program.instances):
        lines.append(f"- id: {i}")
        b = inst.bbox
        lines.append(f"  position_x: {_fmt_number(b.position_x)}")
        lines.append(f"  position_y: {_fmt_number(b.position_y)}")
        lines.append(f"  position_z: {_fmt_number(b.position_z)}")
        lines.append(f"  scale_x: {_fmt_number(b.scale_x)}")
        lines.append(f"  scale_y: {_fmt_number(b.scale_y)}")
        lines.append(f"  scale_z: {_fmt_number(b.scale_z)}")
        lines.append(f"  angle_z: {_fmt_number(b.angle_z)}")
        lines.append(f"  model_id: {model_id_bracketed(inst.model_id)}")
        for k, v in inst.params:
            lines.append(f"  {k}: {_fmt_param(v)}")
    return "\n".join(lines)


def parse_yaml(text: str) -> ShapeProgram:
    """Parse the restricted block-sequence YAML of Figure 11.

    Only the emitted subset is supported: a top-level sequence of
    mappings, each item introduced by ``- <key>: <value>`` and continued by
    indented ``<key>: <value>`` lines.  No nesting, flow style or anchors.
    """
    program = ShapeProgram()
    records: List[Dict[str, str]] = []
    current: Dict[str, str] = {}
    for raw in text.splitlines():
        line = raw.rstrip()
        if not line.strip():
            continue
        stripped = line.lstrip()
        if stripped.startswith("- "):
            if current:
                records.append(current)
            current = {}
            stripped = stripped[2:]
        if ":" not in stripped:
            continue
        key, _, val = stripped.partition(":")
        current[key.strip()] = val.strip()
    if current:
        records.append(current)

    reserved = set(BBOX_FIELDS) | {"id", "model_id"}
    for rec in records:
        try:
            box = Bbox(*(_parse_number(rec[f]) for f in BBOX_FIELDS))
        except KeyError as exc:  # pragma: no cover - defensive
            raise ValueError(f"YAML record missing field {exc}") from None
        model_id = rec.get("model_id", "model_0")
        params = tuple((k, _coerce_value(v)) for k, v in rec.items()
                       if k not in reserved)
        program.instances.append(
            PrimitiveInstance(normalize_model_id(model_id), box, params))
    return program


# --------------------------------------------------------------------------- #
# Canonical pose
# --------------------------------------------------------------------------- #

def program_bounds(program: ShapeProgram
                   ) -> Tuple[Tuple[float, float, float],
                              Tuple[float, float, float]]:
    """Axis-aligned min/max corners over every primitive box in the program."""
    if not program.instances:
        raise ValueError("empty program has no bounds")
    lo = [float("inf")] * 3
    hi = [float("-inf")] * 3
    for inst in program.instances:
        mn, mx = inst.bbox.min_corner(), inst.bbox.max_corner()
        for a in range(3):
            lo[a] = min(lo[a], mn[a])
            hi[a] = max(hi[a], mx[a])
    return (tuple(lo), tuple(hi))


def translate(program: ShapeProgram,
              offset: Tuple[Number, Number, Number]) -> ShapeProgram:
    """Return a copy with every primitive translated by ``offset``."""
    ox, oy, oz = offset
    out = ShapeProgram()
    for inst in program.instances:
        b = inst.bbox
        out.instances.append(PrimitiveInstance(
            inst.model_id,
            Bbox(b.position_x + ox, b.position_y + oy, b.position_z + oz,
                 b.scale_x, b.scale_y, b.scale_z, b.angle_z),
            inst.params))
    return out


def normalize_to_first_octant(program: ShapeProgram) -> ShapeProgram:
    """Move the assembly so its bounding box's min corner sits at the origin.

    The subject is aligned to the main axes with the origin at one corner
    of the bounding box so the whole model lies in the first octant (all
    coordinates non-negative).  This is the deterministic canonical pose used to
    remove translation ambiguity before evaluation.
    """
    (lo, _hi) = program_bounds(program)
    return translate(program, (-lo[0], -lo[1], -lo[2]))
