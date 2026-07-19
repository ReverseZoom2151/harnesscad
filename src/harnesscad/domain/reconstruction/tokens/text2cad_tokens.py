"""CAD-sequence token representation for text-conditioned CAD generation.

This is a sketch-and-extrude token vocabulary that serialises a CAD construction
sequence into a flat stream of 2D tokens for an autoregressive decoder. Every
element of the sequence is a **2-tuple** ``(px, py)``:

* the special / structural tokens carry their id in the first slot and 0 in the
  second, i.e. ``(id, 0)``;
* a *coordinate* token carries a quantised (x, y) pair, both slots populated.

Token id map::

    0            pad                 (0, 0)
    1            start / end-seq     (1, 0)   -- cls (SOS) and EOS share id 1
    2   es       end of sketch       (2, 0)
    3   ef       end of face         (3, 0)
    4   el       end of loop         (4, 0)
    5   ec       end of curve        (5, 0)
    6   ee       end of extrude      (6, 0)
    7..10  boolean New/Cut/Join/Intersect  (beta, 0)
    11..266      quantised value J11..266K  (8-bit, 256 levels)

Continuous 2D coordinates and continuous extrusion parameters are quantised in
**8 bits** -> 256 class labels, then offset by ``COORD_OFFSET = 11`` so quantised
values never collide with the 11 reserved ids ``0..10``. Curves are parameterised
as:

* **Line**  -- start & end coordinate;
* **Arc**   -- start, mid & end coordinate;
* **Circle**-- centre & top-most coordinate (centre shifted by +radius in y).

Loops are always closed. An extrusion block is the 10 parameters
``d+, d-, tx, ty, tz, theta, phi, gamma, sigma, beta`` followed by ``ee``.

This is a self-contained, reversible codec. The repo's other tokenizers
(``ingest.davinci_primitive_tokens`` 8-token fixed blocks, ``reconstruction.
cmt_tokenization`` continuous B-Rep tokens) encode entirely different schemes;
none reproduce this id table or its 11-offset 8-bit coordinate range.
"""

from __future__ import annotations

from dataclasses import dataclass, field

# --- Reserved / structural token ids. ---
PAD = 0
START = 1          # cls (SOS); EOS shares id 1.
END_SEQUENCE = 1
END_SKETCH = 2     # es
END_FACE = 3       # ef
END_LOOP = 4       # el
END_CURVE = 5      # ec
END_EXTRUDE = 6    # ee

# Boolean operation ids 7..10.
BOOL_NEW = 7
BOOL_CUT = 8
BOOL_JOIN = 9
BOOL_INTERSECT = 10
BOOLEAN_IDS = {
    "new": BOOL_NEW,
    "cut": BOOL_CUT,
    "join": BOOL_JOIN,
    "intersect": BOOL_INTERSECT,
}
BOOLEAN_NAMES = {v: k for k, v in BOOLEAN_IDS.items()}

# Quantisation: 8 bits -> 256 levels, offset so values occupy J11..266K.
N_QUANT_LEVELS = 256
COORD_OFFSET = 11
COORD_MIN_TOKEN = COORD_OFFSET                       # 11
COORD_MAX_TOKEN = COORD_OFFSET + N_QUANT_LEVELS - 1  # 266

Token = tuple[int, int]


# --------------------------------------------------------------------------- #
# 8-bit quantisation codec.
# --------------------------------------------------------------------------- #
def _clamp01(x: float) -> float:
    if x < 0.0:
        return 0.0
    if x > 1.0:
        return 1.0
    return x


def quantize(value: float) -> int:
    """Quantise a normalised ``value`` in ``[0, 1]`` to an 8-bit level ``0..255``."""
    return int(round(_clamp01(value) * (N_QUANT_LEVELS - 1)))


def dequantize(level: int) -> float:
    """Inverse of :func:`quantize`: map ``0..255`` back to ``[0, 1]``."""
    if not 0 <= level < N_QUANT_LEVELS:
        raise ValueError(f"quant level out of range: {level}")
    return level / (N_QUANT_LEVELS - 1)


def value_to_token_id(value: float) -> int:
    """Normalised scalar -> token id in ``[11, 266]``."""
    return quantize(value) + COORD_OFFSET


def token_id_to_value(token_id: int) -> float:
    """Token id in ``[11, 266]`` -> normalised scalar in ``[0, 1]``."""
    if not COORD_MIN_TOKEN <= token_id <= COORD_MAX_TOKEN:
        raise ValueError(f"not a value token id: {token_id}")
    return dequantize(token_id - COORD_OFFSET)


def is_value_token_id(token_id: int) -> bool:
    return COORD_MIN_TOKEN <= token_id <= COORD_MAX_TOKEN


def coord_token(x: float, y: float) -> Token:
    """A 2D coordinate token ``(px, py)`` with both slots quantised."""
    return (value_to_token_id(x), value_to_token_id(y))


def special_token(token_id: int) -> Token:
    """A structural token ``(id, 0)`` for ``id`` in ``0..10``."""
    if not 0 <= token_id <= 10:
        raise ValueError(f"not a special token id: {token_id}")
    return (token_id, 0)


def decode_coord_token(token: Token) -> tuple[float, float]:
    """Inverse of :func:`coord_token`."""
    return (token_id_to_value(token[0]), token_id_to_value(token[1]))


def is_coordinate_token(token: Token) -> bool:
    return is_value_token_id(token[0]) and is_value_token_id(token[1])


# --------------------------------------------------------------------------- #
# Curve parameterisation.
# --------------------------------------------------------------------------- #
def circle_topmost(cx: float, cy: float, radius: float) -> tuple[float, float]:
    """Top-most point of a circle = centre shifted ``+radius`` in y."""
    return (cx, cy + radius)


@dataclass(frozen=True)
class Line:
    start: tuple[float, float]
    end: tuple[float, float]

    def tokens(self) -> list[Token]:
        return [coord_token(*self.start), coord_token(*self.end), special_token(END_CURVE)]


@dataclass(frozen=True)
class Arc:
    start: tuple[float, float]
    mid: tuple[float, float]
    end: tuple[float, float]

    def tokens(self) -> list[Token]:
        return [
            coord_token(*self.start),
            coord_token(*self.mid),
            coord_token(*self.end),
            special_token(END_CURVE),
        ]


@dataclass(frozen=True)
class Circle:
    center: tuple[float, float]
    radius: float

    def tokens(self) -> list[Token]:
        return [
            coord_token(*self.center),
            coord_token(*circle_topmost(self.center[0], self.center[1], self.radius)),
            special_token(END_CURVE),
        ]


@dataclass(frozen=True)
class Extrusion:
    """The 10 extrusion parameters, listed in canonical order."""
    d_plus: float
    d_minus: float
    tx: float
    ty: float
    tz: float
    theta: float
    phi: float
    gamma: float
    sigma: float
    boolean: str  # one of BOOLEAN_IDS keys

    def tokens(self) -> list[Token]:
        out = [
            (value_to_token_id(self.d_plus), 0),
            (value_to_token_id(self.d_minus), 0),
            (value_to_token_id(self.tx), 0),
            (value_to_token_id(self.ty), 0),
            (value_to_token_id(self.tz), 0),
            (value_to_token_id(self.theta), 0),
            (value_to_token_id(self.phi), 0),
            (value_to_token_id(self.gamma), 0),
            (value_to_token_id(self.sigma), 0),
            special_token(BOOLEAN_IDS[self.boolean]),
            special_token(END_EXTRUDE),
        ]
        return out


@dataclass
class CadModel:
    """Sketch (faces -> loops -> curves) plus one extrusion block."""
    faces: list[list[list[object]]] = field(default_factory=list)
    extrusion: Extrusion | None = None


def serialize_sketch(faces: list[list[list[object]]]) -> list[Token]:
    """Serialise ``faces[face][loop][curve]`` into tokens with structural markers.

    Emits per-curve tokens (each curve ends with ``ec``), ``el`` after each loop,
    ``ef`` after each face, and a final ``es`` closing the sketch.
    """
    tokens: list[Token] = []
    for face in faces:
        for loop in face:
            for curve in loop:
                tokens.extend(curve.tokens())
            tokens.append(special_token(END_LOOP))
        tokens.append(special_token(END_FACE))
    tokens.append(special_token(END_SKETCH))
    return tokens


def serialize_model(model: CadModel) -> list[Token]:
    """Full construction sequence: ``START, sketch..., extrusion..., EOS``."""
    tokens: list[Token] = [special_token(START)]
    tokens.extend(serialize_sketch(model.faces))
    if model.extrusion is not None:
        tokens.extend(model.extrusion.tokens())
    tokens.append(special_token(END_SEQUENCE))
    return tokens


def vocabulary_size() -> int:
    """Total distinct token ids: 0..266 -> 267."""
    return COORD_MAX_TOKEN + 1
