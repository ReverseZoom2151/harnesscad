"""Text2CAD's exact CAD-sequence <-> vector codec (``CadSeqProc/cad_sequence.py``).

Reference implementation: ``CADSequence.to_vec`` / ``CADSequence.from_vec`` of the
released Text2CAD code (Khan et al., NeurIPS 2024), together with the ``to_vec`` /
``from_vec`` of ``SketchSequence``, ``FaceSequence``, ``LoopSequence``,
``ExtrudeSequence``, ``Line``, ``Arc``, ``Circle`` and the helpers ``split_array`` /
``merge_end_tokens_from_loop`` in ``CadSeqProc/utility/utils.py``.

Why this is not ``reconstruction.text2cad2_sequence_tokens``
-----------------------------------------------------------
That module models the token *vocabulary* as presented in the paper (Table 3): the
same 11 reserved ids and the same 8-bit +11 coordinate offset, but it serialises each
curve with **all** of its points (line -> start & end, arc -> start, mid & end, circle
-> centre & top-most point). The released code does something materially different and
load-bearing for anyone reading the real ``cad_vec`` arrays:

* loops are **closed and chained** -- a curve emits only the coordinates that are not
  implied by its successor. A line emits *one* token (its start point); an arc emits
  *two* (start, mid); the end point of every curve is the start point of the next
  curve in the loop, wrapping around. So an N-curve polygon costs N coordinate tokens,
  not 2N;
* a **circle** is the only curve that emits two tokens (centre, ``pt1``) and it is
  recognised structurally: a loop holding exactly one curve is a circle (radius is
  recovered as ``|pt1 - centre|``);
* every curve is closed by ``END_CURVE``, every loop by ``END_LOOP``, every face by
  ``END_FACE``, every sketch by ``END_SKETCH``;
* an extrusion is exactly **11 tokens**, in the order
  ``(e1, e2, ox, oy, oz, theta, phi, gamma, b, s, END_EXTRUSION)``, where the boolean
  ``b`` is offset by ``END_PAD = 7`` only and every other value by
  ``END_PAD + BOOLEAN_PAD = 11``;
* the whole stream is wrapped in ``START`` ... ``START`` (id 1 serves as both SOS and
  EOS) and right-padded with id 0 to ``MAX_CAD_SEQUENCE_LENGTH = 272``;
* two auxiliary streams are emitted alongside ``cad_vec`` and consumed by the
  decoder's adaptive layer: ``flag_vec`` (token role: 0 = sketch token, 1..10 = the
  extrusion parameter slots, 11 = padding) and ``index_vec`` (which sketch-extrude
  pair a token belongs to, padding = ``max + 1``).

Everything here operates on already-quantised integers (see
``reconstruction.deepcad2_numericalize`` -- Text2CAD reuses DeepCAD's quantisation
maps unchanged, so they are not repeated). Pure stdlib, deterministic; the codec
round-trips.

Data model (plain dicts / tuples)::

    point   = (x, y)                       ints in 0..255
    curve   = {"type": "line",   "start": p}                    # end = next curve
            | {"type": "arc",    "start": p, "mid": p}          # end = next curve
            | {"type": "circle", "center": p, "pt1": p}         # only curve in loop
    loop    = [curve, ...]
    face    = [loop, ...]
    sketch  = [face, ...]
    extrude = {"extent_one": int, "extent_two": int, "origin": (ox, oy, oz),
               "euler": (theta, phi, gamma), "boolean": int, "sketch_size": int}
    part    = {"sketch": sketch, "extrusion": extrude}
    model   = [part, ...]

Decoding fills in the implied ``end`` of lines/arcs and the ``radius`` of circles.
"""

from __future__ import annotations

from dataclasses import dataclass

# --- token ids (CadSeqProc/utility/macro.py) --------------------------------
END_TOKEN: tuple[str, ...] = (
    "PADDING", "START", "END_SKETCH", "END_FACE", "END_LOOP", "END_CURVE",
    "END_EXTRUSION",
)
PADDING = 0
START = 1
END_SKETCH = 2
END_FACE = 3
END_LOOP = 4
END_CURVE = 5
END_EXTRUSION = 6

END_PAD = 7            # number of reserved structural ids
BOOLEAN_PAD = 4        # number of boolean-operation ids (7..10)
COORD_OFFSET = END_PAD + BOOLEAN_PAD    # 11 -- offset of every quantised value
BOOLEAN_OFFSET = END_PAD                # 7  -- offset of the boolean token

N_BIT = 8
N_LEVELS = 2 ** N_BIT                   # 256
MAX_CAD_SEQUENCE_LENGTH = 272
MAX_EXTRUSION = 10
ONE_EXT_SEQ_LENGTH = 10                 # extrusion params, END_EXTRUSION excluded
EXT_BLOCK_LENGTH = ONE_EXT_SEQ_LENGTH + 1

FLAG_SKETCH = 0
FLAG_PAD = ONE_EXT_SEQ_LENGTH + 1       # 11

CAD_CLASS_INFO: dict[str, int] = {
    "one_hot_size": END_PAD + BOOLEAN_PAD + N_LEVELS,   # 267
    "index_size": MAX_EXTRUSION + 1,                    # 11
    "flag_size": ONE_EXT_SEQ_LENGTH + 2,                # 12
}

EXTRUDE_OPERATIONS: tuple[str, ...] = (
    "NewBodyFeatureOperation", "JoinFeatureOperation",
    "CutFeatureOperation", "IntersectFeatureOperation",
)

Token = tuple[int, int]


class CadVecError(ValueError):
    """Raised for malformed models or malformed token streams."""


# --- encoding helpers -------------------------------------------------------
def _coord_token(point) -> Token:
    x, y = point
    for v in (x, y):
        if not isinstance(v, int) or not 0 <= v < N_LEVELS:
            raise CadVecError(f"coordinate {point} is not a quantised 0..255 pair")
    return (x + COORD_OFFSET, y + COORD_OFFSET)


def _decode_coord(token: Token) -> tuple[int, int]:
    return (token[0] - COORD_OFFSET, token[1] - COORD_OFFSET)


def encode_curve(curve: dict) -> list[Token]:
    """Tokens for one curve, ending with ``END_CURVE``.

    A line contributes its start point only; an arc its start and mid points; a
    circle its centre and ``pt1`` (the reference point on the circumference).
    """
    kind = curve["type"]
    if kind == "line":
        pts = [curve["start"]]
    elif kind == "arc":
        pts = [curve["start"], curve["mid"]]
    elif kind == "circle":
        pts = [curve["center"], curve["pt1"]]
    else:
        raise CadVecError(f"unknown curve type {kind!r}")
    return [_coord_token(p) for p in pts] + [(END_CURVE, 0)]


def encode_loop(loop: list[dict]) -> list[Token]:
    if not loop:
        raise CadVecError("loop has no curves")
    if any(c["type"] == "circle" for c in loop) and len(loop) != 1:
        raise CadVecError("a circle must be the only curve in its loop")
    tokens: list[Token] = []
    for curve in loop:
        tokens += encode_curve(curve)
    tokens.append((END_LOOP, 0))
    return tokens


def encode_face(face: list[list[dict]]) -> list[Token]:
    if not face:
        raise CadVecError("face has no loops")
    tokens: list[Token] = []
    for loop in face:
        tokens += encode_loop(loop)
    tokens.append((END_FACE, 0))
    return tokens


def encode_sketch(sketch: list[list[list[dict]]]) -> list[Token]:
    if not sketch:
        raise CadVecError("sketch has no faces")
    tokens: list[Token] = []
    for face in sketch:
        tokens += encode_face(face)
    tokens.append((END_SKETCH, 0))
    return tokens


def _value_token(value: int) -> Token:
    if not isinstance(value, int) or not 0 <= value < N_LEVELS:
        raise CadVecError(f"extrusion value {value!r} is not a quantised 0..255 int")
    return (value + COORD_OFFSET, 0)


def encode_extrusion(extrusion: dict) -> list[Token]:
    """The 11-token extrusion block ``(e1, e2, ox, oy, oz, t, p, g, b, s, ee)``."""
    origin = tuple(extrusion["origin"])
    euler = tuple(extrusion["euler"])
    if len(origin) != 3 or len(euler) != 3:
        raise CadVecError("origin and euler must each hold 3 values")
    boolean = extrusion["boolean"]
    if not isinstance(boolean, int) or not 0 <= boolean < BOOLEAN_PAD:
        raise CadVecError(f"boolean {boolean!r} outside 0..{BOOLEAN_PAD - 1}")
    tokens = [
        _value_token(extrusion["extent_one"]),
        _value_token(extrusion["extent_two"]),
    ]
    tokens += [_value_token(v) for v in origin]
    tokens += [_value_token(v) for v in euler]
    tokens.append((boolean + BOOLEAN_OFFSET, 0))
    tokens.append(_value_token(extrusion["sketch_size"]))
    tokens.append((END_EXTRUSION, 0))
    return tokens


@dataclass(frozen=True)
class CadVectors:
    """The three parallel streams consumed by the Text2CAD decoder."""

    cad_vec: list[Token]
    flag_vec: list[int]
    index_vec: list[int]

    def __len__(self) -> int:
        return len(self.cad_vec)


def encode_model(
    model: list[dict],
    *,
    padding: bool = False,
    max_cad_seq_len: int = MAX_CAD_SEQUENCE_LENGTH,
) -> CadVectors:
    """Serialise a list of sketch-extrude parts into ``(cad_vec, flag_vec, index_vec)``."""
    if not model:
        raise CadVecError("model has no parts")
    if len(model) > MAX_EXTRUSION:
        raise CadVecError(f"at most {MAX_EXTRUSION} extrusions supported")

    cad_vec: list[Token] = [(START, 0)]
    flag_vec: list[int] = [FLAG_SKETCH]
    index_vec: list[int] = [0]

    for i, part in enumerate(model):
        skt = encode_sketch(part["sketch"])
        ext = encode_extrusion(part["extrusion"])
        cad_vec += skt + ext
        flag_vec += [FLAG_SKETCH] * len(skt)
        flag_vec += [1] + list(range(1, ONE_EXT_SEQ_LENGTH + 1))
        index_vec += [i] * (len(skt) + len(ext))

    cad_vec.append((START, 0))
    flag_vec.append(FLAG_SKETCH)
    index_vec.append(index_vec[-1])

    if padding:
        num_pad = max_cad_seq_len - len(cad_vec)
        if num_pad < 0:
            raise CadVecError(
                f"sequence of {len(cad_vec)} tokens exceeds max_cad_seq_len={max_cad_seq_len}"
            )
        cad_vec += [(PADDING, PADDING)] * num_pad
        flag_vec += [FLAG_PAD] * num_pad
        index_vec += [max(index_vec) + 1] * num_pad

    return CadVectors(cad_vec=cad_vec, flag_vec=flag_vec, index_vec=index_vec)


# --- decoding ---------------------------------------------------------------
def split_tokens(tokens: list[Token], value: int) -> list[list[Token]]:
    """Split on every token whose first slot equals ``value``; the token is dropped.

    Mirrors ``utils.split_array(arr, val, include_val=False)``: one chunk is produced
    per occurrence of ``value`` (a trailing remainder without a terminator is not a
    chunk), so a well-formed stream terminates with ``value``.
    """
    chunks: list[list[Token]] = []
    current: list[Token] = []
    for tok in tokens:
        if tok[0] == value:
            chunks.append(current)
            current = []
        else:
            current.append(tok)
    return chunks


def strip_padding(cad_vec: list[Token]) -> list[Token]:
    return [t for t in cad_vec if t[0] != PADDING]


def decode_loop(tokens: list[Token]) -> list[dict]:
    """Rebuild a closed loop from its coordinate tokens (``END_LOOP`` removed).

    Reproduces ``merge_end_tokens_from_loop``: a single curve group is a circle;
    otherwise each group is chained with the first token of the next group (wrapping),
    a 1-token group becoming a line and a 2-token group an arc.
    """
    groups = split_tokens(tokens, END_CURVE)
    if not groups:
        raise CadVecError("loop has no curves")
    if len(groups) == 1:
        group = groups[0]
        if len(group) != 2:
            raise CadVecError(f"single-curve loop must be a circle (2 tokens), got {len(group)}")
        center = _decode_coord(group[0])
        pt1 = _decode_coord(group[1])
        radius = ((pt1[0] - center[0]) ** 2 + (pt1[1] - center[1]) ** 2) ** 0.5
        return [{"type": "circle", "center": center, "pt1": pt1, "radius": radius}]

    curves: list[dict] = []
    n = len(groups)
    for i, group in enumerate(groups):
        nxt = groups[(i + 1) % n]
        if not nxt:
            raise CadVecError("empty curve group in loop")
        end = _decode_coord(nxt[0])
        if len(group) == 1:
            curves.append({"type": "line", "start": _decode_coord(group[0]), "end": end})
        elif len(group) == 2:
            curves.append({
                "type": "arc",
                "start": _decode_coord(group[0]),
                "mid": _decode_coord(group[1]),
                "end": end,
            })
        else:
            raise CadVecError(f"invalid curve token group of length {len(group)}")
    return curves


def decode_sketch(tokens: list[Token]) -> list[list[list[dict]]]:
    """Rebuild ``[face][loop][curve]`` from the sketch tokens (``END_SKETCH`` removed)."""
    faces = []
    for face_tokens in split_tokens(tokens, END_FACE):
        loops = [decode_loop(lp) for lp in split_tokens(face_tokens, END_LOOP)]
        if not loops:
            raise CadVecError("face has no loops")
        faces.append(loops)
    if not faces:
        raise CadVecError("sketch has no faces")
    return faces


def decode_extrusion(tokens: list[Token]) -> dict:
    """Rebuild the extrusion dict from its 10 parameter tokens (``END_EXTRUSION`` removed)."""
    if len(tokens) == EXT_BLOCK_LENGTH and tokens[-1][0] == END_EXTRUSION:
        tokens = tokens[:-1]
    if len(tokens) != ONE_EXT_SEQ_LENGTH:
        raise CadVecError(f"extrusion block must hold {ONE_EXT_SEQ_LENGTH} tokens")
    vals = [t[0] - COORD_OFFSET for t in tokens]
    boolean = tokens[8][0] - BOOLEAN_OFFSET
    if not 0 <= boolean < BOOLEAN_PAD:
        raise CadVecError(f"invalid boolean token {tokens[8][0]}")
    return {
        "extent_one": vals[0],
        "extent_two": vals[1],
        "origin": (vals[2], vals[3], vals[4]),
        "euler": (vals[5], vals[6], vals[7]),
        "boolean": boolean,
        "sketch_size": vals[9],
    }


def decode_model(cad_vec: list[Token]) -> list[dict]:
    """Inverse of :func:`encode_model` (padding tolerated, ``START`` wrapper required)."""
    tokens = strip_padding(list(cad_vec))
    if len(tokens) < 2 or tokens[0][0] != START or tokens[-1][0] != START:
        raise CadVecError("stream must start and end with the START token")
    body = tokens[1:-1]
    parts: list[dict] = []
    for chunk in split_tokens(body, END_EXTRUSION):
        if len(chunk) < ONE_EXT_SEQ_LENGTH + 1:
            raise CadVecError("sketch-extrude chunk is too short")
        sketch_tokens = split_tokens(chunk, END_SKETCH)
        if not sketch_tokens:
            raise CadVecError("chunk has no END_SKETCH token")
        sketch = decode_sketch(sketch_tokens[0])
        extrusion = decode_extrusion(chunk[-ONE_EXT_SEQ_LENGTH:])
        parts.append({"sketch": sketch, "extrusion": extrusion})
    if not parts:
        raise CadVecError("no sketch-extrude pair found")
    return parts


def boolean_name(index: int) -> str:
    """Map a boolean-operation index to its Fusion360/DeepCAD operation name."""
    if not 0 <= index < len(EXTRUDE_OPERATIONS):
        raise CadVecError(f"boolean index {index} out of range")
    return EXTRUDE_OPERATIONS[index]
