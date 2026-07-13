"""SkexGen extrude-branch token vector (19 tokens) and its field-flag stream.

SkexGen encodes every sketch-extrude pair's extrusion as a *fixed-length* run of
19 tokens (``utils/utils.py::process_obj_se``)::

    idx   0  1 | 2  3  4 | 5 .. 13        | 14   | 15    | 16 17  | 18
    field ext_v | origin  | rotation (3x3) | op   | scale | offset | end

    ext_v   2 tokens  extrude distances (+/-), quantised over [-1, 1]
    origin  3 tokens  sketch-plane origin, quantised over [-1, 1]
    rot     9 tokens  the plane's x/y/z axes, each component rounded and
                      clipped to {-1, 0, 1} then shifted by R_PAD = 2  ->  {1,2,3}
                      (SkexGen assumes axis-aligned sketch planes)
    op      1 token   1 = add (join/new body), 2 = cut, 3 = intersect
    scale   1 token   sketch normalisation scale, quantised over [0, SCALE_R]
    offset  2 tokens  sketch bbox centre, quantised over [-OFFSET_R, OFFSET_R]
    end     1 token   0, which becomes the SE_END token (1) after EXTRA_PAD

All value tokens carry ``EXT_PAD = 1`` (rotation carries ``R_PAD = 2``) so that
0 stays free for padding.  A parallel *flag* stream
``[1,1,2,2,2,3,3,3,3,3,3,3,3,3,4,5,6,6,7]`` tags each position with its field
id; the extrude transformer embeds it alongside the value token.

This is a distinct scheme from DeepCAD's single ``EXT`` row (theta/phi/gamma
Euler angles + 11 args) already in ``reconstruction/deepcad2_vector_layout``:
SkexGen stores the *raw rotation matrix* trit-quantised, and it stores the
sketch scale/offset in the extrude branch (DeepCAD keeps them as args ``s``).

Deterministic, stdlib only.
"""
from __future__ import annotations

from typing import Dict, List, Sequence

from harnesscad.domain.reconstruction.skexgen_token_format import (
    BIT, EXTRA_PAD, EXTRUDE_R, EXT_PAD, OFFSET_R, R_PAD, SCALE_R, SKETCH_R,
    dequantize, quantize,
)

EXT_SEQ_LEN = 19

# field-id ("flag") stream, one entry per token of an extrude block
EXT_FLAGS: List[int] = [1, 1, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3, 3, 4, 5, 6, 6, 7]

# slices into the 19-token block
SLICE_VALUE = slice(0, 2)
SLICE_ORIGIN = slice(2, 5)
SLICE_ROTATION = slice(5, 14)
INDEX_OP = 14
INDEX_SCALE = 15
SLICE_OFFSET = slice(16, 18)
INDEX_END = 18

OP_ADD = 1
OP_CUT = 2
OP_INTERSECT = 3

SET_OP_TO_TOKEN = {
    "JoinFeatureOperation": OP_ADD,
    "NewBodyFeatureOperation": OP_ADD,
    "CutFeatureOperation": OP_CUT,
    "IntersectFeatureOperation": OP_INTERSECT,
}
OP_NAME = {OP_ADD: "add", OP_CUT: "cut", OP_INTERSECT: "intersect"}


def extrude_vocab_size(bit: int = BIT) -> int:
    """Merged-stream vocabulary size of the extrude branch."""
    return 2 ** bit + EXT_PAD + EXTRA_PAD


def op_token(set_op: str) -> int:
    """Map a Fusion 360 / DeepCAD set-operation name to SkexGen's op token."""
    try:
        return SET_OP_TO_TOKEN[set_op]
    except KeyError:
        raise ValueError("unknown set operation: %r" % (set_op,))


def _rot_token(component: float) -> int:
    """Round-half-even, clip to {-1, 0, 1}, shift by R_PAD (numpy ``rint``)."""
    r = int(round(component))          # python round() is half-even, like np.rint
    if r < -1:
        r = -1
    if r > 1:
        r = 1
    return r + R_PAD


def encode_extrude(extrude_value: Sequence[float],
                   origin: Sequence[float],
                   x_axis: Sequence[float],
                   y_axis: Sequence[float],
                   z_axis: Sequence[float],
                   set_op: str,
                   scale: float,
                   offset: Sequence[float],
                   bit: int = BIT) -> List[int]:
    """Build the raw (pre-EXTRA_PAD) 19-token extrude block."""
    if len(extrude_value) != 2:
        raise ValueError("extrude_value must hold 2 distances")
    if len(origin) != 3:
        raise ValueError("origin must be 3d")
    for axis in (x_axis, y_axis, z_axis):
        if len(axis) != 3:
            raise ValueError("axes must be 3d")
    if len(offset) != 2:
        raise ValueError("offset must be 2d")

    tokens: List[int] = []
    tokens += [quantize(v, bit, -EXTRUDE_R, EXTRUDE_R) + EXT_PAD for v in extrude_value]
    tokens += [quantize(v, bit, -SKETCH_R, SKETCH_R) + EXT_PAD for v in origin]
    for axis in (x_axis, y_axis, z_axis):
        tokens += [_rot_token(c) for c in axis]
    tokens.append(op_token(set_op))
    tokens.append(quantize(scale, bit, 0.0, SCALE_R) + EXT_PAD)
    tokens += [quantize(v, bit, -OFFSET_R, OFFSET_R) + EXT_PAD for v in offset]
    tokens.append(0)                       # end-of-SE marker (-> 1 after EXTRA_PAD)
    assert len(tokens) == EXT_SEQ_LEN
    return tokens


def decode_extrude(tokens: Sequence[int], bit: int = BIT,
                   shifted: bool = True) -> Dict:
    """Decode a 19-token extrude block back to float parameters.

    ``shifted=True`` expects merged-stream tokens (``EXTRA_PAD`` applied, so the
    block ends with ``1``); ``shifted=False`` expects the raw block.
    """
    if len(tokens) != EXT_SEQ_LEN:
        raise ValueError("extrude block must hold %d tokens" % EXT_SEQ_LEN)
    pad = EXTRA_PAD if shifted else 0
    end = int(tokens[INDEX_END])
    if end != pad:
        raise ValueError("extrude block must end with the SE end token")

    def val(i: int) -> int:
        return int(tokens[i]) - EXT_PAD - pad

    value = [dequantize(val(i), bit, -EXTRUDE_R, EXTRUDE_R) for i in range(2)]
    origin = [dequantize(val(i), bit, -EXTRUDE_R, EXTRUDE_R) for i in range(2, 5)]
    rot = [int(tokens[i]) - R_PAD - pad for i in range(5, 14)]
    op = int(tokens[INDEX_OP]) - pad
    if op not in OP_NAME:
        raise ValueError("invalid extrude operation token: %d" % op)
    scale = dequantize(val(INDEX_SCALE), bit, 0.0, SCALE_R)
    offset = [dequantize(val(i), bit, -OFFSET_R, OFFSET_R) for i in (16, 17)]

    return {
        "value": value,
        "origin": origin,
        "rotation": rot,
        "x_axis": rot[0:3],
        "y_axis": rot[3:6],
        "z_axis": rot[6:9],
        "op": op,
        "op_name": OP_NAME[op],
        "scale": scale,
        "offset": offset,
    }


def extrude_flags(num_se: int) -> List[int]:
    """The field-flag stream for ``num_se`` extrude blocks, plus the final 0."""
    if num_se < 0:
        raise ValueError("num_se must be non-negative")
    return EXT_FLAGS * num_se + [0]


def flag_field_positions(flag: int) -> List[int]:
    """Positions inside one extrude block that carry ``flag``."""
    return [i for i, f in enumerate(EXT_FLAGS) if f == flag]


def is_valid_extrude_block(tokens: Sequence[int], bit: int = BIT,
                           shifted: bool = True) -> bool:
    """Cheap structural validity check for one extrude block."""
    try:
        decode_extrude(tokens, bit, shifted)
    except (ValueError, IndexError):
        return False
    pad = EXTRA_PAD if shifted else 0
    hi = 2 ** bit - 1 + EXT_PAD + pad
    for i in list(range(0, 5)) + [INDEX_SCALE, 16, 17]:
        if not pad <= int(tokens[i]) <= hi:
            return False
    for i in range(5, 14):
        if int(tokens[i]) - pad not in (R_PAD - 1, R_PAD, R_PAD + 1):
            return False
    return True
