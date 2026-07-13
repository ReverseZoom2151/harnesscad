"""Sketch2CAD (SIGGRAPH Asia 2020) multi-channel training-block codec.

The Sketch2CAD data pipeline stores every (context, stroke, operation) sample as
a single zlib-compressed block of ``H x W x 17`` little-endian float32 values in
row-major, channel-interleaved order.  The C++ custom TensorFlow op
``DecodeBlock`` (``networkTraining/libs/decode_block_op.cc``) inflates that
stream and splits it into a 6-channel *input* tensor and a 15-channel *label*
tensor, deriving four of the label channels on the fly.  All of that is pure,
deterministic data marshalling; this module reimplements it in stdlib Python.

Raw channel layout (17)::

    0  user_stroke          8  profile_curve
    1  scaffold_lines       9  offset_curve
    2  context_normal_x    10  shape_mask
    3  context_normal_y    11  offset_distance
    4  context_normal_z    12  offset_direction_x
    5  context_depth       13  offset_direction_y
    6  stitching_face      14  offset_direction_z
    7  base_curve          15  offset_sign
                           16  operation_type

Derived label channels (the decoder's contribution, not stored on disk):

  * ``curve_class`` — a 3-channel one-hot-ish (base, offset, profile) map built
    from the three curve masks with a fixed conflict priority
    ``base > offset > profile``: a pixel claimed by ``base`` is removed from
    ``offset`` and ``profile``; a pixel claimed by ``offset`` is removed from
    ``profile``.  Overlapping strokes therefore never produce ambiguous labels.
  * ``curve_reg`` — the same decision collapsed to a single scalar index map
    (0 = base/background, 1 = offset, 2 = profile), used by the curve-regression
    heads.

Stdlib-only, deterministic.
"""
from __future__ import annotations

import struct
import zlib
from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

RAW_CHANNELS: Tuple[str, ...] = (
    "user_stroke",
    "scaffold_lines",
    "context_normal_x",
    "context_normal_y",
    "context_normal_z",
    "context_depth",
    "stitching_face",
    "base_curve",
    "profile_curve",
    "offset_curve",
    "shape_mask",
    "offset_distance",
    "offset_direction_x",
    "offset_direction_y",
    "offset_direction_z",
    "offset_sign",
    "operation_type",
)
NUM_RAW_CHANNELS = len(RAW_CHANNELS)  # 17
NUM_INPUT_CHANNELS = 6
NUM_LABEL_CHANNELS = 15

#: raw-channel index of each of the six input channels, in decoder order.
INPUT_CHANNEL_MAP: Tuple[int, ...] = (0, 1, 2, 3, 4, 5)
#: raw-channel index of the first eleven label channels, in decoder order.
LABEL_CHANNEL_MAP: Tuple[int, ...] = (6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16)

#: curve-class ids used by the derived channels (priority order).
CURVE_BASE = 0
CURVE_OFFSET = 1
CURVE_PROFILE = 2
CURVE_CLASS_NAMES: Tuple[str, ...] = ("base", "offset", "profile")


class BlockFormatError(ValueError):
    """Raised when a byte stream does not match the declared block shape."""


@dataclass(frozen=True)
class BlockShape:
    height: int
    width: int
    channels: int = NUM_RAW_CHANNELS

    def __post_init__(self) -> None:
        if self.height <= 0 or self.width <= 0 or self.channels <= 0:
            raise BlockFormatError("block dimensions must be positive")

    @property
    def size(self) -> int:
        return self.height * self.width * self.channels

    @property
    def pixels(self) -> int:
        return self.height * self.width


def channel_index(name: str) -> int:
    """Return the raw-channel index of ``name``."""
    try:
        return RAW_CHANNELS.index(name)
    except ValueError:
        raise BlockFormatError("unknown channel: {}".format(name)) from None


# ---------------------------------------------------------------------------
# encode / decode
# ---------------------------------------------------------------------------
def encode_block(values: Sequence[float], shape: BlockShape, level: int = 6) -> bytes:
    """Pack a flat channel-interleaved float list into a zlib-compressed stream."""
    if len(values) != shape.size:
        raise BlockFormatError(
            "expected {} values, got {}".format(shape.size, len(values))
        )
    raw = struct.pack("<{}f".format(shape.size), *[float(v) for v in values])
    return zlib.compress(raw, level)


def inflate_block(stream: bytes, shape: BlockShape) -> List[float]:
    """Inflate a compressed stream into a flat float list (float32 precision).

    Mirrors ``DecodeBlockOp::inflation_byte`` including its size check.
    """
    try:
        raw = zlib.decompress(stream)
    except zlib.error as exc:  # pragma: no cover - exercised via test
        raise BlockFormatError("zlib inflation error: {}".format(exc)) from None
    if len(raw) % 4 != 0:
        raise BlockFormatError("inflated stream is not a whole number of floats")
    count = len(raw) // 4
    if count != shape.size:
        raise BlockFormatError(
            "inflated data mismatch, got {} floats, want {}".format(count, shape.size)
        )
    return list(struct.unpack("<{}f".format(count), raw))


def pixel_slice(values: Sequence[float], shape: BlockShape, row: int, col: int) -> List[float]:
    """All channels of one pixel."""
    if not (0 <= row < shape.height and 0 <= col < shape.width):
        raise BlockFormatError("pixel out of range")
    base = (row * shape.width + col) * shape.channels
    return list(values[base:base + shape.channels])


# ---------------------------------------------------------------------------
# the decoder split: input tensor / label tensor / derived channels
# ---------------------------------------------------------------------------
def _curve_labels(base: float, profile: float, offset: float) -> Tuple[float, float, float, float]:
    """Conflict-resolved (base, offset, profile) one-hot plus scalar reg label."""
    b = base
    o = 0.0 if base > 0 else offset
    p = 0.0 if (base > 0 or offset > 0) else profile
    if base > 0:
        reg = float(CURVE_BASE)
    elif offset > 0:
        reg = float(CURVE_OFFSET)
    elif profile > 0:
        reg = float(CURVE_PROFILE)
    else:
        reg = float(CURVE_BASE)
    return b, o, p, reg


def split_block(values: Sequence[float], shape: BlockShape) -> Tuple[List[float], List[float]]:
    """Split raw values into (input_data 6ch, label_data 15ch), flat and interleaved.

    Reimplements ``DecodeBlockOp::Compute``.
    """
    if len(values) != shape.size:
        raise BlockFormatError("value count does not match shape")
    inp: List[float] = [0.0] * (shape.pixels * NUM_INPUT_CHANNELS)
    lab: List[float] = [0.0] * (shape.pixels * NUM_LABEL_CHANNELS)
    for idx in range(shape.pixels):
        src = idx * shape.channels
        dst_i = idx * NUM_INPUT_CHANNELS
        for k, c in enumerate(INPUT_CHANNEL_MAP):
            inp[dst_i + k] = values[src + c]
        dst_l = idx * NUM_LABEL_CHANNELS
        for k, c in enumerate(LABEL_CHANNEL_MAP):
            lab[dst_l + k] = values[src + c]
        b, o, p, reg = _curve_labels(
            values[src + 7], values[src + 8], values[src + 9]
        )
        lab[dst_l + 11] = b
        lab[dst_l + 12] = o
        lab[dst_l + 13] = p
        lab[dst_l + 14] = reg
    return inp, lab


def decode_block(stream: bytes, shape: BlockShape) -> Tuple[List[float], List[float]]:
    """Inflate + split, the full ``DecodeBlock`` op."""
    return split_block(inflate_block(stream, shape), shape)


# ---------------------------------------------------------------------------
# named maps ("cook_raw_inputs")
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class CookedSample:
    """Named per-pixel maps, each a flat row-major list of ``H*W`` values.

    Vector maps (``context_normal``, ``offset_direction``, ``curve_class``) are
    lists of tuples.  ``operation_label`` is the scalar operation id, taken from
    pixel (0, 0) exactly as the network does.
    """
    shape: BlockShape
    user_stroke: List[float]
    scaffold_lines: List[float]
    context_normal: List[Tuple[float, float, float]]
    context_depth: List[float]
    stitching_face: List[float]
    base_curve: List[float]
    profile_curve: List[float]
    offset_curve: List[float]
    shape_mask: List[float]
    offset_distance: List[float]
    offset_direction: List[Tuple[float, float, float]]
    offset_sign: List[float]
    operation_label: int
    curve_class: List[Tuple[float, float, float]]
    curve_reg: List[float]

    def stroke_mask(self) -> List[float]:
        """``1 - user_stroke`` — the pixels the network is scored on."""
        return [1.0 - v for v in self.user_stroke]

    def as_dict(self) -> Dict[str, object]:
        return {
            "user_stroke": self.user_stroke,
            "scaffold_lines": self.scaffold_lines,
            "context_normal": self.context_normal,
            "context_depth": self.context_depth,
            "stitching_face": self.stitching_face,
            "base_curve": self.base_curve,
            "profile_curve": self.profile_curve,
            "offset_curve": self.offset_curve,
            "shape_mask": self.shape_mask,
            "offset_distance": self.offset_distance,
            "offset_direction": self.offset_direction,
            "offset_sign": self.offset_sign,
            "operation_label": self.operation_label,
            "curve_class": self.curve_class,
            "curve_reg": self.curve_reg,
        }


def cook_raw_inputs(values: Sequence[float], shape: BlockShape) -> CookedSample:
    """Turn a raw (uncompressed) block into named maps."""
    inp, lab = split_block(values, shape)
    n = shape.pixels

    def icol(k: int) -> List[float]:
        return [inp[i * NUM_INPUT_CHANNELS + k] for i in range(n)]

    def lcol(k: int) -> List[float]:
        return [lab[i * NUM_LABEL_CHANNELS + k] for i in range(n)]

    normals = [
        (
            inp[i * NUM_INPUT_CHANNELS + 2],
            inp[i * NUM_INPUT_CHANNELS + 3],
            inp[i * NUM_INPUT_CHANNELS + 4],
        )
        for i in range(n)
    ]
    off_dir = [
        (
            lab[i * NUM_LABEL_CHANNELS + 6],
            lab[i * NUM_LABEL_CHANNELS + 7],
            lab[i * NUM_LABEL_CHANNELS + 8],
        )
        for i in range(n)
    ]
    curve_cls = [
        (
            lab[i * NUM_LABEL_CHANNELS + 11],
            lab[i * NUM_LABEL_CHANNELS + 12],
            lab[i * NUM_LABEL_CHANNELS + 13],
        )
        for i in range(n)
    ]
    op_label = int(lab[10])  # pixel (0,0), label channel 10
    return CookedSample(
        shape=shape,
        user_stroke=icol(0),
        scaffold_lines=icol(1),
        context_normal=normals,
        context_depth=icol(5),
        stitching_face=lcol(0),
        base_curve=lcol(1),
        profile_curve=lcol(2),
        offset_curve=lcol(3),
        shape_mask=lcol(4),
        offset_distance=lcol(5),
        offset_direction=off_dir,
        offset_sign=lcol(9),
        operation_label=op_label,
        curve_class=curve_cls,
        curve_reg=lcol(14),
    )


def build_raw_block(channels: Dict[str, Sequence[float]], shape: BlockShape) -> List[float]:
    """Interleave named per-pixel channel maps into a flat raw block.

    Missing channels default to zero.  Each supplied map must have ``H*W`` values.
    """
    values = [0.0] * shape.size
    for name, data in channels.items():
        c = channel_index(name)
        if len(data) != shape.pixels:
            raise BlockFormatError(
                "channel {} needs {} values, got {}".format(name, shape.pixels, len(data))
            )
        for i, v in enumerate(data):
            values[i * shape.channels + c] = float(v)
    return values
