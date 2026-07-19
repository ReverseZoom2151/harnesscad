"""Flat sketch token format.

This scheme encodes a sketch-and-extrude CAD model as *three parallel token
streams* plus an extrude stream, which is a different representation from the
fixed-row command/argument matrix already in the harness
(``reconstruction/deepcad2_vector_layout``) and from the 8-bit quantisation
scheme (``reconstruction/gencad2_sketch_quantize``).

Differences worth stating explicitly:

* the 16-slot fixed-row layout: fixed 17-column rows ``(command, 16 args)``,
  args quantised to 256 levels, one row per curve.
* the 8-bit quantisation scheme: same row layout, 8-bit args, separate
  sketch/shape normalisation.
* this scheme: *variable-length flat* token streams.  Sketch geometry is
  quantised to ``2**bit`` levels per axis (bit = 6 -> 64 levels) and then
  flattened to a single **pixel** token ``y * 2**bit + x``.  A curve is a
  variable number of pixel tokens (line = 1, arc = 2, circle = 4) terminated by
  a CURVE_END token; loops, faces and the sketch each get their own end token.
  The curve *type* lives in a parallel **command** stream, and the raw ``(x, y)``
  pair lives in a parallel **coordinate** stream.  The end token for a curve is
  implicit: the *next* curve's first pixel is the current curve's end vertex
  (a "curve-gen" style loop encoding).

Token values (as they appear in the merged stream fed to the transformer, i.e.
after the ``EXTRA_PAD`` shift)::

    0                 padding / end-of-model
    1                 end of one sketch-extrude pair (SE_END)
    2                 end of face
    3                 end of loop
    4                 end of curve
    5 .. 5+2**(2b)-1  pixel token (5 + y * 2**b + x)

The raw (unshifted) streams use the pads
``PIX_PAD = 4``, ``COORD_PAD = 4``, ``CMD_PAD = 3``; the ``+1`` EXTRA_PAD is
applied when the streams are merged.  Both forms are produced here.

Deterministic, stdlib only.
"""
from __future__ import annotations

from math import sqrt
from typing import Dict, List, Sequence, Tuple

Vec2 = Tuple[float, float]

# --- constants ---------------------------------------------------------------
BIT = 6                 # quantisation bits per sketch axis
SKETCH_R = 1.0          # sketch coords are normalised to [-1, 1]
EXTRUDE_R = 1.0
SCALE_R = 1.4
OFFSET_R = 0.9

PIX_PAD = 4
CMD_PAD = 3
COORD_PAD = 4
EXT_PAD = 1
EXTRA_PAD = 1
R_PAD = 2

# --- merged-stream structural tokens ----------------------------------------
PAD = 0
SE_END = 1
FACE_END = 2
LOOP_END = 3
CURVE_END = 4
PIX_OFFSET = PIX_PAD + EXTRA_PAD          # 5: first real pixel token

# raw (pre-EXTRA_PAD) sentinels used inside the un-merged streams
RAW_CURVE_END = -1
RAW_LOOP_END = -2
RAW_FACE_END = -3
RAW_SKETCH_END = -4

# command stream values (raw, before CMD_PAD)
CMD_LINE = 0
CMD_ARC = 1
CMD_CIRCLE = 2

CURVE_NUM_POINTS = {"line": 1, "arc": 2, "circle": 4}
CURVE_CMD = {"line": CMD_LINE, "arc": CMD_ARC, "circle": CMD_CIRCLE}


# --- quantisation -----------------------------------------------------------
def quantize(value: float, bit: int = BIT, min_range: float = -SKETCH_R,
             max_range: float = SKETCH_R) -> int:
    """Quantisation: affine map to ``[0, 2**bit - 1]``, clipped.

    Truncates after clipping (values are non-negative post-clip, so
    truncation == floor).
    """
    levels = 2 ** bit - 1
    q = (value - min_range) * levels / (max_range - min_range)
    if q < 0.0:
        q = 0.0
    if q > levels:
        q = float(levels)
    return int(q)


def dequantize(level: float, bit: int = BIT, min_range: float = -SKETCH_R,
               max_range: float = SKETCH_R) -> float:
    """Inverse of :func:`quantize`."""
    levels = 2 ** bit - 1
    return level * (max_range - min_range) / levels + min_range


def quantize_point(point: Vec2, bit: int = BIT) -> Tuple[int, int]:
    return (quantize(point[0], bit), quantize(point[1], bit))


def dequantize_point(level: Sequence[int], bit: int = BIT) -> Vec2:
    return (dequantize(level[0], bit), dequantize(level[1], bit))


# --- pixel token <-> xy ------------------------------------------------------
def pixel_from_xy(x: int, y: int, bit: int = BIT) -> int:
    """Flatten a quantised ``(x, y)`` to the raster ``pixel`` index."""
    n = 2 ** bit
    if not (0 <= x < n and 0 <= y < n):
        raise ValueError("xy out of range for bit=%d: %r" % (bit, (x, y)))
    return y * n + x


def xy_from_pixel(pixel: int, bit: int = BIT) -> Tuple[int, int]:
    n = 2 ** bit
    if not 0 <= pixel < n * n:
        raise ValueError("pixel out of range for bit=%d: %r" % (bit, pixel))
    return (pixel % n, pixel // n)


def pixel_vocab_size(bit: int = BIT) -> int:
    """Size of the merged-stream pixel vocabulary (pads + all raster cells)."""
    return 2 ** (2 * bit) + PIX_OFFSET


def coord_vocab_size(bit: int = BIT) -> int:
    return 2 ** bit + COORD_PAD + EXTRA_PAD


def command_vocab_size() -> int:
    return CMD_CIRCLE + CMD_PAD + EXTRA_PAD + 1     # 7


# --- normalisation ----------------------------------------------------------
def curve_points(curve: Dict) -> List[Vec2]:
    """The points a curve *emits as tokens* (the end vertex is implicit)."""
    t = curve["type"]
    if t == "line":
        return [tuple(curve["start"])]
    if t == "arc":
        return [tuple(curve["start"]), tuple(curve["mid"])]
    if t == "circle":
        return [tuple(curve["pt%d" % i]) for i in (1, 2, 3, 4)]
    raise ValueError("unknown curve type: %r" % (t,))


def curve_vertices(curve: Dict) -> List[Vec2]:
    """All vertices of a curve (used for the sketch bbox / normalisation)."""
    t = curve["type"]
    if t == "line":
        return [tuple(curve["start"]), tuple(curve["end"])]
    if t == "arc":
        return [tuple(curve["start"]), tuple(curve["mid"]), tuple(curve["end"])]
    if t == "circle":
        return [tuple(curve["pt%d" % i]) for i in (1, 2, 3, 4)]
    raise ValueError("unknown curve type: %r" % (t,))


def sketch_vertices(sketch: Sequence[Sequence[Sequence[Dict]]]) -> List[Vec2]:
    out: List[Vec2] = []
    for face in sketch:
        for loop in face:
            for curve in loop:
                out.extend(curve_vertices(curve))
    if not out:
        raise ValueError("empty sketch")
    return out


def center_vertices(vertices: Sequence[Vec2]) -> Vec2:
    """Bounding-box centre."""
    xs = [v[0] for v in vertices]
    ys = [v[1] for v in vertices]
    return (0.5 * (min(xs) + max(xs)), 0.5 * (min(ys) + max(ys)))


def normalize_scale(vertices: Sequence[Vec2]) -> float:
    """Half the bbox diagonal."""
    xs = [v[0] for v in vertices]
    ys = [v[1] for v in vertices]
    ex = max(xs) - min(xs)
    ey = max(ys) - min(ys)
    return 0.5 * sqrt(ex * ex + ey * ey)


def sketch_center_scale(sketch: Sequence[Sequence[Sequence[Dict]]]) -> Tuple[Vec2, float]:
    verts = sketch_vertices(sketch)
    center = center_vertices(verts)
    centered = [(v[0] - center[0], v[1] - center[1]) for v in verts]
    return center, normalize_scale(centered)


# --- encoding ---------------------------------------------------------------
def encode_sketch(sketch: Sequence[Sequence[Sequence[Dict]]], bit: int = BIT) -> Dict:
    """Sketch -> (xy, pix, cmd) streams.

    ``sketch`` is a list of faces; a face is a list of loops (outer first); a
    loop is a list of curve dicts, already in canonical order (see
    ``reconstruction/skexgen_canonical_order``).

    Returns the *padded raw* streams (``+COORD_PAD`` / ``+PIX_PAD`` /
    ``+CMD_PAD``) as stored on disk, plus the sketch ``center`` and ``scale``.
    """
    center, scale = sketch_center_scale(sketch)
    if scale <= 0.0:
        raise ValueError("degenerate sketch (zero scale)")

    xy: List[Tuple[int, int]] = []
    pix: List[int] = []
    cmd: List[int] = []

    def emit(point: Vec2) -> None:
        qx = quantize((point[0] - center[0]) / scale, bit)
        qy = quantize((point[1] - center[1]) / scale, bit)
        xy.append((qx, qy))
        pix.append(pixel_from_xy(qx, qy, bit))

    for face in sketch:
        for loop in face:
            for curve in loop:
                for point in curve_points(curve):
                    emit(point)
                xy.append((RAW_CURVE_END, RAW_CURVE_END))
                pix.append(RAW_CURVE_END)
                cmd.append(CURVE_CMD[curve["type"]])
            xy.append((RAW_LOOP_END, RAW_LOOP_END))
            pix.append(RAW_LOOP_END)
            cmd.append(-1)
        xy.append((RAW_FACE_END, RAW_FACE_END))
        pix.append(RAW_FACE_END)
        cmd.append(-2)
    xy.append((RAW_SKETCH_END, RAW_SKETCH_END))
    pix.append(RAW_SKETCH_END)
    cmd.append(-3)

    return {
        "xy": [(a + COORD_PAD, b + COORD_PAD) for a, b in xy],
        "pix": [p + PIX_PAD for p in pix],
        "cmd": [c + CMD_PAD for c in cmd],
        "center": center,
        "scale": scale,
    }


def shift_stream(stream: Sequence[int]) -> List[int]:
    """Apply ``EXTRA_PAD`` (the ``+1`` used when streams are merged)."""
    return [int(t) + EXTRA_PAD for t in stream]


def merge_se(pix_streams: Sequence[Sequence[int]],
             ext_streams: Sequence[Sequence[int]]) -> List[int]:
    """Interleave per-SE pixel and extrude streams into one flat token list.

    ``[pix_0, ext_0, pix_1, ext_1, ..., 0]`` with ``EXTRA_PAD`` applied; the
    trailing ``0`` is the end-of-model token.
    """
    if len(pix_streams) != len(ext_streams):
        raise ValueError("pix/ext stream count mismatch")
    out: List[int] = []
    for pix, ext in zip(pix_streams, ext_streams):
        out.extend(shift_stream(pix))
        out.extend(shift_stream(ext))
    out.append(PAD)
    return out


def strip_padding(tokens: Sequence[int]) -> List[int]:
    """Everything before the first ``PAD`` token."""
    out: List[int] = []
    for t in tokens:
        if t == PAD:
            break
        out.append(int(t))
    return out


def split_on(tokens: Sequence[int], sentinel: int) -> List[List[int]]:
    """Split *inclusive* of the sentinel; the trailing empty group is dropped.

    Requires the sequence to end with the sentinel.
    """
    groups: List[List[int]] = []
    cur: List[int] = []
    for t in tokens:
        cur.append(int(t))
        if t == sentinel:
            groups.append(cur)
            cur = []
    if cur:
        raise ValueError("tokens do not terminate with sentinel %d" % sentinel)
    return groups
