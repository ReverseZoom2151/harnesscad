"""Flat CAD command/parameter vectorization.

The companion module :mod:`reconstruction.hnc_spl_tree` captures the hierarchical
S-P-L *code tree* and the 6-bit coordinate quantization. It does **not** capture
the separate *flat* command/parameter sequence that feeds the CAD
reconstruction stage. That flat encoding differs from the fixed-row schemes
(:mod:`reconstruction.deepcad_command_spec`,
:mod:`reconstruction.skexgen_token_format`) in four correctness-relevant ways:

1. **Vocabulary of six integer commands with structural end-tokens promoted to
   command *types*** (not a single ``SOL``/``EOS`` pair)::

       1 = SKETCH_END   2 = FACE_END   3 = LOOP_END
       4 = LINE         5 = ARC        6 = CIRCLE

   Loop / face / sketch closure are three distinct commands, encoding the S-P-L
   hierarchy directly in the flat stream.

2. **Implicit-endpoint (start-only) curve encoding.** The fixed-row scheme stores
   each curve's *end* point and implies the start from the previous curve. This
   scheme stores the *new leading* points and implies the endpoint from the *next* curve's start
   (the loop is pre-chained so curve[i].end == curve[i+1].start):

       * LINE   -> start point only            (2 numbers)
       * ARC    -> start + mid                 (4 numbers)  <- no end point
       * CIRCLE -> 4 cardinal points N,S,E,W   (8 numbers)

   Every curve pads to a fixed **8-slot** parameter vector with the ``-1``
   sentinel (the fixed-row vector is 16-slot).

3. **Circle as four axis cardinal points** ``(cx, cy+r), (cx, cy-r), (cx+r, cy),
   (cx-r, cy)`` -- not ``(cx, cy, r)``.

4. **An 11-slot extrude vector** ``[center(3), scale(1), ext_v(2), t_orig(3),
   rot_idx(1), op(1)]`` where ``rot_idx`` is the 25-way categorical orientation
   index from :mod:`reconstruction.hnc_rotation_codebook` (not continuous angles)
   and ``op`` is ``1=add, 2=cut, 3=intersect``.

It also uses **two different sketch normalizations**: the loop-level codes use the
half bounding-box *diagonal* (:func:`normalize_diagonal`), while the flat CAD
sequence uses *half the largest extent* (:func:`normalize_max_extent`).

Pure stdlib, deterministic. No numpy, no learned components.
"""

from __future__ import annotations

import math
from typing import List, Sequence, Tuple

from harnesscad.domain.reconstruction.tokens.hnc_rotation_codebook import quantize_orientation

Vec2 = Tuple[float, float]

# --- command vocabulary (integers, structural tokens are command types) -----
SKETCH_END = 1
FACE_END = 2
LOOP_END = 3
LINE = 4
ARC = 5
CIRCLE = 6

PARAM_WIDTH = 8          # fixed parameter-vector width for curve commands
SENTINEL = -1

SKETCH_R = 1.0           # coordinate range

# extrude boolean operation codes
OP_ADD = 1               # join / new-body
OP_CUT = 2               # cut
OP_INTERSECT = 3         # intersect

_SET_OP_MAP = {
    "JoinFeatureOperation": OP_ADD,
    "NewBodyFeatureOperation": OP_ADD,
    "CutFeatureOperation": OP_CUT,
    "IntersectFeatureOperation": OP_INTERSECT,
}


def set_op_code(set_op: str) -> int:
    """Map a set-operation string to the ``1/2/3`` extrude op code."""
    try:
        return _SET_OP_MAP[set_op]
    except KeyError:
        raise ValueError(f"unknown set operation {set_op!r}")


# --- quantization ------------------------------------------------------------
def quantize(value: float, n_bits: int = 8, min_range: float = -1.0,
             max_range: float = 1.0) -> int:
    """Quantize a scalar in ``[min_range, max_range]`` to ``[0, 2**n_bits - 1]``.

    Includes the final clip. Note the level count is ``2**n_bits - 1``
    (255 for 8-bit), not ``n_bits**2 - 1``.
    """
    range_quantize = (1 << n_bits) - 1
    q = (value - min_range) * range_quantize / (max_range - min_range)
    q = int(math.floor(q))  # truncation toward zero; q >= 0 here so floor == trunc
    if q < 0:
        q = 0
    elif q > range_quantize:
        q = range_quantize
    return q


def quantize_point(point: Sequence[float], n_bits: int = 8,
                   min_range: float = -SKETCH_R,
                   max_range: float = SKETCH_R) -> Tuple[int, int]:
    return (quantize(point[0], n_bits, min_range, max_range),
            quantize(point[1], n_bits, min_range, max_range))


# --- normalization (two distinct schemes) -----------------------------------
def _bbox(vertices: Sequence[Sequence[float]]) -> Tuple[Vec2, Vec2]:
    xs = [v[0] for v in vertices]
    ys = [v[1] for v in vertices]
    return (min(xs), min(ys)), (max(xs), max(ys))


def center_of(vertices: Sequence[Sequence[float]]) -> Vec2:
    """Bounding-box centre ``0.5*(min+max)``."""
    (xmin, ymin), (xmax, ymax) = _bbox(vertices)
    return (0.5 * (xmin + xmax), 0.5 * (ymin + ymax))


def normalize_diagonal(vertices: Sequence[Sequence[float]]) -> float:
    """Loop-level scale: half the bounding-box diagonal.

    ``0.5*sqrt(w**2 + h**2)``. Assumes ``vertices`` already centred.
    """
    (xmin, ymin), (xmax, ymax) = _bbox(vertices)
    w, h = xmax - xmin, ymax - ymin
    return 0.5 * math.sqrt(w * w + h * h)


def normalize_max_extent(vertices: Sequence[Sequence[float]]) -> float:
    """CAD-sequence-level scale: half the largest extent.

    ``max(w, h) / (2*SKETCH_R)``. Assumes centred.
    """
    (xmin, ymin), (xmax, ymax) = _bbox(vertices)
    w, h = xmax - xmin, ymax - ymin
    return max(w, h) / (2 * SKETCH_R)


# --- circle cardinal points -------------------------------------------------
def circle_cardinal_points(center: Sequence[float], radius: float
                           ) -> Tuple[Vec2, Vec2, Vec2, Vec2]:
    """The four axis cardinal points in order: N, S, E, W."""
    cx, cy = center[0], center[1]
    return ((cx, cy + radius), (cx, cy - radius),
            (cx + radius, cy), (cx - radius, cy))


def _pad(values: List[int]) -> List[int]:
    if len(values) > PARAM_WIDTH:
        raise ValueError("too many parameters for an 8-slot vector")
    return values + [SENTINEL] * (PARAM_WIDTH - len(values))


# --- curve encoding (implicit-endpoint / start-only) ------------------------
def encode_line(start: Sequence[float], center: Vec2, scale: float,
                n_bits: int = 8) -> Tuple[int, List[int]]:
    s = quantize_point(_norm_pt(start, center, scale), n_bits)
    return LINE, _pad([s[0], s[1]])


def encode_arc(start: Sequence[float], mid: Sequence[float], center: Vec2,
               scale: float, n_bits: int = 8) -> Tuple[int, List[int]]:
    s = quantize_point(_norm_pt(start, center, scale), n_bits)
    m = quantize_point(_norm_pt(mid, center, scale), n_bits)
    return ARC, _pad([s[0], s[1], m[0], m[1]])


def encode_circle(circ_center: Sequence[float], radius: float, center: Vec2,
                  scale: float, n_bits: int = 8) -> Tuple[int, List[int]]:
    pts = circle_cardinal_points(circ_center, radius)
    flat: List[int] = []
    for p in pts:
        q = quantize_point(_norm_pt(p, center, scale), n_bits)
        flat.extend([q[0], q[1]])
    return CIRCLE, _pad(flat)


def _norm_pt(point: Sequence[float], center: Vec2, scale: float) -> Vec2:
    return ((point[0] - center[0]) / scale, (point[1] - center[1]) / scale)


# --- full flat sketch encoding ----------------------------------------------
def encode_sketch(faces: Sequence[Sequence[Sequence[dict]]], center: Vec2,
                  scale: float, n_bits: int = 8
                  ) -> Tuple[List[int], List[List[int]]]:
    """Encode a chained, ordered sketch into flat ``(cmds, params)``.

    ``faces`` is a list of faces; each face is a list of loops; each loop is a
    list of curve dicts. A curve dict is one of::

        {"type": "line",   "start": (x, y)}
        {"type": "arc",    "start": (x, y), "mid": (x, y)}
        {"type": "circle", "center": (x, y), "radius": r}

    Emits LOOP_END / FACE_END / SKETCH_END structural tokens, with a trailing
    sketch-end after the last face.
    """
    cmds: List[int] = []
    params: List[List[int]] = []
    for face in faces:
        for loop in face:
            for curve in loop:
                t = curve["type"]
                if t == "line":
                    c, p = encode_line(curve["start"], center, scale, n_bits)
                elif t == "arc":
                    c, p = encode_arc(curve["start"], curve["mid"], center,
                                      scale, n_bits)
                elif t == "circle":
                    c, p = encode_circle(curve["center"], curve["radius"],
                                         center, scale, n_bits)
                else:
                    raise ValueError(f"unknown curve type {t!r}")
                cmds.append(c)
                params.append(p)
            cmds.append(LOOP_END)
            params.append([SENTINEL] * PARAM_WIDTH)
        cmds.append(FACE_END)
        params.append([SENTINEL] * PARAM_WIDTH)
    cmds.append(SKETCH_END)
    params.append([SENTINEL] * PARAM_WIDTH)
    return cmds, params


# --- extrude vector packing -------------------------------------------------
def encode_extrude(center: Sequence[float], scale: float,
                   ext_values: Sequence[float], t_orig: Sequence[float],
                   t_x: Sequence[float], t_y: Sequence[float],
                   t_z: Sequence[float], set_op: str,
                   n_bits: int = 8) -> List[int]:
    """Pack the 11-slot extrude parameter vector.

    ``[center(3), scale(1), ext_v(2), t_orig(3), rot_idx(1), op(1)]``.
    """
    if len(center) != 3:
        raise ValueError("center must have length 3")
    if len(ext_values) != 2:
        raise ValueError("ext_values must have length 2 (two-sided extrude)")
    if len(t_orig) != 3:
        raise ValueError("t_orig must have length 3")

    out: List[int] = []
    out += [quantize(c, n_bits, -1.0, 1.0) for c in center]     # center 3
    out += [quantize(scale, n_bits, 0.0, 1.0)]                  # scale 1
    out += [quantize(e, n_bits, -1.0, 1.0) for e in ext_values]  # ext_v 2
    out += [quantize(t, n_bits, -SKETCH_R, SKETCH_R) for t in t_orig]  # t_orig 3
    out += [quantize_orientation(t_x, t_y, t_z)]                # rot_idx 1
    out += [set_op_code(set_op)]                                # op 1
    return out
