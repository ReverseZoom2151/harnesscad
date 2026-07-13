"""Vitruvion primitive-token codec: the exact vocabulary, quantiser and 3-stream layout.

Vitruvion (Seff et al., ICLR 2022 -- ``img2cad/dataset.py`` + ``img2cad/data_utils.py``)
tokenises a *normalised* sketch (see ``geometry.vitruvion_sketch_norm``) into **three
parallel streams** of equal length, all consumed as embeddings that are summed:

  * ``val``   -- the value: a control token, a quantised coordinate bin, or a flag.
  * ``coord`` -- *which parameter slot* the value fills (x of the arc start, the radius,
    ...).  This is what lets one flat stream carry heterogeneous primitives.
  * ``pos``   -- *which primitive* the value belongs to (a 1-based primitive index,
    shared by every token of that primitive).

Vocabulary (``num_bins = n``, default 64)::

    0            Pad
    1            Start
    2            Stop
    3..6         Arc, Circle, Line, Point         (entity-type tokens)
    7 .. 7+n-1   quantised coordinate bins        (bin + len(Token), len(Token) == 7)
    7+n          isConstruction = True
    7+n+1        isConstruction = False

``coord`` vocabulary: ``0`` pad, ``1`` NON_COORD (control/type/flag tokens), then one id
per parameter slot, allocated by walking ``[Arc, Circle, Line, Point]`` in order:
Arc 2..7 (6 params), Circle 8..10 (3), Line 11..14 (4), Point 15..16 (2).

QUANTISATION -- differs from every other sketch quantiser in this harness
------------------------------------------------------------------------
Domain ``[-0.5, 0.5]`` (guaranteed by the long-axis-1 normalisation), and::

    bin   = int((v + 0.5) / 1.0 * n)          # TRUNCATION, then clamp n -> n-1
    value = (bin + 0.5) / n - 0.5             # BIN CENTRE, not the bin's left edge

So Vitruvion is a **floor/truncating quantiser with bin-centre reconstruction**, at a
default of ``n = 64`` (6 bits).  Contrast with what is already in this harness:

  * ``reconstruction.deepcad2_numericalize`` -- DeepCAD: 256 levels (8 bit) over
    ``[-1, 1]``, ``round`` (half-to-even), reconstruction at the *level*, not a centre.
  * ``reconstruction.gencad2_sketch_quantize`` -- GenCAD: DeepCAD's rounding scheme.
  * SkexGen -- 6-bit *truncating* like Vitruvion, but reconstructs at the level.

The bin-centre dequantisation means Vitruvion's round-trip error is bounded by half a
bin (``1/(2n)``) and is *unbiased*, whereas a floor-quantise/floor-dequantise pair (as
in SkexGen) biases every coordinate downward by half a bin.  Feeding Vitruvion bins to a
DeepCAD-style dequantiser (or vice versa) shifts every primitive by ``1/(2n)`` of the
sketch's long axis -- a silent, systematic offset.  A value exactly at the top of the
domain (``+0.5``) would land in bin ``n`` and is clamped back to ``n - 1``.  Values are
rounded to 10 decimals before the range check, so float noise at the boundary does not
trip the guard.

Pointer targets (``gather_idxs``)
---------------------------------
The constraint model (see ``reconstruction.vitruvion_constraint_tokens``) refers to a
primitive *or a specific point of it* by pointing at a position in this primitive stream.
The positions are fixed per type, as offsets from the entity-type token:

    Arc [0, 1, 3, 5]   (entity, start.x, mid.x, end.x)
    Line [0, 1, 3]     (entity, start.x, end.x)
    Circle [0, 1]      (entity, centre.x)
    Point [0]          (entity)

which is exactly the sub-node structure of SketchGraphs (an arc owns centre/start/end
sub-nodes).  Index 0 of ``gather_idxs`` is reserved for the "external" node.

Pure stdlib.
"""

from __future__ import annotations

import enum
from typing import Dict, List, Optional, Sequence, Tuple

from harnesscad.domain.geometry.vitruvion_sketch_norm import (
    NUM_PARAMS,
    VArc,
    VCircle,
    VLine,
    VPoint,
    entity_from_params,
    parameterize_entity,
)

__all__ = [
    "Token",
    "NON_COORD_TOKEN",
    "COORD_TOKEN_MAP",
    "GATHER_MAP",
    "MIN_VAL",
    "MAX_VAL",
    "DEFAULT_NUM_BINS",
    "quantize_params",
    "dequantize_params",
    "construction_tokens",
    "coordinate_token_range",
    "vocabulary_size",
    "pad_or_truncate",
    "tokenize_sketch",
    "param_seq_from_tokens",
    "entities_from_tokens",
]


class Token(enum.IntEnum):
    """Non-parameter value tokens of the primitive model."""

    Pad = 0
    Start = 1
    Stop = 2
    Arc = 3
    Circle = 4
    Line = 5
    Point = 6


TOKEN_BY_TYPE = {
    VArc: Token.Arc,
    VCircle: Token.Circle,
    VLine: Token.Line,
    VPoint: Token.Point,
}

TYPE_BY_TOKEN = {v: k for k, v in TOKEN_BY_TYPE.items()}

NON_COORD_TOKEN = 1  # 0 is reserved for padding

# Coordinate-slot ids, allocated by walking the types in this fixed order.
COORD_TOKEN_MAP: Dict[type, List[int]] = {}
_tok = NON_COORD_TOKEN + 1
for _ent_type in (VArc, VCircle, VLine, VPoint):
    COORD_TOKEN_MAP[_ent_type] = list(range(_tok, _tok + NUM_PARAMS[_ent_type]))
    _tok += NUM_PARAMS[_ent_type]

# Pointer offsets (from the entity-type token) that the constraint model may address.
GATHER_MAP: Dict[type, List[int]] = {
    VArc: [0, 1, 3, 5],
    VCircle: [0, 1],
    VLine: [0, 1, 3],
    VPoint: [0],
}

MIN_VAL = -0.5
MAX_VAL = 0.5
DEFAULT_NUM_BINS = 64


# ---------------------------------------------------------------------------
# Quantisation
# ---------------------------------------------------------------------------
def quantize_params(params: Sequence[float], n_bins: int = DEFAULT_NUM_BINS) -> List[int]:
    """Truncating quantiser of ``[-0.5, 0.5]`` into ``[0, n_bins - 1]``.

    Raises ``ValueError`` when a parameter (rounded to 10 decimals) escapes the domain.
    """
    if n_bins < 1:
        raise ValueError("n_bins must be positive")

    rounded = [round(float(p), 10) for p in params]
    for value in rounded:
        if value < MIN_VAL or value > MAX_VAL:
            raise ValueError(
                "Parameters must be in [{}, {}]. Got {}.".format(MIN_VAL, MAX_VAL, value)
            )

    bins = []
    for value in rounded:
        # int() truncates; the shifted value is non-negative, so this is a floor.
        b = int((value - MIN_VAL) / (MAX_VAL - MIN_VAL) * n_bins)
        if b == n_bins:  # only reachable for value == MAX_VAL
            b -= 1
        bins.append(b)
    return bins


def clip_params(params: Sequence[float]) -> List[float]:
    """Clamp parameters into the quantiser domain (Vitruvion's fallback path)."""
    return [min(MAX_VAL, max(MIN_VAL, float(p))) for p in params]


def dequantize_params(
    bins: Sequence[int], n_bins: int = DEFAULT_NUM_BINS
) -> List[float]:
    """Inverse of :func:`quantize_params`, reconstructing at the **centre** of the bin."""
    for b in bins:
        if not isinstance(b, int) or isinstance(b, bool):
            raise ValueError("quantized params must be ints")
        if b < 0 or b > n_bins - 1:
            raise ValueError("Invalid quantized param: {}".format(b))
    return [(b + 0.5) / n_bins * (MAX_VAL - MIN_VAL) + MIN_VAL for b in bins]


def construction_tokens(n_bins: int = DEFAULT_NUM_BINS) -> Dict[bool, int]:
    """The two ``isConstruction`` flag tokens, which sit just above the coordinate bins."""
    return {True: len(Token) + n_bins, False: len(Token) + n_bins + 1}


def coordinate_token_range(n_bins: int = DEFAULT_NUM_BINS) -> Tuple[int, int]:
    """Inclusive ``(lowest, highest)`` value token that encodes a coordinate bin."""
    return (len(Token), len(Token) + n_bins - 1)


def vocabulary_size(n_bins: int = DEFAULT_NUM_BINS) -> int:
    """Total number of distinct value tokens: controls + types + bins + 2 flags."""
    return len(Token) + n_bins + 2


# ---------------------------------------------------------------------------
# Tokenisation
# ---------------------------------------------------------------------------
def pad_or_truncate(tokens: Sequence[int], max_length: Optional[int] = None) -> List[int]:
    """Pad with ``Token.Pad`` or truncate to ``max_length`` (``None`` leaves as is)."""
    values = list(tokens)
    if max_length is None:
        return values
    if len(values) > max_length:
        return values[:max_length]
    return values + [int(Token.Pad)] * (max_length - len(values))


def tokenize_sketch(
    entities: Sequence[object],
    num_bins: int = DEFAULT_NUM_BINS,
    max_length: Optional[int] = None,
    include_construction: bool = True,
    include_stop: bool = True,
) -> Tuple[Dict[str, List[int]], List[int]]:
    """Tokenise a normalised sketch into the ``val`` / ``coord`` / ``pos`` streams.

    Returns ``(streams, gather_idxs)``.  ``gather_idxs`` maps a SketchGraphs node index
    (1-based; 0 is the external node) to the position in the *unpadded* ``val`` stream
    that the constraint model points at.  Parameters that escape the quantiser domain
    are clamped rather than raising -- Vitruvion's own fallback, which exists because
    zero-length segments can produce out-of-range values after normalisation.
    """
    construction_map = construction_tokens(num_bins)

    val: List[int] = [int(Token.Start)]
    coord: List[int] = [NON_COORD_TOKEN]
    pos_idx = 1  # 0 is reserved for padding
    pos: List[int] = [pos_idx]

    gather_idxs: List[int] = [0]  # index 0: the external node

    for entity in entities:
        ent_type = type(entity)
        if ent_type not in TOKEN_BY_TYPE:
            raise ValueError("unsupported entity type: {!r}".format(ent_type))

        gather_idxs.extend(len(val) + offset for offset in GATHER_MAP[ent_type])

        val.append(int(TOKEN_BY_TYPE[ent_type]))
        coord.append(NON_COORD_TOKEN)
        pos_idx += 1
        pos.append(pos_idx)

        params = parameterize_entity(entity)
        try:
            bins = quantize_params(params, num_bins)
        except ValueError:
            bins = quantize_params(clip_params(params), num_bins)

        val.extend(b + len(Token) for b in bins)
        coord.extend(COORD_TOKEN_MAP[ent_type])
        pos.extend([pos_idx] * len(bins))

        if include_construction:
            val.append(construction_map[bool(entity.is_construction)])
            coord.append(NON_COORD_TOKEN)
            pos.append(pos_idx)

    if include_stop:
        val.append(int(Token.Stop))
        coord.append(NON_COORD_TOKEN)
        pos.append(pos_idx + 1)

    streams = {
        "val": pad_or_truncate(val, max_length),
        "coord": pad_or_truncate(coord, max_length),
        "pos": pad_or_truncate(pos, max_length),
    }
    return streams, gather_idxs


# ---------------------------------------------------------------------------
# Decoding
# ---------------------------------------------------------------------------
def param_seq_from_tokens(
    tokens: Sequence[int], num_bins: int = DEFAULT_NUM_BINS
) -> List[Tuple[List[int], bool]]:
    """Split a ``val`` stream back into ``(quantised params, isConstruction)`` per entity.

    Decoding is *type-free*: the entity type is recovered later from the parameter count,
    exactly as in the reference.  Decoding stops at the first ``Stop`` or ``Pad``.
    """
    reverse_construction = {v: k for k, v in construction_tokens(num_bins).items()}
    low, high = coordinate_token_range(num_bins)

    all_params: List[Tuple[List[int], bool]] = []
    current: List[int] = []
    is_construction = False

    for token in tokens:
        token = int(token)
        if token == int(Token.Start):
            continue
        if token < len(Token):
            if current:
                all_params.append((current, is_construction))
                current = []
        if token in (int(Token.Stop), int(Token.Pad)):
            break
        if token >= len(Token):
            if low <= token <= high:
                is_construction = False  # reset: the flag follows the coordinates
                current.append(token - len(Token))
            else:
                is_construction = reverse_construction[token]

    if current:
        all_params.append((current, is_construction))
    return all_params


def entities_from_tokens(
    tokens: Sequence[int], num_bins: int = DEFAULT_NUM_BINS
) -> List[object]:
    """Rebuild entities from a ``val`` stream (degenerate arcs decode to ``None``)."""
    out: List[object] = []
    for idx, (bins, is_construction) in enumerate(param_seq_from_tokens(tokens, num_bins)):
        entity = entity_from_params(dequantize_params(bins, num_bins), str(idx + 1))
        if entity is not None:
            entity.is_construction = is_construction
        out.append(entity)
    return out
