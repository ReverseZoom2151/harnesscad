"""TriPE: triplane positional encoding for TAR3D.

From "TAR3D: Creating High-Quality 3D Assets via Next-Part Prediction" (Zhang et
al., 2024), Section 3.2 ("TriPE") and Fig. 3. TriPE is the position encoding the
GPT attaches to the triplane index sequence. Although it feeds a learned model,
TriPE itself is a deterministic construction on top of Rotary Position Embedding
(RoPE) [Su et al.], and the paper specifies it exactly:

  * "it is a fusion of 2D position encoding and 1D position encoding based on the
    Rotary Position Embedding (RoPE)."
  * ``P2D in R^{h.w}`` is the RoPE for the 2D feature map; ``P1D in R^3`` the RoPE
    for the 1D sequence of 3 planes.
  * "we repeat the unit element of P2D three times and place the two newly
    emerged elements adjacent to their original element" -> ``TriP2D in R^{3.h.w}``.
  * "we repeat the three elements in P1D for h x w times to highlight the
    difference of the three feature maps" -> ``TriP1D in R^{3.h.w}``.
  * "we calculate the TriPE by performing element-wise addition of TriP2D and
    TriP1D."

This module builds the per-token ``(pos2d, pos1d)`` id layout, the RoPE encoding
vectors, and their element-wise-additive fusion, plus the actual RoPE rotation of
a feature vector so callers can verify the encoding is well-formed. The token
order matches ``reconstruction.tar3d_part_sequence.sequence_positions``: raster
over cells, three planes adjacent. Stdlib-only, deterministic.
"""

from __future__ import annotations

import math
from typing import List, Tuple

_NUM_PLANES = 3


def tripe_position_ids(h: int, w: int) -> List[Tuple[int, int]]:
    """Return ``(pos2d, pos1d)`` for each of the ``3*h*w`` sequence tokens.

    ``pos2d`` is the raster index of the cell (repeated three times adjacently,
    realising TriP2D); ``pos1d`` cycles ``0, 1, 2`` over the three planes at that
    cell (realising TriP1D).
    """
    if h <= 0 or w <= 0:
        raise ValueError("h and w must be positive")
    out: List[Tuple[int, int]] = []
    cell = 0
    for _r in range(h):
        for _c in range(w):
            for plane in range(_NUM_PLANES):
                out.append((cell, plane))
            cell += 1
    return out


def rope_frequencies(dim: int, base: float = 10000.0) -> List[float]:
    """Inverse frequencies for a RoPE of even width ``dim`` (``dim/2`` pairs)."""
    if dim <= 0 or dim % 2 != 0:
        raise ValueError("dim must be a positive even number")
    half = dim // 2
    return [base ** (-(2.0 * i) / dim) for i in range(half)]


def rope_vector(pos: int, dim: int, base: float = 10000.0) -> List[float]:
    """RoPE encoding of a scalar ``pos`` as ``[cos, sin]`` pairs, length ``dim``.

    This is the ``P`` vector the paper adds element-wise; ``TriP2D`` and
    ``TriP1D`` are built from it and summed in :func:`tripe_encoding`.
    """
    freqs = rope_frequencies(dim, base)
    vec: List[float] = []
    for f in freqs:
        angle = pos * f
        vec.append(math.cos(angle))
        vec.append(math.sin(angle))
    return vec


def tripe_encoding(h: int, w: int, dim: int,
                   base: float = 10000.0) -> List[List[float]]:
    """Build TriPE: element-wise sum of the TriP2D and TriP1D RoPE vectors.

    Returns one length-``dim`` vector per token, in sequence order.
    """
    ids = tripe_position_ids(h, w)
    out: List[List[float]] = []
    for pos2d, pos1d in ids:
        v2 = rope_vector(pos2d, dim, base)
        v1 = rope_vector(pos1d, dim, base)
        out.append([a + b for a, b in zip(v2, v1)])
    return out


def apply_rope(vec: List[float], pos: int, base: float = 10000.0) -> List[float]:
    """Rotate a feature ``vec`` by RoPE for position ``pos`` (dim/2 rotations).

    Pairs ``(vec[2i], vec[2i+1])`` rotate by angle ``pos * freq_i``. Provided so
    tests can confirm RoPE's defining property (rotation by 0 is the identity and
    rotations compose additively in position).
    """
    dim = len(vec)
    freqs = rope_frequencies(dim, base)
    out = [0.0] * dim
    for i, f in enumerate(freqs):
        angle = pos * f
        cos, sin = math.cos(angle), math.sin(angle)
        x, y = vec[2 * i], vec[2 * i + 1]
        out[2 * i] = x * cos - y * sin
        out[2 * i + 1] = x * sin + y * cos
    return out
