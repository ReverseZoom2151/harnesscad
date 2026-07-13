"""Next-part-prediction sequence representation for TAR3D.

From "TAR3D: Creating High-Quality 3D Assets via Next-Part Prediction" (Zhang et
al., 2024), Section 3.2 ("Sequence builder", "Autoregressive generator"). The
learned GPT is external, but the *sequence formulation* it consumes is a fully
deterministic serialization of the triplane codebook indices, and the paper
pins the ordering rule exactly (Fig. 2b):

  * "the 3D shapes are encoded as discrete triplane features, which can be
    represented with their indices in the codebook."
  * "we organize the indices within each plane in a raster scan order and the
    indices at the same positions of the three planes in the adjacent orders."
  * the tokenized sequence ``s in {0, ..., K}^{3.h.w}`` is generated as an
    "autoregressive next-index prediction": ``p(s | c) = prod_t p(s_t | s_<t, c)``
    with prompt embedding ``c`` prefilled at the front.

This module builds that sequence from a triplane index grid, inverts it, emits
the teacher-forcing ``(prefix, next)`` targets that define next-part prediction,
and validates a sequence (right length, indices inside the codebook). It is the
symbolic counterpart to ``geometry.tar3d_triplane_grid`` (which holds the plane
geometry) and reuses no learned component. Stdlib-only, deterministic.
"""

from __future__ import annotations

from typing import List, Sequence, Tuple

# Plane order inside each interleaved position triple (TAR3D: XY, YZ, XZ).
PLANE_ORDER = ("XY", "YZ", "XZ")
_NUM_PLANES = 3


class TriplaneIndexGrid:
    """Three ``h x w`` codebook-index maps (one per axis plane).

    Each ``planes[p]`` is a list of ``h`` rows, each of ``w`` integer indices in
    ``{0, ..., K-1}``. All three planes share the same ``h`` and ``w`` (TAR3D
    upsamples every plane to a common resolution).
    """

    def __init__(self, planes, h: int, w: int, codebook_size: int):
        if h <= 0 or w <= 0:
            raise ValueError("h and w must be positive")
        if codebook_size <= 0:
            raise ValueError("codebook_size must be positive")
        if set(planes.keys()) != set(PLANE_ORDER):
            raise ValueError("planes must be keyed by %r" % (PLANE_ORDER,))
        for p in PLANE_ORDER:
            grid = planes[p]
            if len(grid) != h or any(len(row) != w for row in grid):
                raise ValueError("plane %s is not %dx%d" % (p, h, w))
            for row in grid:
                for idx in row:
                    if not (0 <= idx < codebook_size):
                        raise ValueError("index %r outside codebook" % (idx,))
        self.planes = {p: [list(r) for r in planes[p]] for p in PLANE_ORDER}
        self.h = h
        self.w = w
        self.codebook_size = codebook_size


def sequence_positions(h: int, w: int) -> List[Tuple[str, int, int]]:
    """Return the ``(plane, row, col)`` of every token, in sequence order.

    Positions run in raster order over ``(row, col)``; at each cell the three
    planes appear adjacently in ``PLANE_ORDER`` -- exactly TAR3D's rule.
    """
    out: List[Tuple[str, int, int]] = []
    for r in range(h):
        for c in range(w):
            for p in PLANE_ORDER:
                out.append((p, r, c))
    return out


def build_sequence(grid: TriplaneIndexGrid) -> List[int]:
    """Serialize a triplane index grid into a length ``3*h*w`` token sequence."""
    seq: List[int] = []
    for (p, r, c) in sequence_positions(grid.h, grid.w):
        seq.append(grid.planes[p][r][c])
    return seq


def detokenize(seq: Sequence[int], h: int, w: int,
               codebook_size: int) -> TriplaneIndexGrid:
    """Invert :func:`build_sequence` back into a :class:`TriplaneIndexGrid`."""
    expected = _NUM_PLANES * h * w
    if len(seq) != expected:
        raise ValueError("sequence length %d != %d" % (len(seq), expected))
    planes = {p: [[0] * w for _ in range(h)] for p in PLANE_ORDER}
    for token, (p, r, c) in zip(seq, sequence_positions(h, w)):
        planes[p][r][c] = token
    return TriplaneIndexGrid(planes, h, w, codebook_size)


def is_valid_sequence(seq: Sequence[int], h: int, w: int,
                      codebook_size: int) -> bool:
    """True iff ``seq`` has length ``3*h*w`` and every token is in the codebook."""
    if len(seq) != _NUM_PLANES * h * w:
        return False
    return all(isinstance(t, int) and 0 <= t < codebook_size for t in seq)


def next_part_targets(seq: Sequence[int],
                      prompt: Sequence[int] = ()) -> List[Tuple[Tuple[int, ...], int]]:
    """Emit the ``(prefix, next_token)`` pairs of next-part prediction.

    Realises ``p(s_t | s_<t, c)``: the optional ``prompt`` tokens ``c`` are
    prefilled, so the first target predicts ``s_0`` from the prompt alone, and
    each subsequent target extends the prefix by one already-seen token. The
    prompt itself is never a prediction target.
    """
    prefix: List[int] = list(prompt)
    pairs: List[Tuple[Tuple[int, ...], int]] = []
    for token in seq:
        pairs.append((tuple(prefix), token))
        prefix.append(token)
    return pairs


def prefill(prompt: Sequence[int], seq: Sequence[int]) -> List[int]:
    """Concatenate the prompt prefix with the token sequence (GPT prefilling)."""
    return list(prompt) + list(seq)


def teacher_forcing_accuracy(predicted: Sequence[int],
                             target: Sequence[int]) -> float:
    """Fraction of positions where ``predicted`` matches ``target`` exactly.

    A part-level sequence metric for next-part prediction: 1.0 is a perfect
    reproduction of the triplane index sequence. Both must share a length.
    """
    if len(predicted) != len(target):
        raise ValueError("predicted and target must be the same length")
    if not target:
        return 1.0
    hits = sum(1 for a, b in zip(predicted, target) if a == b)
    return hits / len(target)
