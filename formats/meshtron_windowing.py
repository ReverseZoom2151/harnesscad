"""Meshtron hourglass token layout + truncated / sliding-window sequencing.

Deterministic reconstructions of Meshtron's sequence-shaping machinery (Hao et
al., 2024, Secs. 3.1-3.2). None of this needs the learned model; it is all
index arithmetic over a mesh token sequence:

* **Special-token framing** (Sec. 2). A mesh sequence is wrapped with special
  tokens *in groups of 9* to keep the vertex/face structure aligned: 9
  start-of-sequence tokens are prepended and 9 end-of-sequence tokens appended;
  a padding token fills batched sequences to a common length.

* **Hierarchical grouping** (Sec. 3.1, Eq. 1). Mesh tokens form a two-level
  hierarchy -- every 3 coordinate tokens make a vertex, every 3 vertices (9
  coordinate tokens) make a triangle. Meshtron uses an Hourglass Transformer
  with two shortening stages of factor 3 (coordinate -> vertex -> face).

* **Static routing / shortening indices** (Sec. 3.1). For a shortening factor
  ``s`` only every ``s``-th token is processed by the inner Transformer stack,
  while the other tokens bypass it. This module computes exactly which token
  positions are routed through each stage.

* **Truncated-sequence training + sliding-window inference** (Sec. 3.2). Very
  long mesh sequences are cut into fixed-length truncated segments for training,
  and generated at inference with a rolling window (KV-cache buffer equal to the
  attention window). This module produces the segment / window index sets.

Pure stdlib, deterministic (no wall clock, no RNG).
"""

from __future__ import annotations

from typing import List, Sequence, Tuple

# Special tokens. Kept as short strings so a token sequence can mix them with
# integer coordinate ids.
SOS = "S"   # start-of-sequence
EOS = "E"   # end-of-sequence
PAD = "P"   # padding

SPECIAL_GROUP = 9         # special tokens always come in groups of 9
COORDS_PER_VERTEX = 3     # every 3 coordinate tokens form a vertex
VERTICES_PER_FACE = 3     # every 3 vertices form a triangle
COORDS_PER_FACE = COORDS_PER_VERTEX * VERTICES_PER_FACE  # 9


# --------------------------------------------------------------------------- #
# Special-token framing
# --------------------------------------------------------------------------- #
def frame_sequence(coords: Sequence[object]) -> List[object]:
    """Prepend 9 ``SOS`` and append 9 ``EOS`` tokens to a coordinate sequence."""
    return [SOS] * SPECIAL_GROUP + list(coords) + [EOS] * SPECIAL_GROUP


def pad_to_length(tokens: Sequence[object], length: int) -> List[object]:
    """Right-pad ``tokens`` with ``PAD`` up to ``length`` (error if too long)."""
    if len(tokens) > length:
        raise ValueError("sequence longer than requested pad length")
    return list(tokens) + [PAD] * (length - len(tokens))


def pad_batch(batch: Sequence[Sequence[object]]) -> List[List[object]]:
    """Pad every sequence in ``batch`` to the longest length with ``PAD``."""
    if not batch:
        return []
    width = max(len(seq) for seq in batch)
    return [pad_to_length(seq, width) for seq in batch]


# --------------------------------------------------------------------------- #
# Hierarchical grouping
# --------------------------------------------------------------------------- #
def group(tokens: Sequence[object], size: int) -> List[List[object]]:
    """Split ``tokens`` into consecutive groups of ``size`` (last may be short)."""
    if size <= 0:
        raise ValueError("group size must be positive")
    return [list(tokens[i:i + size]) for i in range(0, len(tokens), size)]


def group_coords_to_vertices(coords: Sequence[object]) -> List[List[object]]:
    """Group a coordinate stream into vertices (3 coordinates each)."""
    if len(coords) % COORDS_PER_VERTEX != 0:
        raise ValueError("coordinate count must be a multiple of 3")
    return group(coords, COORDS_PER_VERTEX)


def group_vertices_to_faces(vertices: Sequence[object]) -> List[List[object]]:
    """Group a vertex stream into faces (3 vertices each)."""
    if len(vertices) % VERTICES_PER_FACE != 0:
        raise ValueError("vertex count must be a multiple of 3")
    return group(vertices, VERTICES_PER_FACE)


def hierarchy_levels(coords: Sequence[object]) -> Tuple[int, int, int]:
    """Return ``(n_coords, n_vertices, n_faces)`` for a coordinate stream."""
    if len(coords) % COORDS_PER_FACE != 0:
        raise ValueError("coordinate count must be a multiple of 9 (whole faces)")
    n = len(coords)
    return n, n // COORDS_PER_VERTEX, n // COORDS_PER_FACE


# --------------------------------------------------------------------------- #
# Static routing / shortening
# --------------------------------------------------------------------------- #
def shortening_indices(length: int, factor: int = 3) -> List[int]:
    """Positions routed through the inner stack for one shortening stage.

    For factor ``s`` the ``s``-th token of every group (0-based positions
    ``s-1, 2s-1, ...``) is processed by the inner Transformer stack; the rest
    bypass it (Sec. 3.1). With ``s=3`` this is every 3rd position.
    """
    if factor <= 0:
        raise ValueError("factor must be positive")
    return [i for i in range(length) if (i + 1) % factor == 0]


def shortened_length(length: int, factor: int = 3) -> int:
    """Length after one shortening stage (number of routed positions)."""
    if factor <= 0:
        raise ValueError("factor must be positive")
    return length // factor


def hourglass_stage_lengths(
    length: int, factors: Sequence[int] = (3, 3)
) -> List[int]:
    """Sequence lengths at each Hourglass stage (full, then each shortening).

    Meshtron uses two factor-3 stages, so a coordinate sequence of length ``L``
    becomes ``[L, L/3, L/9]`` -- the coordinate, vertex and face levels.
    """
    lengths = [length]
    cur = length
    for factor in factors:
        cur = shortened_length(cur, factor)
        lengths.append(cur)
    return lengths


# --------------------------------------------------------------------------- #
# Truncated-sequence training + sliding-window inference
# --------------------------------------------------------------------------- #
def truncate_segments(
    tokens: Sequence[object],
    window: int,
    align: int = COORDS_PER_FACE,
    pad: bool = True,
) -> List[List[object]]:
    """Cut ``tokens`` into non-overlapping fixed-length training segments.

    ``window`` must be a multiple of ``align`` (face boundary, default 9) so a
    segment never splits a triangle. The final short segment is right-padded
    with ``PAD`` when ``pad`` is True.
    """
    if window <= 0:
        raise ValueError("window must be positive")
    if window % align != 0:
        raise ValueError("window must be a multiple of the alignment")
    segments: List[List[object]] = []
    for start in range(0, len(tokens), window):
        seg = list(tokens[start:start + window])
        if pad and len(seg) < window:
            seg = pad_to_length(seg, window)
        segments.append(seg)
    return segments


def sliding_windows(
    tokens: Sequence[object], window: int, step: int = 1
) -> List[Tuple[int, int]]:
    """Rolling ``(start, end)`` index spans of size ``window`` over ``tokens``.

    Models the inference-time rolling KV-cache: each generated position attends
    to at most ``window`` previous tokens. Spans before the sequence fills the
    window start at 0 and grow; afterwards the start rolls forward by ``step``.
    """
    if window <= 0 or step <= 0:
        raise ValueError("window and step must be positive")
    spans: List[Tuple[int, int]] = []
    for end in range(1, len(tokens) + 1, step):
        start = max(0, end - window)
        spans.append((start, end))
    return spans


def receptive_field_size(position: int, window: int) -> int:
    """Number of tokens visible to ``position`` (0-based) under a rolling window."""
    if window <= 0:
        raise ValueError("window must be positive")
    return min(position + 1, window)
