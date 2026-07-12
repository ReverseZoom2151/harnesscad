"""rastercad_codec -- deterministic codec / quantisation for RECAD raster sketches.

RECAD (Li et al., "Revisiting CAD Model Generation by Learning Raster Sketch")
represents a sketch not as a curve sequence but as a **binary raster image**: a
pixel value of ``1`` marks an area suitable for extrusion and ``0`` empty space
(Sec. "Raster sketch").  Its learned sketch-image VAE resizes every sketch to
``32 x 32`` and compresses it with a *downsampling factor of 8* into a compact
latent ``4 x 4 x 3`` grid, which is later decoded back to a raster sketch.

The learned VAE weights are external, but the *representation* it wraps -- a
deterministic, lossy block quantisation of a binary sketch canvas plus a lossless
token serialisation -- is a self-contained, buildable idea.  This module
implements exactly that:

* **Block quantisation** (:func:`encode_blocks` / :func:`decode_blocks`): the
  deterministic analogue of the VAE downsample-8 bottleneck.  A binary canvas is
  pooled into ``factor x factor`` blocks; each block stores a quantised
  *occupancy level* (how full the block is, quantised to ``levels`` steps).  This
  mirrors the coarse latent grid and round-trips exactly for block-constant
  inputs.

* **Token codec** (:func:`encode_tokens` / :func:`decode_tokens`): a lossless,
  compact run-length token stream for a binary sketch canvas -- a serialisation
  suitable for the "compact grid / token form" a discrete model would emit.

Pure stdlib, fully deterministic (no randomness, no wall clock).  A binary canvas
is a row-major ``list[list[int]]`` with values in ``{0, 1}``.
"""

from __future__ import annotations

from dataclasses import dataclass


Grid = list[list[int]]


# ---------------------------------------------------------------------------
# Validation helpers.
# ---------------------------------------------------------------------------


def _dims(grid: Grid) -> tuple[int, int]:
    """Return ``(height, width)`` of ``grid`` after validating it is rectangular."""

    if not grid or not grid[0]:
        raise ValueError("grid must be non-empty")
    height = len(grid)
    width = len(grid[0])
    for row in grid:
        if len(row) != width:
            raise ValueError("grid rows must all have the same width")
    return height, width


def _check_binary(grid: Grid) -> None:
    for row in grid:
        for v in row:
            if v not in (0, 1):
                raise ValueError("grid values must be 0 or 1")


def latent_shape(size: int, factor: int) -> int:
    """Downsampled side length for ``size`` under integer ``factor`` (ceil).

    For the RECAD VAE (``size=32``, ``factor=8``) this returns ``4``, matching
    the ``4 x 4`` latent spatial grid.
    """

    if size < 1 or factor < 1:
        raise ValueError("size and factor must be >= 1")
    return (size + factor - 1) // factor


# ---------------------------------------------------------------------------
# Block quantisation -- the deterministic downsample-8 bottleneck analogue.
# ---------------------------------------------------------------------------


def _quantise(fraction: float, levels: int) -> int:
    """Quantise a fraction in ``[0, 1]`` to an integer level in ``[0, levels-1]``."""

    if levels < 2:
        raise ValueError("levels must be >= 2")
    if fraction <= 0.0:
        return 0
    if fraction >= 1.0:
        return levels - 1
    # Round to nearest level; deterministic half-up via +0.5.
    q = int(fraction * (levels - 1) + 0.5)
    if q < 0:
        return 0
    if q > levels - 1:
        return levels - 1
    return q


def encode_blocks(grid: Grid, factor: int = 8, levels: int = 5) -> Grid:
    """Encode a binary canvas into a coarse grid of quantised occupancy levels.

    The canvas is partitioned into ``factor x factor`` blocks (the last row/column
    of blocks may be partial for non-divisible sizes).  Each block becomes one
    cell holding the quantised fraction of ink pixels in that block, an integer in
    ``[0, levels - 1]``.  This is the deterministic counterpart of the VAE's
    downsampling bottleneck.
    """

    height, width = _dims(grid)
    _check_binary(grid)
    if factor < 1:
        raise ValueError("factor must be >= 1")
    out_h = latent_shape(height, factor)
    out_w = latent_shape(width, factor)
    coarse: Grid = [[0] * out_w for _ in range(out_h)]
    for by in range(out_h):
        r0 = by * factor
        r1 = min(r0 + factor, height)
        for bx in range(out_w):
            c0 = bx * factor
            c1 = min(c0 + factor, width)
            ink = 0
            count = 0
            for r in range(r0, r1):
                row = grid[r]
                for c in range(c0, c1):
                    ink += row[c]
                    count += 1
            frac = ink / count if count else 0.0
            coarse[by][bx] = _quantise(frac, levels)
    return coarse


def decode_blocks(
    coarse: Grid,
    factor: int = 8,
    out_height: int | None = None,
    out_width: int | None = None,
    levels: int = 5,
    threshold: float = 0.5,
) -> Grid:
    """Decode a coarse occupancy grid back to a binary canvas.

    Each coarse cell is expanded to a ``factor x factor`` block; a block is filled
    with ``1`` when its dequantised occupancy level meets ``threshold`` (in
    ``[0, 1]``), else ``0``.  ``out_height`` / ``out_width`` clip the reconstructed
    canvas to the original size (defaults to the full ``factor``-expanded grid).
    """

    c_h, c_w = _dims(coarse)
    if levels < 2:
        raise ValueError("levels must be >= 2")
    full_h = c_h * factor
    full_w = c_w * factor
    tgt_h = full_h if out_height is None else out_height
    tgt_w = full_w if out_width is None else out_width
    if tgt_h < 1 or tgt_w < 1:
        raise ValueError("output dimensions must be >= 1")
    out: Grid = [[0] * tgt_w for _ in range(tgt_h)]
    for by in range(c_h):
        for bx in range(c_w):
            level = coarse[by][bx]
            if not (0 <= level <= levels - 1):
                raise ValueError("coarse level out of range for given levels")
            frac = level / (levels - 1)
            fill = 1 if frac >= threshold else 0
            if fill == 0:
                continue
            r0 = by * factor
            r1 = min(r0 + factor, tgt_h)
            c0 = bx * factor
            c1 = min(c0 + factor, tgt_w)
            for r in range(r0, r1):
                row = out[r]
                for c in range(c0, c1):
                    row[c] = 1
    return out


# ---------------------------------------------------------------------------
# Lossless run-length token codec.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TokenStream:
    """A lossless run-length serialisation of a binary sketch canvas.

    ``height`` / ``width`` record the original shape; ``runs`` is a flat list of
    non-negative run lengths over the row-major pixel stream, with the first run
    always describing a run of value ``0``.  A leading zero-length run encodes a
    canvas that starts with ``1``.
    """

    height: int
    width: int
    runs: list[int]


def encode_tokens(grid: Grid) -> TokenStream:
    """Losslessly encode a binary canvas as a run-length :class:`TokenStream`."""

    height, width = _dims(grid)
    _check_binary(grid)
    runs: list[int] = []
    current = 0  # runs always start describing value 0
    run_len = 0
    for row in grid:
        for v in row:
            if v == current:
                run_len += 1
            else:
                runs.append(run_len)
                current = v
                run_len = 1
    runs.append(run_len)
    return TokenStream(height=height, width=width, runs=runs)


def decode_tokens(stream: TokenStream) -> Grid:
    """Decode a :class:`TokenStream` back to the exact original binary canvas."""

    height, width = stream.height, stream.width
    if height < 1 or width < 1:
        raise ValueError("stream dimensions must be >= 1")
    total = height * width
    if sum(stream.runs) != total:
        raise ValueError("token runs do not sum to height * width")
    flat: list[int] = []
    value = 0
    for run_len in stream.runs:
        if run_len < 0:
            raise ValueError("run lengths must be non-negative")
        flat.extend([value] * run_len)
        value ^= 1
    grid: Grid = [flat[r * width:(r + 1) * width] for r in range(height)]
    return grid


def roundtrip_tokens(grid: Grid) -> Grid:
    """Convenience: encode then decode a canvas (lossless identity)."""

    return decode_tokens(encode_tokens(grid))
