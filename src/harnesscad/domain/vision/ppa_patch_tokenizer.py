"""Deterministic image-patch tokenisation of a rasterised CAD sketch
(Wang et al., "Parametric Primitive Analysis of CAD Sketches with Vision
Transformer", IEEE T-II 2024, Sec. III-A "Architecture").

The paper's primitive network front-end is a ViT: *"the sketch image with a
resolution of 128x128 is divided into K non-overlapping square patches ... These
patches are flattened and mapped into a token set ... via a Patch MLP."* The learned
Patch-MLP + Transformer encoder are out of scope, but the *patchification* -- slicing
a square image into a grid of non-overlapping square patches and flattening each into
a token vector -- is a pure deterministic operation and is what this module provides.

Given a square 0/1 (or grey) grid it returns:

  * :func:`patch_grid` -- the number of patches per side and total ``K``;
  * :func:`tokenize` -- the ordered list of ``K`` flattened patch tokens (row-major
    over patches, and row-major within each patch), each of length ``patch_size**2``;
  * :func:`patch_occupancy` -- the mean value per patch (a cheap positional feature /
    useful for tests);
  * :func:`grid_from_pixels` -- adapt a ``(resolution, set-of-lit-pixels)`` raster
    (e.g. :class:`vision.cadvlm_sketch_raster.RasterImage`) to a dense grid.

Everything is integer/stdlib. Tokens are tuples so results are hashable and
deterministic. Table VI of the paper varies the patch count (16 -> 32x32 patches,
etc.); this module accepts any ``patch_size`` that divides the resolution.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class PatchGrid:
    """The patch layout for a ``resolution`` image split into ``patch_size`` patches."""

    resolution: int
    patch_size: int

    @property
    def per_side(self) -> int:
        return self.resolution // self.patch_size

    @property
    def num_patches(self) -> int:
        """``K`` -- total non-overlapping patches."""
        return self.per_side * self.per_side

    @property
    def token_dim(self) -> int:
        """Length of a flattened patch token (single channel)."""
        return self.patch_size * self.patch_size


def patch_grid(resolution: int, patch_size: int) -> PatchGrid:
    """Validate divisibility and return the :class:`PatchGrid` layout."""
    if resolution <= 0 or patch_size <= 0:
        raise ValueError("resolution and patch_size must be positive")
    if resolution % patch_size != 0:
        raise ValueError(
            f"patch_size {patch_size} must divide resolution {resolution}")
    return PatchGrid(resolution, patch_size)


def _check_grid(grid, resolution: int) -> None:
    if len(grid) != resolution:
        raise ValueError(f"grid has {len(grid)} rows, expected {resolution}")
    for row in grid:
        if len(row) != resolution:
            raise ValueError("grid must be square")


def tokenize(grid, patch_size: int) -> tuple[tuple, ...]:
    """Split a square ``grid`` into flattened patch tokens.

    ``grid`` is a ``resolution x resolution`` row-major sequence of numbers. Returns
    ``K`` tokens in row-major patch order; within each token the patch's pixels are
    row-major (top-to-bottom, left-to-right). Deterministic and lossless -- the tokens
    are exactly the patch pixels.
    """
    resolution = len(grid)
    layout = patch_grid(resolution, patch_size)
    _check_grid(grid, resolution)
    tokens = []
    n = layout.per_side
    for pr in range(n):
        for pc in range(n):
            r0, c0 = pr * patch_size, pc * patch_size
            token = tuple(
                grid[r0 + dr][c0 + dc]
                for dr in range(patch_size)
                for dc in range(patch_size)
            )
            tokens.append(token)
    return tuple(tokens)


def patch_occupancy(grid, patch_size: int) -> tuple[float, ...]:
    """Mean pixel value of each patch, in the same row-major patch order as tokens."""
    return tuple(sum(tok) / len(tok) for tok in tokenize(grid, patch_size))


def detokenize(tokens, patch_size: int, per_side: int) -> tuple[tuple, ...]:
    """Inverse of :func:`tokenize`: reassemble tokens into the dense square grid."""
    if len(tokens) != per_side * per_side:
        raise ValueError("token count does not match per_side")
    resolution = per_side * patch_size
    grid = [[0] * resolution for _ in range(resolution)]
    for idx, token in enumerate(tokens):
        pr, pc = divmod(idx, per_side)
        r0, c0 = pr * patch_size, pc * patch_size
        for k, val in enumerate(token):
            dr, dc = divmod(k, patch_size)
            grid[r0 + dr][c0 + dc] = val
    return tuple(tuple(row) for row in grid)


def grid_from_pixels(resolution: int, pixels) -> tuple[tuple, ...]:
    """Dense 0/1 grid from a set of lit ``(x, y)`` pixels (row-major, y then x).

    Compatible with :class:`vision.cadvlm_sketch_raster.RasterImage` (pass its
    ``resolution`` and ``pixels``). Out-of-range pixels are ignored.
    """
    lit = set(pixels)
    return tuple(
        tuple(1 if (x, y) in lit else 0 for x in range(resolution))
        for y in range(resolution)
    )
