"""Deterministic ViT-MAE-style patch masking for CadVLM sketch images.

To bridge natural images and CAD sketches, CadVLM (Wu et al., Sec 4.1) fine-tunes a
ViT-MAE on rendered sketches with a **one-epoch image-reconstruction task at a 75%
masking ratio**, using patch size and stride of 32 on ``224 x 224`` renders (so a
``7 x 7`` grid of 49 patches). Training the autoencoder is out of scope, but two
pieces of that pipeline are pure and deterministic and are not covered anywhere in
the repository:

* **patchify / unpatchify** -- splitting a square grid into non-overlapping
  ``patch x patch`` tiles and reassembling them; and
* **seeded patch masking** -- choosing which ratio of patches to hide, plus the
  Image Decoding Loss (IDL) the paper defines as the pixel-level MSE between the
  reconstructed and ground-truth images (Eq. 6).

Randomness is confined to ``random.Random(seed)`` so a ``(n_patches, ratio, seed)``
triple always selects the same masked set; no wall clock is read.
"""

from __future__ import annotations

import random
from dataclasses import dataclass


def patchify(grid, patch: int = 32):
    """Split a square grid (rows of values) into ``patch x patch`` tiles.

    Returns a mapping ``(patch_row, patch_col) -> tuple-of-rows``. Requires the grid
    to be square with side divisible by ``patch``.
    """
    n = len(grid)
    if n == 0 or any(len(row) != n for row in grid):
        raise ValueError("grid must be square and non-empty")
    if n % patch != 0:
        raise ValueError(f"grid side {n} not divisible by patch {patch}")
    per = n // patch
    tiles = {}
    for pr in range(per):
        for pc in range(per):
            tiles[(pr, pc)] = tuple(
                tuple(grid[pr * patch + r][pc * patch + c] for c in range(patch))
                for r in range(patch)
            )
    return tiles


def unpatchify(tiles, patch: int = 32):
    """Inverse of :func:`patchify`: reassemble tiles into a dense square grid."""
    per = max(pr for pr, _ in tiles) + 1
    n = per * patch
    grid = [[0] * n for _ in range(n)]
    for (pr, pc), tile in tiles.items():
        for r in range(patch):
            for c in range(patch):
                grid[pr * patch + r][pc * patch + c] = tile[r][c]
    return tuple(tuple(row) for row in grid)


def patch_count(resolution: int, patch: int = 32) -> int:
    """Number of patches in a ``resolution`` square grid at the given patch size."""
    if resolution % patch != 0:
        raise ValueError(f"resolution {resolution} not divisible by patch {patch}")
    per = resolution // patch
    return per * per


def mask_count(n_patches: int, ratio: float = 0.75) -> int:
    """Number of patches hidden at ``ratio`` (``round(n_patches * ratio)``, clamped)."""
    if not 0.0 <= ratio <= 1.0:
        raise ValueError("ratio must be in [0, 1]")
    return min(n_patches, max(0, round(n_patches * ratio)))


def masked_indices(n_patches: int, ratio: float = 0.75, seed: int = 0) -> tuple:
    """Deterministically pick the masked patch indices ``0..n_patches-1``.

    ``round(n_patches * ratio)`` patches are hidden, chosen with
    ``random.Random(seed)``; the returned indices are sorted for a stable contract.
    """
    if not 0.0 <= ratio <= 1.0:
        raise ValueError("ratio must be in [0, 1]")
    k = mask_count(n_patches, ratio)
    rng = random.Random(seed)
    return tuple(sorted(rng.sample(range(n_patches), k)))


@dataclass(frozen=True)
class MaskedImage:
    """A patch-masked grid plus the bookkeeping needed to score reconstruction."""

    grid: tuple            # masked grid (hidden patches filled with ``fill``)
    patch: int
    masked: tuple          # sorted masked patch indices (row-major)
    order: tuple           # row-major (patch_row, patch_col) for each flat index

    @property
    def visible(self) -> tuple:
        """Flat indices of the patches left visible."""
        hidden = set(self.masked)
        return tuple(i for i in range(len(self.order)) if i not in hidden)


def apply_mask(grid, patch: int = 32, ratio: float = 0.75, seed: int = 0,
               fill: float = 0.0) -> MaskedImage:
    """Patchify ``grid``, hide a seeded ``ratio`` of patches, and refill them."""
    tiles = patchify(grid, patch)
    order = tuple(sorted(tiles))                      # row-major patch coords
    idx = masked_indices(len(order), ratio, seed)
    hidden = {order[i] for i in idx}
    blank = tuple(tuple(fill for _ in range(patch)) for _ in range(patch))
    masked_tiles = {coord: (blank if coord in hidden else tile)
                    for coord, tile in tiles.items()}
    return MaskedImage(grid=unpatchify(masked_tiles, patch), patch=patch,
                       masked=idx, order=order)


def mse(reconstructed, ground_truth) -> float:
    """Pixel-level mean squared error between two equally shaped grids (Eq. 6)."""
    if len(reconstructed) != len(ground_truth):
        raise ValueError("grid height mismatch")
    total = 0.0
    count = 0
    for row_r, row_g in zip(reconstructed, ground_truth):
        if len(row_r) != len(row_g):
            raise ValueError("grid width mismatch")
        for a, b in zip(row_r, row_g):
            diff = float(a) - float(b)
            total += diff * diff
            count += 1
    return total / count if count else 0.0


def masked_mse(reconstructed, ground_truth, masked: MaskedImage) -> float:
    """Image Decoding Loss restricted to the hidden patches (MAE-style scoring)."""
    patch = masked.patch
    hidden = {masked.order[i] for i in masked.masked}
    total = 0.0
    count = 0
    for (pr, pc) in hidden:
        for r in range(patch):
            for c in range(patch):
                y = pr * patch + r
                x = pc * patch + c
                diff = float(reconstructed[y][x]) - float(ground_truth[y][x])
                total += diff * diff
                count += 1
    return total / count if count else 0.0
