"""Deterministic ViT patch-feature manipulation for LAS-Diffusion.

From "Locally Attentional SDF Diffusion for Controllable 3D Shape Generation"
(Zheng et al., ACM TOG 2023), Section 4.2 ("Shape generation via ViT feature
manipulation", Fig. 16). Because LAS-Diffusion conditions on *local* image-patch
features, one can synthesise novel shapes without drawing a new sketch by
**stitching patch features of two existing sketches** -- e.g. "replace the
top-half patch features of a sketch with the bottom-half features of another",
or the left/right split used for the car+airplane example.

The neural encoding of a patch is learned, but *which* patch index comes from
*which* source grid is a deterministic partition of the ViT patch grid. This
module implements that partition and stitching over an abstract patch grid whose
cells hold arbitrary feature payloads (token ids, vectors, tuples -- opaque here).

Regions supported (over an ``n x n`` grid of ``(row, col)`` patches):
  * ``"top"`` / ``"bottom"`` -- horizontal split by row.
  * ``"left"`` / ``"right"`` -- vertical split by column.
  * an explicit rectangle ``(row0, col0, row1, col1)`` (half-open).

This is separate from ``geometry.lasdiff_local_attention_mask`` (which builds the
voxel<->patch adjacency); here we recombine the patch payloads themselves.
Stdlib-only, deterministic.
"""

from __future__ import annotations

from typing import Any, Dict, Mapping, Set, Tuple

Patch = Tuple[int, int]
PatchGrid = Mapping[Patch, Any]


def _check_grid(grid: PatchGrid, n: int) -> None:
    if n <= 0:
        raise ValueError("n must be positive")
    if set(grid.keys()) != {(r, c) for r in range(n) for c in range(n)}:
        raise ValueError("grid must contain exactly the n x n patch coordinates")


def region_patches(n: int, region: str) -> Set[Patch]:
    """Patch coordinates selected by a named half-region of an ``n x n`` grid.

    Rows/cols are split at ``n // 2``; the *upper* half takes the larger share
    when ``n`` is odd (rows ``[n//2, n)`` for ``bottom``).
    """
    if n <= 0:
        raise ValueError("n must be positive")
    half = n // 2
    all_cells = {(r, c) for r in range(n) for c in range(n)}
    if region == "top":
        return {(r, c) for (r, c) in all_cells if r < half}
    if region == "bottom":
        return {(r, c) for (r, c) in all_cells if r >= half}
    if region == "left":
        return {(r, c) for (r, c) in all_cells if c < half}
    if region == "right":
        return {(r, c) for (r, c) in all_cells if c >= half}
    raise ValueError("region must be one of top/bottom/left/right")


def rect_patches(rect: Tuple[int, int, int, int]) -> Set[Patch]:
    """Patch coordinates in a half-open rectangle ``(row0, col0, row1, col1)``."""
    r0, c0, r1, c1 = rect
    if r1 <= r0 or c1 <= c0:
        raise ValueError("rectangle must have positive extent")
    return {(r, c) for r in range(r0, r1) for c in range(c0, c1)}


def stitch(base: PatchGrid, other: PatchGrid, n: int, region: str) -> Dict[Patch, Any]:
    """Take ``region`` patches from ``other`` and the rest from ``base``.

    Both grids must be complete ``n x n`` grids over identical coordinates.
    """
    _check_grid(base, n)
    _check_grid(other, n)
    take = region_patches(n, region)
    return {p: (other[p] if p in take else base[p]) for p in base}


def stitch_rect(base: PatchGrid, other: PatchGrid, n: int,
                rect: Tuple[int, int, int, int]) -> Dict[Patch, Any]:
    """Like :func:`stitch` but the swapped region is an explicit rectangle."""
    _check_grid(base, n)
    _check_grid(other, n)
    take = rect_patches(rect)
    if not take.issubset(base.keys()):
        raise ValueError("rectangle falls outside the grid")
    return {p: (other[p] if p in take else base[p]) for p in base}


def provenance(base: PatchGrid, stitched: PatchGrid) -> Dict[Patch, str]:
    """Label each patch ``"base"`` if unchanged from ``base`` else ``"other"``.

    Useful for verifying a stitch touched exactly the intended region.
    """
    return {p: ("base" if stitched[p] == base[p] else "other") for p in base}
