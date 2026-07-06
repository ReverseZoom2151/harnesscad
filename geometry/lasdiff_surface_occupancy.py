"""Discrete surface-occupancy shells from a signed distance grid (LAS-Diffusion).

From "Locally Attentional SDF Diffusion for Controllable 3D Shape Generation"
(Zheng et al., ACM TOG 2023), Section 3.1 and Section 3.3. The paper builds a
two-stage diffusion pipeline whose *first* stage generates a coarse discrete
**surface-occupancy function** approximating the thin shell of a shape, and
whose ground-truth occupancy is derived deterministically from a high-resolution
discrete SDF. Those derivations are pure geometry and are what this module
implements (the diffusion / U-Net part is learned and lives outside).

Definitions taken directly from the paper:

  * **Discrete SDF** ``g : z in Z -> R`` on a regular grid, recording the signed
    distance from each cell centre to the surface (Section 3.1).
  * **Surface-occupancy function** ``o : z -> {0, 1}`` with
    ``o(z) = 1 iff |g(z)| <= delta`` for a threshold ``delta > 0`` (Section 3.1).
    The occupied set ``Omega_o = {z : o(z) = 1}`` approximates the shell only.
  * **Coarse-from-fine occupancy** (Section 3.3): a cell of the coarse 64^3 grid
    contains 8 sub-voxels of the fine 128^3 grid, and the coarse occupancy is
    ``o(z) = 1`` iff *some* sub-voxel has stored SDF value ``|v| <= 1/32``.

This is distinct from ``geometry.cadmorph_tsdf`` (which clamps distances into a
*truncated* SDF and does Boolean CSG algebra) and from
``numeric.flatcad_sdf_derivatives`` (SDF gradient/Hessian). Here we only threshold
and pool SDF values into occupancy shells.

Grids are represented sparsely as ``dict[(i, j, k) -> float]`` (a dense grid is
just a full dict). Everything is stdlib-only and deterministic.
"""

from __future__ import annotations

from typing import Dict, Iterable, Mapping, Set, Tuple

Coord = Tuple[int, int, int]
SdfGrid = Mapping[Coord, float]


def surface_occupancy(sdf: SdfGrid, delta: float) -> Dict[Coord, int]:
    """Threshold an SDF grid into a discrete surface-occupancy function.

    ``o(z) = 1`` iff ``|g(z)| <= delta`` (Section 3.1). Returns a dict over the
    same coordinates as ``sdf`` with values in ``{0, 1}``.
    """
    if delta <= 0.0:
        raise ValueError("delta must be positive")
    return {z: (1 if abs(v) <= delta else 0) for z, v in sdf.items()}


def occupied_cells(sdf: SdfGrid, delta: float) -> Set[Coord]:
    """Return ``Omega_o = {z : |g(z)| <= delta}`` -- the occupied shell set."""
    if delta <= 0.0:
        raise ValueError("delta must be positive")
    return {z for z, v in sdf.items() if abs(v) <= delta}


def shell_thickness(sdf: SdfGrid, delta: float) -> int:
    """Number of occupied shell cells (``|Omega_o|``)."""
    return len(occupied_cells(sdf, delta))


def coarsen_occupancy(
    fine_sdf: SdfGrid,
    threshold: float = 1.0 / 32.0,
    factor: int = 2,
) -> Set[Coord]:
    """Pool a fine SDF grid into a coarse occupancy set (Section 3.3).

    Each coarse cell ``(i, j, k)`` covers the block of fine cells
    ``[factor*i, factor*i + factor)`` along every axis (``factor = 2`` gives the
    paper's "8 sub-voxels" rule for 64^3 vs 128^3). A coarse cell is occupied iff
    *any* fine sub-voxel it contains has ``|v| <= threshold``.
    """
    if factor < 1:
        raise ValueError("factor must be >= 1")
    if threshold <= 0.0:
        raise ValueError("threshold must be positive")
    occupied: Set[Coord] = set()
    for (i, j, k), v in fine_sdf.items():
        if abs(v) <= threshold:
            occupied.add((i // factor, j // factor, k // factor))
    return occupied


def subvoxels_of(coarse: Coord, factor: int = 2) -> Iterable[Coord]:
    """Yield the ``factor**3`` fine sub-voxel coordinates inside a coarse cell.

    Inverse of the block mapping used by :func:`coarsen_occupancy`.
    """
    if factor < 1:
        raise ValueError("factor must be >= 1")
    ci, cj, ck = coarse
    base_i, base_j, base_k = ci * factor, cj * factor, ck * factor
    for a in range(factor):
        for b in range(factor):
            for c in range(factor):
                yield (base_i + a, base_j + b, base_k + c)


def occupancy_iou(a: Iterable[Coord], b: Iterable[Coord]) -> float:
    """Intersection-over-union of two occupancy sets (1.0 when both empty)."""
    sa, sb = set(a), set(b)
    if not sa and not sb:
        return 1.0
    inter = len(sa & sb)
    union = len(sa | sb)
    return inter / union if union else 1.0
