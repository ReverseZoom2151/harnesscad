"""Two-stage sparse-voxel subdivision for LAS-Diffusion's SDF stage.

From "Locally Attentional SDF Diffusion for Controllable 3D Shape Generation"
(Zheng et al., ACM TOG 2023), Sections 3.2-3.4. LAS-Diffusion runs a *coarse*
occupancy-diffusion at 64^3, then a *fine* SDF-diffusion restricted to the
occupied shell at 128^3. The bridge between the two stages is pure, deterministic
book-keeping (the diffusion sampling itself is learned and lives elsewhere):

  * **Reserve occupied coarse voxels** -- "we reserve the voxels whose predicted
    surface-occupancy values are larger than 0.5" (Section 3.3 inference).
  * **Subdivide once** -- "subdivide them once to obtain a set of sub-voxels in
    128^3 resolution": each reserved 64^3 voxel becomes 8 fine sub-voxels
    (Section 3.4). Only these fine voxels are active in the sparse SDF stage;
    the rest of the 128^3 grid is never allocated.
  * **Fill with noise** -- "the subdivided voxels ... are initialized with
    Gaussian noise" before the reverse process (Section 3.4). We do this
    deterministically with ``random.Random(seed)`` so callers can reproduce a
    sparse noisy grid without a wall clock.

This complements ``geometry.lasdiff_surface_occupancy`` (which derives occupancy
*from* an SDF). Here we go the other way: from a coarse occupancy probability
field to the *active fine-voxel set* and a shell-masked fine SDF. Stdlib-only.
"""

from __future__ import annotations

import random
from typing import Dict, Iterable, Mapping, Set, Tuple

Coord = Tuple[int, int, int]


def reserve_occupied(prob_grid: Mapping[Coord, float], threshold: float = 0.5) -> Set[Coord]:
    """Coarse voxels whose predicted occupancy probability exceeds ``threshold``.

    Uses a strict ``> threshold`` comparison, matching the paper's "larger than
    0.5" wording (Section 3.3).
    """
    return {z for z, p in prob_grid.items() if p > threshold}


def subdivide(coarse_occupied: Iterable[Coord], factor: int = 2) -> Set[Coord]:
    """Subdivide each occupied coarse voxel into ``factor**3`` fine sub-voxels.

    Returns the *active* fine-voxel coordinate set for the sparse SDF stage
    (``factor = 2`` -> 8 sub-voxels, i.e. 64^3 -> 128^3).
    """
    if factor < 1:
        raise ValueError("factor must be >= 1")
    active: Set[Coord] = set()
    for (ci, cj, ck) in coarse_occupied:
        bi, bj, bk = ci * factor, cj * factor, ck * factor
        for a in range(factor):
            for b in range(factor):
                for c in range(factor):
                    active.add((bi + a, bj + b, bk + c))
    return active


def mask_sdf_to_shell(fine_sdf: Mapping[Coord, float], active: Iterable[Coord]) -> Dict[Coord, float]:
    """Keep only fine SDF values that fall inside the active (subdivided) shell.

    Values outside the occupied region are dropped -- the sparse SDF stage never
    stores them.
    """
    active_set = set(active)
    return {z: v for z, v in fine_sdf.items() if z in active_set}


def fill_gaussian_noise(active: Iterable[Coord], seed: int, sigma: float = 1.0) -> Dict[Coord, float]:
    """Initialise every active fine voxel with deterministic Gaussian noise.

    Coordinates are sorted before sampling so the result depends only on the
    coordinate set and ``seed`` (never on iteration/hash order).
    """
    if sigma < 0.0:
        raise ValueError("sigma must be non-negative")
    rng = random.Random(seed)
    return {z: rng.gauss(0.0, sigma) for z in sorted(active)}


def subdivision_ratio(coarse_occupied: Iterable[Coord], total_coarse: int, factor: int = 2) -> float:
    """Fraction of the full fine grid that is actually allocated (sparsity gain).

    ``= (|occupied| * factor**3) / (total_coarse * factor**3)`` which reduces to
    the occupied fraction of the coarse grid -- reported here as the share of the
    dense fine grid the sparse representation avoids materialising.
    """
    if total_coarse <= 0:
        raise ValueError("total_coarse must be positive")
    n = len(set(coarse_occupied))
    if n > total_coarse:
        raise ValueError("occupied count exceeds total_coarse")
    return n / total_coarse
