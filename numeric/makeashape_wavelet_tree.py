"""Wavelet-tree representation: subband filtering, packing, adaptive masks.

From *Make-A-Shape* (Hui, Sanghi, Rampini et al., 2024), Secs. 4-6.  On top of
the raw multi-level 3D DWT (see ``numeric.makeashape_wavelet_transform``), the
paper builds a compact **wavelet-tree representation** through three
deterministic constructions, all implemented here:

  * **Subband coefficient filtering** (Sec. 4, Fig. 7).  Because sibling detail
    coefficients are positively correlated, for every spatial location the
    paper takes the *largest-magnitude* coefficient across the sibling detail
    subbands as that location's "information" measure, then keeps the top-K
    locations.  ``sibling_information`` and ``top_k_locations`` implement this;
    ``truncate_top_k`` zeroes every detail coefficient outside the kept set,
    yielding the lossy-but-faithful representation.

  * **Subband adaptive coordinate sets** (Sec. 6, Eq. 2).  For each detail
    subband the paper finds the max magnitude ``v`` and marks every coefficient
    with magnitude ``> v/32`` as important, unioning the sibling sets into a
    single coordinate set ``P0`` (and its complement ``P0'``).  These drive the
    adaptive training loss.  ``importance_mask`` and ``adaptive_coordinate_set``
    build them, and ``coordinate_set_as_binary_mask`` gives the fixed-size mask
    the paper uses for efficient loss computation.

  * **Subband coefficient packing** (Sec. 5, Fig. 8).  Siblings and their
    children are channel-wise concatenated so the representation collapses onto
    the low-resolution coarse grid with many channels (in 3D: ``1`` coarse
    ``+ 7`` D0 siblings ``+ 7*8`` D1 descendants ``= 64`` channels), making it a
    regular grid a diffusion model can consume.  ``pack_diffusible`` /
    ``unpack_diffusible`` perform this rearrangement and its exact inverse.

Everything is stdlib-only and deterministic (random selection, where the paper
uses it, is seeded via ``random.Random``).
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Dict, List, Sequence, Set, Tuple

from numeric.makeashape_wavelet_transform import (
    Grid3D, DETAIL_NAMES, WaveletDecomposition,
)

Coord = Tuple[int, int, int]


# --------------------------------------------------------------------------- #
# Subband coefficient filtering (top-K by sibling magnitude)                   #
# --------------------------------------------------------------------------- #

def sibling_information(detail: Dict[str, Grid3D]) -> Dict[Coord, float]:
    """Per-location information = max |coeff| across the sibling detail bands.

    ``detail`` is one level's dict of the seven detail subbands (all sharing
    the same dims).  Returns a mapping from ``(ix, iy, iz)`` to the largest
    sibling magnitude at that location (Sec. 4: "selecting the one with the
    largest magnitude ... as the measure of information").
    """
    names = [n for n in DETAIL_NAMES if n in detail]
    if not names:
        raise ValueError("no detail subbands provided")
    dims = detail[names[0]].dims
    for n in names:
        if detail[n].dims != dims:
            raise ValueError("detail subbands have mismatched dims")
    nx, ny, nz = dims
    info: Dict[Coord, float] = {}
    for ix in range(nx):
        for iy in range(ny):
            for iz in range(nz):
                m = 0.0
                for n in names:
                    a = abs(detail[n].get(ix, iy, iz))
                    if a > m:
                        m = a
                info[(ix, iy, iz)] = m
    return info


def top_k_locations(detail: Dict[str, Grid3D], k: int) -> List[Coord]:
    """The ``k`` most information-rich locations, deterministically ordered.

    Ties break by coordinate so the result is stable across runs.
    """
    if k < 0:
        raise ValueError("k must be non-negative")
    info = sibling_information(detail)
    ordered = sorted(info.items(), key=lambda kv: (-kv[1], kv[0]))
    return [coord for coord, _ in ordered[:k]]


def truncate_top_k(detail: Dict[str, Grid3D], k: int) -> Dict[str, Grid3D]:
    """Return a copy of ``detail`` with all but the top-K locations zeroed.

    Every sibling subband keeps its coefficient at a kept location and is set to
    zero elsewhere (Sec. 4 subband coefficient filtering).
    """
    keep: Set[Coord] = set(top_k_locations(detail, k))
    names = [n for n in DETAIL_NAMES if n in detail]
    dims = detail[names[0]].dims
    nx, ny, nz = dims
    out: Dict[str, Grid3D] = {}
    for n in names:
        src = detail[n]
        data = [0.0] * (nx * ny * nz)
        for coord in keep:
            ix, iy, iz = coord
            data[(ix * ny + iy) * nz + iz] = src.get(ix, iy, iz)
        out[n] = Grid3D(dims, data)
    return out


def compress_decomposition_top_k(
    decomp: WaveletDecomposition, k: int
) -> WaveletDecomposition:
    """Apply top-K subband filtering to every detail level of a decomposition."""
    new_details = [truncate_top_k(level, k) for level in decomp.details]
    return WaveletDecomposition(
        decomp.input_dims, decomp.wavelet, decomp.coarse, new_details
    )


def detail_coefficient_count(decomp: WaveletDecomposition) -> int:
    """Total number of (nonzero-capable) detail coefficient slots."""
    total = 0
    for level in decomp.details:
        for n in DETAIL_NAMES:
            if n in level:
                total += len(level[n].data)
    return total


def nonzero_detail_count(decomp: WaveletDecomposition, tol: float = 0.0) -> int:
    total = 0
    for level in decomp.details:
        for n in DETAIL_NAMES:
            if n in level:
                total += sum(1 for v in level[n].data if abs(v) > tol)
    return total


# --------------------------------------------------------------------------- #
# Subband adaptive coordinate sets (Eq. 2)                                     #
# --------------------------------------------------------------------------- #

def importance_mask(subband: Grid3D, ratio: float = 32.0) -> Set[Coord]:
    """Coords whose magnitude exceeds ``v/ratio`` where ``v`` is the band max.

    The paper uses ``v/32`` (Sec. 6).  A constant/zero band yields the empty set.
    """
    if ratio <= 0:
        raise ValueError("ratio must be positive")
    v = subband.max_abs()
    if v == 0.0:
        return set()
    thresh = v / ratio
    nx, ny, nz = subband.dims
    out: Set[Coord] = set()
    for ix in range(nx):
        for iy in range(ny):
            for iz in range(nz):
                if abs(subband.get(ix, iy, iz)) > thresh:
                    out.add((ix, iy, iz))
    return out


def adaptive_coordinate_set(detail: Dict[str, Grid3D], ratio: float = 32.0) -> Set[Coord]:
    """Union of the per-subband importance masks -> the set ``P0`` (Eq. 2)."""
    names = [n for n in DETAIL_NAMES if n in detail]
    p0: Set[Coord] = set()
    for n in names:
        p0 |= importance_mask(detail[n], ratio)
    return p0


def complement_coordinate_set(detail: Dict[str, Grid3D], p0: Set[Coord]) -> Set[Coord]:
    """``P0'``: the spatial complement of ``P0`` within the subband domain."""
    names = [n for n in DETAIL_NAMES if n in detail]
    dims = detail[names[0]].dims
    nx, ny, nz = dims
    full = {(ix, iy, iz) for ix in range(nx) for iy in range(ny) for iz in range(nz)}
    return full - p0


def coordinate_set_as_binary_mask(coords: Set[Coord], dims: Tuple[int, int, int]) -> Grid3D:
    """Fixed-size 0/1 mask for a coordinate set (Sec. 6 efficient loss)."""
    nx, ny, nz = dims
    data = [0.0] * (nx * ny * nz)
    for (ix, iy, iz) in coords:
        data[(ix * ny + iy) * nz + iz] = 1.0
    return Grid3D(dims, data)


def sample_complement(
    p_complement: Set[Coord], count: int, seed: int
) -> List[Coord]:
    """Randomly pick ``count`` coords from ``P0'`` (the ``R(.)`` op in Eq. 2).

    Deterministic given ``seed``; picks ``min(count, |P0'|)`` distinct coords.
    """
    ordered = sorted(p_complement)
    rng = random.Random(seed)
    if count >= len(ordered):
        return ordered
    return sorted(rng.sample(ordered, count))


# --------------------------------------------------------------------------- #
# Subband coefficient packing (Fig. 8): collapse onto coarse grid + channels   #
# --------------------------------------------------------------------------- #

# Channel layout per coarse cell:
#   0                : C0 coarse coefficient
#   1 .. 7           : the 7 D0 sibling detail coefficients (same-res band)
#   8 .. 63          : the 7 * 8 D1 descendants (each sibling's 2x2x2 children)
_D1_CHILD_OFFSETS = [
    (dx, dy, dz) for dx in (0, 1) for dy in (0, 1) for dz in (0, 1)
]  # 8 children in a 2x2x2 block
PACKED_CHANNELS = 1 + len(DETAIL_NAMES) + len(DETAIL_NAMES) * len(_D1_CHILD_OFFSETS)


@dataclass
class PackedRepresentation:
    """Diffusible packed grid: coarse-resolution dims, ``PACKED_CHANNELS`` deep."""

    dims: Tuple[int, int, int]   # == coarse (C0 / D0) dims
    channels: int
    data: List[float]            # length nx*ny*nz*channels, channel-fastest

    def channel_vector(self, ix: int, iy: int, iz: int) -> List[float]:
        nx, ny, nz = self.dims
        base = ((ix * ny + iy) * nz + iz) * self.channels
        return self.data[base:base + self.channels]


def _coarse_detail_layers(
    decomp: WaveletDecomposition,
) -> Tuple[Grid3D, Dict[str, Grid3D], Dict[str, Grid3D]]:
    """Extract (C0, D0, D1): coarse band, coarsest detail, next-finer detail."""
    if len(decomp.details) < 2:
        raise ValueError("packing needs at least 2 detail levels (D0 and D1)")
    c0 = decomp.coarse
    d0 = decomp.details[-1]   # coarsest detail, same dims as C0
    d1 = decomp.details[-2]   # next finer, twice the resolution
    if d0[DETAIL_NAMES[0]].dims != c0.dims:
        raise ValueError("D0 dims must match coarse dims")
    cd = c0.dims
    if d1[DETAIL_NAMES[0]].dims != (cd[0] * 2, cd[1] * 2, cd[2] * 2):
        raise ValueError("D1 dims must be twice the coarse dims")
    return c0, d0, d1


def pack_diffusible(decomp: WaveletDecomposition) -> PackedRepresentation:
    """Pack (C0, D0, D1) into a coarse-resolution grid with 64 channels."""
    c0, d0, d1 = _coarse_detail_layers(decomp)
    nx, ny, nz = c0.dims
    ch = PACKED_CHANNELS
    data = [0.0] * (nx * ny * nz * ch)
    for ix in range(nx):
        for iy in range(ny):
            for iz in range(nz):
                base = ((ix * ny + iy) * nz + iz) * ch
                data[base] = c0.get(ix, iy, iz)
                off = 1
                for name in DETAIL_NAMES:
                    data[base + off] = d0[name].get(ix, iy, iz)
                    off += 1
                for name in DETAIL_NAMES:
                    band = d1[name]
                    for (dx, dy, dz) in _D1_CHILD_OFFSETS:
                        data[base + off] = band.get(2 * ix + dx, 2 * iy + dy, 2 * iz + dz)
                        off += 1
    return PackedRepresentation(c0.dims, ch, data)


def unpack_diffusible(
    packed: PackedRepresentation,
) -> Tuple[Grid3D, Dict[str, Grid3D], Dict[str, Grid3D]]:
    """Inverse of :func:`pack_diffusible`; returns (C0, D0, D1)."""
    if packed.channels != PACKED_CHANNELS:
        raise ValueError("unexpected channel count")
    nx, ny, nz = packed.dims
    c0_data = [0.0] * (nx * ny * nz)
    d0: Dict[str, List[float]] = {n: [0.0] * (nx * ny * nz) for n in DETAIL_NAMES}
    d1dims = (nx * 2, ny * 2, nz * 2)
    d1: Dict[str, List[float]] = {n: [0.0] * (d1dims[0] * d1dims[1] * d1dims[2]) for n in DETAIL_NAMES}
    ch = packed.channels
    for ix in range(nx):
        for iy in range(ny):
            for iz in range(nz):
                base = ((ix * ny + iy) * nz + iz) * ch
                c0_data[(ix * ny + iy) * nz + iz] = packed.data[base]
                off = 1
                for name in DETAIL_NAMES:
                    d0[name][(ix * ny + iy) * nz + iz] = packed.data[base + off]
                    off += 1
                for name in DETAIL_NAMES:
                    for (dx, dy, dz) in _D1_CHILD_OFFSETS:
                        cx, cy, cz = 2 * ix + dx, 2 * iy + dy, 2 * iz + dz
                        d1[name][(cx * d1dims[1] + cy) * d1dims[2] + cz] = packed.data[base + off]
                        off += 1
    c0 = Grid3D(packed.dims, c0_data)
    d0_grids = {n: Grid3D(packed.dims, d0[n]) for n in DETAIL_NAMES}
    d1_grids = {n: Grid3D(d1dims, d1[n]) for n in DETAIL_NAMES}
    return c0, d0_grids, d1_grids
