"""Separable 3D discrete wavelet transform for TSDF shape grids.

The central deterministic idea is a **wavelet-tree representation**: a shape is
encoded as a truncated signed distance field (TSDF) on a regular grid, then
decomposed with a wavelet transform into a coarse coefficient subband ``C0``
and a set of multiscale detail coefficient subbands.  A learned diffusion
*generator* is out of scope, but the transform itself is pure, deterministic
linear algebra and is *bijective*: the representation is lossless and can be
bijectively converted back to a TSDF through inverse wavelet transforms.

This module implements that transform from scratch, stdlib-only:

  * a 1D two-channel filter bank (analysis -> low/high, synthesis -> full) with
    two wavelet families that both give **perfect reconstruction**:
      - ``haar`` -- the orthonormal Haar wavelet;
      - ``bior53`` -- the Le Gall 5/3 *biorthogonal* wavelet (the JPEG-2000
        lifting wavelet), a stand-in for the paper's biorthogonal choice
        ("biorthogonal wavelets with 6 and 8 moments", p. 6 footnote);
  * the **separable single-level 3D DWT** producing the eight octant subbands
    ``LLL, HLL, LHL, HHL, LLH, HLH, LHH, HHH`` (the 3D case has *seven* detail
    subbands per level plus the all-low coarse band -- Fig. 5), and its inverse;
  * the **multi-level decomposition** that recursively transforms the coarse
    ``LLL`` band, yielding the ``C0`` root plus per-level detail subbands
    (Fig. 5's ``C2 -> C1 -> C0`` cascade), and its exact inverse round-trip.

All boundary handling is periodic so reconstruction is exact to floating point.
No wall clock, no randomness.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, Dict, List, Sequence, Tuple

Dims = Tuple[int, int, int]

# Names of the eight octant subbands, ordered (x, y, z) low/high bits.  "LLL" is
# the all-low coarse band; the other seven are detail subbands.
SUBBAND_NAMES: Tuple[str, ...] = (
    "LLL", "HLL", "LHL", "HHL", "LLH", "HLH", "LHH", "HHH",
)
DETAIL_NAMES: Tuple[str, ...] = tuple(n for n in SUBBAND_NAMES if n != "LLL")


@dataclass
class Grid3D:
    """A dense 3D array, row-major with index ``((ix*ny)+iy)*nz+iz``."""

    dims: Dims
    data: List[float]

    def __post_init__(self) -> None:
        nx, ny, nz = self.dims
        if nx <= 0 or ny <= 0 or nz <= 0:
            raise ValueError("dims must be positive")
        if len(self.data) != nx * ny * nz:
            raise ValueError("data length does not match dims")

    @classmethod
    def from_function(cls, dims: Dims, fn: Callable[[int, int, int], float]) -> "Grid3D":
        nx, ny, nz = dims
        data = [0.0] * (nx * ny * nz)
        p = 0
        for ix in range(nx):
            for iy in range(ny):
                for iz in range(nz):
                    data[p] = float(fn(ix, iy, iz))
                    p += 1
        return cls(dims, data)

    def get(self, ix: int, iy: int, iz: int) -> float:
        nx, ny, nz = self.dims
        return self.data[(ix * ny + iy) * nz + iz]

    def max_abs(self) -> float:
        return max((abs(v) for v in self.data), default=0.0)


# --------------------------------------------------------------------------- #
# 1D two-channel filter banks (analysis: full -> (low, high); synthesis back)  #
# --------------------------------------------------------------------------- #

_SQRT2 = math.sqrt(2.0)


def _haar_forward(x: Sequence[float]) -> Tuple[List[float], List[float]]:
    n = len(x)
    if n % 2 != 0:
        raise ValueError("Haar transform needs even length")
    low: List[float] = []
    high: List[float] = []
    for i in range(0, n, 2):
        a, b = x[i], x[i + 1]
        low.append((a + b) / _SQRT2)
        high.append((a - b) / _SQRT2)
    return low, high


def _haar_inverse(low: Sequence[float], high: Sequence[float]) -> List[float]:
    if len(low) != len(high):
        raise ValueError("low/high length mismatch")
    out: List[float] = [0.0] * (2 * len(low))
    for i, (l, h) in enumerate(zip(low, high)):
        out[2 * i] = (l + h) / _SQRT2
        out[2 * i + 1] = (l - h) / _SQRT2
    return out


def _le_gall_forward(x: Sequence[float]) -> Tuple[List[float], List[float]]:
    """Le Gall 5/3 biorthogonal wavelet via the lifting scheme, periodic."""
    n = len(x)
    if n % 2 != 0:
        raise ValueError("Le Gall 5/3 transform needs even length")
    half = n // 2
    even = [float(x[2 * i]) for i in range(half)]
    odd = [float(x[2 * i + 1]) for i in range(half)]
    # Predict: detail = odd - 0.5*(even[i] + even[i+1])
    detail = [odd[i] - 0.5 * (even[i] + even[(i + 1) % half]) for i in range(half)]
    # Update: approx = even + 0.25*(detail[i-1] + detail[i])
    approx = [even[i] + 0.25 * (detail[(i - 1) % half] + detail[i]) for i in range(half)]
    return approx, detail


def _le_gall_inverse(low: Sequence[float], high: Sequence[float]) -> List[float]:
    if len(low) != len(high):
        raise ValueError("low/high length mismatch")
    half = len(low)
    approx = list(low)
    detail = list(high)
    even = [approx[i] - 0.25 * (detail[(i - 1) % half] + detail[i]) for i in range(half)]
    odd = [detail[i] + 0.5 * (even[i] + even[(i + 1) % half]) for i in range(half)]
    out: List[float] = [0.0] * (2 * half)
    for i in range(half):
        out[2 * i] = even[i]
        out[2 * i + 1] = odd[i]
    return out


_FORWARD: Dict[str, Callable[[Sequence[float]], Tuple[List[float], List[float]]]] = {
    "haar": _haar_forward,
    "bior53": _le_gall_forward,
}
_INVERSE: Dict[str, Callable[[Sequence[float], Sequence[float]], List[float]]] = {
    "haar": _haar_inverse,
    "bior53": _le_gall_inverse,
}

WAVELETS: Tuple[str, ...] = tuple(_FORWARD.keys())


def wavelet_forward_1d(x: Sequence[float], wavelet: str = "haar") -> Tuple[List[float], List[float]]:
    """Public 1D analysis: full signal -> (low, high) half-length bands."""
    if wavelet not in _FORWARD:
        raise ValueError("unknown wavelet: %r" % (wavelet,))
    return _FORWARD[wavelet](x)


def wavelet_inverse_1d(low: Sequence[float], high: Sequence[float], wavelet: str = "haar") -> List[float]:
    """Public 1D synthesis: (low, high) -> full signal."""
    if wavelet not in _INVERSE:
        raise ValueError("unknown wavelet: %r" % (wavelet,))
    return _INVERSE[wavelet](low, high)


# --------------------------------------------------------------------------- #
# Separable transforms along a single axis                                     #
# --------------------------------------------------------------------------- #

def _idx(dims: Dims, ix: int, iy: int, iz: int) -> int:
    return (ix * dims[1] + iy) * dims[2] + iz


def _transform_axis(
    values: Sequence[float], dims: Dims, axis: int, wavelet: str
) -> Tuple[List[float], List[float], Dims]:
    """Apply the 1D analysis along ``axis`` to every fiber. Returns low, high."""
    n = dims[axis]
    if n % 2 != 0:
        raise ValueError("axis %d length %d is not even" % (axis, n))
    half = n // 2
    out_dims: Dims = tuple(half if i == axis else dims[i] for i in range(3))  # type: ignore[assignment]
    size = out_dims[0] * out_dims[1] * out_dims[2]
    low = [0.0] * size
    high = [0.0] * size
    others = [i for i in range(3) if i != axis]
    fwd = _FORWARD[wavelet]
    for a in range(dims[others[0]]):
        for b in range(dims[others[1]]):
            coord = [0, 0, 0]
            coord[others[0]] = a
            coord[others[1]] = b
            fiber = []
            for k in range(n):
                coord[axis] = k
                fiber.append(values[_idx(dims, coord[0], coord[1], coord[2])])
            lo, hi = fwd(fiber)
            for k in range(half):
                coord[axis] = k
                j = _idx(out_dims, coord[0], coord[1], coord[2])
                low[j] = lo[k]
                high[j] = hi[k]
    return low, high, out_dims


def _inverse_axis(
    low: Sequence[float], high: Sequence[float], low_dims: Dims, axis: int, wavelet: str
) -> Tuple[List[float], Dims]:
    full_n = low_dims[axis] * 2
    out_dims: Dims = tuple(full_n if i == axis else low_dims[i] for i in range(3))  # type: ignore[assignment]
    size = out_dims[0] * out_dims[1] * out_dims[2]
    out = [0.0] * size
    others = [i for i in range(3) if i != axis]
    inv = _INVERSE[wavelet]
    half = low_dims[axis]
    for a in range(low_dims[others[0]]):
        for b in range(low_dims[others[1]]):
            coord = [0, 0, 0]
            coord[others[0]] = a
            coord[others[1]] = b
            lo = []
            hi = []
            for k in range(half):
                coord[axis] = k
                j = _idx(low_dims, coord[0], coord[1], coord[2])
                lo.append(low[j])
                hi.append(high[j])
            full = inv(lo, hi)
            for k in range(full_n):
                coord[axis] = k
                out[_idx(out_dims, coord[0], coord[1], coord[2])] = full[k]
    return out, out_dims


# --------------------------------------------------------------------------- #
# Single-level 3D DWT -> eight octant subbands                                 #
# --------------------------------------------------------------------------- #

def dwt3_level(grid: Grid3D, wavelet: str = "haar") -> Tuple[Dict[str, Grid3D], Dims]:
    """One level of separable 3D DWT. Returns the eight subbands and their dims.

    Each dimension must be even. Output subbands each have half the resolution.
    """
    for n in grid.dims:
        if n % 2 != 0:
            raise ValueError("all dims must be even for a DWT level: %r" % (grid.dims,))
    d = grid.dims
    # Split along z (axis 2): zLow, zHigh
    zlow, zhigh, dz = _transform_axis(grid.data, d, 2, wavelet)
    # Split along y (axis 1)
    zl_ylow, zl_yhigh, dzy = _transform_axis(zlow, dz, 1, wavelet)
    zh_ylow, zh_yhigh, _ = _transform_axis(zhigh, dz, 1, wavelet)
    # Split along x (axis 0)
    lll, hll, sub = _transform_axis(zl_ylow, dzy, 0, wavelet)
    lhl, hhl, _ = _transform_axis(zl_yhigh, dzy, 0, wavelet)
    llh, hlh, _ = _transform_axis(zh_ylow, dzy, 0, wavelet)
    lhh, hhh, _ = _transform_axis(zh_yhigh, dzy, 0, wavelet)
    subbands = {
        "LLL": Grid3D(sub, lll), "HLL": Grid3D(sub, hll),
        "LHL": Grid3D(sub, lhl), "HHL": Grid3D(sub, hhl),
        "LLH": Grid3D(sub, llh), "HLH": Grid3D(sub, hlh),
        "LHH": Grid3D(sub, lhh), "HHH": Grid3D(sub, hhh),
    }
    return subbands, sub


def idwt3_level(subbands: Dict[str, Grid3D], wavelet: str = "haar") -> Grid3D:
    """Invert one level of 3D DWT from the eight octant subbands."""
    missing = [n for n in SUBBAND_NAMES if n not in subbands]
    if missing:
        raise ValueError("missing subbands: %r" % (missing,))
    sub_dims = subbands["LLL"].dims
    for name in SUBBAND_NAMES:
        if subbands[name].dims != sub_dims:
            raise ValueError("subband %s has mismatched dims" % name)
    # Invert x (axis 0): combine (L*, H*) pairs sharing y,z bits.
    zl_ylow, dyx = _inverse_axis(subbands["LLL"].data, subbands["HLL"].data, sub_dims, 0, wavelet)
    zl_yhigh, _ = _inverse_axis(subbands["LHL"].data, subbands["HHL"].data, sub_dims, 0, wavelet)
    zh_ylow, _ = _inverse_axis(subbands["LLH"].data, subbands["HLH"].data, sub_dims, 0, wavelet)
    zh_yhigh, _ = _inverse_axis(subbands["LHH"].data, subbands["HHH"].data, sub_dims, 0, wavelet)
    # Invert y (axis 1)
    zlow, dy = _inverse_axis(zl_ylow, zl_yhigh, dyx, 1, wavelet)
    zhigh, _ = _inverse_axis(zh_ylow, zh_yhigh, dyx, 1, wavelet)
    # Invert z (axis 2)
    full, full_dims = _inverse_axis(zlow, zhigh, dy, 2, wavelet)
    return Grid3D(full_dims, full)


# --------------------------------------------------------------------------- #
# Multi-level decomposition (recurse on the coarse LLL band)                   #
# --------------------------------------------------------------------------- #

@dataclass
class WaveletDecomposition:
    """Multi-level wavelet-tree decomposition.

    ``coarse`` is the final all-low ``C0`` band.  ``details`` is a list of
    per-level dicts of the seven detail subbands, ordered *finest first*
    (``details[0]`` are the highest-frequency subbands at half the input
    resolution; ``details[-1]`` are the coarsest, at the resolution of
    ``coarse``).  This mirrors the paper's ``C2 -> C1 -> C0`` cascade where
    ``D0`` (coarsest detail) sits beside ``C0``.
    """

    input_dims: Dims
    wavelet: str
    coarse: Grid3D
    details: List[Dict[str, Grid3D]]


def dwt3(grid: Grid3D, levels: int, wavelet: str = "haar") -> WaveletDecomposition:
    if levels < 1:
        raise ValueError("levels must be >= 1")
    factor = 1 << levels
    for n in grid.dims:
        if n % factor != 0:
            raise ValueError("each dim must be divisible by 2**levels=%d" % factor)
    current = grid
    details: List[Dict[str, Grid3D]] = []
    for _ in range(levels):
        subbands, _sub = dwt3_level(current, wavelet)
        detail = {n: subbands[n] for n in DETAIL_NAMES}
        details.append(detail)
        current = subbands["LLL"]
    return WaveletDecomposition(grid.dims, wavelet, current, details)


def idwt3(decomp: WaveletDecomposition) -> Grid3D:
    current = decomp.coarse
    for detail in reversed(decomp.details):
        subbands = dict(detail)
        subbands["LLL"] = current
        current = idwt3_level(subbands, decomp.wavelet)
    return current
