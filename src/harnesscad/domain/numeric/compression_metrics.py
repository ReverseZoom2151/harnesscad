"""Compression and reconstruction-error metrics for wavelet shape encoding.

Tables 1 & 8 and
Sec. 4.  The paper evaluates its wavelet-tree representation with two families
of numbers, both fully deterministic:

  * **Representation compactness** -- how many floating-point "input variables"
    the representation carries versus the raw ``256^3`` SDF grid (Table 1's
    "Input Variables" column and the 44.5% data-loading reduction claim).

  * **Reconstruction fidelity** -- how faithfully the (lossy, top-K filtered)
    representation reconstructs the original TSDF, reported as an
    Intersection-over-Union of the reconstructed occupancy against ground truth
    (Table 1's IOU of 99.56%, Sec. 4's mean IoU claims).

This module implements those metrics on plain grids (works with the
``Grid3D`` produced by ``numeric.makeashape_wavelet_transform`` or with any
equal-length sequences):

  * ``mse`` / ``rmse`` / ``relative_l2_error`` / ``psnr`` -- signal error;
  * ``occupancy_from_sdf`` / ``occupancy_iou`` -- the IoU the paper reports,
    thresholding a (T)SDF at an iso level to get inside/outside occupancy;
  * ``compression_ratio`` / ``coefficient_reduction_fraction`` /
    ``streaming_reduction_fraction`` -- compactness ratios;
  * ``evaluate_top_k`` -- the end-to-end curve: decompose, keep top-K detail
    coefficients, reconstruct, and report both compactness and fidelity for a
    given K.

Everything is stdlib-only and deterministic.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Sequence, Tuple

from harnesscad.domain.numeric.wavelet_transform import Grid3D, dwt3, idwt3
from harnesscad.domain.numeric.wavelet_tree import (
    compress_decomposition_top_k, detail_coefficient_count, nonzero_detail_count,
)


def _values(x) -> Sequence[float]:
    return x.data if isinstance(x, Grid3D) else x


def _check(a: Sequence[float], b: Sequence[float]) -> None:
    if len(a) != len(b):
        raise ValueError("length mismatch: %d vs %d" % (len(a), len(b)))
    if not a:
        raise ValueError("empty input")


# --------------------------------------------------------------------------- #
# Signal-error metrics                                                          #
# --------------------------------------------------------------------------- #

def mse(a, b) -> float:
    a, b = _values(a), _values(b)
    _check(a, b)
    return sum((x - y) ** 2 for x, y in zip(a, b)) / len(a)


def rmse(a, b) -> float:
    return math.sqrt(mse(a, b))


def relative_l2_error(a, b) -> float:
    """``||a-b|| / ||a||`` (0 when identical; the reference is ``a``)."""
    a, b = _values(a), _values(b)
    _check(a, b)
    num = math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))
    den = math.sqrt(sum(x * x for x in a))
    if den == 0.0:
        return 0.0 if num == 0.0 else float("inf")
    return num / den


def psnr(a, b, value_range: float) -> float:
    """Peak signal-to-noise ratio in dB; ``inf`` for a perfect match."""
    if value_range <= 0:
        raise ValueError("value_range must be positive")
    e = mse(a, b)
    if e == 0.0:
        return float("inf")
    return 20.0 * math.log10(value_range) - 10.0 * math.log10(e)


# --------------------------------------------------------------------------- #
# Occupancy IoU (the paper's headline reconstruction metric)                    #
# --------------------------------------------------------------------------- #

def occupancy_from_sdf(sdf, iso: float = 0.0) -> List[bool]:
    """Inside/outside occupancy: a cell is occupied when ``value <= iso``."""
    return [v <= iso for v in _values(sdf)]


def occupancy_iou(a, b, iso: float = 0.0) -> float:
    """Intersection-over-Union of the occupancy of two (T)SDF grids.

    Returns 1.0 when both are empty (degenerate but well-defined).
    """
    oa = occupancy_from_sdf(a, iso)
    ob = occupancy_from_sdf(b, iso)
    _check(oa, ob)
    inter = sum(1 for x, y in zip(oa, ob) if x and y)
    union = sum(1 for x, y in zip(oa, ob) if x or y)
    if union == 0:
        return 1.0
    return inter / union


# --------------------------------------------------------------------------- #
# Compactness ratios                                                            #
# --------------------------------------------------------------------------- #

def compression_ratio(original_count: int, kept_count: int) -> float:
    """``original / kept`` -- how many times smaller the representation is."""
    if kept_count <= 0:
        raise ValueError("kept_count must be positive")
    if original_count < 0:
        raise ValueError("original_count must be non-negative")
    return original_count / kept_count


def coefficient_reduction_fraction(original_count: int, kept_count: int) -> float:
    """Fraction of coefficients dropped: ``1 - kept/original``."""
    if original_count <= 0:
        raise ValueError("original_count must be positive")
    if kept_count < 0:
        raise ValueError("kept_count must be non-negative")
    return 1.0 - kept_count / original_count


def streaming_reduction_fraction(original_bytes: float, representation_bytes: float) -> float:
    """The paper's 44.5% data-loading reduction is exactly this fraction."""
    if original_bytes <= 0:
        raise ValueError("original_bytes must be positive")
    if representation_bytes < 0:
        raise ValueError("representation_bytes must be non-negative")
    return 1.0 - representation_bytes / original_bytes


# --------------------------------------------------------------------------- #
# End-to-end top-K evaluation                                                   #
# --------------------------------------------------------------------------- #

@dataclass
class TopKReport:
    k: int
    wavelet: str
    levels: int
    kept_detail_coeffs: int
    total_detail_coeffs: int
    reduction_fraction: float
    mse: float
    relative_l2_error: float
    occupancy_iou: float


def evaluate_top_k(
    grid: Grid3D, k: int, levels: int = 2, wavelet: str = "haar", iso: float = 0.0
) -> TopKReport:
    """Decompose, keep top-K detail coeffs per level, reconstruct, and score.

    Ties compactness (how many detail coefficients survive) to fidelity (MSE,
    relative L2, occupancy IoU) for a single K -- the core Table-1 style trade.
    """
    decomp = dwt3(grid, levels=levels, wavelet=wavelet)
    comp = compress_decomposition_top_k(decomp, k)
    recon = idwt3(comp)
    total = detail_coefficient_count(decomp)
    kept = nonzero_detail_count(comp)
    return TopKReport(
        k=k,
        wavelet=wavelet,
        levels=levels,
        kept_detail_coeffs=kept,
        total_detail_coeffs=total,
        reduction_fraction=(coefficient_reduction_fraction(total, kept) if total else 0.0),
        mse=mse(grid, recon),
        relative_l2_error=relative_l2_error(grid, recon),
        occupancy_iou=occupancy_iou(grid, recon, iso),
    )
