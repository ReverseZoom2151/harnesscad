"""Deterministic front-to-back splat compositing + tile binning (Turbo3D / 3DGS).

Turbo3D (Hu et al., 2024) generates a 3D Gaussian Splatting (3DGS) asset and
renders it with the standard tile-based rasteriser. The *learned* generator and
reconstructor are out of scope, but the rasteriser's compositing stage is pure,
deterministic math and is **not** covered by ``geometry.gaussiancad_splatting``
(which only provides the per-Gaussian forward math: covariance, projection,
footprint bounding box, 2D density). This module adds the pieces that turn a set
of already-projected Gaussians into a pixel:

  * ``tile_bins`` -- the 3DGS rasteriser splits the image into a grid of square
    tiles and assigns each Gaussian to every tile its footprint bounding box
    overlaps (tile binning / culling);
  * ``alpha_from_kernel`` -- a splat's alpha is ``opacity * gaussian_kernel``,
    clamped to ``[0, 1)`` (the 3DGS alpha, Kerbl et al. Eq. 3 companion);
  * ``composite_front_to_back`` -- the volumetric over operator accumulated in
    depth order,  ``C = sum_i c_i * alpha_i * T_i``  with transmittance
    ``T_i = prod_{j<i} (1 - alpha_j)``, ``T_0 = 1`` -- with early ray termination
    once transmittance drops below a threshold (the standard 3DGS optimisation);
  * ``sort_front_to_back`` -- the depth sort (ascending view-space depth) the
    compositing relies on.

Everything is closed-form and reproducible: no learned model, no wall clock, no
randomness. Colours are plain tuples/lists of channel floats.
"""

from __future__ import annotations

from typing import List, Sequence, Tuple


# --------------------------------------------------------------------------- #
# Tile binning (rasteriser culling)
# --------------------------------------------------------------------------- #
def tile_bins(
    bbox: Sequence[float],
    image_width: int,
    image_height: int,
    tile_size: int,
) -> List[Tuple[int, int]]:
    """Tiles overlapped by an axis-aligned footprint ``bbox`` in pixel space.

    ``bbox`` is ``(u_min, v_min, u_max, v_max)`` (e.g. from
    ``gaussiancad_splatting.footprint_bbox``). The image is divided into a grid
    of ``tile_size``-pixel squares; the last row/column may be partial. Returns
    the ``(tile_col, tile_row)`` indices of every tile the box overlaps, in
    row-major order. A box fully outside the image yields an empty list (the
    Gaussian is culled).
    """
    if image_width <= 0 or image_height <= 0:
        raise ValueError("image dimensions must be positive")
    if tile_size <= 0:
        raise ValueError("tile_size must be positive")
    u_min, v_min, u_max, v_max = (float(v) for v in bbox)
    if u_max < u_min or v_max < v_min:
        raise ValueError("bbox must have max >= min on each axis")

    # Clip to image bounds [0, W) x [0, H).
    lo_u = max(0.0, u_min)
    lo_v = max(0.0, v_min)
    hi_u = min(float(image_width) - 1.0, u_max)
    hi_v = min(float(image_height) - 1.0, v_max)
    if hi_u < lo_u or hi_v < lo_v:
        return []

    n_cols = (image_width + tile_size - 1) // tile_size
    n_rows = (image_height + tile_size - 1) // tile_size
    c0 = min(int(lo_u) // tile_size, n_cols - 1)
    c1 = min(int(hi_u) // tile_size, n_cols - 1)
    r0 = min(int(lo_v) // tile_size, n_rows - 1)
    r1 = min(int(hi_v) // tile_size, n_rows - 1)
    return [(c, r) for r in range(r0, r1 + 1) for c in range(c0, c1 + 1)]


def tile_grid_shape(image_width: int, image_height: int, tile_size: int) -> Tuple[int, int]:
    """Return ``(n_cols, n_rows)`` of the tile grid covering the image."""
    if image_width <= 0 or image_height <= 0:
        raise ValueError("image dimensions must be positive")
    if tile_size <= 0:
        raise ValueError("tile_size must be positive")
    return (
        (image_width + tile_size - 1) // tile_size,
        (image_height + tile_size - 1) // tile_size,
    )


# --------------------------------------------------------------------------- #
# Alpha from a Gaussian kernel value
# --------------------------------------------------------------------------- #
def alpha_from_kernel(opacity: float, kernel: float) -> float:
    """3DGS splat alpha ``alpha = clamp(opacity * kernel, 0, 0.999)``.

    ``opacity`` is the Gaussian's learned opacity in ``[0, 1]`` and ``kernel`` is
    the unnormalised Gaussian value in ``[0, 1]`` (e.g. from
    ``gaussiancad_splatting.evaluate_gaussian_2d``). The alpha is clamped strictly
    below one so transmittance can never reach exactly zero mid-ray (matching the
    reference rasteriser, which skips fully opaque contributions).
    """
    if not 0.0 <= opacity <= 1.0:
        raise ValueError("opacity must be in [0, 1]")
    if not 0.0 <= kernel <= 1.0:
        raise ValueError("kernel must be in [0, 1]")
    a = opacity * kernel
    if a < 0.0:
        return 0.0
    if a > 0.999:
        return 0.999
    return a


# --------------------------------------------------------------------------- #
# Depth sorting
# --------------------------------------------------------------------------- #
def sort_front_to_back(
    depths: Sequence[float],
) -> List[int]:
    """Indices ordering ``depths`` front-to-back (ascending view-space depth).

    Ties keep the original order (stable sort), matching a stable key sort in the
    rasteriser. Returns a permutation of ``range(len(depths))``.
    """
    return sorted(range(len(depths)), key=lambda i: float(depths[i]))


# --------------------------------------------------------------------------- #
# Front-to-back alpha compositing (the "over" operator)
# --------------------------------------------------------------------------- #
def composite_front_to_back(
    colors: Sequence[Sequence[float]],
    alphas: Sequence[float],
    background: Sequence[float] | None = None,
    min_transmittance: float = 1e-4,
) -> Tuple[List[float], float]:
    """Composite ordered splats with the front-to-back ``over`` operator.

    ``colors`` and ``alphas`` are already in front-to-back (near-to-far) order --
    use :func:`sort_front_to_back` first if needed. The accumulation is

        C = sum_i  c_i * alpha_i * T_i,     T_i = prod_{j<i} (1 - alpha_j)

    with ``T_0 = 1``. Rays terminate early once ``T_i`` falls below
    ``min_transmittance`` (the remaining splats contribute negligibly). Any
    residual transmittance blends the optional ``background`` colour. Returns
    ``(composited_color, accumulated_alpha)`` where ``accumulated_alpha`` is the
    coverage ``1 - T_final`` (before adding the background).
    """
    if len(colors) != len(alphas):
        raise ValueError("colors and alphas must have equal length")
    if not 0.0 <= min_transmittance <= 1.0:
        raise ValueError("min_transmittance must be in [0, 1]")

    n_channels = len(colors[0]) if len(colors) else (len(background) if background else 0)
    accum = [0.0] * n_channels
    transmittance = 1.0
    for color, alpha in zip(colors, alphas):
        a = float(alpha)
        if not 0.0 <= a <= 1.0:
            raise ValueError("alpha must be in [0, 1]")
        if len(color) != n_channels:
            raise ValueError("all colours must have the same channel count")
        weight = a * transmittance
        for ch in range(n_channels):
            accum[ch] += weight * float(color[ch])
        transmittance *= (1.0 - a)
        if transmittance < min_transmittance:
            break

    accumulated_alpha = 1.0 - transmittance
    if background is not None:
        if len(background) != n_channels:
            raise ValueError("background must match colour channel count")
        for ch in range(n_channels):
            accum[ch] += transmittance * float(background[ch])
    return accum, accumulated_alpha
