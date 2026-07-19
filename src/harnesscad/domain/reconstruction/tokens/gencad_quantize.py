"""Exact CAD normalisation + quantisation pipeline (dataset processing).

A companion command specification states *that* continuous parameters are
normalised and quantised to 256 levels; it does not carry the exact affine
constants. Those constants matter -- a decoder that de-quantises with the wrong
offset reconstructs a shifted solid. This module fixes the constants and formulas
for the shape-, sketch- and extrude-level maps:

Shape-level normalisation, applied before anything else::

    scale = size * NORM_FACTOR / max(|bbox|)      NORM_FACTOR = 0.75, size = 1.0

  The 0.75 factor deliberately leaves head-room so that data augmentation cannot
  push a coordinate out of the ``[-1, 1]`` cube.

Sketch-level normalisation maps a profile into a ``size x size`` raster
with the profile's *start point* at the raster centre::

    scale = (size / 2 * NORM_FACTOR - 1) / bbox_size          bbox_size measured
    p'    = (p - start_point) * scale + (size / 2, size / 2)   from the start point

Parameter quantisation (``numericalize`` / ``denumericalize``), all to ``n = 256``::

    sketch coordinate  q = clip(round(p), 0, n-1)          (already in raster space)
    radius             q = clip(round(r), 1, n-1)          (a radius is never 0)
    unit coordinate    q = clip(round((v + 1) / 2 * n), 0, n-1)   v in [-1, 1]
    angle (rad)        q = clip(round((a / pi + 1) / 2 * n), 0, n-1)   a in [-pi, pi]
    sketch size s      q = clip(round(s / 2 * n), 0, n-1)   (s in [0, 2], no offset)

Note the asymmetry: extents/origins carry the ``+1`` offset, the sketch *size* does
not. Pure standard library, deterministic.
"""

from __future__ import annotations

import math
from typing import Sequence, Tuple

Vec2 = Tuple[float, float]
Vec3 = Tuple[float, float, float]

NORM_FACTOR = 0.75    # GenCAD macro.NORM_FACTOR
ARGS_DIM = 256        # GenCAD macro.ARGS_DIM
SKETCH_DIM = 256      # default raster size for a normalised sketch


# --- shape-level normalisation ----------------------------------------------
def shape_normalize_scale(bbox: Sequence[Sequence[float]], size: float = 1.0) -> float:
    """``CADSequence.normalize``: ``size * NORM_FACTOR / max(|bbox|)``.

    ``bbox`` is any iterable of 3D points (the reference passes ``[max_point,
    min_point]``); the divisor is the largest absolute *coordinate*, so the shape ends
    up inside the ``[-0.75, 0.75]`` cube for ``size = 1``.
    """
    peak = max(abs(v) for point in bbox for v in point)
    if peak == 0.0:
        raise ValueError("degenerate bounding box: all coordinates are zero")
    return size * NORM_FACTOR / peak


def normalize_shape_point(point: Vec3, scale: float) -> Vec3:
    """Apply the shape scale to a 3D point (the reference translates by 0.0)."""
    return (point[0] * scale, point[1] * scale, point[2] * scale)


# --- sketch-level normalisation ---------------------------------------------
def sketch_normalize_scale(bbox_size: float, size: int = SKETCH_DIM) -> float:
    """``(size / 2 * NORM_FACTOR - 1) / bbox_size`` (``SketchBase.normalize``)."""
    if bbox_size <= 0.0:
        raise ValueError("bbox_size must be positive")
    return (size / 2 * NORM_FACTOR - 1) / bbox_size


def sketch_denormalize_scale(bbox_size: float, size: int = SKETCH_DIM) -> float:
    """Inverse of :func:`sketch_normalize_scale` (``SketchBase.denormalize``)."""
    return bbox_size / (size / 2 * NORM_FACTOR - 1)


def normalize_sketch_point(point: Vec2, start_point: Vec2, bbox_size: float,
                           size: int = SKETCH_DIM) -> Vec2:
    """Move ``start_point`` to the raster centre and scale by the sketch factor."""
    scale = sketch_normalize_scale(bbox_size, size)
    half = size / 2
    return ((point[0] - start_point[0]) * scale + half,
            (point[1] - start_point[1]) * scale + half)


def denormalize_sketch_point(point: Vec2, bbox_size: float,
                             size: int = SKETCH_DIM) -> Vec2:
    """Inverse of :func:`normalize_sketch_point` (start point back at the origin)."""
    scale = sketch_denormalize_scale(bbox_size, size)
    half = size / 2
    return ((point[0] - half) * scale, (point[1] - half) * scale)


def normalize_sketch_length(length: float, bbox_size: float,
                            size: int = SKETCH_DIM) -> float:
    """Scale a length (e.g. a circle radius) into raster units -- no translation."""
    return length * sketch_normalize_scale(bbox_size, size)


def bbox_size_from_bbox(bbox: Tuple[float, float, float, float],
                        start_point: Vec2) -> float:
    """``SketchBase.bbox_size``: max |box corner - start point| across x and y."""
    min_x, min_y, max_x, max_y = bbox
    sx, sy = start_point
    return max(abs(max_x - sx), abs(max_y - sy), abs(min_x - sx), abs(min_y - sy))


# --- parameter quantisation --------------------------------------------------
def _clip(v: int, lo: int, hi: int) -> int:
    return max(lo, min(hi, v))


def quantize_coordinate(value: float, n: int = ARGS_DIM) -> int:
    """Raster-space sketch coordinate: ``clip(round(v), 0, n-1)``."""
    return _clip(int(round(value)), 0, n - 1)


def quantize_radius(value: float, n: int = ARGS_DIM) -> int:
    """Raster-space radius: like a coordinate but floored at 1 (never zero)."""
    return _clip(int(round(value)), 1, n - 1)


def quantize_unit(value: float, n: int = ARGS_DIM) -> int:
    """Unit-cube value in ``[-1, 1]`` -> ``clip(round((v + 1) / 2 * n), 0, n-1)``."""
    return _clip(int(round((value + 1.0) / 2 * n)), 0, n - 1)


def dequantize_unit(level: int, n: int = ARGS_DIM) -> float:
    """Inverse of :func:`quantize_unit`: ``q / n * 2 - 1``."""
    return level / n * 2 - 1.0


def quantize_angle(value: float, n: int = ARGS_DIM) -> int:
    """Angle in ``[-pi, pi]`` -> ``clip(round((a / pi + 1) / 2 * n), 0, n-1)``."""
    return _clip(int(round((value / math.pi + 1.0) / 2 * n)), 0, n - 1)


def dequantize_angle(level: int, n: int = ARGS_DIM) -> float:
    """Inverse of :func:`quantize_angle`: ``(q / n * 2 - 1) * pi``."""
    return (level / n * 2 - 1.0) * math.pi


def quantize_sketch_size(value: float, n: int = ARGS_DIM) -> int:
    """Sketch size ``s`` in ``[0, 2]`` -> ``clip(round(s / 2 * n), 0, n-1)``.

    Unlike origins and extents this carries *no* ``+1`` offset -- a size is already
    non-negative. Getting this wrong scales every profile by roughly two.
    """
    return _clip(int(round(value / 2 * n)), 0, n - 1)


def dequantize_sketch_size(level: int, n: int = ARGS_DIM) -> float:
    """Inverse of :func:`quantize_sketch_size`: ``q / n * 2``."""
    return level / n * 2


def check_extent_range(extent: float) -> float:
    """The reference asserts ``-2 <= extent <= 2`` before quantising an extrude."""
    if not -2.0 <= extent <= 2.0:
        raise ValueError("extent out of quantisable range [-2, 2]: {}".format(extent))
    return extent


def quantize_coord_system(origin: Vec3, theta: float, phi: float, gamma: float,
                          n: int = ARGS_DIM) -> Tuple[int, int, int, int, int, int]:
    """``CoordSystem.numericalize``: origin as unit values, angles as angles."""
    return (quantize_unit(origin[0], n), quantize_unit(origin[1], n),
            quantize_unit(origin[2], n), quantize_angle(theta, n),
            quantize_angle(phi, n), quantize_angle(gamma, n))


def dequantize_coord_system(levels: Sequence[int],
                            n: int = ARGS_DIM) -> Tuple[Vec3, float, float, float]:
    """``CoordSystem.denumericalize``: inverse of :func:`quantize_coord_system`."""
    if len(levels) != 6:
        raise ValueError("expected 6 levels (ox, oy, oz, theta, phi, gamma)")
    origin = (dequantize_unit(levels[0], n), dequantize_unit(levels[1], n),
              dequantize_unit(levels[2], n))
    return (origin, dequantize_angle(levels[3], n), dequantize_angle(levels[4], n),
            dequantize_angle(levels[5], n))


def quantization_step(n: int = ARGS_DIM) -> float:
    """Width of one quantisation bucket in unit-cube space: ``2 / n``."""
    return 2.0 / n


def max_quantization_error(n: int = ARGS_DIM) -> float:
    """Worst-case round-trip error of a unit-cube value: half a bucket."""
    return quantization_step(n) / 2
