"""Exact CAD normalisation / quantisation numerics (reference implementation).

``reconstruction.deepcad_command_spec`` already models the *command vocabulary* and a
generic symmetric quantiser ``round((v - low)/(high - low) * (n - 1))``. That is NOT
what the reference implementation does: it uses a *family* of field-specific affine
maps with ``n`` (not ``n - 1``) in the denominator and a ``clip(0, n-1)``, and each
field family (unit-cube coordinate, angle, size, sketch pixel, sweep angle) has its
own map. This module reproduces those maps exactly, so a vector produced here is
bit-comparable with the released ``.h5`` data.

The five quantisation families
------------------------------
=========================  ================================  =====================
field                      forward (n = 256)                 inverse
=========================  ================================  =====================
unit-cube coord / extent   ``round((x+1)/2*n).clip(0,n-1)``  ``q/n*2 - 1``
plane angle (-pi..pi)      ``round((a/pi+1)/2*n).clip(..)``  ``(q/n*2 - 1)*pi``
sketch size (0..2)         ``round(s/2*n).clip(0,n-1)``      ``q/n*2``
sketch pixel (0..n-1)      ``round(x).clip(0,n-1)``          identity
arc sweep angle            ``round(a/(2*pi)*n).clip(0,n-1)`` ``q/n*2*pi``
=========================  ================================  =====================

Note the asymmetry deliberately kept from the reference: the forward map divides by
``n`` and rounds, so ``+1.0`` clips down to level ``n-1`` -- the round-trip is lossy
by design and *not* self-inverse at the top of the range.

Normalisation
-------------
* ``NORM_FACTOR = 0.75`` -- shrink factor leaving head-room for data augmentation.
* Shape: ``scale = size * NORM_FACTOR / max(|bbox|)`` scales the whole CAD sequence
  into the cube ``(-0.75 .. 0.75)`` for ``size = 1``.
* Sketch profile: mapped into a ``size x size`` (default 256) pixel canvas with the
  loop's *start point* at the canvas centre and
  ``scale = (size/2 * NORM_FACTOR - 1) / bbox_size``, where ``bbox_size`` is measured
  **relative to the start point** (max abs deviation), not the bbox diagonal.

Pure stdlib, deterministic. Learned models and OCC B-Rep export are out of scope.
"""

from __future__ import annotations

import math
from typing import Iterable, Sequence

# --- constants ---------------------------------------------------------------
NORM_FACTOR = 0.75
ARGS_DIM = 256          # quantisation levels
SKETCH_DIM = 256        # sketch canvas size


# --- generic helpers --------------------------------------------------------
def _clip(value: int, low: int, high: int) -> int:
    return max(low, min(high, value))


def _round_half_even(value: float) -> int:
    """numpy's ``round`` (banker's rounding), which Python's ``round`` also uses."""
    return int(round(value))


# --- family 1: unit-cube coordinates / extrude extents ----------------------
def numericalize_unit(value: float, n: int = ARGS_DIM) -> int:
    """``x`` in ``[-1, 1]`` -> level ``round((x+1)/2*n)`` clipped to ``0..n-1``."""
    return _clip(_round_half_even((value + 1.0) / 2 * n), 0, n - 1)


def denumericalize_unit(level: int, n: int = ARGS_DIM) -> float:
    """Inverse of :func:`numericalize_unit`: ``q/n*2 - 1``."""
    return level / n * 2 - 1.0


# --- family 2: sketch-plane angles (theta, phi, gamma) ----------------------
def numericalize_angle(angle: float, n: int = ARGS_DIM) -> int:
    """Angle in ``[-pi, pi]`` -> level ``round((a/pi + 1)/2*n)`` clipped."""
    return _clip(_round_half_even((angle / math.pi + 1.0) / 2 * n), 0, n - 1)


def denumericalize_angle(level: int, n: int = ARGS_DIM) -> float:
    """Inverse of :func:`numericalize_angle`: ``(q/n*2 - 1) * pi``."""
    return (level / n * 2 - 1.0) * math.pi


# --- family 3: sketch size (a positive scalar in 0..2) ----------------------
def numericalize_size(size: float, n: int = ARGS_DIM) -> int:
    """Sketch size -> level ``round(s/2*n)`` clipped to ``0..n-1``."""
    return _clip(_round_half_even(size / 2 * n), 0, n - 1)


def denumericalize_size(level: int, n: int = ARGS_DIM) -> float:
    """Inverse of :func:`numericalize_size`: ``q/n*2``."""
    return level / n * 2


# --- family 4: sketch pixel coordinates (already on the 0..n-1 canvas) ------
def numericalize_pixel(value: float, n: int = ARGS_DIM) -> int:
    """Canvas coordinate -> ``round(x)`` clipped to ``0..n-1``."""
    return _clip(_round_half_even(value), 0, n - 1)


def numericalize_radius(radius: float, n: int = ARGS_DIM) -> int:
    """Circle radius -> ``round(r)`` clipped to ``1..n-1`` (radius 0 is invalid)."""
    return _clip(_round_half_even(radius), 1, n - 1)


# --- family 5: arc sweep angle ---------------------------------------------
def numericalize_sweep(angle: float, n: int = ARGS_DIM) -> int:
    """Sweep angle in ``[0, 2pi]`` -> level ``round(a/(2pi)*n)`` clipped."""
    return _clip(_round_half_even(angle / (2 * math.pi) * n), 0, n - 1)


def denumericalize_sweep(level: int, n: int = ARGS_DIM) -> float:
    """Inverse of :func:`numericalize_sweep`: ``q/n*2*pi``."""
    return level / n * 2 * math.pi


# --- shape normalisation ------------------------------------------------------
def shape_scale(bbox: Sequence[Sequence[float]], size: float = 1.0) -> float:
    """``size * NORM_FACTOR / max |bbox|`` -- the CAD-sequence normalising scale.

    ``bbox`` is any iterable of 3D points (the reference stacks
    ``[max_point, min_point]``).
    Raises ``ValueError`` on a degenerate (all-zero) bounding box.
    """
    peak = max(abs(c) for point in bbox for c in point)
    if peak == 0:
        raise ValueError("degenerate bounding box")
    return size * NORM_FACTOR / peak


def normalize_shape(points: Iterable[Sequence[float]],
                    bbox: Sequence[Sequence[float]],
                    size: float = 1.0) -> list[tuple[float, ...]]:
    """Scale 3D points by :func:`shape_scale` (translation is 0 in the reference)."""
    scale = shape_scale(bbox, size)
    return [tuple(c * scale for c in p) for p in points]


# --- sketch normalisation -----------------------------------------------------
def sketch_bbox_size(points: Iterable[Sequence[float]],
                     start_point: Sequence[float]) -> float:
    """The reference ``bbox_size``: max abs deviation of the bbox corners from *start*.

    ``max(|bbox_max - start|, |bbox_min - start|)`` over both axes -- i.e. the
    half-width of the smallest start-point-centred square containing the sketch.
    """
    pts = list(points)
    if not pts:
        raise ValueError("empty sketch")
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    corners = ((min(xs), min(ys)), (max(xs), max(ys)))
    return max(abs(c[i] - start_point[i]) for c in corners for i in (0, 1))


def sketch_normalize_scale(bbox_size: float, size: float = SKETCH_DIM) -> float:
    """``(size/2 * NORM_FACTOR - 1) / bbox_size`` -- the sketch normalising scale.

    The ``-1`` is the reference's overflow guard: after augmentation the profile must
    still fit inside the ``size x size`` canvas.
    """
    if bbox_size <= 0:
        raise ValueError("bbox_size must be positive")
    return (size / 2 * NORM_FACTOR - 1) / bbox_size


def normalize_sketch(points: Sequence[Sequence[float]],
                     start_point: Sequence[float] | None = None,
                     size: float = SKETCH_DIM) -> list[tuple[float, float]]:
    """Map a sketch into the ``size x size`` canvas, start point at the centre.

    ``p -> (p - start) * scale + (size/2, size/2)`` with ``scale`` from
    :func:`sketch_normalize_scale`. ``start_point`` defaults to ``points[0]``.
    """
    if not points:
        raise ValueError("empty sketch")
    start = tuple(start_point) if start_point is not None else tuple(points[0])
    scale = sketch_normalize_scale(sketch_bbox_size(points, start), size)
    half = size / 2
    return [((p[0] - start[0]) * scale + half, (p[1] - start[1]) * scale + half)
            for p in points]


def denormalize_sketch(points: Sequence[Sequence[float]], bbox_size: float,
                       size: float = SKETCH_DIM) -> list[tuple[float, float]]:
    """Inverse of :func:`normalize_sketch` up to the start-point translation.

    ``p -> (p - (size/2, size/2)) * bbox_size / (size/2 * NORM_FACTOR - 1)``, giving
    sketch-local coordinates relative to the (now origin-placed) start point.
    """
    scale = bbox_size / (size / 2 * NORM_FACTOR - 1)
    half = size / 2
    return [((p[0] - half) * scale, (p[1] - half) * scale) for p in points]


# --- the extrude parameter block -------------------------------------------
#: Order of the 11 extrusion parameters in the command vector.
EXT_PARAM_NAMES: tuple[str, ...] = (
    "theta", "phi", "gamma",     # sketch-plane orientation (angles)
    "px", "py", "pz",            # sketch-plane origin (unit-cube coords)
    "s",                         # sketch size
    "e1", "e2",                  # extrude extents (unit-cube coords)
    "b", "u",                    # boolean op, extent type (categorical: kept as-is)
)


def numericalize_extrude(params: dict, n: int = ARGS_DIM) -> dict:
    """Quantise a full extrude parameter block, each field with its own family.

    ``e1``/``e2`` must lie in ``[-2, 2]`` (the reference asserts this after
    :func:`normalize_shape`); ``b``/``u`` are categorical indices, passed through.
    """
    for key in ("e1", "e2"):
        if not -2.0 <= params[key] <= 2.0:
            raise ValueError(f"{key} out of range: {params[key]}")
    out = {}
    for name in EXT_PARAM_NAMES:
        value = params[name]
        if name in ("theta", "phi", "gamma"):
            out[name] = numericalize_angle(value, n)
        elif name in ("px", "py", "pz", "e1", "e2"):
            out[name] = numericalize_unit(value, n)
        elif name == "s":
            out[name] = numericalize_size(value, n)
        else:  # b, u
            out[name] = int(value)
    return out


def denumericalize_extrude(params: dict, n: int = ARGS_DIM) -> dict:
    """Inverse of :func:`numericalize_extrude`."""
    out = {}
    for name in EXT_PARAM_NAMES:
        level = params[name]
        if name in ("theta", "phi", "gamma"):
            out[name] = denumericalize_angle(level, n)
        elif name in ("px", "py", "pz", "e1", "e2"):
            out[name] = denumericalize_unit(level, n)
        elif name == "s":
            out[name] = denumericalize_size(level, n)
        else:  # b, u
            out[name] = int(level)
    return out
