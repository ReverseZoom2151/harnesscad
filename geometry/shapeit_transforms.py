"""Spatial and amplitude transforms for SHAPE-IT height patterns.

SHAPE-IT's *Animation* element (Section 3.3) is defined as "enhancements to
primitives, altering geometry parameters to achieve dynamic and continuous
motion" -- basic motion (translations: rising, falling), transformations
(morphing), and pulsing.  Its worked example (Section 5.1, Step 2) enumerates
the controllable parameters of a shape as ``positionX``, ``positionY``,
``scale``, ``height``, and later ``rotation``.

This module implements exactly those deterministic operators on a
:class:`geometry.shapeit_heightfield.HeightField`.  Each returns a *new* field
of the same resolution and stroke range (the source is never mutated).  Pins
that map outside the source read the field's floor (``min_height``).  Sampling
is nearest-neighbour so results are exact and reproducible; no randomness.

Operators
---------
``translate``          integer/fractional pin shift (positionX / positionY).
``scale``              resample about a centre (uniform or per-axis scale).
``rotate``             rotate the pattern about a centre by an angle (radians).
``rotate90``          exact k*90-degree rotation (no interpolation).
``scale_amplitude``    scale heights about the floor (pulsing / height gain).
``vertical_offset``    raise/lower every pin by a constant (rising / falling).
"""

from __future__ import annotations

from math import cos, sin
from typing import Optional

from geometry.shapeit_heightfield import HeightField


def _like(src: HeightField) -> HeightField:
    """A fresh floor-filled field matching ``src``'s resolution and stroke."""
    return HeightField(src.rows, src.cols, src.min_height, src.max_height)


def _sample_nn(src: HeightField, row: float, col: float) -> float:
    """Nearest-neighbour sample; out-of-bounds reads the floor."""
    r = int(round(row))
    c = int(round(col))
    if src.in_bounds(r, c):
        return src.get(r, c)
    return src.min_height


def translate(src: HeightField, d_row: float, d_col: float) -> HeightField:
    """Shift the pattern by ``(d_row, d_col)`` pins (positionY / positionX)."""
    out = _like(src)
    for r in range(src.rows):
        for c in range(src.cols):
            out.set(r, c, _sample_nn(src, r - d_row, c - d_col))
    return out


def scale(
    src: HeightField,
    factor_row: float,
    factor_col: Optional[float] = None,
    center_row: Optional[float] = None,
    center_col: Optional[float] = None,
) -> HeightField:
    """Resample the pattern scaled about a centre.

    ``factor > 1`` enlarges, ``0 < factor < 1`` shrinks.  ``factor_col``
    defaults to ``factor_row`` (uniform scale).  Centre defaults to the grid
    middle.  Scale factors must be positive.
    """
    fc = factor_row if factor_col is None else factor_col
    if factor_row <= 0.0 or fc <= 0.0:
        raise ValueError("scale factors must be > 0")
    cr = (src.rows - 1) / 2.0 if center_row is None else center_row
    cc = (src.cols - 1) / 2.0 if center_col is None else center_col
    out = _like(src)
    for r in range(src.rows):
        for c in range(src.cols):
            sr = cr + (r - cr) / factor_row
            sc = cc + (c - cc) / fc
            out.set(r, c, _sample_nn(src, sr, sc))
    return out


def rotate(
    src: HeightField,
    angle: float,
    center_row: Optional[float] = None,
    center_col: Optional[float] = None,
) -> HeightField:
    """Rotate the pattern by ``angle`` radians (counter-clockwise) about a
    centre, sampling nearest-neighbour.  Centre defaults to the grid middle.
    """
    cr = (src.rows - 1) / 2.0 if center_row is None else center_row
    cc = (src.cols - 1) / 2.0 if center_col is None else center_col
    ca = cos(angle)
    sa = sin(angle)
    out = _like(src)
    for r in range(src.rows):
        for c in range(src.cols):
            dr = r - cr
            dc = c - cc
            # inverse rotation to find the source pin
            sr = cr + ca * dr + sa * dc
            sc = cc - sa * dr + ca * dc
            out.set(r, c, _sample_nn(src, sr, sc))
    return out


def rotate90(src: HeightField, k: int = 1) -> HeightField:
    """Exact rotation by ``k`` multiples of 90 degrees (counter-clockwise),
    with no interpolation.  ``k`` may be negative.
    """
    k = k % 4
    rows = src.to_rows()
    if k == 0:
        result = rows
    elif k == 1:
        # CCW: columns (right to left) become rows
        result = [
            [rows[r][c] for r in range(src.rows)]
            for c in range(src.cols - 1, -1, -1)
        ]
    elif k == 2:
        result = [list(reversed(rows[r])) for r in range(src.rows - 1, -1, -1)]
    else:  # k == 3, CW
        result = [
            [rows[r][c] for r in range(src.rows - 1, -1, -1)]
            for c in range(src.cols)
        ]
    return HeightField.from_rows(result, src.min_height, src.max_height)


def scale_amplitude(src: HeightField, gain: float) -> HeightField:
    """Scale every pin's height about the floor by ``gain`` (pulsing / gain).

    ``h -> floor + gain * (h - floor)``.  ``gain > 1`` amplifies the relief,
    ``0 <= gain < 1`` flattens it toward the floor.  Results are clamped.
    """
    if gain < 0.0:
        raise ValueError("gain must be >= 0")
    floor = src.min_height
    out = _like(src)
    for r in range(src.rows):
        for c in range(src.cols):
            out.set(r, c, floor + gain * (src.get(r, c) - floor))
    return out


def vertical_offset(src: HeightField, delta: float) -> HeightField:
    """Raise (``delta > 0``) or lower every pin by a constant, clamped."""
    out = _like(src)
    for r in range(src.rows):
        for c in range(src.cols):
            out.set(r, c, src.get(r, c) + delta)
    return out
