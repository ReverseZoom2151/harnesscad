"""Primitive height patterns for SHAPE-IT pin-grid displays.

SHAPE-IT's formative study identifies *Primitive* -- "the basic geometry that
constitutes target shape display behavior ... basic shapes, geometries, or
movement patterns" -- as the foundational generative element (Section 3.2).
The paper repeatedly names the *numerically describable* primitives its
code-generation approach handles well: "wave, square, cone" (Section 9.2), plus
"circles, squares, triangles" and gradient/ramp layouts.

This module renders those primitives deterministically onto a
:class:`geometry.shapeit_heightfield.HeightField`.  Each drawer takes an
existing field and *stamps* a pattern into it (raising pins), returning the
same field for chaining.  All coordinates are pin ``(row, col)`` indices; all
heights are clamped by the field's stroke range.  Stdlib-only, no randomness.

Drawers
-------
``draw_rectangle``   filled axis-aligned rectangle at a constant height.
``draw_disc``        filled circle (Euclidean radius) at a constant height.
``draw_line``        Bresenham pin line at a constant height.
``draw_linear_gradient``  ramp of heights along a direction (a "basic layout").
``draw_cone``        radial cone/pyramid peaking at an apex.
``draw_wave``        sinusoidal ripple pattern (the paper's "wave" primitive).
"""

from __future__ import annotations

from math import cos, hypot, pi
from typing import Tuple

from geometry.shapeit_heightfield import HeightField

TAU = 2.0 * pi


def _stamp(hf: HeightField, row: int, col: int, height: float, additive: bool) -> None:
    if not hf.in_bounds(row, col):
        return
    if additive:
        hf.add(row, col, height)
    else:
        hf.set(row, col, height)


def draw_rectangle(
    hf: HeightField,
    top: int,
    left: int,
    height_rows: int,
    width_cols: int,
    value: float,
    additive: bool = False,
) -> HeightField:
    """Stamp a filled axis-aligned rectangle of pins at ``value``.

    ``(top, left)`` is the top-left pin; the rectangle spans ``height_rows`` x
    ``width_cols`` pins.  Cells outside the grid are silently skipped.
    """
    if height_rows < 0 or width_cols < 0:
        raise ValueError("rectangle extents must be >= 0")
    for r in range(top, top + height_rows):
        for c in range(left, left + width_cols):
            _stamp(hf, r, c, value, additive)
    return hf


def draw_disc(
    hf: HeightField,
    center_row: float,
    center_col: float,
    radius: float,
    value: float,
    additive: bool = False,
) -> HeightField:
    """Stamp a filled circle (all pins within ``radius`` of the centre)."""
    if radius < 0.0:
        raise ValueError("radius must be >= 0")
    r_lo = int(center_row - radius)
    r_hi = int(center_row + radius) + 1
    c_lo = int(center_col - radius)
    c_hi = int(center_col + radius) + 1
    for r in range(r_lo, r_hi + 1):
        for c in range(c_lo, c_hi + 1):
            if hypot(r - center_row, c - center_col) <= radius:
                _stamp(hf, r, c, value, additive)
    return hf


def draw_line(
    hf: HeightField,
    row0: int,
    col0: int,
    row1: int,
    col1: int,
    value: float,
    additive: bool = False,
) -> HeightField:
    """Stamp a straight pin line via integer Bresenham rasterisation."""
    r0, c0, r1, c1 = int(row0), int(col0), int(row1), int(col1)
    dr = abs(r1 - r0)
    dc = abs(c1 - c0)
    sr = 1 if r0 < r1 else -1
    sc = 1 if c0 < c1 else -1
    err = dc - dr
    r, c = r0, c0
    while True:
        _stamp(hf, r, c, value, additive)
        if r == r1 and c == c1:
            break
        e2 = 2 * err
        if e2 > -dr:
            err -= dr
            c += sc
        if e2 < dc:
            err += dc
            r += sr
    return hf


def draw_linear_gradient(
    hf: HeightField,
    low_value: float,
    high_value: float,
    axis: str = "row",
) -> HeightField:
    """Fill the whole field with a linear ramp of heights.

    ``axis='row'`` ramps top-to-bottom, ``axis='col'`` left-to-right.  Row 0 /
    col 0 gets ``low_value`` and the last row / col gets ``high_value``.
    """
    if axis not in ("row", "col"):
        raise ValueError("axis must be 'row' or 'col'")
    span = (hf.rows - 1) if axis == "row" else (hf.cols - 1)
    for r in range(hf.rows):
        for c in range(hf.cols):
            idx = r if axis == "row" else c
            t = 0.0 if span == 0 else idx / span
            hf.set(r, c, low_value + t * (high_value - low_value))
    return hf


def draw_cone(
    hf: HeightField,
    apex_row: float,
    apex_col: float,
    radius: float,
    peak_value: float,
    base_value: float = 0.0,
    additive: bool = False,
) -> HeightField:
    """Stamp a radial cone peaking at the apex and falling to ``base_value``
    at ``radius`` (linear in distance).  Pins past ``radius`` are untouched.
    """
    if radius <= 0.0:
        raise ValueError("radius must be > 0")
    r_lo = int(apex_row - radius)
    r_hi = int(apex_row + radius) + 1
    c_lo = int(apex_col - radius)
    c_hi = int(apex_col + radius) + 1
    for r in range(r_lo, r_hi + 1):
        for c in range(c_lo, c_hi + 1):
            d = hypot(r - apex_row, c - apex_col)
            if d <= radius:
                t = d / radius
                _stamp(hf, r, c, peak_value + t * (base_value - peak_value), additive)
    return hf


def draw_wave(
    hf: HeightField,
    amplitude: float,
    wavelength: float,
    axis: str = "col",
    phase: float = 0.0,
    offset: float = 0.0,
    additive: bool = False,
) -> HeightField:
    """Stamp a sinusoidal ripple: ``offset + amplitude * cos(2*pi*x/lambda +
    phase)`` where ``x`` runs along ``axis`` ('row' or 'col').

    This is the paper's canonical "wave" primitive/animation base.  With
    ``additive=True`` it superimposes onto an existing field.
    """
    if wavelength <= 0.0:
        raise ValueError("wavelength must be > 0")
    if axis not in ("row", "col"):
        raise ValueError("axis must be 'row' or 'col'")
    for r in range(hf.rows):
        for c in range(hf.cols):
            x = r if axis == "row" else c
            value = offset + amplitude * cos(TAU * x / wavelength + phase)
            _stamp(hf, r, c, value, additive)
    return hf
