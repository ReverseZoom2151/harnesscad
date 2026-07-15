"""Parametric roof generation and structural-consistency metrics (ShellMaker).

Mined from *ShellMaker: Language-Guided Exterior Completion under Structural
Constraints*. ShellMaker's stylisation uses trained 3D generators, but two of its
pillars are deterministic and stdlib-portable:

*   **parametric roof generation** subject to the fixed footprint -- a gable roof is
    a ridge line lifted above the footprint's bounding box at a given pitch; and
*   **structural-consistency metrics** -- the paper's quantitative measures of
    *footprint violation* (does the generated exterior stay within the immutable
    footprint?) and *opening preservation* (do window/door openings remain at their
    prescribed positions?).

This module ports those. Footprints are axis-aligned bounding boxes over 2D
polygons for simplicity; openings are 2D points on the wall boundary. Deterministic,
stdlib-only.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import List, Sequence, Tuple

__all__ = [
    "bounding_box",
    "GableRoof",
    "generate_gable_roof",
    "footprint_violation",
    "opening_preservation",
]

Pt2 = Tuple[float, float]
Pt3 = Tuple[float, float, float]


def bounding_box(polygon: Sequence[Pt2]) -> Tuple[Pt2, Pt2]:
    """Axis-aligned ``(min, max)`` corners of a 2D footprint polygon."""
    if not polygon:
        raise ValueError("polygon must be non-empty")
    xs = [p[0] for p in polygon]
    ys = [p[1] for p in polygon]
    return (min(xs), min(ys)), (max(xs), max(ys))


@dataclass(frozen=True)
class GableRoof:
    """A gable roof: two ridge endpoints and the four eave corners."""

    ridge: Tuple[Pt3, Pt3]
    eaves: Tuple[Pt3, Pt3, Pt3, Pt3]
    height: float


def generate_gable_roof(
    footprint: Sequence[Pt2], wall_height: float, pitch_deg: float
) -> GableRoof:
    """Generate a gable roof over ``footprint`` at eave level ``wall_height``.

    The ridge runs along the footprint's longer axis at its centre; the roof rises
    by ``(half-span) * tan(pitch)`` above the eaves. The footprint is respected
    exactly -- eaves sit on the footprint bounding box.
    """
    if wall_height < 0:
        raise ValueError("wall_height must be non-negative")
    if not 0 < pitch_deg < 90:
        raise ValueError("pitch_deg must be in (0, 90)")
    (x0, y0), (x1, y1) = bounding_box(footprint)
    span_x = x1 - x0
    span_y = y1 - y0
    pitch = math.radians(pitch_deg)
    if span_x >= span_y:
        # ridge along x at mid-y
        half = span_y / 2.0
        rise = half * math.tan(pitch)
        mid_y = (y0 + y1) / 2.0
        ridge = ((x0, mid_y, wall_height + rise), (x1, mid_y, wall_height + rise))
    else:
        half = span_x / 2.0
        rise = half * math.tan(pitch)
        mid_x = (x0 + x1) / 2.0
        ridge = ((mid_x, y0, wall_height + rise), (mid_x, y1, wall_height + rise))
    eaves = (
        (x0, y0, wall_height), (x1, y0, wall_height),
        (x1, y1, wall_height), (x0, y1, wall_height),
    )
    return GableRoof(ridge=ridge, eaves=eaves, height=rise)


def footprint_violation(
    generated_points: Sequence[Pt2], footprint: Sequence[Pt2], tol: float = 1e-6
) -> float:
    """Fraction of generated exterior points that fall OUTSIDE the footprint box.

    0.0 means the exterior stays perfectly within the immutable footprint.
    """
    if not generated_points:
        raise ValueError("need at least one generated point")
    (x0, y0), (x1, y1) = bounding_box(footprint)
    outside = 0
    for x, y in generated_points:
        if x < x0 - tol or x > x1 + tol or y < y0 - tol or y > y1 + tol:
            outside += 1
    return outside / len(generated_points)


def opening_preservation(
    generated_openings: Sequence[Pt2],
    required_openings: Sequence[Pt2],
    tol: float,
) -> float:
    """Fraction of required openings matched by a generated opening within ``tol``.

    Each required opening is matched to at most one generated opening (greedy by
    index). 1.0 means every prescribed window/door is preserved in place.
    """
    if not required_openings:
        raise ValueError("need at least one required opening")
    if tol < 0:
        raise ValueError("tol must be non-negative")
    used = [False] * len(generated_openings)
    matched = 0
    for rx, ry in required_openings:
        for j, (gx, gy) in enumerate(generated_openings):
            if used[j]:
                continue
            if math.hypot(gx - rx, gy - ry) <= tol:
                used[j] = True
                matched += 1
                break
    return matched / len(required_openings)
