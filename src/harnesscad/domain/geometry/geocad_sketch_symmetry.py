"""Sketch-level symmetric editing for GeoCAD (Zhang et al. 2025, appendix H).

Although GeoCAD generates one *loop* at a time, it bootstraps *sketch-level* editing
from loop-level editing via a deterministic symmetry procedure (appendix H, Fig. A1):

    "1) Estimate the center point of each original loop by averaging its coordinate
        points, which are extracted using string matching.
     2) Determine the symmetry axis based on the two center points.
     3) Generate a new local loop through GeoCAD replacing one of the original loops.
     4) Reflect the newly generated loop across the symmetry axis to produce the
        second symmetric loop, thereby replacing both original loops."

This module implements the deterministic geometry of steps 1, 2 and 4 (the loop
generation in step 3 is the LLM and is supplied by the caller). Given two symmetric
loops, it recovers their symmetry axis; given a newly generated loop for one side, it
reflects it to synthesise the partner loop. Pure computational geometry.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

Point = tuple[float, float]


def centroid(points: list[Point]) -> Point:
    """Center point of a loop = the average of its coordinate points (step 1)."""
    if not points:
        raise ValueError("empty loop")
    n = len(points)
    return (sum(p[0] for p in points) / n, sum(p[1] for p in points) / n)


@dataclass(frozen=True)
class SymmetryAxis:
    """Perpendicular-bisector symmetry axis between two loop centers (step 2).

    Represented as a point on the axis (the midpoint of the two centers) and a unit
    direction vector along the axis.
    """

    point: Point
    direction: Point  # unit vector


def symmetry_axis(center_a: Point, center_b: Point) -> SymmetryAxis:
    """Axis of mirror symmetry between two loops = perpendicular bisector of centers."""
    mx = (center_a[0] + center_b[0]) / 2.0
    my = (center_a[1] + center_b[1]) / 2.0
    # Vector joining the centers; the axis is perpendicular to it.
    jx, jy = center_b[0] - center_a[0], center_b[1] - center_a[1]
    if jx == 0.0 and jy == 0.0:
        raise ValueError("coincident loop centers: symmetry axis undefined")
    # Perpendicular direction, normalised.
    dx, dy = -jy, jx
    norm = math.hypot(dx, dy)
    return SymmetryAxis((mx, my), (dx / norm, dy / norm))


def reflect_point(p: Point, axis: SymmetryAxis) -> Point:
    """Mirror a point across the symmetry axis (step 4)."""
    ax, ay = axis.point
    dx, dy = axis.direction
    # Project (p - point) onto the axis direction, then reflect the normal part.
    px, py = p[0] - ax, p[1] - ay
    t = px * dx + py * dy          # component along the axis
    proj = (t * dx, t * dy)        # parallel component
    # Reflected point = axis.point + parallel - perpendicular
    perp = (px - proj[0], py - proj[1])
    return (ax + proj[0] - perp[0], ay + proj[1] - perp[1])


def reflect_loop(points: list[Point], axis: SymmetryAxis) -> list[Point]:
    """Reflect an entire loop across the symmetry axis (produces the partner loop)."""
    return [reflect_point(p, axis) for p in points]


def synthesise_symmetric_pair(new_loop: list[Point],
                              axis: SymmetryAxis) -> tuple[list[Point], list[Point]]:
    """From a newly generated loop and the axis, return both symmetric loops (step 4)."""
    return list(new_loop), reflect_loop(new_loop, axis)
