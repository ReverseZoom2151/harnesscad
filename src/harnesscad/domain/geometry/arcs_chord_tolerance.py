"""Sagitta / chord-tolerance arc tessellation (from the ``arcs`` Rust CAD core).

``arcs-core/src/algorithms/approximate.rs`` derives the number of line segments
needed to approximate an arc to within a given *tolerance* (the maximum
deviation between the chord and the arc, i.e. the **sagitta**):

    draw a chord between two points A, B on a circle of centre C and radius R;
    bisect the angle ACB, hitting the chord at D.  From triangle DCB

        cos(theta / 2) = |CD| / R = 1 - sagitta / R
        theta          = 2 * acos(1 - tolerance / R)
        N              = ceil(sweep / theta)          (at least 2 segments)

and then walks the arc in ``sweep / N`` steps, emitting ``N + 1`` points so the
exact start and end points are always present.

The harness already tessellates arcs with a *fixed* sample count
(``geometry.gencad2_arc_vector.sample_arc_points(n=32)``); what it lacked is the
tolerance-driven direction -- "how many segments do I need for a 0.01 mm chord
error", the question every mesher/DXF-flattener/GCode post actually asks -- plus
the inverse (error attained by N segments) and the degenerate guards
(``tolerance <= 0`` or ``radius <= tolerance`` collapse to a single chord).

Pure standard library, deterministic. Points are ``(x, y)`` float tuples.
"""

from __future__ import annotations

import math
from typing import List, Tuple

from harnesscad.domain.geometry.arcs_closest_point import Arc2D

Point = Tuple[float, float]

TWO_PI = 2.0 * math.pi

__all__ = [
    "TWO_PI",
    "approximate_arc",
    "approximate_circle",
    "chord_error",
    "chord_length",
    "sagitta",
    "segment_angle_for_tolerance",
    "segments_for_tolerance",
]


def sagitta(radius: float, theta: float) -> float:
    """Chord-to-arc deviation of a segment subtending ``theta`` radians."""
    if radius <= 0.0:
        raise ValueError("radius must be positive")
    return radius * (1.0 - math.cos(abs(theta) / 2.0))


def chord_length(radius: float, theta: float) -> float:
    """Length of the chord subtending ``theta`` radians."""
    if radius <= 0.0:
        raise ValueError("radius must be positive")
    return 2.0 * radius * math.sin(abs(theta) / 2.0)


def segment_angle_for_tolerance(radius: float, tolerance: float) -> float:
    """Largest segment angle whose sagitta stays within ``tolerance``."""
    if radius <= 0.0:
        raise ValueError("radius must be positive")
    if tolerance <= 0.0 or radius <= tolerance:
        return TWO_PI
    return 2.0 * math.acos(1.0 - tolerance / radius)


def segments_for_tolerance(
    radius: float, sweep_angle: float, tolerance: float
) -> int:
    """Number of straight segments needed to hit ``tolerance`` (>= 1)."""
    if radius <= 0.0:
        raise ValueError("radius must be positive")
    if tolerance <= 0.0 or radius <= tolerance:
        return 1
    sweep = abs(sweep_angle)
    if sweep == 0.0:
        return 1

    theta = segment_angle_for_tolerance(radius, tolerance)
    count = sweep / theta
    # arcs: `let line_segment_count = f64::max(line_segment_count, 2.0);`
    return int(math.ceil(max(count, 2.0)))


def chord_error(radius: float, sweep_angle: float, segments: int) -> float:
    """Worst chord-to-arc deviation when the arc is split into ``segments``."""
    if segments < 1:
        raise ValueError("segments must be >= 1")
    return sagitta(radius, abs(sweep_angle) / segments)


def approximate_arc(arc: Arc2D, tolerance: float) -> List[Point]:
    """Polyline approximating ``arc`` within ``tolerance`` of chord error.

    The first and last points are exactly ``arc.start()`` and ``arc.end()``.
    """
    if arc.radius <= 0.0:
        raise ValueError("radius must be positive")

    if tolerance <= 0.0 or arc.radius <= tolerance:
        steps = 1
    else:
        steps = segments_for_tolerance(arc.radius, arc.sweep_angle, tolerance)

    step = arc.sweep_angle / steps
    return [arc.point_at(i * step) for i in range(steps + 1)]


def approximate_circle(
    centre: Point, radius: float, tolerance: float
) -> List[Point]:
    """Closed polygon approximating a full circle within ``tolerance``.

    The returned ring is *not* duplicated: the last point is distinct from the
    first (callers close the loop themselves).
    """
    arc = Arc2D(centre, radius, 0.0, TWO_PI)
    points = approximate_arc(arc, tolerance)
    return points[:-1]
