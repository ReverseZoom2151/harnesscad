"""autocad_linetype_dash -- apply dashed linetype patterns to polylines.

The ``AutoCAD.py`` ``LineStyle`` enum names the standard AutoCAD linetypes
(CONTINUOUS, DASHED, DOTTED, CENTER, HIDDEN, PHANTOM, DASHDOT, BORDER, DIVIDE,
...) but leaves the actual dash geometry to the CAD host. That geometry is a
classic deterministic algorithm: a linetype is a repeating sequence of signed
lengths -- a positive length is a drawn *dash*, a negative length is a *gap*,
and a zero length is a *dot* -- and rendering it means marching that pattern
along the arc length of a polyline and emitting the visible sub-segments.

This module implements that pattern-application algorithm plus the canonical
element definitions for the named AutoCAD linetypes (the ``acad.lin`` metric
family), so a continuous polyline can be turned into the list of short segments
(and dot points) a renderer would actually stroke. Stdlib-only, deterministic.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Sequence, Tuple

Point = Tuple[float, float]
Segment = Tuple[float, float, float, float]

# Canonical element sequences (metric-ish, unit scale). Positive = dash,
# negative = gap, 0.0 = dot. Values follow the standard acad.lin proportions.
NAMED_PATTERNS: Dict[str, List[float]] = {
    "CONTINUOUS": [],  # empty == solid, no dashing
    "DASHED": [0.5, -0.25],
    "DOTTED": [0.0, -0.25],
    "DOT2": [0.0, -0.125],
    "DOTX2": [0.0, -0.5],
    "HIDDEN": [0.25, -0.125],
    "CENTER": [1.25, -0.25, 0.25, -0.25],
    "PHANTOM": [1.25, -0.25, 0.25, -0.25, 0.25, -0.25],
    "DASHDOT": [0.5, -0.25, 0.0, -0.25],
    "BORDER": [0.5, -0.25, 0.5, -0.25, 0.0, -0.25],
    "DIVIDE": [0.5, -0.25, 0.0, -0.25, 0.0, -0.25],
    "TRACKING": [-0.25, 0.0],
}


@dataclass(frozen=True)
class StrokedLine:
    """Result of applying a linetype: visible dash segments and dot points."""

    segments: List[Segment] = field(default_factory=list)
    dots: List[Point] = field(default_factory=list)


def pattern_length(pattern: Sequence[float]) -> float:
    """Total (absolute) length of one repetition of ``pattern``."""
    return sum(abs(x) for x in pattern)


def _polyline_length(points: Sequence[Point]) -> float:
    total = 0.0
    for i in range(1, len(points)):
        total += math.hypot(points[i][0] - points[i - 1][0],
                            points[i][1] - points[i - 1][1])
    return total


def point_at_arclen(points: Sequence[Point], s: float) -> Point:
    """Point at arc length ``s`` along the polyline (clamped to its ends)."""
    if len(points) < 2:
        raise ValueError("polyline needs at least two points")
    if s <= 0.0:
        return points[0]
    acc = 0.0
    for i in range(1, len(points)):
        a, b = points[i - 1], points[i]
        seg = math.hypot(b[0] - a[0], b[1] - a[1])
        if seg == 0.0:
            continue
        if acc + seg >= s:
            t = (s - acc) / seg
            return (a[0] + t * (b[0] - a[0]), a[1] + t * (b[1] - a[1]))
        acc += seg
    return points[-1]


def apply_pattern(points: Sequence[Point], pattern: Sequence[float],
                  scale: float = 1.0) -> StrokedLine:
    """March ``pattern`` along the polyline, returning drawn dashes and dots.

    ``scale`` multiplies every element length. An empty pattern (or CONTINUOUS)
    yields the polyline's own segments unchanged.
    """
    pts = list(points)
    if len(pts) < 2:
        raise ValueError("polyline needs at least two points")
    total = _polyline_length(pts)
    pat = [x * scale for x in pattern if True]
    if not pat or pattern_length(pattern) == 0.0:
        segs = [(pts[i - 1][0], pts[i - 1][1], pts[i][0], pts[i][1])
                for i in range(1, len(pts))]
        return StrokedLine(segments=segs, dots=[])

    segments: List[Segment] = []
    dots: List[Point] = []
    s = 0.0
    k = 0
    # Guard against pathological zero-only patterns (all dots, no advance).
    if pattern_length(pat) == 0.0:
        return StrokedLine(segments=[], dots=[pts[0]])

    while s < total - 1e-12:
        elem = pat[k % len(pat)]
        k += 1
        if elem == 0.0:
            dots.append(point_at_arclen(pts, s))
            continue
        length = abs(elem)
        s_end = min(s + length, total)
        if elem > 0.0:
            segments.append(_subpath_segment(pts, s, s_end))
        s = s_end
    return StrokedLine(segments=segments, dots=dots)


def _subpath_segment(points: Sequence[Point], s0: float, s1: float) -> Segment:
    a = point_at_arclen(points, s0)
    b = point_at_arclen(points, s1)
    return (a[0], a[1], b[0], b[1])


def apply_named(points: Sequence[Point], name: str,
                scale: float = 1.0) -> StrokedLine:
    """Apply a named linetype from :data:`NAMED_PATTERNS` (case-insensitive)."""
    key = name.upper()
    if key not in NAMED_PATTERNS:
        raise ValueError(f"unknown linetype '{name}'")
    return apply_pattern(points, NAMED_PATTERNS[key], scale=scale)


def dashed_length(stroked: StrokedLine) -> float:
    """Total drawn (inked) length across all dash segments."""
    return sum(math.hypot(s[2] - s[0], s[3] - s[1]) for s in stroked.segments)
