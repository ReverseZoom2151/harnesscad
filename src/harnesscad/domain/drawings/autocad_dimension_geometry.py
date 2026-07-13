"""autocad_dimension_geometry -- deterministic 2D dimension measurement + placement.

Transferred from the ``AutoCAD.py`` COM library (manufino/AutoCAD), whose
``add_dimension`` / ``add_overall_dimensions`` helpers delegate the actual
numbers to AutoCAD's Automation API (``AddDimAligned``, ``AddDimRotated``,
``AddDimAngular``, ``AddDimRadial``, ``AddDimDiametric``). That host is
Windows/COM-only, but the *geometry* each of those dimension types encodes is a
small, exact computation:

  * **aligned**   -- true distance between two feature points, dimension line
    drawn parallel to them at an offset;
  * **rotated/linear** -- distance between the two points *projected onto* a
    direction at a given angle (angle 0 measures the horizontal extent);
  * **angular**   -- angle at a vertex between the two rays to the points;
  * **radial**    -- radius from a centre to a point on a circle;
  * **diametric** -- diameter across two chord-endpoints;
  * **overall bbox** -- horizontal + vertical size dimensions of an entity's
    axis-aligned extents.

This module recomputes those values and returns the full draftable geometry
(dimension line, extension lines, arrow anchor points, text anchor) without any
CAD host. Stdlib-only, deterministic, no wall clock.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

Point = Tuple[float, float]
Segment = Tuple[float, float, float, float]  # (x1, y1, x2, y2)


def _sub(a: Point, b: Point) -> Point:
    return (a[0] - b[0], a[1] - b[1])


def _add(a: Point, b: Point) -> Point:
    return (a[0] + b[0], a[1] + b[1])


def _scale(a: Point, s: float) -> Point:
    return (a[0] * s, a[1] * s)


def _norm(a: Point) -> float:
    return math.hypot(a[0], a[1])


def _unit(a: Point) -> Point:
    n = _norm(a)
    if n == 0.0:
        return (0.0, 0.0)
    return (a[0] / n, a[1] / n)


def _perp(a: Point) -> Point:
    """90-degree counter-clockwise rotation."""
    return (-a[1], a[0])


def _signed_offset(p1: Point, p2: Point, ref: Point) -> float:
    """Signed perpendicular distance of ``ref`` from the line p1->p2."""
    d = _unit(_sub(p2, p1))
    n = _perp(d)
    return _sub(ref, p1)[0] * n[0] + _sub(ref, p1)[1] * n[1]


@dataclass(frozen=True)
class DimensionGeometry:
    """A fully placed linear/aligned dimension."""

    measured: float
    dimension_line: Segment
    extension_a: Segment
    extension_b: Segment
    text_anchor: Point
    arrow_a: Point
    arrow_b: Point

    def to_dict(self) -> dict:
        return {
            "measured": self.measured,
            "dimension_line": self.dimension_line,
            "extension_a": self.extension_a,
            "extension_b": self.extension_b,
            "text_anchor": self.text_anchor,
            "arrow_a": self.arrow_a,
            "arrow_b": self.arrow_b,
        }


def aligned_dimension(p1: Point, p2: Point, offset: float = 10.0) -> DimensionGeometry:
    """Aligned dimension: measures the true distance |p2 - p1|.

    The dimension line is parallel to ``p1->p2`` and displaced by ``offset``
    along the counter-clockwise perpendicular (negative offset places it on the
    other side).
    """
    d = _unit(_sub(p2, p1))
    if d == (0.0, 0.0):
        raise ValueError("aligned_dimension needs two distinct points")
    n = _perp(d)
    disp = _scale(n, offset)
    a2 = _add(p1, disp)
    b2 = _add(p2, disp)
    measured = _norm(_sub(p2, p1))
    text = _scale(_add(a2, b2), 0.5)
    return DimensionGeometry(
        measured=measured,
        dimension_line=(a2[0], a2[1], b2[0], b2[1]),
        extension_a=(p1[0], p1[1], a2[0], a2[1]),
        extension_b=(p2[0], p2[1], b2[0], b2[1]),
        text_anchor=text,
        arrow_a=a2,
        arrow_b=b2,
    )


def rotated_dimension(p1: Point, p2: Point, angle: float,
                      offset: float = 10.0) -> DimensionGeometry:
    """Rotated (linear) dimension at ``angle`` radians.

    Measures the distance between the two points projected onto the direction
    ``(cos angle, sin angle)``. Angle 0 gives the horizontal extent ``|dx|``,
    angle pi/2 the vertical extent ``|dy|`` -- matching AutoCAD's
    ``AddDimRotated``. The dimension line runs along the measure direction; the
    extension lines drop from each point to it.
    """
    u = (math.cos(angle), math.sin(angle))
    n = _perp(u)
    # Projected coordinates of each point along the measure axis.
    t1 = p1[0] * u[0] + p1[1] * u[1]
    t2 = p2[0] * u[0] + p2[1] * u[1]
    measured = abs(t2 - t1)
    # Base offset line: displace along n by the further of the two points plus
    # the requested offset so the dimension line clears the geometry.
    s1 = p1[0] * n[0] + p1[1] * n[1]
    s2 = p2[0] * n[0] + p2[1] * n[1]
    base_s = max(s1, s2) + offset
    a2 = (u[0] * t1 + n[0] * base_s, u[1] * t1 + n[1] * base_s)
    b2 = (u[0] * t2 + n[0] * base_s, u[1] * t2 + n[1] * base_s)
    text = _scale(_add(a2, b2), 0.5)
    return DimensionGeometry(
        measured=measured,
        dimension_line=(a2[0], a2[1], b2[0], b2[1]),
        extension_a=(p1[0], p1[1], a2[0], a2[1]),
        extension_b=(p2[0], p2[1], b2[0], b2[1]),
        text_anchor=text,
        arrow_a=a2,
        arrow_b=b2,
    )


def angular_dimension(vertex: Point, p1: Point, p2: Point) -> float:
    """Return the unsigned angle (radians, in [0, pi]) at ``vertex``.

    Matches ``AddDimAngular(vertex, p1, p2, ...)``: the angle between rays
    ``vertex->p1`` and ``vertex->p2``.
    """
    a = _sub(p1, vertex)
    b = _sub(p2, vertex)
    na, nb = _norm(a), _norm(b)
    if na == 0.0 or nb == 0.0:
        raise ValueError("angular_dimension needs points distinct from vertex")
    dot = (a[0] * b[0] + a[1] * b[1]) / (na * nb)
    dot = max(-1.0, min(1.0, dot))
    return math.acos(dot)


def radial_dimension(center: Point, point_on_circle: Point) -> float:
    """Radius = distance from ``center`` to a point on the circle."""
    return _norm(_sub(point_on_circle, center))


def diametric_dimension(p1: Point, p2: Point) -> float:
    """Diameter = distance across the two chord endpoints (a full diameter)."""
    return _norm(_sub(p2, p1))


def bounding_box(points: Sequence[Point]) -> Tuple[float, float, float, float]:
    """Axis-aligned extents ``(minx, miny, maxx, maxy)`` of ``points``."""
    if not points:
        raise ValueError("bounding_box needs at least one point")
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    return (min(xs), min(ys), max(xs), max(ys))


def overall_dimensions(points: Sequence[Point],
                       offset: float = 5.0
                       ) -> Tuple[DimensionGeometry, DimensionGeometry]:
    """Horizontal (width) and vertical (height) dimensions of the bbox.

    Mirrors ``add_overall_dimensions``: a width dimension below the box and a
    height dimension to its left.
    """
    minx, miny, maxx, maxy = bounding_box(points)
    width = aligned_dimension((minx, miny), (maxx, miny), offset=-offset)
    height = aligned_dimension((minx, maxy), (minx, miny), offset=-offset)
    return width, height
