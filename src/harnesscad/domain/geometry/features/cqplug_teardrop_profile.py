"""Teardrop (self-supporting) hole profiles for filament 3D printing.

Source rule: the ``teardrop`` plugin of the CadQuery community plugin
collection.  A horizontal cylindrical hole printed with fused-filament
fabrication has an unsupported roof: near the top of the bore the wall
tangent approaches horizontal and the material droops.  The deterministic
fix is to replace the circular cross-section with a *teardrop*: keep the
circle up to the point where its tangent reaches the maximum printable
overhang angle (45 degrees from vertical by default), then continue with
two straight flanks at exactly that angle until they meet at an apex.
Every surface of the resulting profile is therefore at or below the
overhang limit.

An optional ``clip`` truncates the apex with a horizontal chord, giving a
flat-topped ("truncated") teardrop: the small flat span is bridged by the
printer instead of being supported, which saves headroom.  Three regimes
follow from the clip height ``c`` measured from the circle centre:

* ``c is None``          -- full teardrop, apex at ``y = r*sqrt(2)``
  (for the 45 degree default; in general ``r / sin(theta)`` ... see below).
* ``yjoin < c < ymax``   -- flanks are truncated by a horizontal chord.
* ``c <= yjoin``         -- the chord cuts into the circular arc itself, so
  the characteristic 45 degree flanks disappear (``c = 0`` gives a
  half circle).

``yjoin``/``xjoin`` is the arc-line junction, at angle ``theta`` (the
overhang angle) around the circle: ``(-r*cos(theta), +r*sin(theta))`` and
its mirror.  The apex sits where the two flanks of slope ``tan(theta)``
intersect the vertical axis: ``y = yjoin + xjoin*tan(theta)``, which for
``theta = 45`` reduces to ``r*sqrt(2)``.

This module is a *printability-aware refinement* of
``geometry/cqcontrib_hole_features.py``: that module models the axial
(side-view) profile of plain / counterbored / countersunk holes and their
volumes; this one models the cross-sectional (bore) outline and the
overhang rule.  The two compose -- a teardrop bore can carry any of the
axial schedules.

Stdlib only, deterministic.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

Point = Tuple[float, float]

DEFAULT_OVERHANG_ANGLE = 45.0


class TeardropError(ValueError):
    """Raised for teardrop parameters that cannot produce a valid profile."""


def _check(radius: float, overhang_angle: float) -> None:
    if radius <= 0.0:
        raise TeardropError("radius must be positive")
    if not 0.0 < overhang_angle < 90.0:
        raise TeardropError("overhang_angle must lie in (0, 90) degrees")


def apex_height(radius: float,
                overhang_angle: float = DEFAULT_OVERHANG_ANGLE) -> float:
    """Height of the teardrop vertex above the bore centre."""
    _check(radius, overhang_angle)
    t = math.radians(overhang_angle)
    return radius * math.sin(t) + radius * math.cos(t) * math.tan(t)


def junction_point(radius: float,
                   overhang_angle: float = DEFAULT_OVERHANG_ANGLE) -> Point:
    """Right-hand arc/flank junction ``(x, y)``; the left one is its mirror."""
    _check(radius, overhang_angle)
    t = math.radians(overhang_angle)
    return (radius * math.cos(t), radius * math.sin(t))


def clip_bounds(radius: float,
                overhang_angle: float = DEFAULT_OVERHANG_ANGLE
                ) -> Tuple[float, float]:
    """Exclusive ``(min, max)`` bounds for the ``clip`` argument."""
    return (-radius, apex_height(radius, overhang_angle))


@dataclass(frozen=True)
class TeardropProfile:
    """A closed teardrop outline.

    ``points`` is the closed polygon approximation (first point is repeated
    only implicitly -- the polygon closes back to ``points[0]``).
    ``arc`` records the three-point arc of the circular part as
    ``(start, mid, end)`` so an exact arc can be rebuilt by a CAD kernel.
    ``flanks`` lists the straight segments in order.
    """

    radius: float
    overhang_angle: float
    clip: Optional[float]
    points: Tuple[Point, ...]
    arc: Tuple[Point, Point, Point]
    flanks: Tuple[Tuple[Point, Point], ...]

    @property
    def height(self) -> float:
        """Total vertical extent of the profile."""
        ys = [p[1] for p in self.points]
        return max(ys) - min(ys)

    @property
    def width(self) -> float:
        xs = [p[0] for p in self.points]
        return max(xs) - min(xs)


def _rotate(points: Sequence[Point], degrees: float) -> Tuple[Point, ...]:
    if degrees == 0.0:
        return tuple((float(x), float(y)) for x, y in points)
    a = math.radians(degrees)
    ca, sa = math.cos(a), math.sin(a)
    return tuple((x * ca - y * sa, x * sa + y * ca) for x, y in points)


def _arc_points(radius: float, a_start: float, a_end: float,
                segments: int) -> List[Point]:
    """Sample the circle counter-clockwise from ``a_start`` to ``a_end``.

    ``a_end`` is lifted by whole turns until it is at or above ``a_start``,
    so the sampled sweep always runs the short way through the bottom of the
    bore (the supported side) rather than over the roof.
    """
    while a_end < a_start:
        a_end += 2.0 * math.pi
    out: List[Point] = []
    for i in range(segments + 1):
        a = a_start + (a_end - a_start) * (i / segments)
        out.append((radius * math.cos(a), radius * math.sin(a)))
    return out


def teardrop_profile(radius: float,
                     rotate: float = 0.0,
                     clip: Optional[float] = None,
                     overhang_angle: float = DEFAULT_OVERHANG_ANGLE,
                     segments: int = 32) -> TeardropProfile:
    """Build a teardrop cross-section of the given bore ``radius``.

    ``rotate`` turns the whole profile (degrees, counter-clockwise) so the
    apex can be aimed at the build-up direction of the part.  ``clip`` (in
    the *unrotated* frame, measured from the centre) truncates the apex.
    """
    _check(radius, overhang_angle)
    if segments < 4:
        raise TeardropError("segments must be at least 4")

    t = math.radians(overhang_angle)
    xj, yj = junction_point(radius, overhang_angle)
    ymax = apex_height(radius, overhang_angle)

    # Arc spans the *lower* part of the circle: from the left junction,
    # clockwise through the bottom, to the right junction.
    left = (-xj, yj)
    right = (xj, yj)
    bottom = (0.0, -radius)

    if clip is None:
        apex = (0.0, ymax)
        a_left = math.pi - t          # angle of the left junction
        a_right = t                   # angle of the right junction
        arc_pts = _arc_points(radius, a_left, a_right, segments)
        pts = arc_pts + [apex]
        flanks = ((right, apex), (apex, left))
        arc = (left, bottom, right)
    else:
        lo, hi = clip_bounds(radius, overhang_angle)
        if clip >= hi:
            raise TeardropError(
                "clip must be less than the apex height %.6f" % hi)
        if clip <= lo:
            raise TeardropError("clip must be greater than %.6f" % lo)

        if clip > yj:
            # Flat top cutting across the two straight flanks.
            xflat = (clip - yj) / math.tan(t)
            p_left = (left[0] + xflat, clip)
            p_right = (right[0] - xflat, clip)
            a_left = math.pi - t
            a_right = t
            arc_pts = _arc_points(radius, a_left, a_right, segments)
            pts = arc_pts + [p_right, p_left]
            flanks = ((right, p_right), (p_right, p_left), (p_left, left))
            arc = (left, bottom, right)
        else:
            # Chord cuts the arc: no 45-degree flanks survive.
            xflat = math.sqrt(max(radius * radius - clip * clip, 0.0))
            left = (-xflat, clip)
            right = (xflat, clip)
            a_left = math.atan2(clip, -xflat)
            a_right = math.atan2(clip, xflat)
            arc_pts = _arc_points(radius, a_left, a_right, segments)
            pts = arc_pts
            flanks = ((right, left),)
            arc = (left, bottom, right)

    return TeardropProfile(
        radius=float(radius),
        overhang_angle=float(overhang_angle),
        clip=None if clip is None else float(clip),
        points=_rotate(pts, rotate),
        arc=tuple(_rotate(arc, rotate)),  # type: ignore[arg-type]
        flanks=tuple(
            (_rotate([a], rotate)[0], _rotate([b], rotate)[0])
            for a, b in flanks),
    )


def polygon_area(points: Sequence[Point]) -> float:
    """Signed-area magnitude of a closed polygon (shoelace)."""
    n = len(points)
    acc = 0.0
    for i in range(n):
        x1, y1 = points[i]
        x2, y2 = points[(i + 1) % n]
        acc += x1 * y2 - x2 * y1
    return abs(acc) / 2.0


def max_overhang_of_profile(points: Sequence[Point]) -> float:
    """Worst overhang angle, in degrees from vertical, of an upward-facing wall.

    For each polygon edge whose outward-facing side looks *up* (i.e. the edge
    forms part of the roof of a hole), the angle between the edge and the
    vertical build direction is measured.  A plain circular bore returns 90
    (a horizontal roof tangent); a teardrop returns its overhang angle.
    """
    worst = 0.0
    n = len(points)
    for i in range(n):
        x1, y1 = points[i]
        x2, y2 = points[(i + 1) % n]
        # Only the upper half of the bore roofs over the void.
        if y1 <= 0.0 and y2 <= 0.0:
            continue
        dx = x2 - x1
        dy = y2 - y1
        if dx == 0.0 and dy == 0.0:
            continue
        # Angle from the vertical axis.
        ang = math.degrees(math.atan2(abs(dx), abs(dy)))
        mid_y = 0.5 * (y1 + y2)
        if mid_y > 0.0:
            worst = max(worst, ang)
    return worst


def is_self_supporting(profile: TeardropProfile,
                       limit: float = DEFAULT_OVERHANG_ANGLE,
                       bridge_tolerance: float = 1e-9) -> bool:
    """True when no roof surface exceeds ``limit`` degrees from vertical.

    A truncated teardrop has a perfectly horizontal top chord; that chord is
    a *bridge*, not an overhang, so it is exempt.  Any other near-horizontal
    roof surface (a plain round hole) fails.
    """
    n = len(profile.points)
    for i in range(n):
        x1, y1 = profile.points[i]
        x2, y2 = profile.points[(i + 1) % n]
        if 0.5 * (y1 + y2) <= 0.0:
            continue
        dx, dy = x2 - x1, y2 - y1
        if dx == 0.0 and dy == 0.0:
            continue
        if profile.clip is not None and abs(dy) <= bridge_tolerance:
            continue  # the bridged flat top
        ang = math.degrees(math.atan2(abs(dx), abs(dy)))
        if ang > limit + 1e-9:
            return False
    return True


def bridge_span(radius: float,
                clip: float,
                overhang_angle: float = DEFAULT_OVERHANG_ANGLE) -> float:
    """Width of the flat top that the printer must bridge, for a given clip."""
    _check(radius, overhang_angle)
    lo, hi = clip_bounds(radius, overhang_angle)
    if not lo < clip < hi:
        raise TeardropError("clip out of range (%.6f, %.6f)" % (lo, hi))
    t = math.radians(overhang_angle)
    xj, yj = junction_point(radius, overhang_angle)
    if clip > yj:
        xflat = (clip - yj) / math.tan(t)
        return 2.0 * (xj - xflat)
    return 2.0 * math.sqrt(max(radius * radius - clip * clip, 0.0))


def headroom_saved(radius: float,
                   clip: float,
                   overhang_angle: float = DEFAULT_OVERHANG_ANGLE) -> float:
    """Vertical space recovered by truncating the apex at ``clip``."""
    return apex_height(radius, overhang_angle) - clip
