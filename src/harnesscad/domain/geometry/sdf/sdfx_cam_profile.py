"""Exact 2D signed distance fields for mechanical cam profiles (sdfx).

Reimplementation of the cam generators from deadsy/sdfx (``sdf/cams.go``).  Two
classic radial cam profiles, each an *exact* 2D SDF (closed-form nearest-feature
distance) plus a design-parameter constructor that solves the geometry from
follower lift / duration / maximum diameter:

* :class:`FlatFlankCam` -- base circle, smaller nose circle, and straight
  tangent flanks (the profile decomposes into two arcs and a line);
* :class:`ThreeArcCam` -- base circle, nose circle, and *circular* flank arcs
  tangent to both, with a tunable nose roundness ``k``.

Both are symmetric about the y-axis with the base circle centered at the origin
and the nose on the +y axis.  ``evaluate((x, y))`` returns the signed distance
(negative inside).  These are genuine manufacturing cam curves not present in
the harness's SDF stack (curv / libfive / sdf-csg provide no cam primitive).

Pure stdlib, deterministic.
"""

from __future__ import annotations

import math
from typing import Tuple

__all__ = [
    "FlatFlankCam",
    "ThreeArcCam",
    "make_flat_flank_cam",
    "make_three_arc_cam",
]

Vec2 = Tuple[float, float]


def _line_intersect(p0: Vec2, d0: Vec2, p1: Vec2, d1: Vec2) -> Tuple[float, Vec2]:
    """Intersect two parametric lines p0 + t*d0 and p1 + s*d1.

    Returns (t, point).  Raises ValueError if the lines are parallel.
    """
    # p0 + t d0 = p1 + s d1  ->  solve for t via 2x2 determinant.
    det = d0[0] * (-d1[1]) - (-d1[0]) * d0[1]
    if det == 0.0:
        raise ValueError("parallel lines do not intersect")
    rx = p1[0] - p0[0]
    ry = p1[1] - p0[1]
    t = (rx * (-d1[1]) - (-d1[0]) * ry) / det
    return t, (p0[0] + t * d0[0], p0[1] + t * d0[1])


class FlatFlankCam:
    """Cam from a base circle, a nose circle and flat tangent flanks."""

    def __init__(self, distance: float, base_radius: float, nose_radius: float):
        if distance <= 0:
            raise ValueError("distance must be > 0")
        if base_radius <= 0 or nose_radius <= 0:
            raise ValueError("radii must be > 0")
        self.distance = distance
        self.base_radius = base_radius
        self.nose_radius = nose_radius
        # flank line tangent to both circles (positive-x side)
        sin = (base_radius - nose_radius) / distance
        if abs(sin) > 1.0:
            raise ValueError("no tangent flank for these radii/distance")
        cos = math.sqrt(1.0 - sin * sin)
        # first point on the flank line (on the base circle)
        self.ax = cos * base_radius
        self.ay = sin * base_radius
        # second point (on the nose circle, offset up by distance)
        bx = cos * nose_radius
        by = sin * nose_radius + distance
        ux = bx - self.ax
        uy = by - self.ay
        self.length = math.hypot(ux, uy)
        self.ux = ux / self.length
        self.uy = uy / self.length

    def evaluate(self, p: Vec2) -> float:
        """Signed distance from point ``p`` to the cam profile."""
        # symmetry about the y-axis
        px = abs(p[0])
        py = p[1]
        vx = px - self.ax
        vy = py - self.ay
        t = vx * self.ux + vy * self.uy
        if t < 0.0:
            # nearest point on the base circle
            return math.hypot(px, py) - self.base_radius
        if t <= self.length:
            # nearest point on the flank line: normal component (uy, -ux)
            return vx * self.uy - vy * self.ux
        # nearest point on the nose circle
        return math.hypot(px, py - self.distance) - self.nose_radius


class ThreeArcCam:
    """Cam from a base circle, a nose circle and circular flank arcs."""

    def __init__(self, distance: float, base_radius: float,
                 nose_radius: float, flank_radius: float):
        if distance <= 0:
            raise ValueError("distance must be > 0")
        if base_radius <= 0 or nose_radius <= 0:
            raise ValueError("radii must be > 0")
        if flank_radius < (base_radius + distance + nose_radius) / 2.0:
            raise ValueError("flankRadius too small")
        self.distance = distance
        self.base_radius = base_radius
        self.nose_radius = nose_radius
        self.flank_radius = flank_radius
        # flank arc center is where circles about base/nose centers intersect
        r0 = flank_radius - base_radius
        r1 = flank_radius - nose_radius
        y = ((r0 * r0) - (r1 * r1) + (distance * distance)) / (2.0 * distance)
        x = -math.sqrt(max(0.0, (r0 * r0) - (y * y)))  # +x flank arc
        self.fcx = x
        self.fcy = y
        # angle (wrt flank center) at which flank arc meets the base circle
        self.theta_base = math.atan2(0.0 - y, 0.0 - x)
        # angle at which flank arc meets the nose circle
        self.theta_nose = math.atan2(distance - y, 0.0 - x)

    def evaluate(self, p: Vec2) -> float:
        """Signed distance from point ``p`` to the cam profile."""
        px = abs(p[0])
        py = p[1]
        vx = px - self.fcx
        vy = py - self.fcy
        t = math.atan2(vy, vx)
        if t < self.theta_base:
            return math.hypot(px, py) - self.base_radius
        if t > self.theta_nose:
            return math.hypot(px, py - self.distance) - self.nose_radius
        return math.hypot(vx, vy) - self.flank_radius


def make_flat_flank_cam(lift: float, duration: float,
                        max_diameter: float) -> FlatFlankCam:
    """Design a flat-flank cam from follower lift, duration and max diameter.

    ``lift`` is the follower rise above the base circle, ``duration`` the angle
    (radians, 0 < duration < pi) over which the follower lifts, ``max_diameter``
    the maximum rotation diameter.
    """
    if max_diameter <= 0:
        raise ValueError("max_diameter must be > 0")
    if lift <= 0:
        raise ValueError("lift must be > 0")
    if duration <= 0 or duration >= math.pi:
        raise ValueError("invalid duration")
    base_radius = (max_diameter / 2.0) - lift
    if base_radius <= 0:
        raise ValueError("base_radius <= 0")
    delta = duration / 2.0
    c = math.cos(delta)
    nose_radius = base_radius - (lift * c) / (1.0 - c)
    if nose_radius <= 0:
        raise ValueError("nose_radius <= 0")
    distance = base_radius + lift - nose_radius
    return FlatFlankCam(distance, base_radius, nose_radius)


def make_three_arc_cam(lift: float, duration: float, max_diameter: float,
                       k: float) -> ThreeArcCam:
    """Design a three-arc cam; ``k`` (>1) tunes nose roundness (e.g. 1.05)."""
    if max_diameter <= 0:
        raise ValueError("max_diameter must be > 0")
    if lift <= 0:
        raise ValueError("lift must be > 0")
    if duration <= 0:
        raise ValueError("invalid duration")
    if k <= 1.0:
        raise ValueError("invalid k")
    base_radius = (max_diameter / 2.0) - lift
    if base_radius <= 0:
        raise ValueError("base_radius <= 0")
    # flank arc intersects the base circle at this point
    theta = (math.pi - duration) / 2.0
    p0 = (math.cos(theta) * base_radius, math.sin(theta) * base_radius)
    # line from p0 back through the origin toward the flank arc center
    d0 = (-p0[0], -p0[1])
    # the flank arc intersects the y axis above the lift height
    p1 = (0.0, k * (base_radius + lift))
    # perpendicular bisector of p0 and p1 passes through the flank arc center
    p_mid = ((p0[0] + p1[0]) * 0.5, (p0[1] + p1[1]) * 0.5)
    u = (p1[0] - p0[0], p1[1] - p0[1])
    d1 = (u[1], -u[0])
    _, flank_center = _line_intersect(p0, d0, p_mid, d1)
    cx, cy = flank_center
    # the flank arc passes through p0, so its radius is |center - p0|
    flank_radius = math.hypot(cx - p0[0], cy - p0[1])
    # nose circle tangent to the flank arcs and the lift line
    j = base_radius + lift
    f = flank_radius
    nose_radius = ((cx * cx) + (cy * cy) - (f * f) + (j * j) - (2 * cy * j)) \
        / (2 * (j - f - cy))
    distance = base_radius + lift - nose_radius
    return ThreeArcCam(distance, base_radius, nose_radius, flank_radius)
