"""Exact 2D signed distance field for an Archimedean spiral (sdfx).

Reimplementation of ``ArcSpiral2D`` from deadsy/sdfx (``sdf/spiral.go``).  The
spiral is the curve r = a*theta + k in polar coordinates, bounded to the angular
range [start, end]; the field is the distance to that curve, minus an offset
``d`` (so a positive ``d`` gives the spiral a finite thickness / becomes a
groove of half-width ``d``).

The distance is computed in polar space.  For a query point (given as polar
(r, theta)) the algorithm inverts the spiral to find the theta value whose
radius equals the point's radius, then -- because the polar angle is only
defined modulo 2*pi -- checks that solution shifted by whole turns to land
inside [start, end], taking the minimum polar distance.  The two spiral
endpoints are always candidates so the field stays correct beyond the winding
range.  ``polar_dist2`` is the law-of-cosines squared distance between two polar
points.

No spiral primitive exists in the harness's SDF stack (curv / libfive /
sdf-csg), so this is genuinely new.  Pure stdlib, deterministic.
"""

from __future__ import annotations

import math
from typing import Tuple

__all__ = [
    "ArcSpiral",
    "polar_dist2",
    "to_polar",
]

TAU = 2.0 * math.pi
Vec2 = Tuple[float, float]


def polar_dist2(r0: float, t0: float, r1: float, t1: float) -> float:
    """Squared distance between two polar points, via the law of cosines."""
    return r0 * r0 + r1 * r1 - 2.0 * r0 * r1 * math.cos(t0 - t1)


def to_polar(p: Vec2) -> Tuple[float, float]:
    """Convert cartesian (x, y) to polar (r, theta)."""
    return math.hypot(p[0], p[1]), math.atan2(p[1], p[0])


class ArcSpiral:
    """Archimedean spiral r = a*theta + k over [start, end], offset by d."""

    def __init__(self, a: float, k: float, start: float, end: float,
                 d: float = 0.0):
        if start == end:
            raise ValueError("start == end")
        if a == 0:
            raise ValueError("a == 0")
        self.a = a
        self.k = k
        self.d = d
        if start > end:
            start, end = end, start
        self.start = start
        self.end = end
        self.start_r = self._radius(start)
        self.end_r = self._radius(end)

    def _radius(self, theta: float) -> float:
        return self.a * theta + self.k

    def _theta(self, radius: float) -> float:
        # invert r = a*theta + k
        return (radius - self.k) / self.a

    def evaluate(self, p: Vec2) -> float:
        """Signed distance from cartesian point ``p`` to the spiral."""
        pr, pt = to_polar(p)

        # the endpoints are always distance candidates
        d2 = min(
            polar_dist2(pr, pt, self.start_r, self.start),
            polar_dist2(pr, pt, self.end_r, self.end),
        )

        theta = self._theta(pr)
        # shift theta by whole turns so it is nearest the query angle
        n = round((pt - theta) / TAU)
        theta = pt - TAU * n

        if self.start <= theta <= self.end:
            d2 = min(d2, polar_dist2(pr, pt, self._radius(theta), theta))
        else:
            if theta < self.start:
                th = theta
                while th < self.start:
                    th += TAU
                if th < self.end:
                    d2 = min(d2, polar_dist2(pr, pt, self._radius(th), th))
            if theta > self.end:
                th = theta
                while th > self.end:
                    th -= TAU
                if th > self.start:
                    d2 = min(d2, polar_dist2(pr, pt, self._radius(th), th))

        return math.sqrt(d2) - self.d
