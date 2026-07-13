"""Closest-point queries with multiplicity (from the ``arcs`` Rust CAD core).

``arcs-core/src/algorithms/closest_point.rs`` models the closest point on a
primitive as a three-way result rather than a single point:

* ``ONE``      -- a unique closest point (the common case);
* ``MANY``     -- a tie, e.g. a target exactly halfway between the two ends of
                  an arc that does not face it;
* ``INFINITE`` -- every point of the primitive is equidistant, i.e. the target
                  is the centre of an arc.

The harness only had ``geometry.joinable_joint_axis.closest_point_on_line``
(projection onto an *infinite* 3-D line, single result). This module adds the
clamped 2-D segment projection, the arc case (radial projection when the arc
subtends the target's bearing, otherwise the nearer endpoint, with the tie
detected exactly), and the tie/degenerate bookkeeping.

Arc convention follows ``arcs``: centre, radius, ``start_angle`` and a signed
``sweep_angle`` (positive = anticlockwise). ``contains_angle`` is implemented on
the *normalised* sweep parameter rather than the raw comparison used by the
Rust source, so arcs which cross the +/-pi branch cut behave correctly.

Pure standard library, deterministic. Points are ``(x, y)`` float tuples.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Sequence, Tuple

Point = Tuple[float, float]

TWO_PI = 2.0 * math.pi
EPSILON = 1e-9

INFINITE = "infinite"
ONE = "one"
MANY = "many"

__all__ = [
    "EPSILON",
    "INFINITE",
    "MANY",
    "ONE",
    "TWO_PI",
    "Arc2D",
    "Closest",
    "closest_point_on_arc",
    "closest_point_on_polyline",
    "closest_point_on_segment",
    "distance_to_arc",
    "distance_to_segment",
]


@dataclass(frozen=True)
class Closest:
    """Result of a closest-point query."""

    kind: str
    points: Tuple[Point, ...] = ()

    @staticmethod
    def one(point: Point) -> "Closest":
        return Closest(ONE, (point,))

    @staticmethod
    def many(points: Sequence[Point]) -> "Closest":
        return Closest(MANY, tuple(points))

    @staticmethod
    def infinite() -> "Closest":
        return Closest(INFINITE, ())

    @property
    def is_infinite(self) -> bool:
        return self.kind == INFINITE

    def single(self) -> Point:
        """The unique closest point; raises unless the result is ``ONE``."""
        if self.kind != ONE:
            raise ValueError("closest point is not unique: " + self.kind)
        return self.points[0]


@dataclass(frozen=True)
class Arc2D:
    """A circle segment: centre + radius + start angle + signed sweep."""

    centre: Point
    radius: float
    start_angle: float
    sweep_angle: float

    @property
    def end_angle(self) -> float:
        return self.start_angle + self.sweep_angle

    @property
    def is_anticlockwise(self) -> bool:
        return self.sweep_angle > 0.0

    @property
    def is_minor_arc(self) -> bool:
        return abs(self.sweep_angle) <= math.pi

    @property
    def is_major_arc(self) -> bool:
        return not self.is_minor_arc

    def point_at(self, offset: float) -> Point:
        """Point at ``offset`` radians past :attr:`start_angle`."""
        angle = self.start_angle + offset
        return (
            self.centre[0] + self.radius * math.cos(angle),
            self.centre[1] + self.radius * math.sin(angle),
        )

    def start(self) -> Point:
        return self.point_at(0.0)

    def end(self) -> Point:
        return self.point_at(self.sweep_angle)

    def contains_angle(self, angle: float, tol: float = EPSILON) -> bool:
        """Does the arc sweep through the absolute bearing ``angle``?"""
        sweep = self.sweep_angle
        if abs(sweep) >= TWO_PI - tol:
            return True
        if sweep >= 0.0:
            # forward from the start angle
            delta = (angle - self.start_angle) % TWO_PI
        else:
            # clockwise: walk backwards from the start angle
            delta = (self.start_angle - angle) % TWO_PI
        # `delta` near TWO_PI means the bearing sits just *before* the start
        return delta <= abs(sweep) + tol or delta >= TWO_PI - tol

    def length(self) -> float:
        return abs(self.sweep_angle) * self.radius


def closest_point_on_segment(start: Point, end: Point, target: Point) -> Closest:
    """Closest point on the *segment* ``start``-``end`` (clamped projection)."""
    dx = end[0] - start[0]
    dy = end[1] - start[1]
    squared = dx * dx + dy * dy

    if squared <= EPSILON * EPSILON:
        return Closest.one(start)

    t = ((target[0] - start[0]) * dx + (target[1] - start[1]) * dy) / squared
    if t <= 0.0:
        return Closest.one(start)
    if t >= 1.0:
        return Closest.one(end)
    return Closest.one((start[0] + t * dx, start[1] + t * dy))


def closest_point_on_arc(arc: Arc2D, target: Point) -> Closest:
    """Closest point(s) on ``arc``; ``INFINITE`` when ``target`` is the centre."""
    rx = target[0] - arc.centre[0]
    ry = target[1] - arc.centre[1]
    radial = math.hypot(rx, ry)

    if radial <= EPSILON:
        return Closest.infinite()

    bearing = math.atan2(ry, rx)
    ideal = (
        arc.centre[0] + rx / radial * arc.radius,
        arc.centre[1] + ry / radial * arc.radius,
    )

    if arc.contains_angle(bearing):
        return Closest.one(ideal)

    start = arc.start()
    end = arc.end()
    to_start = math.hypot(start[0] - ideal[0], start[1] - ideal[1])
    to_end = math.hypot(end[0] - ideal[0], end[1] - ideal[1])

    if abs(to_start - to_end) <= EPSILON:
        return Closest.many([start, end])
    if to_start < to_end:
        return Closest.one(start)
    return Closest.one(end)


def closest_point_on_polyline(points: Sequence[Point], target: Point) -> Closest:
    """Closest point(s) on a polyline; ties are reported as ``MANY``."""
    if not points:
        raise ValueError("polyline is empty")
    if len(points) == 1:
        return Closest.one(points[0])

    best = float("inf")
    winners: List[Point] = []
    for i in range(1, len(points)):
        candidate = closest_point_on_segment(
            points[i - 1], points[i], target
        ).single()
        d = math.hypot(candidate[0] - target[0], candidate[1] - target[1])
        if d < best - EPSILON:
            best = d
            winners = [candidate]
        elif abs(d - best) <= EPSILON:
            if all(
                abs(candidate[0] - w[0]) > EPSILON
                or abs(candidate[1] - w[1]) > EPSILON
                for w in winners
            ):
                winners.append(candidate)
            best = min(best, d)

    if len(winners) == 1:
        return Closest.one(winners[0])
    return Closest.many(winners)


def distance_to_segment(start: Point, end: Point, target: Point) -> float:
    """Euclidean distance from ``target`` to the segment ``start``-``end``."""
    p = closest_point_on_segment(start, end, target).single()
    return math.hypot(p[0] - target[0], p[1] - target[1])


def distance_to_arc(arc: Arc2D, target: Point) -> float:
    """Euclidean distance from ``target`` to ``arc`` (the radius, at the centre)."""
    result = closest_point_on_arc(arc, target)
    if result.is_infinite:
        return arc.radius
    p = result.points[0]
    return math.hypot(p[0] - target[0], p[1] - target[1])
