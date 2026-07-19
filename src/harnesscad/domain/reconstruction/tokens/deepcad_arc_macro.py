"""Arc macro encoding: (end point, sweep angle, ccw flag) <-> full geometry.

``reconstruction.deepcad_command_spec`` records that an ``Arc`` command carries
``(x, y, alpha, f)`` and ``reconstruction.deepcad_profile_assembly`` chains endpoints
into segments -- but neither ever *decodes* the arc: the centre, radius and reference
vector implied by ``(start, end, alpha, f)`` are never recovered, and the arc's bulge
is explicitly excluded from ``loop_bbox`` ("endpoints only"). This module supplies
exactly that missing geometry layer:

* :func:`arc_from_macro` -- the reference's closed form
  ``r = (|s->e| / 2) / sin(alpha/2)`` and
  ``centre = mid(s,e) - v_perp * r * cos(alpha/2)``, with ``v_perp`` the left normal
  of ``s->e`` negated when the ccw flag is 0.
* :func:`clock_sign` -- the encoder's flag: ``cross(s->m, s->e) >= 0``.
* :func:`arc_bbox` -- the *true* bbox including the bulge, by testing which of the
  four axis extreme points of the circle the sweep covers.
* :func:`sample_arc_points` / :func:`sample_circle_points` -- the reference's
  deterministic uniform sampling used to build the evaluation point clouds.

Pure stdlib, deterministic. No numpy, no OCC.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

Vec2 = tuple[float, float]

TWO_PI = 2 * math.pi


# --- planar helpers ---------------------------------------------------------
def _sub(a: Sequence[float], b: Sequence[float]) -> Vec2:
    return (a[0] - b[0], a[1] - b[1])


def _norm(v: Sequence[float]) -> float:
    return math.hypot(v[0], v[1])


def cross2(a: Sequence[float], b: Sequence[float]) -> float:
    """Scalar 2D cross product ``a.x*b.y - a.y*b.x``."""
    return a[0] * b[1] - a[1] * b[0]


def angle_from_vector_to_x(vec: Sequence[float]) -> float:
    """Angle in ``[0, 2pi)`` between a *unit* vector and the +x axis.

    Reproduces the reference's quadrant-cased ``asin`` implementation exactly.
    """
    x, y = vec[0], vec[1]
    if x >= 0:
        if y >= 0:
            return math.asin(min(1.0, max(-1.0, y)))          # quadrant 1
        return TWO_PI - math.asin(min(1.0, max(-1.0, -y)))    # quadrant 4
    if y >= 0:
        return math.pi - math.asin(min(1.0, max(-1.0, y)))    # quadrant 2
    return math.pi + math.asin(min(1.0, max(-1.0, -y)))       # quadrant 3


# --- the arc ---------------------------------------------------------------
@dataclass(frozen=True)
class Arc:
    """A planar circular arc, fully determined."""
    start_point: Vec2
    end_point: Vec2
    center: Vec2
    radius: float
    ref_vec: Vec2         # unit vector centre -> the angle-0 reference point
    start_angle: float    # always 0 in the macro decoding
    end_angle: float      # == sweep angle

    @property
    def mid_point(self) -> Vec2:
        return arc_mid_point(self.center, self.radius, self.ref_vec,
                             self.start_angle, self.end_angle)


def arc_mid_point(center: Sequence[float], radius: float, ref_vec: Sequence[float],
                  start_angle: float, end_angle: float) -> Vec2:
    """Rotate ``ref_vec`` by the mean angle and step ``radius`` from ``center``."""
    mid = (start_angle + end_angle) / 2
    c, s = math.cos(mid), math.sin(mid)
    vx = c * ref_vec[0] - s * ref_vec[1]
    vy = s * ref_vec[0] + c * ref_vec[1]
    return (center[0] + vx * radius, center[1] + vy * radius)


def clock_sign(start: Sequence[float], mid: Sequence[float],
               end: Sequence[float]) -> int:
    """The encoder's ``f`` flag: 1 when the arc bulges counter-clockwise of s->e.

    Reference: ``cross(s->m, s->e) >= 0``.
    """
    return int(cross2(_sub(mid, start), _sub(end, start)) >= 0)


def arc_from_macro(start: Sequence[float], end: Sequence[float],
                   sweep_angle: float, flag: int) -> Arc:
    """Decode ``(start, end, alpha, f)`` into full arc geometry.

    ``sweep_angle`` is in radians (the caller de-quantises ``alpha`` first, e.g. with
    ``deepcad2_numericalize.denumericalize_sweep``). Raises ``ValueError`` for a
    degenerate arc (coincident endpoints, or a sweep that is a multiple of ``2pi``).
    """
    s2e = _sub(end, start)
    chord = _norm(s2e)
    if chord == 0:
        raise ValueError("degenerate arc: coincident start and end points")
    half = sweep_angle / 2
    sin_half = math.sin(half)
    if sin_half == 0:
        raise ValueError(f"degenerate arc: sweep angle {sweep_angle}")
    radius = (chord / 2) / sin_half

    mid_chord = ((start[0] + end[0]) / 2, (start[1] + end[1]) / 2)
    # cross(s2e, +z) = (s2e.y, -s2e.x) -- the left normal of the chord direction.
    vertical = (s2e[1] / chord, -s2e[0] / chord)
    if flag == 0:
        vertical = (-vertical[0], -vertical[1])
    offset = radius * math.cos(half)
    center = (mid_chord[0] - vertical[0] * offset,
              mid_chord[1] - vertical[1] * offset)

    anchor = end if flag == 0 else start
    ref = _sub(anchor, center)
    ref_len = _norm(ref)
    if ref_len == 0:
        raise ValueError("degenerate arc: endpoint coincides with centre")
    ref_vec = (ref[0] / ref_len, ref[1] / ref_len)
    return Arc(tuple(start), tuple(end), center, radius, ref_vec, 0.0, sweep_angle)


def arc_to_macro(start: Sequence[float], mid: Sequence[float],
                 end: Sequence[float], center: Sequence[float]) -> tuple[Vec2, float, int]:
    """Encode an arc as ``(end_point, sweep_angle, f)`` -- the inverse macro.

    The sweep angle is the counter-clockwise span from :func:`angles_counterclockwise`.
    """
    angle_s, angle_e = angles_counterclockwise(center, start, mid, end)
    return (tuple(end), angle_e - angle_s, clock_sign(start, mid, end))


def angles_counterclockwise(center: Sequence[float], start: Sequence[float],
                            mid: Sequence[float], end: Sequence[float],
                            eps: float = 1e-8) -> tuple[float, float]:
    """The reference ccw-span computation: the ccw span ``(a_s, a_e)``.

    Both angles are measured from +x. ``a_s`` is allowed to go negative (the span is
    shifted by ``-2pi``) so that the arc's mid point always lies strictly between
    them, which is what makes the extreme-point tests in :func:`arc_bbox` valid.
    """
    def unit(p):
        v = _sub(p, center)
        n = _norm(v) + eps
        return (v[0] / n, v[1] / n)

    angle_s = angle_from_vector_to_x(unit(start))
    angle_m = angle_from_vector_to_x(unit(mid))
    angle_e = angle_from_vector_to_x(unit(end))
    angle_s, angle_e = min(angle_s, angle_e), max(angle_s, angle_e)
    if not angle_s < angle_m < angle_e:
        angle_s, angle_e = angle_e - TWO_PI, angle_s
    return angle_s, angle_e


def arc_bbox(arc: Arc) -> tuple[float, float, float, float]:
    """True ``(min_x, min_y, max_x, max_y)`` of an arc, bulge included.

    Endpoints plus whichever of the circle's four axis extreme points
    (``centre +/- r`` on each axis) the swept angular interval covers.
    """
    start, end, center, radius = arc.start_point, arc.end_point, arc.center, arc.radius
    angle_s, angle_e = angles_counterclockwise(center, start, arc.mid_point, end)
    points = [start, end]
    cx, cy = center
    if angle_s < 0 < angle_e:
        points.append((cx + radius, cy))
    if angle_s < math.pi / 2 < angle_e or angle_s < -math.pi / 2 * 3 < angle_e:
        points.append((cx, cy + radius))
    if angle_s < math.pi < angle_e or angle_s < -math.pi < angle_e:
        points.append((cx - radius, cy))
    if angle_s < math.pi / 2 * 3 < angle_e or angle_s < -math.pi / 2 < angle_e:
        points.append((cx, cy - radius))
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    return (min(xs), min(ys), max(xs), max(ys))


# --- deterministic sampling (used to build the eval point clouds) -----------
def sample_arc_points(arc: Arc, n: int = 32) -> list[Vec2]:
    """``n`` points uniformly in angle from start to end (both endpoints included)."""
    if n < 2:
        raise ValueError("n must be >= 2")
    angle_s, angle_e = angles_counterclockwise(arc.center, arc.start_point,
                                               arc.mid_point, arc.end_point)
    step = (angle_e - angle_s) / (n - 1)
    return [(arc.center[0] + math.cos(angle_s + i * step) * arc.radius,
             arc.center[1] + math.sin(angle_s + i * step) * arc.radius)
            for i in range(n)]


def sample_circle_points(center: Sequence[float], radius: float,
                         n: int = 32) -> list[Vec2]:
    """``n`` points around a full circle, endpoint excluded (``linspace`` semantics)."""
    if n < 1:
        raise ValueError("n must be >= 1")
    step = TWO_PI / n
    return [(center[0] + math.cos(i * step) * radius,
             center[1] + math.sin(i * step) * radius)
            for i in range(n)]


def sample_line_points(start: Sequence[float], end: Sequence[float],
                       n: int = 32) -> list[Vec2]:
    """``n`` points uniformly along a segment (both endpoints included)."""
    if n < 2:
        raise ValueError("n must be >= 2")
    return [(start[0] + (end[0] - start[0]) * i / (n - 1),
             start[1] + (end[1] - start[1]) * i / (n - 1))
            for i in range(n)]
