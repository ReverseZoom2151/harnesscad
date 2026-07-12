"""Exact GenCAD/DeepCAD arc<->vector geometry (reference-implementation level).

The GenCAD reference implementation (``cadlib/curves.py``, ``cadlib/math_utils.py``)
encodes a sketch arc in the 16-slot command vector as only three numbers beside its
end-point: the *sweep angle* ``alpha`` (quantised over ``[0, 2*pi)``) and a binary
*counter-clockwise flag* ``f`` (``clock_sign``). Everything else -- centre, radius,
reference vector, mid-point -- is *reconstructed* from the previous curve's
end-point (the implicit start-point) by an exact closed-form procedure. That
procedure, and the exact arc bounding box (which must include the axis-extreme
points the arc sweeps through, not just its endpoints), are implementation details
that the paper-level modules do not carry:

* ``reconstruction.deepcad_command_spec`` gives the 16-slot vector layout only.
* ``reconstruction.deepcad_profile_assembly.loop_bbox`` documents explicitly that
  "arc bulge beyond the chord is not modelled -- endpoints only"; this module
  models it exactly.

Reconstruction (``Arc.from_vector``)::

    sweep  = q / 256 * 2*pi                       (when the vector is quantised)
    r      = (|end - start| / 2) / sin(sweep / 2)
    v      = unit(perp(end - start))              perp(a, b) = (b, -a)
    v      = -v            if clock_sign == 0
    centre = midpoint(start, end) - v * r * cos(sweep / 2)
    ref    = unit(start - centre) if clock_sign else unit(end - centre)
    mid    = centre + R(sweep / 2) @ ref * r

Encoding (``Arc.to_vector``) recovers ``clock_sign`` from the sign of
``cross(start->mid, start->end) >= 0``.

Pure standard library, deterministic, no plotting. All points are ``(x, y)``
tuples of floats.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Optional, Tuple

Vec2 = Tuple[float, float]

TWO_PI = 2.0 * math.pi
ARGS_DIM = 256  # GenCAD macro.ARGS_DIM: quantisation levels for the sweep angle


# --- small vector helpers ---------------------------------------------------
def _sub(a: Vec2, b: Vec2) -> Vec2:
    return (a[0] - b[0], a[1] - b[1])


def _add(a: Vec2, b: Vec2) -> Vec2:
    return (a[0] + b[0], a[1] + b[1])


def _scale(a: Vec2, s: float) -> Vec2:
    return (a[0] * s, a[1] * s)


def _norm(a: Vec2) -> float:
    return math.hypot(a[0], a[1])


def _unit(a: Vec2) -> Vec2:
    n = _norm(a)
    if n == 0.0:
        raise ValueError("cannot normalise a zero-length vector")
    return (a[0] / n, a[1] / n)


def _cross(a: Vec2, b: Vec2) -> float:
    """z-component of the 3D cross product of two 2D vectors."""
    return a[0] * b[1] - a[1] * b[0]


def _perp(a: Vec2) -> Vec2:
    """``cross([ax, ay, 0], [0, 0, 1])`` projected back to 2D, i.e. ``(ay, -ax)``."""
    return (a[1], -a[0])


def angle_from_vector_to_x(vec: Vec2) -> float:
    """Angle in ``[0, 2*pi)`` between a *unit* vector and the positive x-axis.

    Exact port of GenCAD ``math_utils.angle_from_vector_to_x`` (quadrant-wise
    ``asin``). The input is expected to be unit-length; components are clamped to
    ``[-1, 1]`` so that floating-point drift cannot raise a domain error.
    """
    x, y = vec

    def _asin(v: float) -> float:
        return math.asin(max(-1.0, min(1.0, v)))

    if x >= 0:
        if y >= 0:
            return _asin(y)                      # quadrant 1
        return TWO_PI - _asin(-y)                # quadrant 4
    if y >= 0:
        return math.pi - _asin(y)                # quadrant 2
    return math.pi + _asin(-y)                   # quadrant 3


def dequantize_sweep(level: float, n: int = ARGS_DIM) -> float:
    """``alpha = q / n * 2*pi`` -- the exact GenCAD sweep-angle de-quantisation."""
    return float(level) / n * TWO_PI


def quantize_sweep(sweep: float, n: int = ARGS_DIM) -> int:
    """Inverse of :func:`dequantize_sweep`, with GenCAD's ``max(., 1)`` floor.

    GenCAD's ``Arc.to_vector`` emits ``max(abs(start_angle - end_angle), 1)`` on the
    already-quantised angles, so a degenerate zero sweep is never written out.
    """
    q = int(round(sweep / TWO_PI * n))
    q = max(0, min(n - 1, q))
    return max(q, 1)


# --- arc ---------------------------------------------------------------------
@dataclass(frozen=True)
class Arc:
    """Fully-resolved sketch arc, as GenCAD's ``Arc`` after ``from_vector``."""

    start_point: Vec2
    end_point: Vec2
    center: Vec2
    radius: float
    ref_vec: Vec2
    start_angle: float
    end_angle: float
    mid_point: Vec2


def arc_mid_point(center: Vec2, radius: float, ref_vec: Vec2,
                  start_angle: float, end_angle: float) -> Vec2:
    """GenCAD ``Arc.get_mid_point``: rotate ``ref_vec`` by the half sweep."""
    mid_angle = (start_angle + end_angle) / 2.0
    c, s = math.cos(mid_angle), math.sin(mid_angle)
    mid_vec = (c * ref_vec[0] - s * ref_vec[1], s * ref_vec[0] + c * ref_vec[1])
    return _add(center, _scale(mid_vec, radius))


def arc_from_vector(start_point: Vec2, end_point: Vec2, sweep: float,
                    clock_sign: int, is_numerical: bool = True,
                    n: int = ARGS_DIM) -> Optional[Arc]:
    """Reconstruct an arc from ``(end_point, sweep, clock_sign)`` + implicit start.

    ``sweep`` is a quantised level when ``is_numerical`` (the on-the-wire form), or a
    radian angle otherwise. Returns ``None`` for a degenerate arc whose start and end
    coincide -- GenCAD replaces such an arc with a line.
    """
    sweep_angle = dequantize_sweep(sweep, n) if is_numerical else float(sweep)
    s2e = _sub(end_point, start_point)
    chord = _norm(s2e)
    if chord == 0.0:
        return None
    half = sweep_angle / 2.0
    sin_half = math.sin(half)
    if sin_half == 0.0:
        return None
    radius = (chord / 2.0) / sin_half
    s2e_mid = _scale(_add(start_point, end_point), 0.5)
    vertical = _unit(_perp(s2e))
    if not clock_sign:
        vertical = _scale(vertical, -1.0)
    center = _sub(s2e_mid, _scale(vertical, radius * math.cos(half)))

    ref_source = start_point if clock_sign else end_point
    ref_vec = _unit(_sub(ref_source, center))
    mid_point = arc_mid_point(center, radius, ref_vec, 0.0, sweep_angle)
    return Arc(start_point=tuple(map(float, start_point)),
               end_point=tuple(map(float, end_point)),
               center=center, radius=radius, ref_vec=ref_vec,
               start_angle=0.0, end_angle=sweep_angle, mid_point=mid_point)


def arc_clock_sign(start_point: Vec2, mid_point: Vec2, end_point: Vec2) -> int:
    """GenCAD ``Arc.clock_sign``: ``cross(start->mid, start->end) >= 0``."""
    s2m = _sub(mid_point, start_point)
    s2e = _sub(end_point, start_point)
    return 1 if _cross(s2m, s2e) >= 0 else 0


def arc_sweep_angle(start_point: Vec2, mid_point: Vec2, end_point: Vec2,
                    center: Vec2) -> float:
    """Sweep angle in ``(0, 2*pi)`` of the arc start->mid->end about ``center``."""
    angle_s, angle_e = arc_angles_counterclockwise(start_point, mid_point,
                                                   end_point, center)
    return angle_e - angle_s


def arc_angles_counterclockwise(start_point: Vec2, mid_point: Vec2,
                                end_point: Vec2, center: Vec2,
                                eps: float = 1e-8) -> Tuple[float, float]:
    """GenCAD ``Arc.get_angles_counterclockwise``.

    Returns ``(angle_s, angle_e)`` with ``angle_s < angle_e`` such that sweeping
    counter-clockwise from ``angle_s`` to ``angle_e`` traverses the arc through its
    mid-point. ``angle_s`` may be negative (the branch is unwrapped by ``-2*pi``).
    """
    def _ang(p: Vec2) -> float:
        d = _sub(p, center)
        m = _norm(d) + eps
        return angle_from_vector_to_x((d[0] / m, d[1] / m))

    angle_s, angle_m, angle_e = _ang(start_point), _ang(mid_point), _ang(end_point)
    angle_s, angle_e = min(angle_s, angle_e), max(angle_s, angle_e)
    if not angle_s < angle_m < angle_e:
        angle_s, angle_e = angle_e - TWO_PI, angle_s
    return angle_s, angle_e


def arc_bbox(start_point: Vec2, mid_point: Vec2, end_point: Vec2,
             center: Vec2, radius: float) -> Tuple[float, float, float, float]:
    """Exact arc bounding box ``(min_x, min_y, max_x, max_y)`` (GenCAD ``Arc.bbox``).

    Beyond the two endpoints, each cardinal extreme point (``center +/- radius`` on
    each axis) is included when the arc actually sweeps through the corresponding
    angle -- so the bulge is captured, unlike a chord-endpoint-only box.
    """
    angle_s, angle_e = arc_angles_counterclockwise(start_point, mid_point,
                                                   end_point, center)
    cx, cy = center
    points: List[Vec2] = [tuple(map(float, start_point)),
                          tuple(map(float, end_point))]
    if angle_s < 0.0 < angle_e:
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


# --- uniform point sampling (GenCAD ``sample_points``) -----------------------
def sample_line_points(start_point: Vec2, end_point: Vec2, n: int = 32) -> List[Vec2]:
    """``n`` points from start to end inclusive (``linspace`` semantics)."""
    if n < 1:
        raise ValueError("n must be >= 1")
    if n == 1:
        return [tuple(map(float, start_point))]
    out = []
    for i in range(n):
        t = i / (n - 1)
        out.append((start_point[0] + (end_point[0] - start_point[0]) * t,
                    start_point[1] + (end_point[1] - start_point[1]) * t))
    return out


def sample_arc_points(start_point: Vec2, mid_point: Vec2, end_point: Vec2,
                      center: Vec2, radius: float, n: int = 32) -> List[Vec2]:
    """``n`` points along the arc, endpoints included (GenCAD ``Arc.sample_points``)."""
    if n < 1:
        raise ValueError("n must be >= 1")
    angle_s, angle_e = arc_angles_counterclockwise(start_point, mid_point,
                                                   end_point, center)
    if n == 1:
        angles = [angle_s]
    else:
        step = (angle_e - angle_s) / (n - 1)
        angles = [angle_s + step * i for i in range(n)]
    return [(math.cos(a) * radius + center[0], math.sin(a) * radius + center[1])
            for a in angles]


def sample_circle_points(center: Vec2, radius: float, n: int = 32) -> List[Vec2]:
    """``n`` points around a full circle, ``endpoint=False`` (GenCAD ``Circle``)."""
    if n < 1:
        raise ValueError("n must be >= 1")
    return [(math.cos(TWO_PI * i / n) * radius + center[0],
             math.sin(TWO_PI * i / n) * radius + center[1]) for i in range(n)]


def circle_bbox(center: Vec2, radius: float) -> Tuple[float, float, float, float]:
    """``center +/- radius`` box (GenCAD ``Circle.bbox``)."""
    return (center[0] - radius, center[1] - radius,
            center[0] + radius, center[1] + radius)


def line_bbox(start_point: Vec2, end_point: Vec2) -> Tuple[float, float, float, float]:
    """Endpoint box (GenCAD ``Line.bbox``)."""
    return (min(start_point[0], end_point[0]), min(start_point[1], end_point[1]),
            max(start_point[0], end_point[0]), max(start_point[1], end_point[1]))


def circle_start_point(center: Vec2, radius: float) -> Vec2:
    """GenCAD's convention: a circle "starts" at its left-most point."""
    return (center[0] - radius, center[1])


def circle_end_point(center: Vec2, radius: float) -> Vec2:
    """GenCAD's convention: a circle "ends" at its right-most point."""
    return (center[0] + radius, center[1])
