"""Sketch constraint-satisfaction checker (HistCAD, Dong et al. 2026, Table 3).

HistCAD argues that what makes a parametric CAD sequence *reusable* is its explicit
sketch constraints: when a dimension is edited, the native solver re-resolves the model
so required relations still hold. HistCAD's representation encodes **19 constraint
types** (Table 3): coincident, horizontal, vertical, parallel, perpendicular,
concentric, tangent, normal, length, distance, diameter, radius, angle, minor-radius,
major-radius, fix, midpoint, equal, and mirror.

Existing ``sketch.constraints`` is Diffusion-CAD's integer *repair* equations for a
different (axis-aligned) subset; this module is HistCAD's **evaluation** side: given a
concrete sketch geometry, decide for each constraint whether it is currently satisfied
within tolerance. That predicate is exactly what HistCAD's Constraint-Aware Editability
Benchmark (see :mod:`harnesscad.eval.quality.edit.editability`) needs to measure whether
required relations *survive* a parameter edit -- the "closure only vs. with constraints"
distinction in the paper's Fig. 1.

Geometry model (pure numbers, no kernel):

    * a **point** is ``(x, y)``.
    * a **line** is ``((x1, y1), (x2, y2))``.
    * a **circle/arc** is ``dict(center=(cx, cy), radius=r)``.

Each constraint is a small dataclass; :func:`satisfies` dispatches on its ``kind`` and
returns a bool. Everything is closed-form Euclidean geometry -- deterministic.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, Sequence, Tuple

__all__ = [
    "CONSTRAINT_TYPES",
    "Constraint",
    "satisfies",
    "satisfaction_rate",
]

# The 19 HistCAD constraint types (Table 3), lowercased & underscored.
CONSTRAINT_TYPES: Tuple[str, ...] = (
    "coincident",
    "horizontal",
    "vertical",
    "parallel",
    "perpendicular",
    "concentric",
    "tangent",
    "normal",
    "length",
    "distance",
    "diameter",
    "radius",
    "angle",
    "minor_radius",
    "major_radius",
    "fix",
    "midpoint",
    "equal",
    "mirror",
)

Point = Tuple[float, float]


@dataclass(frozen=True)
class Constraint:
    """One sketch constraint.

    ``kind`` is one of :data:`CONSTRAINT_TYPES`. ``entities`` holds the geometry the
    constraint applies to (points/lines/circles, per-kind). ``value`` holds a target
    scalar for dimensional kinds (length/distance/diameter/radius/angle degrees).
    """

    kind: str
    entities: Sequence[Any] = ()
    value: float = 0.0


def satisfies(c: Constraint, *, atol: float = 1e-6, ang_atol_deg: float = 1e-3) -> bool:
    """Return True iff constraint ``c`` holds within tolerance."""
    k = c.kind
    e = c.entities
    if k not in _DISPATCH:
        raise ValueError(f"unknown constraint kind {k!r}")
    return _DISPATCH[k](e, c.value, atol, ang_atol_deg)


def satisfaction_rate(constraints: Sequence[Constraint], **kw) -> float:
    """Fraction of ``constraints`` currently satisfied (0.0 for empty)."""
    if not constraints:
        return 0.0
    ok = sum(1 for c in constraints if satisfies(c, **kw))
    return ok / len(constraints)


# --- geometry helpers ------------------------------------------------------
def _dist(p: Point, q: Point) -> float:
    return math.hypot(q[0] - p[0], q[1] - p[1])


def _dir(line) -> Point:
    (x1, y1), (x2, y2) = line
    return (x2 - x1, y2 - y1)


def _norm(v: Point) -> float:
    return math.hypot(v[0], v[1])


def _angle_between_deg(u: Point, v: Point) -> float:
    nu, nv = _norm(u), _norm(v)
    if nu == 0 or nv == 0:
        return 0.0
    c = (u[0] * v[0] + u[1] * v[1]) / (nu * nv)
    c = max(-1.0, min(1.0, c))
    return math.degrees(math.acos(abs(c)))  # unsigned 0..90 for line pairs


def _circle(c: Dict) -> Tuple[Point, float]:
    return tuple(c["center"]), float(c["radius"])


# --- per-kind predicates ---------------------------------------------------
def _coincident(e, v, atol, aatol):
    p, q = e
    return _dist(p, q) <= atol


def _horizontal(e, v, atol, aatol):
    (x1, y1), (x2, y2) = e[0]
    return abs(y1 - y2) <= atol


def _vertical(e, v, atol, aatol):
    (x1, y1), (x2, y2) = e[0]
    return abs(x1 - x2) <= atol


def _parallel(e, v, atol, aatol):
    return _angle_between_deg(_dir(e[0]), _dir(e[1])) <= aatol


def _perpendicular(e, v, atol, aatol):
    return abs(_angle_between_deg(_dir(e[0]), _dir(e[1])) - 90.0) <= aatol


def _concentric(e, v, atol, aatol):
    (c1, _), (c2, _) = _circle(e[0]), _circle(e[1])
    return _dist(c1, c2) <= atol


def _tangent(e, v, atol, aatol):
    # Two circles tangent: center distance == r1+r2 (external) or |r1-r2| (internal).
    (c1, r1), (c2, r2) = _circle(e[0]), _circle(e[1])
    d = _dist(c1, c2)
    return abs(d - (r1 + r2)) <= atol or abs(d - abs(r1 - r2)) <= atol


def _normal(e, v, atol, aatol):
    # Alias of perpendicular for line/line in HistCAD's set.
    return _perpendicular(e, v, atol, aatol)


def _length(e, v, atol, aatol):
    return abs(_dist(e[0][0], e[0][1]) - v) <= atol


def _distance(e, v, atol, aatol):
    p, q = e
    return abs(_dist(p, q) - v) <= atol


def _diameter(e, v, atol, aatol):
    _, r = _circle(e[0])
    return abs(2 * r - v) <= atol


def _radius(e, v, atol, aatol):
    _, r = _circle(e[0])
    return abs(r - v) <= atol


def _angle(e, v, atol, aatol):
    return abs(_angle_between_deg(_dir(e[0]), _dir(e[1])) - v) <= aatol


def _minor_radius(e, v, atol, aatol):
    # Ellipse minor radius target.
    return abs(float(e[0]["minor_radius"]) - v) <= atol


def _major_radius(e, v, atol, aatol):
    return abs(float(e[0]["major_radius"]) - v) <= atol


def _fix(e, v, atol, aatol):
    # Point pinned to a target location e = (point, target).
    p, target = e
    return _dist(p, target) <= atol


def _midpoint(e, v, atol, aatol):
    # e = (m, a, b): m is the midpoint of segment a-b.
    m, a, b = e
    mid = ((a[0] + b[0]) / 2.0, (a[1] + b[1]) / 2.0)
    return _dist(m, mid) <= atol


def _equal(e, v, atol, aatol):
    # Two lines of equal length, or two circles of equal radius.
    x, y = e
    if isinstance(x, dict):
        return abs(_circle(x)[1] - _circle(y)[1]) <= atol
    return abs(_dist(x[0], x[1]) - _dist(y[0], y[1])) <= atol


def _mirror(e, v, atol, aatol):
    # e = (p, q, axis): p and q are reflections of each other across line ``axis``.
    p, q, axis = e
    return _dist(_reflect(p, axis), q) <= atol


def _reflect(p: Point, axis) -> Point:
    (ax, ay), (bx, by) = axis
    dx, dy = bx - ax, by - ay
    d2 = dx * dx + dy * dy
    if d2 == 0:
        return p
    t = ((p[0] - ax) * dx + (p[1] - ay) * dy) / d2
    foot = (ax + t * dx, ay + t * dy)
    return (2 * foot[0] - p[0], 2 * foot[1] - p[1])


_DISPATCH = {
    "coincident": _coincident,
    "horizontal": _horizontal,
    "vertical": _vertical,
    "parallel": _parallel,
    "perpendicular": _perpendicular,
    "concentric": _concentric,
    "tangent": _tangent,
    "normal": _normal,
    "length": _length,
    "distance": _distance,
    "diameter": _diameter,
    "radius": _radius,
    "angle": _angle,
    "minor_radius": _minor_radius,
    "major_radius": _major_radius,
    "fix": _fix,
    "midpoint": _midpoint,
    "equal": _equal,
    "mirror": _mirror,
}

assert set(_DISPATCH) == set(CONSTRAINT_TYPES)
