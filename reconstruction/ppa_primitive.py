"""Parametric sketch-primitive representation (Wang et al., "Parametric Primitive
Analysis of CAD Sketches with Vision Transformer", IEEE T-II 2024).

This is the *foundation* shared representation for CAD-sketch parametric-primitive
analysis. The paper (Sec. III, Table I) parameterises every sketch primitive by a
triple ``P_j = (tp_j, f_j, pp_j)``:

  * ``tp`` -- the primitive **type** (one of line / circle / arc / point);
  * ``f``  -- a boolean **flag** ("purpose"): whether the primitive is a physical
    entity (``True``) or serves only as a construction reference for constraints
    (``False``);
  * ``pp`` -- a fixed-length vector of 7 **parameters** with ``0`` padding.

Table I lays out the 7-slot parameter row per type::

    Type    flag  x1  y1  x2  y2  x3  y3  r/pad
    Line     f    x1  y1  x2  y2  0   0   0
    Circle   f    x1  y1  0   0   0   0   r
    Arc      f    x1  y1  x2  y2  x3  y3  0
    Point    f    x1  y1  0   0   0   0   0

A **sketch** ``S`` is an *unordered* set ``P = {P_i}`` of such primitives (the paper
stresses "there is no explicit order among primitives"). This module provides the
typed primitive, its Table-I padded-row (de)serialisation, the number of meaningful
(non-padding) parameters ``|pp_i|`` used by the paper's embedding / loss, an
order-invariant canonical form for a sketch, and deterministic point sampling of a
primitive (used downstream for Chamfer-style comparison).

Pure stdlib (``math`` only). Points are ``(x, y)`` float tuples.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable, Sequence

Point = tuple[float, float]

# Primitive type codes (paper's four types; the four Vitruvion/SketchGraphs types).
LINE = "line"
CIRCLE = "circle"
ARC = "arc"
POINT = "point"
TYPES = (LINE, CIRCLE, ARC, POINT)

# Integer type code used by the classification head / cost matrix (stable order).
TYPE_CODE = {t: i for i, t in enumerate(TYPES)}

# Number of parameter slots in the padded Table-I row.
PARAM_SLOTS = 7

# Number of *meaningful* (non-padding) parameters |pp_i| per type -- used by the
# constraint model's embedding (Num = 1 + 1 + |pp_i|) and the parameter loss.
MEANINGFUL_PARAMS = {LINE: 4, CIRCLE: 3, ARC: 6, POINT: 2}


@dataclass(frozen=True)
class Primitive:
    """A single parametric sketch primitive ``(type, flag, params[7])``.

    ``params`` is always the length-7 Table-I row (with ``0.0`` padding). Prefer the
    :func:`line` / :func:`circle` / :func:`arc` / :func:`point` constructors, which
    place coordinates into the correct slots.
    """

    ptype: str
    flag: bool
    params: tuple[float, ...]

    def __post_init__(self) -> None:
        if self.ptype not in TYPES:
            raise ValueError(f"unknown primitive type: {self.ptype!r}")
        if len(self.params) != PARAM_SLOTS:
            raise ValueError(f"params must have {PARAM_SLOTS} slots")

    # -- typed accessors ---------------------------------------------------

    @property
    def type_code(self) -> int:
        return TYPE_CODE[self.ptype]

    @property
    def meaningful(self) -> int:
        """``|pp_i|`` -- the count of non-padding parameters for this type."""
        return MEANINGFUL_PARAMS[self.ptype]

    def control_points(self) -> tuple[Point, ...]:
        """The characteristic 2D points defining the primitive.

        line -> (p1, p2); circle -> (center,); arc -> (p1, p2, p3); point -> (p1,).
        """
        p = self.params
        if self.ptype == LINE:
            return ((p[0], p[1]), (p[2], p[3]))
        if self.ptype == CIRCLE:
            return ((p[0], p[1]),)
        if self.ptype == ARC:
            return ((p[0], p[1]), (p[2], p[3]), (p[4], p[5]))
        return ((p[0], p[1]),)  # POINT

    @property
    def radius(self) -> float:
        """Circle radius (slot 6). Raises for non-circle primitives."""
        if self.ptype != CIRCLE:
            raise ValueError("radius is only defined for circle primitives")
        return self.params[6]

    # -- Table-I row (de)serialisation -------------------------------------

    def to_row(self) -> tuple[str, int, tuple[float, ...]]:
        """Return the ``(type, flag_int, params[7])`` Table-I row (flag as 0/1)."""
        return (self.ptype, 1 if self.flag else 0, tuple(float(x) for x in self.params))

    @classmethod
    def from_row(cls, ptype: str, flag: int, params: Sequence[float]) -> "Primitive":
        return cls(ptype, bool(flag), tuple(float(x) for x in params))


# -- ergonomic constructors (fill the correct Table-I slots) ----------------

def line(p1: Point, p2: Point, flag: bool = True) -> Primitive:
    return Primitive(LINE, flag, (p1[0], p1[1], p2[0], p2[1], 0.0, 0.0, 0.0))


def circle(center: Point, r: float, flag: bool = True) -> Primitive:
    return Primitive(CIRCLE, flag, (center[0], center[1], 0.0, 0.0, 0.0, 0.0, r))


def arc(p1: Point, p2: Point, p3: Point, flag: bool = True) -> Primitive:
    return Primitive(ARC, flag, (p1[0], p1[1], p2[0], p2[1], p3[0], p3[1], 0.0))


def point(p: Point, flag: bool = True) -> Primitive:
    return Primitive(POINT, flag, (p[0], p[1], 0.0, 0.0, 0.0, 0.0, 0.0))


@dataclass(frozen=True)
class Sketch:
    """An unordered set of parametric primitives ``P = {P_i}``.

    Primitives are stored as a tuple (preserving insertion order for convenience) but
    equality / :meth:`canonical` are order-invariant, reflecting the paper's set
    semantics.
    """

    primitives: tuple[Primitive, ...]

    def __init__(self, primitives: Iterable[Primitive] = ()):
        object.__setattr__(self, "primitives", tuple(primitives))

    def __len__(self) -> int:
        return len(self.primitives)

    def __iter__(self):
        return iter(self.primitives)

    def canonical(self) -> tuple:
        """Order-invariant key: sorted ``(type, flag, params)`` rows."""
        return tuple(sorted(p.to_row() for p in self.primitives))

    def __eq__(self, other) -> bool:
        return isinstance(other, Sketch) and self.canonical() == other.canonical()

    def __hash__(self) -> int:
        return hash(self.canonical())


def _arc_angles(c: Point, p1: Point, p3: Point):
    a1 = math.atan2(p1[1] - c[1], p1[0] - c[0])
    a3 = math.atan2(p3[1] - c[1], p3[0] - c[0])
    return a1, a3


def sample_primitive(prim: Primitive, n: int = 16) -> tuple[Point, ...]:
    """Deterministically sample ``n`` points along a primitive.

    Line -> ``n`` points evenly between endpoints. Circle -> ``n`` points around the
    circumference (starting at angle 0). Arc -> ``n`` points along the shorter sweep
    from ``p1`` to ``p3`` passing through the mid point ``p2`` (fitted circle centre
    from the three points). Point -> the single coordinate repeated once.
    ``n`` is clamped to at least 2 for curves.
    """
    if n < 2:
        n = 2
    cps = prim.control_points()
    if prim.ptype == POINT:
        return (cps[0],)
    if prim.ptype == LINE:
        (x1, y1), (x2, y2) = cps
        return tuple(
            (x1 + (x2 - x1) * i / (n - 1), y1 + (y2 - y1) * i / (n - 1))
            for i in range(n)
        )
    if prim.ptype == CIRCLE:
        (cx, cy) = cps[0]
        r = prim.radius
        return tuple(
            (cx + r * math.cos(2 * math.pi * i / n),
             cy + r * math.sin(2 * math.pi * i / n))
            for i in range(n)
        )
    # ARC: circle through p1, p2, p3.
    p1, p2, p3 = cps
    c, r = _circumcenter(p1, p2, p3)
    if c is None:  # degenerate / collinear -> treat as a line p1..p3
        return sample_primitive(line(p1, p3, prim.flag), n)
    a1, a3 = _arc_angles(c, p1, p3)
    a2 = math.atan2(p2[1] - c[1], p2[0] - c[0])
    # Choose the sweep direction from a1 to a3 that passes through a2.
    span = _directed_span(a1, a3, a2)
    return tuple(
        (c[0] + r * math.cos(a1 + span * i / (n - 1)),
         c[1] + r * math.sin(a1 + span * i / (n - 1)))
        for i in range(n)
    )


def _norm_angle(a: float) -> float:
    while a <= -math.pi:
        a += 2 * math.pi
    while a > math.pi:
        a -= 2 * math.pi
    return a


def _directed_span(a1: float, a3: float, through: float) -> float:
    """Signed angular sweep from ``a1`` to ``a3`` passing through ``through``."""
    ccw = _norm_angle(a3 - a1) % (2 * math.pi)  # in [0, 2pi)
    mid_ccw = _norm_angle(through - a1) % (2 * math.pi)
    if mid_ccw <= ccw:
        return ccw           # counter-clockwise sweep contains the mid point
    return ccw - 2 * math.pi  # clockwise sweep


def _circumcenter(p1: Point, p2: Point, p3: Point):
    """Centre and radius of the circle through three points; ``(None, 0)`` if collinear."""
    ax, ay = p1
    bx, by = p2
    cx, cy = p3
    d = 2 * (ax * (by - cy) + bx * (cy - ay) + cx * (ay - by))
    if abs(d) < 1e-12:
        return None, 0.0
    a2 = ax * ax + ay * ay
    b2 = bx * bx + by * by
    c2 = cx * cx + cy * cy
    ux = (a2 * (by - cy) + b2 * (cy - ay) + c2 * (ay - by)) / d
    uy = (a2 * (cx - bx) + b2 * (ax - cx) + c2 * (bx - ax)) / d
    r = math.hypot(ax - ux, ay - uy)
    return (ux, uy), r


def circumcircle(p1: Point, p2: Point, p3: Point):
    """Public wrapper: circle centre + radius through three points."""
    return _circumcenter(p1, p2, p3)
