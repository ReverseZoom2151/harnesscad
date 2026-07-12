"""Primitive-as-point representation for symbol spotting (SymPoint, ECCV 2024).

SymPoint reframes CAD symbol spotting as *point-cloud* panoptic segmentation:
every graphic primitive of a vectorised floor plan (line / arc / circle /
ellipse) collapses to a single point carrying a small feature vector.  The
deterministic front end (``parse_svg.py`` in the reference repo) is what this
module reimplements, stdlib-only and without ``svgpathtools``:

* :func:`sample_args` -- the four *anchor* samples of a primitive.  Paths are
  sampled at parameters ``0, 1/3, 2/3, 1``; closed circles and ellipses are
  sampled at the four cardinal angles ``0, pi/2, pi, 3pi/2`` (with the ellipse
  written major-axis-first, exactly as the reference does).  The resulting
  8-tuple ``(x1,y1,...,x4,y4)`` is SymPoint's ``args`` record.
* :func:`primitive_length` -- arc length: ``2*pi*r`` for a circle, the
  ``2*pi*b + 4*(a-b)`` ellipse approximation, ``r*|sweep|`` for an arc.
* :func:`primitive_point` -- the point a primitive becomes: the *mean of its
  four anchors* (SymPoint uses ``mean(args[0::2]), mean(args[1::2])``), which
  is emphatically not the segment midpoint used by endpoint-graph methods such
  as ``drawings.cadtransformer_primitive_graph``.
* :func:`instance_boxes` -- ground-truth aggregation: an instance's bounding
  box and centre are computed from the *anchors of its member primitives*, so
  no re-evaluation of the curves is needed.

This is the representation half of SymPoint; the feature vector built on top of
it lives in :mod:`drawings.sympoint_point_features`.
"""

from __future__ import annotations

import math
from typing import Dict, List, Sequence, Tuple

LINE = "line"
ARC = "arc"
CIRCLE = "circle"
ELLIPSE = "ellipse"

#: Command order of the reference implementation (``COMMANDS`` in parse_svg.py).
COMMANDS: Tuple[str, ...] = (LINE, ARC, CIRCLE, ELLIPSE)

#: Path sample parameters and closed-curve sample angles.
PATH_PARAMS: Tuple[float, ...] = (0.0, 1.0 / 3.0, 2.0 / 3.0, 1.0)
CARDINAL_ANGLES: Tuple[float, ...] = (0.0, math.pi / 2.0, math.pi, 3.0 * math.pi / 2.0)

Point = Tuple[float, float]
Args = Tuple[float, float, float, float, float, float, float, float]


class Primitive:
    """A vector primitive.

    ``kind`` is one of :data:`COMMANDS`; ``geom`` holds

    * ``line``:    ``(x1, y1, x2, y2)``
    * ``arc``:     ``(cx, cy, r, start_angle, end_angle)`` -- angles in radians,
      the arc sweeps from ``start_angle`` to ``end_angle``
    * ``circle``:  ``(cx, cy, r)``
    * ``ellipse``: ``(cx, cy, rx, ry)``
    """

    __slots__ = ("kind", "geom")

    def __init__(self, kind: str, geom: Sequence[float]) -> None:
        if kind not in COMMANDS:
            raise ValueError("unknown primitive kind: %r" % (kind,))
        expected = {LINE: 4, ARC: 5, CIRCLE: 3, ELLIPSE: 4}[kind]
        geom = tuple(float(v) for v in geom)
        if len(geom) != expected:
            raise ValueError("%s expects %d parameters, got %d" % (kind, expected, len(geom)))
        if kind in (CIRCLE, ARC) and geom[2] < 0:
            raise ValueError("radius must be non-negative")
        if kind == ELLIPSE and (geom[2] < 0 or geom[3] < 0):
            raise ValueError("radii must be non-negative")
        self.kind = kind
        self.geom = geom

    def __repr__(self) -> str:  # pragma: no cover - debug helper
        return "Primitive(%r, %r)" % (self.kind, self.geom)


def command_id(kind: str) -> int:
    """Index of ``kind`` in the SymPoint command vocabulary."""
    return COMMANDS.index(kind)


def ellipse_axes(rx: float, ry: float) -> Tuple[float, float]:
    """Return ``(a, b)`` with ``a`` the major and ``b`` the minor radius."""
    return (rx, ry) if rx > ry else (ry, rx)


def evaluate(prim: Primitive, t: float) -> Point:
    """Point on ``prim`` at normalised parameter ``t`` in ``[0, 1]``."""
    if t < 0.0 or t > 1.0:
        raise ValueError("t must lie in [0, 1]")
    kind, g = prim.kind, prim.geom
    if kind == LINE:
        x1, y1, x2, y2 = g
        return (x1 + (x2 - x1) * t, y1 + (y2 - y1) * t)
    if kind == ARC:
        cx, cy, r, a0, a1 = g
        ang = a0 + (a1 - a0) * t
        return (cx + r * math.cos(ang), cy + r * math.sin(ang))
    if kind == CIRCLE:
        cx, cy, r = g
        ang = 2.0 * math.pi * t
        return (cx + r * math.cos(ang), cy + r * math.sin(ang))
    cx, cy, rx, ry = g
    a, b = ellipse_axes(rx, ry)
    ang = 2.0 * math.pi * t
    return (cx + a * math.cos(ang), cy + b * math.sin(ang))


def sample_args(prim: Primitive) -> Args:
    """The four SymPoint anchors of ``prim``, flattened to ``(x1,y1,...,x4,y4)``.

    Open primitives (line, arc) are sampled at ``0, 1/3, 2/3, 1``; closed ones
    (circle, ellipse) at the four cardinal angles.
    """
    kind, g = prim.kind, prim.geom
    out: List[float] = []
    if kind in (CIRCLE, ELLIPSE):
        if kind == CIRCLE:
            cx, cy, r = g
            a = b = r
        else:
            cx, cy, rx, ry = g
            a, b = ellipse_axes(rx, ry)
        for theta in CARDINAL_ANGLES:
            out.extend((cx + a * math.cos(theta), cy + b * math.sin(theta)))
    else:
        for t in PATH_PARAMS:
            x, y = evaluate(prim, t)
            out.extend((x, y))
    return tuple(out)  # type: ignore[return-value]


def primitive_length(prim: Primitive) -> float:
    """Arc length of ``prim`` (SymPoint ``lengths``)."""
    kind, g = prim.kind, prim.geom
    if kind == LINE:
        x1, y1, x2, y2 = g
        return math.hypot(x2 - x1, y2 - y1)
    if kind == ARC:
        _, _, r, a0, a1 = g
        return r * abs(a1 - a0)
    if kind == CIRCLE:
        return 2.0 * math.pi * g[2]
    a, b = ellipse_axes(g[2], g[3])
    return 2.0 * math.pi * b + 4.0 * (a - b)


def anchor_points(args: Sequence[float]) -> List[Point]:
    """Split a flat ``args`` record into its four ``(x, y)`` anchors."""
    if len(args) != 8:
        raise ValueError("args must hold 8 values (4 anchor points)")
    return [(float(args[2 * i]), float(args[2 * i + 1])) for i in range(4)]


def primitive_point(args: Sequence[float]) -> Point:
    """The point a primitive becomes: the mean of its four anchors."""
    pts = anchor_points(args)
    return (sum(p[0] for p in pts) / 4.0, sum(p[1] for p in pts) / 4.0)


def primitive_record(prim: Primitive) -> Dict[str, object]:
    """Full deterministic record of one primitive: args, length, command, point."""
    args = sample_args(prim)
    return {
        "command": command_id(prim.kind),
        "args": args,
        "length": primitive_length(prim),
        "point": primitive_point(args),
    }


def to_point_set(primitives: Sequence[Primitive]) -> Dict[str, List[object]]:
    """Convert primitives to SymPoint's parallel-array point set."""
    commands: List[int] = []
    args: List[Args] = []
    lengths: List[float] = []
    points: List[Point] = []
    for prim in primitives:
        rec = primitive_record(prim)
        commands.append(rec["command"])  # type: ignore[arg-type]
        args.append(rec["args"])  # type: ignore[arg-type]
        lengths.append(rec["length"])  # type: ignore[arg-type]
        points.append(rec["point"])  # type: ignore[arg-type]
    return {"commands": commands, "args": args, "lengths": lengths, "points": points}


def instance_boxes(args: Sequence[Sequence[float]], instance_ids: Sequence[int],
                   semantic_ids: Sequence[int]) -> List[Dict[str, object]]:
    """Bounding box + centre of every labelled instance, from primitive anchors.

    Primitives with ``instance_id < 0`` (stuff / background) are skipped.  The
    result is sorted by ``(instance_id, semantic_id)`` for determinism.
    """
    if not (len(args) == len(instance_ids) == len(semantic_ids)):
        raise ValueError("args, instance_ids and semantic_ids must be the same length")
    grouped: Dict[Tuple[int, int], List[Point]] = {}
    for arg, ins, sem in zip(args, instance_ids, semantic_ids):
        if ins < 0:
            continue
        grouped.setdefault((int(ins), int(sem)), []).extend(anchor_points(arg))
    out: List[Dict[str, object]] = []
    for (ins, sem) in sorted(grouped):
        pts = grouped[(ins, sem)]
        x1 = min(p[0] for p in pts)
        y1 = min(p[1] for p in pts)
        x2 = max(p[0] for p in pts)
        y2 = max(p[1] for p in pts)
        out.append({
            "instance_id": ins,
            "semantic_id": sem,
            "box": (x1, y1, x2, y2),
            "center": ((x1 + x2) / 2.0, (y1 + y2) / 2.0),
        })
    return out
