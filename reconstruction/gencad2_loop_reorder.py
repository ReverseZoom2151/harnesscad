"""Exact GenCAD/DeepCAD loop + profile canonical ordering (``cadlib/sketch.py``).

The GenCAD reference implementation canonicalises every sketch *before* it is
vectorised, so that one geometry always maps to one command sequence. The exact
procedure (``SketchBase.reorder`` for ``Loop`` and ``Profile``) has three steps the
paper-level modules do not carry:

1. **Endpoint-orientation repair.** Curves arrive from the raw JSON with arbitrary
   start/end order. The first curve is reversed if it *starts* where the second one
   starts or ends; thereafter, curve ``i+1`` is reversed whenever its end coincides
   with curve ``i``'s end. Only after this does the chain read start->end->start.
2. **Left-most rotation.** The loop is cyclically rotated to begin at the curve whose
   start point is left-most, ties broken by the lower ``y`` -- i.e. ordering on
   ``(x, y)`` rounded to 6 decimals.
3. **Counter-clockwise enforcement.** If ``cross(dir_out(last), dir_in(first)) <= 0``
   the loop runs clockwise: *every curve is reversed and the curve list is
   reversed*. Loops whose first or last curve is a circle are exempt (a lone circle
   has no meaningful winding in this encoding).

``Profile.reorder`` then sorts loops by the ``(x, y)`` of their bounding-box minimum,
placing the outer-most (left/bottom-most) loop first.

This differs from ``reconstruction.deepcad_profile_assembly.canonical_loop``, which
rotates on ``(y, x)`` and performs neither the orientation repair nor the CCW
reversal. Pure standard library, deterministic.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import List, Optional, Sequence, Tuple

from geometry.gencad2_arc_vector import (
    arc_bbox,
    circle_bbox,
    circle_end_point,
    circle_start_point,
    line_bbox,
)

Vec2 = Tuple[float, float]

LINE = "Line"
ARC = "Arc"
CIRCLE = "Circle"

ROUND_DIGITS = 6  # the reference compares start points at 6-decimal precision


@dataclass(frozen=True)
class Curve:
    """A resolved sketch curve: line, arc (with mid-point) or circle."""

    kind: str
    start: Vec2 = (0.0, 0.0)
    end: Vec2 = (0.0, 0.0)
    mid: Optional[Vec2] = None
    center: Optional[Vec2] = None
    radius: float = 0.0

    def __post_init__(self) -> None:
        if self.kind not in (LINE, ARC, CIRCLE):
            raise ValueError("unknown curve kind: {}".format(self.kind))
        if self.kind == ARC and self.mid is None:
            raise ValueError("an Arc requires a mid point")
        if self.kind == CIRCLE and self.center is None:
            raise ValueError("a Circle requires a center")


def circle(center: Vec2, radius: float) -> Curve:
    """Build a circle curve with GenCAD's implicit left/right start/end points."""
    return Curve(kind=CIRCLE, start=circle_start_point(center, radius),
                 end=circle_end_point(center, radius),
                 center=center, radius=radius)


def reverse_curve(c: Curve) -> Curve:
    """Swap start and end (a circle is unaffected -- ``Circle.reverse`` is a no-op)."""
    if c.kind == CIRCLE:
        return c
    return replace(c, start=c.end, end=c.start)


def curve_direction(c: Curve, from_start: bool = True) -> Vec2:
    """Tangent-ish direction vector used by the winding test.

    Line: ``end - start``. Arc: ``mid - start`` (in) or ``end - mid`` (out).
    Circle: ``center - start``.
    """
    if c.kind == LINE:
        a, b = c.start, c.end
    elif c.kind == ARC:
        a, b = (c.start, c.mid) if from_start else (c.mid, c.end)
    else:
        a, b = c.start, c.center
    return (b[0] - a[0], b[1] - a[1])


def curve_bbox(c: Curve) -> Tuple[float, float, float, float]:
    """Exact bounding box of a curve (arc bulge included)."""
    if c.kind == LINE:
        return line_bbox(c.start, c.end)
    if c.kind == CIRCLE:
        return circle_bbox(c.center, c.radius)
    return arc_bbox(c.start, c.mid, c.end, c.center, c.radius)


def loop_bbox(curves: Sequence[Curve]) -> Tuple[float, float, float, float]:
    """Axis-aligned box of a whole loop (union of its curve boxes)."""
    if not curves:
        raise ValueError("empty loop has no bounding box")
    boxes = [curve_bbox(c) for c in curves]
    return (min(b[0] for b in boxes), min(b[1] for b in boxes),
            max(b[2] for b in boxes), max(b[3] for b in boxes))


def loop_bbox_size(curves: Sequence[Curve]) -> float:
    """GenCAD ``SketchBase.bbox_size``: max |bbox corner - start_point| over x and y.

    The sketch "size" is measured *relative to the loop's start point*, not as the
    box diagonal or width -- this is the ``s`` slot of the Ext command.
    """
    min_x, min_y, max_x, max_y = loop_bbox(curves)
    sx, sy = curves[0].start
    return max(abs(max_x - sx), abs(max_y - sy), abs(min_x - sx), abs(min_y - sy))


def _key(p: Vec2) -> Tuple[float, float]:
    return (round(p[0], ROUND_DIGITS), round(p[1], ROUND_DIGITS))


def _close(a: Vec2, b: Vec2, tol: float = 1e-8) -> bool:
    return abs(a[0] - b[0]) <= tol and abs(a[1] - b[1]) <= tol


def repair_orientation(curves: Sequence[Curve]) -> List[Curve]:
    """Step 1: make consecutive curves chain end -> start (GenCAD's reverse fixes)."""
    out = list(curves)
    if len(out) <= 1:
        return out
    if _close(out[0].start, out[1].start) or _close(out[0].start, out[1].end):
        out[0] = reverse_curve(out[0])
    for i in range(len(out) - 1):
        if _close(out[i].end, out[i + 1].end):
            out[i + 1] = reverse_curve(out[i + 1])
    return out


def leftmost_index(curves: Sequence[Curve]) -> int:
    """Step 2's pivot: index of the curve with the left-most (then lowest) start."""
    best = 0
    best_key = _key(curves[0].start)
    for i, c in enumerate(curves):
        k = _key(c.start)
        if k < best_key:
            best, best_key = i, k
    return best


def is_counter_clockwise(curves: Sequence[Curve]) -> bool:
    """GenCAD's winding test: ``cross(dir_out(last), dir_in(first)) > 0``."""
    start_vec = curve_direction(curves[0], from_start=True)
    end_vec = curve_direction(curves[-1], from_start=False)
    cross = end_vec[0] * start_vec[1] - end_vec[1] * start_vec[0]
    return cross > 0


def reorder_loop(curves: Sequence[Curve]) -> List[Curve]:
    """Full ``Loop.reorder``: orientation repair, left-most rotation, CCW enforcement."""
    out = repair_orientation(curves)
    if len(out) <= 1:
        return out

    pivot = leftmost_index(out)
    out = out[pivot:] + out[:pivot]

    # A loop bounded by a circle is left as-is (the reference's hard-coded guard).
    if out[0].kind == CIRCLE or out[-1].kind == CIRCLE:
        return out

    if not is_counter_clockwise(out):
        out = [reverse_curve(c) for c in out]
        out.reverse()
    return out


def reorder_profile(loops: Sequence[Sequence[Curve]]) -> List[List[Curve]]:
    """``Profile.reorder``: sort loops by their bbox minimum, ``x`` first then ``y``."""
    if len(loops) <= 1:
        return [list(lp) for lp in loops]
    keyed = []
    for lp in loops:
        min_x, min_y, _, _ = loop_bbox(lp)
        keyed.append(((round(min_x, ROUND_DIGITS), round(min_y, ROUND_DIGITS)),
                      list(lp)))
    keyed.sort(key=lambda kv: kv[0])
    return [lp for _, lp in keyed]


def canonicalize_profile(loops: Sequence[Sequence[Curve]]) -> List[List[Curve]]:
    """Reorder every loop, then sort the loops -- the exact pre-vectorisation form."""
    return reorder_profile([reorder_loop(lp) for lp in loops])


def loop_is_closed(curves: Sequence[Curve], tol: float = 1e-6) -> bool:
    """True when the chain closes: ``end(Ci) == start(Ci+1)`` and ``end(Cn) == start(C1)``."""
    if not curves:
        return False
    if len(curves) == 1:
        return curves[0].kind == CIRCLE or _close(curves[0].start, curves[0].end, tol)
    for i in range(len(curves)):
        nxt = curves[(i + 1) % len(curves)]
        if not _close(curves[i].end, nxt.start, tol):
            return False
    return True
