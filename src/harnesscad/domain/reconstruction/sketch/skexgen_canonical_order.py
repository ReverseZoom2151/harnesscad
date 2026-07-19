"""Canonical sketch ordering (``sort_faces`` / ``sort_loops`` / ``sort_curves``).

Before tokenisation a sketch is put into a *canonical* order so that one
geometry maps to exactly one token sequence (this is what makes the "unique %"
metric meaningful and what a topology codebook can learn):

1. **faces** are sorted by their bounding box's ``(x_min, y_min)``, increasing;
2. **loops** inside a face: the outer loop stays first, the inner loops are
   sorted by ``(x_min, y_min)``;
3. **curves** inside a loop: start from the curve whose bbox ``(x_min, y_min)``
   is smallest (bottom-left), then walk the loop through shared endpoints.  When
   the first step has two choices (both neighbours of the start curve), take the
   one whose bottom-left corner has the *larger* x, i.e. traverse in the
   increasing-x direction.  Finally, flip curves so every curve's end is the
   next curve's start.

Note the bbox used for an arc is the bbox of ``(start, mid, end)``, not the true
arc extremes; this approximation is deliberate and kept stable so the ordering is
reproducible token-for-token.

This differs from ``reconstruction/gencad2_loop_reorder``, which starts from the
leftmost *point* and enforces counter-clockwise winding. The ordering here
enforces neither winding nor a start point -- only bbox order plus connectivity.

Deterministic, stdlib only.
"""
from __future__ import annotations

from typing import Dict, List, Sequence, Tuple

Vec2 = Tuple[float, float]
Curve = Dict
Loop = List[Curve]
Face = List[Loop]
Sketch = List[Face]

TOL_DIGITS = 9


def point_key(point: Sequence[float]) -> Tuple[float, float]:
    """Vertex identity (vertices are rounded to 9 decimals)."""
    return (round(float(point[0]), TOL_DIGITS), round(float(point[1]), TOL_DIGITS))


def circle_rim_points(center: Vec2, radius: float) -> Tuple[Vec2, Vec2, Vec2, Vec2]:
    """The 4 rim points of a circle: top, bottom, right, left."""
    cx, cy = center
    return ((cx, cy + radius), (cx, cy - radius),
            (cx + radius, cy), (cx - radius, cy))


def curve_bbox(curve: Curve) -> Tuple[float, float, float, float]:
    """``(x_min, x_max, y_min, y_max)`` over the curve's defining points."""
    t = curve["type"]
    if t == "line":
        pts = [curve["start"], curve["end"]]
    elif t == "arc":
        pts = [curve["start"], curve["end"], curve["mid"]]
    elif t == "circle":
        if "pt1" in curve:
            pts = [curve["pt%d" % i] for i in (1, 2, 3, 4)]
        else:
            pts = list(circle_rim_points(curve["center"], curve["radius"]))
    else:
        raise ValueError("unknown curve type: %r" % (t,))
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    return (min(xs), max(xs), min(ys), max(ys))


def bottom_left(curve: Curve) -> Vec2:
    b = curve_bbox(curve)
    return (b[0], b[2])


def _merge_bbox(boxes: Sequence[Tuple[float, float, float, float]]):
    return (min(b[0] for b in boxes), max(b[1] for b in boxes),
            min(b[2] for b in boxes), max(b[3] for b in boxes))


def loop_bbox(loop: Loop) -> Tuple[float, float, float, float]:
    return _merge_bbox([curve_bbox(c) for c in loop])


def face_bbox(face: Face) -> Tuple[float, float, float, float]:
    return _merge_bbox([loop_bbox(l) for l in face])


def _sort_by_min_corner(items, bbox_fn):
    """Stable sort by ``(x_min, y_min)`` -- a lexicographic sort on the min corner."""
    keyed = [(bbox_fn(x)[0], bbox_fn(x)[2], i, x) for i, x in enumerate(items)]
    keyed.sort(key=lambda t: (t[0], t[1], t[2]))
    return [t[3] for t in keyed]


def sort_faces(sketch: Sketch) -> Sketch:
    return _sort_by_min_corner(list(sketch), face_bbox)


def sort_loops(face: Face) -> Face:
    """Outer loop first, inner loops sorted by their min corner."""
    if not face:
        raise ValueError("empty face")
    outer, inner = face[0], list(face[1:])
    if not inner:
        return [outer]
    return [outer] + _sort_by_min_corner(inner, loop_bbox)


def endpoints(curve: Curve) -> Tuple[Vec2, Vec2]:
    if curve["type"] == "circle":
        raise ValueError("a circle has no endpoints")
    return (point_key(curve["start"]), point_key(curve["end"]))


def flip_curve(curve: Curve) -> Curve:
    """Reverse a curve's direction (arcs keep their mid point)."""
    if curve["type"] == "circle":
        return dict(curve)
    out = dict(curve)
    out["start"], out["end"] = curve["end"], curve["start"]
    return out


def _adjacent(index: int, curves: Sequence[Curve]) -> List[int]:
    s, e = endpoints(curves[index])
    out = []
    for j, other in enumerate(curves):
        if j == index:
            continue
        os_, oe = endpoints(other)
        if os_ in (s, e) or oe in (s, e):
            out.append(j)
    return out


def sort_curves(loop: Loop) -> Loop:
    """Order the curves of a loop; returns curves whose end == next start."""
    if not loop:
        raise ValueError("empty loop")
    if len(loop) == 1:
        if loop[0]["type"] != "circle":
            raise ValueError("a single-curve loop must be a circle")
        return [dict(loop[0])]
    if any(c["type"] == "circle" for c in loop):
        raise ValueError("a circle must be the only curve of its loop")

    order = [i for i, _ in sorted(enumerate(loop),
                                  key=lambda t: (bottom_left(t[1])[0],
                                                 bottom_left(t[1])[1], t[0]))]
    sorted_idx = [order[0]]
    while True:
        cands = [j for j in _adjacent(sorted_idx[-1], loop) if j not in sorted_idx]
        if not cands:
            break
        if len(cands) > 1:
            # tie at the first step: walk in the increasing-x direction
            cands.sort(key=lambda j: (-bottom_left(loop[j])[0], j))
        sorted_idx.append(cands[0])

    if len(sorted_idx) != len(loop):
        raise ValueError("loop is not a single connected chain")

    ordered = [dict(loop[i]) for i in sorted_idx]
    return _orient_chain(ordered)


def _orient_chain(curves: Loop) -> Loop:
    """Flip curves so each curve's end is the next curve's start (closed)."""
    out = [dict(c) for c in curves]
    for i in range(len(out) - 1):
        prev, nxt = out[i], out[i + 1]
        if endpoints(prev)[1] != endpoints(nxt)[0]:
            shared = set(endpoints(prev)) & set(endpoints(nxt))
            if len(shared) == 2:                 # two-curve loop: flip the next
                out[i + 1] = flip_curve(nxt)
            elif len(shared) == 1:
                v = shared.pop()
                if endpoints(prev)[1] != v:
                    out[i] = flip_curve(prev)
                if endpoints(out[i + 1])[0] != v:
                    out[i + 1] = flip_curve(out[i + 1])
            else:
                raise ValueError("consecutive curves do not share a vertex")
    if endpoints(out[0])[0] != endpoints(out[-1])[1]:
        raise ValueError("loop does not close")
    return out


def canonicalize_sketch(sketch: Sketch) -> Sketch:
    """Full canonicalisation: faces, then loops, then curves."""
    out: Sketch = []
    for face in sort_faces(sketch):
        out.append([sort_curves(loop) for loop in sort_loops(face)])
    return out
