"""Procedural serpentine (meander) spring generator.

From Séquin, *Interactive Procedural Computer-Aided Design*, Section 2.1 (MEMS
resonator suspension). The paper replaces the crooked, asymmetric legs produced
by a genetic-algorithm search with a *specially coded serpentine element* built
from "two or more regular rectilinear hairpin turns" whose free parameters are
just the placement, orientation and a handful of geometric sizes. It further
notes that adding *cross braces between symmetrical pairs of serpentines*
prevents the flare-out that hurts the stiffness ratio.

This module implements the deterministic geometry of that procedural primitive:

* :func:`serpentine_polyline` -- a rectilinear boustrophedon (meander) polyline
  with ``n_turns`` hairpin turns, parameterised by the horizontal beam
  ``length`` and the vertical ``pitch`` between beams;
* :func:`serpentine_segments` -- the polyline as a tuple of segments;
* :func:`wire_length`, :func:`bounding_box`, :func:`beam_endpoints`;
* :func:`cross_brace` / :func:`cross_braces` -- straight connectors between the
  corresponding vertices of two parallel (mirror-image) serpentines.

Everything is pure stdlib and deterministic; no wall clock, no randomness.
"""

from __future__ import annotations

from typing import List, Sequence, Tuple

Point = Tuple[float, float]
Segment = Tuple[Point, Point]


def serpentine_polyline(
    n_turns: int,
    length: float,
    pitch: float,
    origin: Point = (0.0, 0.0),
    *,
    start_dir: int = 1,
) -> Tuple[Point, ...]:
    """Vertices of a rectilinear serpentine spring.

    The spring alternates horizontal beams (of length ``length``) with short
    vertical risers (of height ``pitch``). ``n_turns`` counts the hairpin turns;
    the spring therefore contains ``n_turns + 1`` horizontal beams. ``start_dir``
    is ``+1`` to run the first beam in the ``+x`` direction, ``-1`` for ``-x``.

    Raises ``ValueError`` for a non-positive turn count or size.
    """
    if n_turns < 1:
        raise ValueError("n_turns must be >= 1 (a serpentine needs a turn)")
    if length <= 0 or pitch <= 0:
        raise ValueError("length and pitch must be positive")
    if start_dir not in (1, -1):
        raise ValueError("start_dir must be +1 or -1")

    x, y = float(origin[0]), float(origin[1])
    pts: List[Point] = [(x, y)]
    direction = start_dir
    n_beams = n_turns + 1
    for beam in range(n_beams):
        # horizontal beam
        x = x + direction * length
        pts.append((x, y))
        # riser (skip the trailing riser after the last beam)
        if beam < n_beams - 1:
            y = y + pitch
            pts.append((x, y))
            direction = -direction
    return tuple(pts)


def serpentine_segments(polyline: Sequence[Point]) -> Tuple[Segment, ...]:
    """Consecutive-vertex segments of a polyline."""
    if len(polyline) < 2:
        raise ValueError("need at least two points")
    return tuple((polyline[i], polyline[i + 1]) for i in range(len(polyline) - 1))


def wire_length(polyline: Sequence[Point]) -> float:
    """Total developed length of the meander wire."""
    total = 0.0
    for (ax, ay), (bx, by) in serpentine_segments(polyline):
        total += ((bx - ax) ** 2 + (by - ay) ** 2) ** 0.5
    return total


def bounding_box(polyline: Sequence[Point]) -> Tuple[Point, Point]:
    """Axis-aligned bounding box as ((xmin, ymin), (xmax, ymax))."""
    if not polyline:
        raise ValueError("empty polyline")
    xs = [p[0] for p in polyline]
    ys = [p[1] for p in polyline]
    return (min(xs), min(ys)), (max(xs), max(ys))


def beam_endpoints(polyline: Sequence[Point]) -> Tuple[Point, ...]:
    """The outer vertices of the horizontal beams (the extreme x turns).

    These are the natural attachment points for a cross brace.
    """
    box = bounding_box(polyline)
    xmin, xmax = box[0][0], box[1][0]
    tol = 1e-9 * (abs(xmax - xmin) + 1.0)
    return tuple(p for p in polyline if abs(p[0] - xmin) <= tol or abs(p[0] - xmax) <= tol)


def mirror_x(polyline: Sequence[Point], axis_x: float) -> Tuple[Point, ...]:
    """Reflect a polyline across the vertical line ``x = axis_x``.

    Produces the mirror-image partner of a symmetrical serpentine pair.
    """
    return tuple((2.0 * axis_x - x, y) for (x, y) in polyline)


def cross_brace(a: Point, b: Point) -> Segment:
    """A single straight brace connecting a vertex of one serpentine to the
    corresponding vertex of its symmetric partner."""
    return (tuple(a), tuple(b))  # type: ignore[return-value]


def cross_braces(
    left: Sequence[Point], right: Sequence[Point], indices: Sequence[int]
) -> Tuple[Segment, ...]:
    """Braces between corresponding vertices ``left[i]`` and ``right[i]``.

    ``indices`` picks which vertex pairs to brace. Raises ``ValueError`` if the
    two serpentines have different vertex counts or an index is out of range.
    """
    if len(left) != len(right):
        raise ValueError("serpentines must have matching vertex counts to brace")
    out: List[Segment] = []
    for i in indices:
        if not (0 <= i < len(left)):
            raise IndexError(f"brace index {i} out of range")
        out.append(cross_brace(left[i], right[i]))
    return tuple(out)
