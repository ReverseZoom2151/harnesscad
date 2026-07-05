"""DeepCAD loop / profile assembly from a command sequence (Wu et al., 2021).

DeepCAD's sketch representation nests as ``profile -> loops -> curves`` (paper
Sec. 3.1.1). This module is the deterministic parser that turns a flat DeepCAD
command sequence (see :mod:`reconstruction.deepcad_command_spec`) back into that
structure, reconstructing absolute 2D coordinates and applying the paper's exact
canonical orderings:

  * A **loop** ``Qi = [SOL, C1, ..., Cn]`` always starts with the indicator command
    ``SOL`` followed by its curve commands.
  * "we exclude the curve's starting position from its parameter list; each curve
    always starts from the ending point of its predecessor in the loop" -- so a
    closed loop chains: ``start(C1) = end(Cn)`` and ``start(Ci) = end(Ci-1)``. This
    module rebuilds each curve's ``(start, end)`` from the stored endpoints.
  * "We list all the curves on the loop in counter-clockwise order, beginning with
    the curve whose starting point is at the most bottom-left" -- :func:`canonical_loop`
    rotates the cyclic curve list to begin at the bottom-left-most vertex.
  * "the loops in a profile are sorted according to the bottom-left corners of their
    bounding boxes" -- :func:`sort_loops` orders loops by their bbox corner.

A profile is the run of loops that precedes an extrusion (``Ext``) command;
:func:`split_profiles` groups a whole model into ``(profile, ext_command)`` pairs.

Pure and deterministic, stdlib only. Curves are the ``Command`` objects of the
DeepCAD spec; circles are treated as standalone single-curve loops.
"""

from __future__ import annotations

from dataclasses import dataclass

from reconstruction.deepcad_command_spec import (
    Command, SOL, LINE, ARC, CIRCLE, EXT, EOS,
)

Vec2 = tuple[float, float]

# Sketch-plane origin, where the first loop's first curve starts by convention.
ORIGIN: Vec2 = (0.0, 0.0)


@dataclass(frozen=True)
class Segment:
    """A reconstructed curve with absolute 2D endpoints in the sketch frame."""

    type: str
    start: Vec2
    end: Vec2
    command: Command


# --- loop / profile splitting ----------------------------------------------
def split_loops(commands: list[Command]) -> list[list[Command]]:
    """Partition sketch commands into loops, each beginning at a ``SOL``.

    Curve commands appearing before the first ``SOL`` are tolerated as an implicit
    leading loop. ``Ext``/``EOS`` terminate the sketch region and are ignored here.
    """
    loops: list[list[Command]] = []
    current: list[Command] = []
    for cmd in commands:
        if cmd.type in (EXT, EOS):
            break
        if cmd.type == SOL:
            if current:
                loops.append(current)
            current = [cmd]
        else:
            current.append(cmd)
    if current:
        loops.append(current)
    return loops


def split_profiles(commands: list[Command]) -> list[tuple[list[list[Command]], Command]]:
    """Group a model into ``(loops, ext_command)`` profile/extrusion pairs.

    Every run of sketch commands terminated by an ``Ext`` forms one profile bound to
    that extrusion. A trailing run with no closing ``Ext`` is ignored (incomplete).
    """
    profiles: list[tuple[list[list[Command]], Command]] = []
    sketch: list[Command] = []
    for cmd in commands:
        if cmd.type == EOS:
            break
        if cmd.type == EXT:
            profiles.append((split_loops(sketch), cmd))
            sketch = []
        else:
            sketch.append(cmd)
    return profiles


# --- coordinate reconstruction ---------------------------------------------
def _curves(loop: list[Command]) -> list[Command]:
    return [c for c in loop if c.type != SOL]


def curve_endpoint(cmd: Command) -> Vec2:
    """Absolute 2D endpoint stored by a Line/Arc, or the centre of a Circle."""
    return (cmd.get("x"), cmd.get("y"))


def reconstruct_segments(loop: list[Command]) -> list[Segment]:
    """Rebuild ``(start, end)`` for each curve by predecessor chaining + closure.

    For a poly-curve loop (lines/arcs), ``start(Ci) = end(Ci-1)`` and the loop closes
    so ``start(C1) = end(Cn)``. A lone circle is a self-contained loop whose start and
    end are its centre (it has no free endpoint to chain).
    """
    curves = _curves(loop)
    if not curves:
        return []
    if len(curves) == 1 and curves[0].type == CIRCLE:
        c = curves[0]
        centre = curve_endpoint(c)
        return [Segment(CIRCLE, centre, centre, c)]
    ends = [curve_endpoint(c) for c in curves]
    starts = [ends[-1]] + ends[:-1]  # closure: first starts at last curve's end
    return [Segment(c.type, s, e, c) for c, s, e in zip(curves, starts, ends)]


# --- bounding boxes ---------------------------------------------------------
def loop_bbox(loop: list[Command]) -> tuple[float, float, float, float]:
    """Axis-aligned ``(min_x, min_y, max_x, max_y)`` of a loop.

    Circles contribute their full ``centre +/- r`` extent; line/arc loops use their
    endpoint vertices (arc bulge beyond the chord is not modelled -- endpoints only).
    """
    xs: list[float] = []
    ys: list[float] = []
    for c in _curves(loop):
        cx, cy = curve_endpoint(c)
        if c.type == CIRCLE:
            r = c.get("r")
            xs += [cx - r, cx + r]
            ys += [cy - r, cy + r]
        else:
            xs.append(cx)
            ys.append(cy)
    if not xs:
        raise ValueError("empty loop has no bounding box")
    return (min(xs), min(ys), max(xs), max(ys))


# --- canonical orderings ----------------------------------------------------
def canonical_loop(loop: list[Command]) -> list[Command]:
    """Rotate a loop's curves to begin at the most bottom-left starting vertex.

    "bottom-left" ranks vertices lexicographically by ``(y, x)`` (bottom first, then
    left). The leading ``SOL`` is preserved. Circles (single-curve loops) are
    returned unchanged. The cyclic order is otherwise kept (no reversal).
    """
    curves = _curves(loop)
    if len(curves) <= 1:
        return list(loop)
    segments = reconstruct_segments(loop)
    # Index of the curve whose START vertex is most bottom-left.
    best = min(range(len(segments)),
               key=lambda i: (segments[i].start[1], segments[i].start[0]))
    rotated = curves[best:] + curves[:best]
    return [Command(SOL)] + rotated


def sort_loops(loops: list[list[Command]]) -> list[list[Command]]:
    """Sort loops by the bottom-left corner ``(min_x, min_y)`` of their bbox."""
    return sorted(loops, key=lambda lp: loop_bbox(lp)[:2])


def canonical_profile(loops: list[list[Command]]) -> list[list[Command]]:
    """Full canonicalisation: reorder every loop, then sort loops by bbox corner."""
    return sort_loops([canonical_loop(lp) for lp in loops])
