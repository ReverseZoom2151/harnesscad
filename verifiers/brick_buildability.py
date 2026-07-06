"""Buildability / assembly-order analysis for brick structures (BRICKGPT).

Paper: "Generating Physically Stable and Buildable Brick Structures from Text".
A design is *buildable* if it can be assembled brick-by-brick by a human or
robot (Section 1). BRICKGPT emits bricks in a raster-scan order from bottom to
top (Section 4.1), which doubles as an assembly guide (Appendix C: "since our
method outputs a sequence of intermediate steps, it naturally serves as an
intuitive assembly guide"). Robotic assembly (Appendix B) additionally reorders
bricks so that "each intermediate structure is physically stable by itself".

This module provides the deterministic assembly-order checks:

* :func:`raster_assembly_order` -- the paper's canonical bottom-to-top raster
  order (sort by ``z``, then ``y``, then ``x``).
* :func:`is_supported_order` -- every brick is placed only after the bricks it
  studs onto are already present (no brick placed in mid-air): each prefix is a
  grounded, connected assembly.
* :func:`find_buildable_order` -- greedily grows the assembly one brick at a
  time, always adding a brick that connects to the current partial build,
  starting from baseplate bricks. Returns an order in which every intermediate
  is grounded/connected, or ``None`` if the structure has a floating island that
  can never be reached.
* :func:`is_buildable` -- convenience predicate.
* :func:`is_assembly_stable` -- optional stronger check (Appendix B): every
  intermediate partial build is physically stable, using the stability analysis.

Deterministic, stdlib only.
"""

from __future__ import annotations

from typing import Callable, Optional

from geometry.brick_structure import Brick, BrickStructure
from geometry.brick_connectivity import (
    connection_area,
    grounds,
    is_interconnected,
)


def raster_assembly_order(structure: BrickStructure) -> list[int]:
    """Canonical bottom-to-top raster order (Section 4.1): sort by z, then y, x."""
    idx = list(range(len(structure.bricks)))
    idx.sort(key=lambda i: (structure.bricks[i].z, structure.bricks[i].y, structure.bricks[i].x))
    return idx


def _supports_of(bricks, i: int) -> set[int]:
    """Indices of bricks that ``bricks[i]`` studs directly onto (below it)."""
    target = bricks[i]
    if target.z == 0:
        return set()
    out = set()
    for j, b in enumerate(bricks):
        if j != i and connection_area(b, target) > 0:
            out.add(j)
    return out


def is_supported_order(structure: BrickStructure, order: list[int]) -> bool:
    """True if ``order`` never places a brick before at least one of its supports.

    A brick may be placed once it either rests on the baseplate or has at least
    one already-placed brick directly beneath it to stud onto -- i.e. it is
    never floating in mid-air at the moment of placement.
    """
    bricks = structure.bricks
    if sorted(order) != list(range(len(bricks))):
        raise ValueError("order must be a permutation of all brick indices")
    placed: set[int] = set()
    for i in order:
        b = bricks[i]
        if not grounds(b):
            supports = _supports_of(bricks, i)
            if not (supports & placed):
                return False
        placed.add(i)
    return True


def find_buildable_order(structure: BrickStructure) -> Optional[list[int]]:
    """Greedily find an assembly order where every intermediate build is connected.

    Grows the assembly from the baseplate bricks, at each step adding the
    lowest-index still-connectable brick (deterministic). Returns ``None`` if
    some brick is a floating island unreachable from the baseplate.
    """
    bricks = structure.bricks
    n = len(bricks)
    if n == 0:
        return []
    supports = [_supports_of(bricks, i) for i in range(n)]
    placed: set[int] = set()
    order: list[int] = []
    # Candidates start as all baseplate bricks.
    while len(order) < n:
        progressed = False
        for i in range(n):
            if i in placed:
                continue
            if grounds(bricks[i]) or (supports[i] & placed):
                order.append(i)
                placed.add(i)
                progressed = True
        if not progressed:
            return None  # remaining bricks form an unreachable floating island
    return order


def is_buildable(structure: BrickStructure) -> bool:
    """True if the structure can be assembled brick-by-brick from the baseplate.

    Equivalent to being fully interconnected and grounded (no floating island):
    a valid, support-respecting assembly order then exists.
    """
    if len(structure.bricks) == 0:
        return True
    if not is_interconnected(structure):
        return False
    return find_buildable_order(structure) is not None


def is_assembly_stable(
    structure: BrickStructure,
    stability_predicate: Callable[[BrickStructure], bool],
    order: Optional[list[int]] = None,
) -> bool:
    """True if every intermediate partial build (Appendix B) is stable.

    ``stability_predicate`` maps a :class:`BrickStructure` to a bool (e.g.
    ``verifiers.brick_stability.is_stable``). If ``order`` is omitted, the
    canonical raster order is used. Each prefix ``[b_order[0], ..., b_order[k]]``
    must satisfy the predicate.
    """
    bricks = structure.bricks
    if order is None:
        order = raster_assembly_order(structure)
    partial: list[Brick] = []
    for i in order:
        partial.append(bricks[i])
        sub = BrickStructure(
            tuple(partial), structure.grid_h, structure.grid_w, structure.grid_d
        )
        if not stability_predicate(sub):
            return False
    return True
