"""Stud-into-tube connectivity graph for brick structures (BRICKGPT).

Paper: "Generating Physically Stable and Buildable Brick Structures from Text".
Buildability (Section 1) requires the structure to be "compatible with standard
interconnecting toy brick pieces" -- i.e. every brick must interlock with its
neighbours and the whole assembly must hang together as one connected piece
resting on the baseplate. Related work (Section 2) lists "ensuring that all
bricks are interconnected" as a core heuristic.

Interlocking toy bricks connect vertically: the studs on the top of a lower
brick plug into the tubes on the underside of an upper brick wherever their
footprints overlap by one or more cells. This module builds that adjacency
graph -- bricks plus a virtual ``GROUND`` node (the baseplate, which studs into
every brick sitting on layer ``z = 0``) -- and provides connectivity queries:

* connection area (number of shared studs) between two vertically-adjacent
  bricks,
* connected components of the brick graph,
* whether the structure is a single connected component, and
* whether every brick is grounded (reachable from the baseplate) -- the
  buildable, no-floating-island condition.

Deterministic, stdlib only.
"""

from __future__ import annotations

from typing import Sequence

from geometry.brick_structure import Brick, BrickStructure

GROUND = "GROUND"


def connection_area(lower: Brick, upper: Brick) -> int:
    """Number of shared studs where ``upper`` sits directly on ``lower``.

    A stud-into-tube connection exists only when ``upper`` is exactly one layer
    above ``lower`` (``upper.z == lower.z + 1``) and their footprints overlap.
    Returns the count of overlapping cells (0 = not connected).
    """
    if upper.z != lower.z + 1:
        return 0
    return len(lower.cell_set() & upper.cell_set())


def bricks_connected(a: Brick, b: Brick) -> bool:
    """True if two bricks are stud-connected (one directly above the other)."""
    return connection_area(a, b) > 0 or connection_area(b, a) > 0


def grounds(brick: Brick) -> bool:
    """True if the brick rests on the baseplate (its bottom is layer 0)."""
    return brick.z == 0


def adjacency(structure: BrickStructure) -> dict[object, set[object]]:
    """Build the undirected connectivity graph over brick indices + ``GROUND``.

    Nodes are integer brick indices, plus the string ``GROUND``. An edge means a
    stud-into-tube connection (vertical adjacency with footprint overlap), or a
    brick sitting on the baseplate.
    """
    bricks = structure.bricks
    graph: dict[object, set[object]] = {GROUND: set()}
    for i in range(len(bricks)):
        graph[i] = set()

    # Baseplate connections.
    for i, b in enumerate(bricks):
        if grounds(b):
            graph[GROUND].add(i)
            graph[i].add(GROUND)

    # Vertical brick-brick connections, bucketed by layer for efficiency.
    by_layer: dict[int, list[int]] = {}
    for i, b in enumerate(bricks):
        by_layer.setdefault(b.z, []).append(i)
    for z, lower_indices in by_layer.items():
        upper_indices = by_layer.get(z + 1, [])
        for li in lower_indices:
            lower = bricks[li]
            lcells = lower.cell_set()
            for ui in upper_indices:
                if lcells & bricks[ui].cell_set():
                    graph[li].add(ui)
                    graph[ui].add(li)
    return graph


def _component_of(
    graph: dict[object, set[object]], start: object
) -> set[object]:
    seen: set[object] = set()
    stack = [start]
    while stack:
        node = stack.pop()
        if node in seen:
            continue
        seen.add(node)
        stack.extend(graph.get(node, ()))
    return seen


def connected_components(structure: BrickStructure) -> list[list[int]]:
    """Connected components of the *brick* graph (excluding ``GROUND``).

    Each component is a sorted list of brick indices. Two bricks are in the same
    component if they are joined through stud connections and/or the shared
    baseplate.
    """
    graph = adjacency(structure)
    n = len(structure.bricks)
    seen: set[int] = set()
    components: list[list[int]] = []
    for i in range(n):
        if i in seen:
            continue
        comp_nodes = _component_of(graph, i)
        comp = sorted(node for node in comp_nodes if isinstance(node, int))
        seen.update(comp)
        components.append(comp)
    components.sort()
    return components


def is_single_component(structure: BrickStructure) -> bool:
    """True if all bricks form one connected component (trivially true if empty)."""
    return len(connected_components(structure)) <= 1


def grounded_bricks(structure: BrickStructure) -> set[int]:
    """Indices of bricks reachable from the baseplate through stud connections."""
    graph = adjacency(structure)
    reachable = _component_of(graph, GROUND)
    return {node for node in reachable if isinstance(node, int)}


def is_grounded(structure: BrickStructure) -> bool:
    """True if every brick is connected (directly or transitively) to the baseplate.

    This is the buildable, no-floating-island condition: nothing hovers
    disconnected from the plate.
    """
    return len(grounded_bricks(structure)) == len(structure.bricks)


def is_interconnected(structure: BrickStructure) -> bool:
    """Buildability connectivity criterion: single component *and* grounded."""
    return is_single_component(structure) and is_grounded(structure)


def floating_bricks(structure: BrickStructure) -> list[int]:
    """Indices of bricks that are *not* connected to the baseplate (they float)."""
    grounded = grounded_bricks(structure)
    return sorted(i for i in range(len(structure.bricks)) if i not in grounded)


def supporting_indices(structure: BrickStructure, index: int) -> list[int]:
    """Indices of bricks directly below ``index`` that it studs onto (its supports)."""
    bricks = structure.bricks
    target = bricks[index]
    out = []
    for i, b in enumerate(bricks):
        if i == index:
            continue
        if connection_area(b, target) > 0:
            out.append(i)
    return sorted(out)


def total_connection_area(structure: BrickStructure) -> int:
    """Sum of stud-connection areas over all vertical brick-brick contacts."""
    bricks = structure.bricks
    by_layer: dict[int, list[int]] = {}
    for i, b in enumerate(bricks):
        by_layer.setdefault(b.z, []).append(i)
    total = 0
    for z, lower_indices in by_layer.items():
        for li in lower_indices:
            for ui in by_layer.get(z + 1, []):
                total += connection_area(bricks[li], bricks[ui])
    return total
