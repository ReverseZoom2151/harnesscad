"""Cell-adjacency graph and plausible-sequence generation for surface CSG.

After decomposition, the model is turned into a **graph** whose nodes are the
half-spaces / cells the model is made of; two nodes are connected if they touch
each other in the model. This graph is then used to generate plausible sequences
that build the same 3D geometry while avoiding geometry that is unconnected, and
enumerating all plausible sequences maximises the amount of training data.

This module builds that graph deterministically and enumerates the plausible
build orderings. A *plausible sequence* is an ordering of the cells such that
every cell after the first is adjacent (touches) at least one already-placed
cell -- i.e. the placed prefix is always a connected sub-graph. This is exactly
the avoid-unconnected-geometry constraint, and enumerating every such ordering
is an order-augmentation strategy that varies the order of the sequence, which
lifts model accuracy.

Adjacency is decided geometrically with no external tools: two cells touch if
they share a surface (reference the same surface object) or their occupied
bounding boxes are contact-adjacent within a tolerance. Everything is stdlib and
deterministic.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Set, Tuple

from harnesscad.domain.geometry.sdf.halfspace_csg import (
    Box,
    Cell,
    CSGModel,
    Vec3,
    bounding_box,
)


def _cell_bbox(cell: Cell, probe: Box, res: int) -> Optional[Box]:
    return bounding_box(CSGModel((cell,)), probe, res)


def _boxes_touch(a: Box, b: Box, tol: float) -> bool:
    """True if two axis-aligned boxes overlap or touch within ``tol`` on every
    axis (i.e. their inflated-by-tol extents intersect on all three axes)."""
    for i in range(3):
        if a[1][i] + tol < b[0][i] or b[1][i] + tol < a[0][i]:
            return False
    return True


def _shared_surface(a: Cell, b: Cell) -> bool:
    sa = {id(s) for s in a.surfaces()}
    return any(id(s) in sa for s in b.surfaces())


@dataclass
class CellGraph:
    """Undirected adjacency over a model's cells."""

    n: int
    adjacency: Dict[int, Set[int]]

    def neighbours(self, i: int) -> Set[int]:
        return self.adjacency.get(i, set())

    def edges(self) -> List[Tuple[int, int]]:
        out: List[Tuple[int, int]] = []
        for i in sorted(self.adjacency):
            for j in sorted(self.adjacency[i]):
                if i < j:
                    out.append((i, j))
        return out

    def is_connected(self) -> bool:
        """True if the whole cell set forms one connected component."""
        if self.n <= 1:
            return True
        seen: Set[int] = {0}
        stack = [0]
        while stack:
            cur = stack.pop()
            for nb in self.neighbours(cur):
                if nb not in seen:
                    seen.add(nb)
                    stack.append(nb)
        return len(seen) == self.n

    def components(self) -> List[List[int]]:
        seen: Set[int] = set()
        comps: List[List[int]] = []
        for start in range(self.n):
            if start in seen:
                continue
            comp: List[int] = []
            stack = [start]
            seen.add(start)
            while stack:
                cur = stack.pop()
                comp.append(cur)
                for nb in self.neighbours(cur):
                    if nb not in seen:
                        seen.add(nb)
                        stack.append(nb)
            comps.append(sorted(comp))
        return comps


def build_cell_graph(
    model: CSGModel, probe: Box, res: int = 12, tol: Optional[float] = None
) -> CellGraph:
    """Build the touch-adjacency graph of ``model``'s cells.

    Two cells are adjacent if they share a surface or their sampled bounding
    boxes are contact-adjacent within ``tol`` (default: one probe voxel).
    """
    cells = model.cells
    n = len(cells)
    if tol is None:
        span = max(
            probe[1][0] - probe[0][0],
            probe[1][1] - probe[0][1],
            probe[1][2] - probe[0][2],
        )
        # Grid sampling shrinks each occupied box inward by ~half a voxel per
        # side, so genuinely face-adjacent cells read ~one voxel apart. Use 1.5
        # voxels of slack to catch contact while still separating cells that are
        # a clear gap apart.
        tol = 1.5 * span / res
    bboxes = [_cell_bbox(c, probe, res) for c in cells]
    adj: Dict[int, Set[int]] = {i: set() for i in range(n)}
    for i in range(n):
        for j in range(i + 1, n):
            touch = _shared_surface(cells[i], cells[j])
            if not touch and bboxes[i] is not None and bboxes[j] is not None:
                touch = _boxes_touch(bboxes[i], bboxes[j], tol)
            if touch:
                adj[i].add(j)
                adj[j].add(i)
    return CellGraph(n, adj)


def plausible_sequences(
    graph: CellGraph, limit: Optional[int] = None
) -> List[Tuple[int, ...]]:
    """Enumerate every connected build ordering of the cells.

    A sequence is plausible when each cell after the first touches at least one
    already-placed cell (so the placed prefix stays connected -- the paper's
    "avoid generating geometry that is unconnected"). Orderings are produced in a
    deterministic (lexicographic) order; ``limit`` caps the count to guard
    against combinatorial blow-up.
    """
    n = graph.n
    if n == 0:
        return [()]
    results: List[Tuple[int, ...]] = []

    def extend(order: List[int], placed: Set[int], frontier: Set[int]) -> bool:
        # returns False to signal the limit was reached (stop all recursion)
        if len(order) == n:
            results.append(tuple(order))
            return not (limit is not None and len(results) >= limit)
        # candidates: cells adjacent to the placed prefix (or any cell if empty)
        if not order:
            candidates = list(range(n))
        else:
            candidates = sorted(frontier)
        for c in candidates:
            new_frontier = (frontier | graph.neighbours(c)) - (placed | {c})
            placed.add(c)
            order.append(c)
            keep_going = extend(order, placed, new_frontier)
            order.pop()
            placed.discard(c)
            if not keep_going:
                return False
        return True

    extend([], set(), set())
    return results


def is_plausible_sequence(graph: CellGraph, order: Sequence[int]) -> bool:
    """Validate that ``order`` is a permutation whose every prefix is connected."""
    if sorted(order) != list(range(graph.n)):
        return False
    placed: Set[int] = set()
    for idx, c in enumerate(order):
        if idx > 0 and not (graph.neighbours(c) & placed):
            return False
        placed.add(c)
    return True
