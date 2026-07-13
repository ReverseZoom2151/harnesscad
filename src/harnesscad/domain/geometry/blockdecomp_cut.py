"""Axis-aligned cut / split / merge operations for block decomposition.

From *Reinforcement Learning for Block Decomposition of CAD Models* (DiPrete
et al., AAAI-2022). The single geometric modification the agent performs is a
**full cut**: "slice the geometric model into two or more parts using an
infinite line" restricted to "originate from a model vertex and be aligned with
the X- or Y-axis" (Sec. "Methodology"). A cut goes fully through the shape and
splits it into two or more parts (Sec. "Training Phase"); parts that are
quadrilaterals are set aside and the rest are pushed onto a processing queue.
"Crucially, at the end of the decomposition, all the shapes are merged back
together while retaining the boundaries between them" (Sec. "Deploying the
Trained Framework") so that internal cut boundaries are imprinted for meshing.

This module implements those deterministic operations on the
:class:`~geometry.blockdecomp_domain.Shape` cell representation:

  * ``cut_candidates`` -- the legal (vertex, direction) cut actions: from every
    model vertex, a cut along the X-axis (horizontal line) or the Y-axis
    (vertical line);
  * ``full_cut`` -- slice a shape by an axis-aligned line through a coordinate,
    returning the resulting parts (connected components with adjacency severed
    across the cut line). One part means the cut did not affect the model (a cut
    along a side) -- the paper's penalised action;
  * ``cut_from_vertex`` -- the paper's restricted action: cut from a model
    vertex along the X- or Y-axis;
  * ``split_step`` -- apply a cut and partition the parts into finished quad
    blocks vs. non-quad parts still needing decomposition;
  * ``merge`` -- reunite parts into one shape while recording the retained
    internal boundary segments (imprint-and-merge).

Pure stdlib; deterministic (no randomness, no wall clock).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, FrozenSet, List, Sequence, Tuple

from harnesscad.domain.geometry.blockdecomp_domain import Cell, Shape, Vec2

_EPS = 1e-9


@dataclass(frozen=True)
class CutAction:
    """A candidate cut: an axis-aligned line through a model vertex."""

    vertex: Vec2
    direction: str  # "x" -> cut line along X-axis (horizontal y = vy);
    #                  "y" -> cut line along Y-axis (vertical x = vx)

    @property
    def orientation(self) -> str:
        return "horizontal" if self.direction == "x" else "vertical"

    @property
    def coord(self) -> float:
        return self.vertex[1] if self.direction == "x" else self.vertex[0]


def _line_index(shape: Shape, orientation: str, coord: float) -> int:
    """Index of the grid line at ``coord`` for the given orientation, or -1."""
    lines = shape.ys if orientation == "horizontal" else shape.xs
    for k, v in enumerate(lines):
        if abs(v - coord) < _EPS:
            return k
    return -1


def refine(shape: Shape, orientation: str, coord: float) -> Shape:
    """Return ``shape`` with the cut line inserted into its coordinate mesh.

    Inserting a grid line strictly inside the region splits the straddling
    column/row of cells so a subsequent cut along that line can separate them.
    A no-op if the line already exists or lies outside the mesh extent.
    """
    lines = list(shape.ys if orientation == "horizontal" else shape.xs)
    if _line_index(shape, orientation, coord) >= 0:
        return shape
    if coord <= lines[0] + _EPS or coord >= lines[-1] - _EPS:
        return shape  # on/outside the mesh boundary: nothing to refine
    pos = 0
    while pos < len(lines) and lines[pos] < coord:
        pos += 1
    # 'pos' is the index of the split column/row (line pos-1 < coord < line pos).
    split = pos - 1
    new_lines = lines[:pos] + [coord] + lines[pos:]
    new_cells = set()
    for (i, j) in shape.cells:
        if orientation == "vertical":
            if i < split:
                new_cells.add((i, j))
            elif i == split:
                new_cells.add((i, j))
                new_cells.add((i + 1, j))
            else:
                new_cells.add((i + 1, j))
        else:
            if j < split:
                new_cells.add((i, j))
            elif j == split:
                new_cells.add((i, j))
                new_cells.add((i, j + 1))
            else:
                new_cells.add((i, j + 1))
    if orientation == "vertical":
        return Shape(tuple(new_lines), shape.ys, frozenset(new_cells))
    return Shape(shape.xs, tuple(new_lines), frozenset(new_cells))


def full_cut(shape: Shape, orientation: str, coord: float) -> List[Shape]:
    """Split ``shape`` by an infinite axis-aligned line into parts.

    ``orientation`` is ``"horizontal"`` (line ``y = coord``) or ``"vertical"``
    (line ``x = coord``). Returns the resulting parts as shapes on the same mesh.
    Adjacency across the cut line is severed, so parts are the connected
    components under that severed adjacency. If the line does not pass through
    the interior (a cut along a side or outside), a single part is returned.
    """
    if orientation not in ("horizontal", "vertical"):
        raise ValueError("orientation must be 'horizontal' or 'vertical'")
    shape = refine(shape, orientation, coord)
    idx = _line_index(shape, orientation, coord)
    cells = shape.cells
    remaining = set(cells)
    comps: List[FrozenSet[Cell]] = []
    while remaining:
        seed = next(iter(remaining))
        stack = [seed]
        remaining.discard(seed)
        comp = {seed}
        while stack:
            i, j = stack.pop()
            neighbours = ((i + 1, j), (i - 1, j), (i, j + 1), (i, j - 1))
            for ni, nj in neighbours:
                if (ni, nj) not in remaining:
                    continue
                if _severed(orientation, idx, i, j, ni, nj):
                    continue
                remaining.discard((ni, nj))
                comp.add((ni, nj))
                stack.append((ni, nj))
        comps.append(frozenset(comp))
    # Deterministic ordering: by (min i, min j) of each component.
    comps.sort(key=lambda c: (min(i for i, _ in c), min(j for _, j in c)))
    return [shape.with_cells(c) for c in comps]


def _severed(orientation: str, idx: int, i: int, j: int, ni: int, nj: int) -> bool:
    """True if the adjacency (i,j)-(ni,nj) crosses the cut line at index idx."""
    if idx < 0:
        return False
    if orientation == "vertical":
        # cut at x = xs[idx]; horizontal neighbours crossing column boundary idx
        if nj == j:
            hi = max(i, ni)
            return hi == idx
        return False
    else:  # horizontal, cut at y = ys[idx]
        if ni == i:
            hi = max(j, nj)
            return hi == idx
        return False


def cut_candidates(shape: Shape) -> List[CutAction]:
    """All legal cut actions: from each model vertex, an X- and a Y-axis cut."""
    actions: List[CutAction] = []
    seen = set()
    for corner in shape.corners():
        for direction in ("x", "y"):
            key = (corner.pos, direction)
            if key in seen:
                continue
            seen.add(key)
            actions.append(CutAction(vertex=corner.pos, direction=direction))
    return actions


def cut_from_vertex(shape: Shape, action: CutAction) -> List[Shape]:
    """Apply the paper's restricted action: axis-aligned cut from a vertex."""
    return full_cut(shape, action.orientation, action.coord)


@dataclass(frozen=True)
class SplitResult:
    """Outcome of one cut: finished quad blocks and remaining non-quad parts."""

    parts: Tuple[Shape, ...]
    quads: Tuple[Shape, ...]
    non_quads: Tuple[Shape, ...]

    @property
    def num_parts(self) -> int:
        return len(self.parts)

    @property
    def is_effective(self) -> bool:
        """A cut that actually subdivided the shape (produced >= 2 parts)."""
        return len(self.parts) >= 2


def split_step(shape: Shape, action: CutAction) -> SplitResult:
    """Cut and classify parts into quads (done) vs. non-quads (to process)."""
    parts = cut_from_vertex(shape, action)
    quads = tuple(p for p in parts if p.is_quad())
    non_quads = tuple(p for p in parts if not p.is_quad())
    return SplitResult(parts=tuple(parts), quads=quads, non_quads=non_quads)


def merge(parts: Sequence[Shape]) -> Tuple[Shape, List[Tuple[Vec2, Vec2]]]:
    """Reunite parts into one shape, recording retained internal boundaries.

    Returns ``(merged_shape, internal_edges)`` where ``internal_edges`` are the
    grid edge segments shared between two different parts -- the imprinted cut
    boundaries the paper keeps so adjacent blocks share vertices.
    """
    if not parts:
        raise ValueError("nothing to merge")
    xs = parts[0].xs
    ys = parts[0].ys
    for p in parts:
        if p.xs != xs or p.ys != ys:
            raise ValueError("parts must share the same coordinate mesh")
    owner: Dict[Cell, int] = {}
    all_cells = set()
    for k, p in enumerate(parts):
        for c in p.cells:
            owner[c] = k
            all_cells.add(c)
    internal: List[Tuple[Vec2, Vec2]] = []
    seen_edges = set()
    for (i, j), k in owner.items():
        # right neighbour -> shared vertical edge at x = xs[i+1]
        rn = (i + 1, j)
        if rn in owner and owner[rn] != k:
            edge = ((xs[i + 1], ys[j]), (xs[i + 1], ys[j + 1]))
            if edge not in seen_edges:
                seen_edges.add(edge)
                internal.append(edge)
        # top neighbour -> shared horizontal edge at y = ys[j+1]
        tn = (i, j + 1)
        if tn in owner and owner[tn] != k:
            edge = ((xs[i], ys[j + 1]), (xs[i + 1], ys[j + 1]))
            if edge not in seen_edges:
                seen_edges.add(edge)
                internal.append(edge)
    merged = Shape(xs, ys, frozenset(all_cells))
    internal.sort()
    return merged, internal
