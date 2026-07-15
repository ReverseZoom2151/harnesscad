"""Half-edge (co-edge) topological walk over a B-Rep, A2Z style.

A2Z-10M+ (Jena et al., 2026), Sec. 3, treats a B-Rep as a chain complex and augments
it with the *co-edge* (half-edge / radial-edge, Weiler 1986) traversal that the
existing :mod:`harnesscad.domain.reconstruction.brep.chain_complex` (a ComplexGen-style
incidence-matrix model) does not carry. A2Z stores three per-co-edge transition
vectors::

    N : next   -- the next co-edge around the same loop
    P : parent -- the face (or loop) that owns the co-edge
    M : mate   -- the co-edge on the adjacent face sharing the same edge

and shows that a topological walk composes them, e.g. "to know the mating face id of
another face that is next to my current co-edge id ``e`` ... F[M[N[E[e]]]]" (their
Sec. 3). This module builds that half-edge structure from an oriented-loop description
of a solid and exposes the deterministic walks (next / previous / mate, loop cycles,
face recovery, and mate-face queries). It is the traversal layer that machining-feature
and boundary-detection code can walk without touching incidence matrices.

Input: for each face, an ordered list of directed edges, each ``(v_start, v_end)``.
Two co-edges are *mates* when they traverse the same undirected vertex pair in opposite
directions (the manifold assumption: exactly two co-edges per edge). Stdlib only,
deterministic; co-edge ids are assigned in a stable face-then-loop-order.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Mapping, Optional, Sequence

__all__ = [
    "CoEdge",
    "HalfEdgeStructure",
    "build_half_edges",
    "loop_of",
    "walk_loop",
    "mate_face",
    "is_manifold",
    "boundary_co_edges",
]


@dataclass(frozen=True)
class CoEdge:
    """A directed use of an edge by one face (A2Z co-edge)."""

    id: int
    face: int
    loop: int
    v_start: int
    v_end: int
    next: int          # N: next co-edge in the same loop
    prev: int          # previous co-edge in the same loop
    mate: Optional[int]  # M: opposite co-edge on the adjacent face (None on boundary)

    @property
    def edge_key(self) -> tuple[int, int]:
        """Undirected vertex pair identifying the shared geometric edge."""
        return (self.v_start, self.v_end) if self.v_start < self.v_end else (self.v_end, self.v_start)


@dataclass(frozen=True)
class HalfEdgeStructure:
    """Immutable half-edge topology with A2Z next/prev/parent/mate transitions."""

    co_edges: tuple[CoEdge, ...]
    faces: tuple[int, ...]
    _by_id: Mapping[int, CoEdge] = field(default_factory=dict)

    def co_edge(self, ce_id: int) -> CoEdge:
        return self._by_id[ce_id]

    def face_of(self, ce_id: int) -> int:
        """P: the parent face of a co-edge."""
        return self._by_id[ce_id].face


def build_half_edges(
    face_loops: Mapping[int, Sequence[Sequence[tuple[int, int]]]] | Mapping[int, Sequence[tuple[int, int]]],
) -> HalfEdgeStructure:
    """Assemble the half-edge structure from oriented face loops.

    ``face_loops`` maps ``face_id -> loops``, where each loop is an ordered sequence of
    directed edges ``(v_start, v_end)`` and consecutive edges must share a vertex. A
    face may be given either as a single loop (sequence of edges) or as a sequence of
    loops. Co-edge ids are assigned deterministically in ascending face id, then loop
    order, then edge order.
    """
    co_edges: list[CoEdge] = []
    # First pass: create co-edges with next/prev inside each loop.
    pending: list[dict] = []
    faces_seen: list[int] = []
    next_id = 0
    for face in sorted(face_loops):
        faces_seen.append(face)
        loops = face_loops[face]
        # Normalise "single loop" (sequence of (int,int) tuples) into list-of-loops.
        if loops and isinstance(loops[0], tuple) and len(loops[0]) == 2 and isinstance(loops[0][0], int):
            loops = [loops]  # type: ignore[list-item]
        for loop_idx, loop in enumerate(loops):
            loop = list(loop)
            if len(loop) < 2:
                raise ValueError(f"face {face} loop {loop_idx} has fewer than 2 edges")
            start_id = next_id
            m = len(loop)
            for k, (vs, ve) in enumerate(loop):
                cid = start_id + k
                pending.append({
                    "id": cid, "face": face, "loop": loop_idx,
                    "v_start": vs, "v_end": ve,
                    "next": start_id + (k + 1) % m,
                    "prev": start_id + (k - 1) % m,
                })
            next_id += m

    # Second pass: resolve mates by undirected edge key with opposite direction.
    #  key (a,b) directed a->b ; its mate is the co-edge directed b->a.
    directed: dict[tuple[int, int], list[int]] = {}
    for p in pending:
        directed.setdefault((p["v_start"], p["v_end"]), []).append(p["id"])

    mate_of: dict[int, Optional[int]] = {}
    for p in pending:
        opp = directed.get((p["v_end"], p["v_start"]), [])
        mate_of[p["id"]] = opp[0] if opp else None

    for p in pending:
        co_edges.append(CoEdge(
            id=p["id"], face=p["face"], loop=p["loop"],
            v_start=p["v_start"], v_end=p["v_end"],
            next=p["next"], prev=p["prev"], mate=mate_of[p["id"]],
        ))

    by_id = {ce.id: ce for ce in co_edges}
    return HalfEdgeStructure(co_edges=tuple(co_edges), faces=tuple(faces_seen), _by_id=by_id)


def loop_of(struct: HalfEdgeStructure, ce_id: int) -> tuple[int, ...]:
    """Return the co-edge ids forming the loop containing ``ce_id`` (via N transitions)."""
    return walk_loop(struct, ce_id)


def walk_loop(struct: HalfEdgeStructure, ce_id: int) -> tuple[int, ...]:
    """Walk N (next) from ``ce_id`` until returning to it; the ordered loop cycle."""
    start = struct.co_edge(ce_id)
    order = [start.id]
    cur = start
    guard = len(struct.co_edges) + 1
    while True:
        cur = struct.co_edge(cur.next)
        if cur.id == start.id:
            break
        order.append(cur.id)
        guard -= 1
        if guard < 0:
            raise ValueError("loop walk did not terminate (malformed next-pointers)")
    return tuple(order)


def mate_face(struct: HalfEdgeStructure, ce_id: int) -> Optional[int]:
    """A2Z F[M[e]]: the face on the far side of a co-edge's shared edge (or None)."""
    mate = struct.co_edge(ce_id).mate
    return None if mate is None else struct.face_of(mate)


def is_manifold(struct: HalfEdgeStructure) -> bool:
    """True when every co-edge has exactly one mate (each edge used by two co-edges)."""
    return all(ce.mate is not None for ce in struct.co_edges)


def boundary_co_edges(struct: HalfEdgeStructure) -> tuple[int, ...]:
    """Co-edges with no mate: the open boundary of a non-watertight solid."""
    return tuple(ce.id for ce in struct.co_edges if ce.mate is None)
