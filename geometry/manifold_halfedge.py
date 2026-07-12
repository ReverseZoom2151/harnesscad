"""Half-edge triangle mesh with Manifold's manifoldness invariants.

Manifold (Emmett Lalish) stores every solid as a *half-edge* triangle mesh and
guarantees it is a closed, oriented 2-manifold at every step of a boolean.  The
core data structure (``src/impl.h`` / ``src/shared.h``) is deliberately
index-arithmetic based rather than pointer based:

* triangle ``t`` owns exactly three consecutive half-edges ``3t, 3t+1, 3t+2``;
* the next half-edge inside a face is ``NextHalfedge`` (``+1`` with wrap at the
  triangle boundary) and the previous is ``PrevHalfedge``;
* each half-edge stores ``startVert``, ``endVert`` and ``pairedHalfedge`` -- the
  opposite half-edge of the adjacent triangle -- so face and vertex circulation
  are pure index walks with no per-vertex adjacency lists.

Its validity checks (``Manifold::Impl::IsManifold`` / ``Is2Manifold`` in
``src/properties.cpp``) are the manifold contract:

* every half-edge count is a multiple of three (whole triangles);
* every directed edge is paired, the pairing is an involution
  (``Pair(Pair(e)) == e``), the pair runs the opposite direction
  (``Start == End(pair)`` and ``End == Start(pair)``), and no half-edge is a
  loop (``start != end``);
* the mesh is a **2-manifold** iff, additionally, no undirected edge is shared
  by more than two triangles -- i.e. sorting the half-edges leaves no duplicate
  ``(start, end)`` directed edge, so each undirected edge has exactly two
  half-edges of opposite direction.

This module reimplements that structure and its checks in stdlib Python:

* :class:`HalfedgeMesh` built from a vertex list + triangle-index list, with
  the exact Manifold half-edge indexing, :meth:`next_halfedge`,
  :meth:`prev_halfedge`, :meth:`pair`, :meth:`face_circulation`,
  :meth:`vertex_ring` (one-ring of outgoing half-edges around a vertex), and
  :meth:`boundary_halfedges`;
* :meth:`is_manifold` / :meth:`is_2manifold` matching Manifold's predicates,
  returning structured :class:`MeshIssue` records;
* Euler characteristic, genus (``1 - chi/2`` for a closed orientable surface)
  and boundary-loop extraction.

Overlap with :mod:`geometry.angelcad_polyhedron`: that module validates an
*explicit polygon-face* solid via an undirected edge-use table (counts, boundary
and non-manifold edges) and computes mass properties; it is a set-based check on
polygon faces and has **no half-edge structure, no paired-half-edge involution,
no face/vertex circulation, and no pairing-consistency (orientation-of-pair)
test**.  This module is the true half-edge kernel: circulation walks and the
``Pair(Pair(e)) == e`` involution the boolean engine relies on.  The two are
complementary -- ``angelcad_polyhedron`` gates arbitrary polygon input,
``manifold_halfedge`` is the triangle-mesh runtime representation.

Pure stdlib, deterministic.
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Sequence, Tuple

__all__ = [
    "MeshIssue",
    "HalfedgeMesh",
    "tetrahedron_mesh",
    "cube_mesh",
]

Point = Tuple[float, float, float]
Tri = Tuple[int, int, int]


class MeshIssue:
    """One manifoldness finding."""

    __slots__ = ("code", "message", "halfedge", "edge")

    def __init__(self, code, message, halfedge=None, edge=None):
        self.code = code
        self.message = message
        self.halfedge = halfedge
        self.edge = edge

    def key(self):
        return (self.code, self.message, self.halfedge, self.edge)

    def __eq__(self, other):
        return isinstance(other, MeshIssue) and other.key() == self.key()

    def __hash__(self):
        return hash(self.key())

    def __repr__(self):
        return "MeshIssue(%s: %s)" % (self.code, self.message)

    __str__ = __repr__


def next_halfedge(h: int) -> int:
    """Next half-edge inside the same triangle (Manifold ``NextHalfedge``)."""
    return h - 2 if h % 3 == 2 else h + 1


def prev_halfedge(h: int) -> int:
    """Previous half-edge inside the same triangle (Manifold ``PrevHalfedge``)."""
    return h + 2 if h % 3 == 0 else h - 1


class HalfedgeMesh:
    """A triangle mesh in Manifold's half-edge representation.

    ``tris[t] = (i, j, k)`` produces half-edges ``3t = i->j``, ``3t+1 = j->k``,
    ``3t+2 = k->i``.  Pairing is computed by matching each directed edge
    ``(a, b)`` to its reverse ``(b, a)``; an unmatched half-edge is a boundary
    half-edge whose pair is ``-1``.
    """

    __slots__ = ("vertices", "tris", "starts", "ends", "pairs", "_directed")

    def __init__(self, vertices: Sequence[Sequence[float]], tris: Sequence[Sequence[int]]):
        self.vertices: List[Point] = [
            (float(p[0]), float(p[1]), float(p[2])) for p in vertices
        ]
        self.tris: List[Tri] = [(int(t[0]), int(t[1]), int(t[2])) for t in tris]
        n = len(self.tris)
        self.starts: List[int] = [0] * (3 * n)
        self.ends: List[int] = [0] * (3 * n)
        self.pairs: List[int] = [-1] * (3 * n)
        # directed edge (a, b) -> list of half-edge indices with that direction
        directed: Dict[Tuple[int, int], List[int]] = {}
        for t, (i, j, k) in enumerate(self.tris):
            verts = (i, j, k)
            for e in range(3):
                h = 3 * t + e
                a = verts[e]
                b = verts[(e + 1) % 3]
                self.starts[h] = a
                self.ends[h] = b
                directed.setdefault((a, b), []).append(h)
        self._directed = directed
        # Pair each half-edge (a, b) with a reverse (b, a).  When exactly one of
        # each exists the pairing is unambiguous; otherwise leave -1 and let the
        # validator flag the non-manifold / boundary condition.
        for (a, b), hs in directed.items():
            if a >= b:
                continue
            rev = directed.get((b, a), [])
            if len(hs) == 1 and len(rev) == 1:
                self.pairs[hs[0]] = rev[0]
                self.pairs[rev[0]] = hs[0]

    # -- counts -----------------------------------------------------------
    def num_vert(self) -> int:
        return len(self.vertices)

    def num_tri(self) -> int:
        return len(self.tris)

    def num_halfedge(self) -> int:
        return 3 * len(self.tris)

    def num_edge(self) -> int:
        """Number of undirected edges."""
        seen = set()
        for (a, b) in self._directed:
            seen.add((a, b) if a < b else (b, a))
        return len(seen)

    # -- topology accessors ----------------------------------------------
    def start(self, h: int) -> int:
        return self.starts[h]

    def end(self, h: int) -> int:
        return self.ends[h]

    def pair(self, h: int) -> int:
        return self.pairs[h]

    def tri_of(self, h: int) -> int:
        return h // 3

    def face_circulation(self, h: int) -> List[int]:
        """The three half-edges of the triangle owning ``h``, starting at ``h``."""
        return [h, next_halfedge(h), next_halfedge(next_halfedge(h))]

    def vertex_ring(self, h: int) -> List[int]:
        """Outgoing half-edges around the start vertex of ``h`` (one-ring walk).

        Walks ``h -> pair(prev(h))`` which rotates to the next outgoing
        half-edge sharing the same start vertex.  Returns the cycle starting at
        ``h``.  Raises :class:`ValueError` if the ring hits a boundary (an
        unpaired half-edge), since the walk is only closed on a 2-manifold
        interior vertex.
        """
        ring = []
        cur = h
        v = self.starts[h]
        for _ in range(self.num_halfedge() + 1):
            ring.append(cur)
            p = self.pairs[prev_halfedge(cur)]
            if p < 0:
                raise ValueError("vertex ring hits a boundary at half-edge %d" % cur)
            cur = p
            if cur == h:
                return ring
            if self.starts[cur] != v:  # pragma: no cover - defensive
                raise ValueError("inconsistent vertex ring")
        raise ValueError("vertex ring did not close")  # pragma: no cover

    def boundary_halfedges(self) -> List[int]:
        """Half-edges with no pair (edges on an open boundary)."""
        return [h for h in range(self.num_halfedge()) if self.pairs[h] < 0]

    # -- validity ---------------------------------------------------------
    def is_manifold(self) -> bool:
        """True iff this is a consistent oriented manifold (Manifold ``IsManifold``)."""
        return not self._manifold_issues()

    def _manifold_issues(self) -> List[MeshIssue]:
        issues: List[MeshIssue] = []
        nh = self.num_halfedge()
        if nh % 3 != 0:  # pragma: no cover - constructor enforces triangles
            issues.append(MeshIssue("not-triangles", "half-edge count not a multiple of 3"))
            return issues
        nv = self.num_vert()
        for h in range(nh):
            a, b = self.starts[h], self.ends[h]
            if a == b:
                issues.append(MeshIssue("loop", "half-edge %d is a loop (start==end)" % h, h))
            if a < 0 or a >= nv or b < 0 or b >= nv:
                issues.append(MeshIssue("index-range", "half-edge %d has out-of-range vertex" % h, h))
            p = self.pairs[h]
            if p < 0:
                issues.append(MeshIssue("boundary", "half-edge %d is unpaired (open edge)" % h, h,
                                        (a, b) if a < b else (b, a)))
                continue
            if self.pairs[p] != h:
                issues.append(MeshIssue("pair-involution", "Pair(Pair(%d)) != %d" % (h, h), h))
            if not (self.starts[h] == self.ends[p] and self.ends[h] == self.starts[p]):
                issues.append(MeshIssue("pair-orientation",
                                        "half-edge %d and its pair are not oppositely directed" % h, h))
        return issues

    def is_2manifold(self) -> Tuple[bool, List[MeshIssue]]:
        """True iff a 2-manifold; also returns the issue list.

        Beyond :meth:`is_manifold`, requires that no undirected edge is used by
        more than two triangles, i.e. every directed edge is unique.
        """
        issues = list(self._manifold_issues())
        for (a, b), hs in self._directed.items():
            if len(hs) > 1:
                issues.append(MeshIssue("nonmanifold-edge",
                                        "directed edge (%d,%d) used by %d half-edges" % (a, b, len(hs)),
                                        None, (a, b)))
        # dedup while preserving deterministic order
        seen = set()
        uniq = []
        for it in issues:
            if it.key() not in seen:
                seen.add(it.key())
                uniq.append(it)
        return (not uniq, uniq)

    # -- global invariants ------------------------------------------------
    def euler_characteristic(self) -> int:
        """V - E + F."""
        return self.num_vert() - self.num_edge() + self.num_tri()

    def genus(self) -> int:
        """Genus for a closed orientable surface: ``1 - chi/2``.

        Only meaningful when :meth:`boundary_halfedges` is empty.
        """
        chi = self.euler_characteristic()
        return 1 - chi // 2

    def is_closed(self) -> bool:
        return not self.boundary_halfedges()

    def boundary_loops(self) -> List[List[int]]:
        """Ordered vertex loops of the open boundary (empty for a closed mesh)."""
        bnd = set(self.boundary_halfedges())
        # next boundary half-edge around a boundary vertex: walk forward through
        # faces via pair until another boundary half-edge is found.
        loops: List[List[int]] = []
        visited = set()
        # map start vertex -> boundary half-edge for successor lookup
        by_start: Dict[int, int] = {}
        for h in bnd:
            by_start[self.starts[h]] = h
        for h0 in sorted(bnd):
            if h0 in visited:
                continue
            loop: List[int] = []
            h = h0
            for _ in range(len(bnd) + 1):
                if h in visited:
                    break
                visited.add(h)
                loop.append(self.starts[h])
                nxt = by_start.get(self.ends[h])
                if nxt is None:
                    break
                h = nxt
                if h == h0:
                    break
            loops.append(loop)
        return loops


# --------------------------------------------------------------------------
# canonical meshes
# --------------------------------------------------------------------------


def tetrahedron_mesh(size: float = 1.0) -> HalfedgeMesh:
    s = float(size)
    verts = [(0, 0, 0), (s, 0, 0), (0, s, 0), (0, 0, s)]
    tris = [(0, 2, 1), (0, 1, 3), (1, 2, 3), (2, 0, 3)]
    return HalfedgeMesh(verts, tris)


def cube_mesh(size: float = 1.0) -> HalfedgeMesh:
    s = float(size)
    v = [
        (0, 0, 0), (s, 0, 0), (s, s, 0), (0, s, 0),
        (0, 0, s), (s, 0, s), (s, s, s), (0, s, s),
    ]
    quads = [
        (0, 3, 2, 1),  # bottom -z
        (4, 5, 6, 7),  # top +z
        (0, 1, 5, 4),  # -y
        (1, 2, 6, 5),  # +x
        (2, 3, 7, 6),  # +y
        (3, 0, 4, 7),  # -x
    ]
    tris = []
    for a, b, c, d in quads:
        tris.append((a, b, c))
        tris.append((a, c, d))
    return HalfedgeMesh(v, tris)
