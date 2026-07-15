"""Coedge topology and topological walks (BRepNet's B-Rep message-passing kernel).

Ported from *BRepNet: A Topological Message Passing System for Solid Models*
(Lambourne et al., CVPR 2021).  BRepNet's insight is that a B-Rep carries
orientation information that arbitrary graphs do not: each face boundary is an
ordered loop of oriented edges called **coedges**, and every coedge has four
canonical topological relations --

* ``n`` -- the *next* coedge around its face loop,
* ``p`` -- the *previous* coedge around its face loop,
* ``m`` -- the *mate*: the coedge of the adjacent face on the same undirected
  edge (running the opposite direction),
* ``f`` -- the *parent face*,
* ``e`` -- the *parent (undirected) edge*.

A **topological walk** is a string of these instructions applied left to right
from a starting coedge, e.g. ``"mn"`` = "go to my mate, then to that coedge's
next".  A **kernel** is a set of walks that names an oriented neighbourhood
around every coedge -- BRepNet's ``winged_edge`` kernel, for instance, names the
faces ``["f", "mf"]``, edges ``["e", "ne", "pe", "mne", "mpe"]`` and coedges
``["", "m", "n", "p", "mn", "mp"]``.  The reference resolves these walks as
index-array compositions (``c = n[c]`` etc.); that resolution is the transferable,
deterministic core reimplemented here.  The neural convolution on top is *not*.

Why this helps the harness: the topological-naming problem (``topology/
topological_naming.py``, ``topology/face_adjacency.py``) needs *stable,
orientation-aware* names for faces/edges under regeneration.  The existing
face-adjacency graph is undirected and knows nothing about loop order or coedge
mating.  Coedge walks give a directed, orientation-aware neighbourhood per entity,
from which this module derives:

* :meth:`CoedgeTopology.walk` / :meth:`kernel_neighbourhood` -- BRepNet walk
  resolution and kernel application;
* :meth:`coedge_signature` / :meth:`face_signature` -- a deterministic
  Weisfeiler-Lehman-style refinement over the coedge relations that yields a
  relabelling-invariant topological fingerprint (a stronger persistent name than
  undirected adjacency alone);
* :meth:`canonical_bfs_face_order` -- a canonical BFS ordering of the faces over
  mate-adjacency from a deterministic seed.  This is exactly the *BFS order of
  the B-Rep topology graph* that AutoBrep (Xu et al., SIGGRAPH Asia 2025) uses to
  serialise a B-Rep into a canonical token sequence; here it gives a canonical
  face relabelling for naming/serialisation, with no model involved.

Input model: a solid is given as a list of faces, each face an ordered loop of
vertex indices (its boundary polygon, CCW seen from outside).  Coedges are the
directed boundary edges in loop order.  Two coedges on the same undirected edge
are mates.  Pure stdlib, deterministic throughout.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

__all__ = [
    "CoedgeTopology",
    "WINGED_EDGE_KERNEL",
    "SIMPLE_EDGE_KERNEL",
    "WINGED_EDGE_PLUS_KERNEL",
]

# BRepNet kernel definitions (verbatim from kernels/*.json).
SIMPLE_EDGE_KERNEL: Dict[str, List[str]] = {
    "faces": ["f", "mf"],
    "edges": ["e"],
    "coedges": ["", "m"],
}
WINGED_EDGE_KERNEL: Dict[str, List[str]] = {
    "faces": ["f", "mf"],
    "edges": ["e", "ne", "pe", "mne", "mpe"],
    "coedges": ["", "m", "n", "p", "mn", "mp"],
}
WINGED_EDGE_PLUS_KERNEL: Dict[str, List[str]] = {
    "faces": ["f", "mf", "nmf", "pmf"],
    "edges": ["e", "ne", "pe", "mne", "mpe"],
    "coedges": ["", "n", "p", "m", "mn", "mp", "nm", "pm"],
}


@dataclass(frozen=True)
class CoedgeTopology:
    """Index-array coedge topology built from face boundary loops.

    Arrays are indexed by coedge id.  ``n``/``p``/``m`` map coedge -> coedge;
    ``f`` maps coedge -> face; ``e`` maps coedge -> undirected edge.  A coedge
    with no mate (a boundary edge of an open surface) is its own mate and is
    listed in :attr:`boundary_coedges`.
    """

    n: Tuple[int, ...]
    p: Tuple[int, ...]
    m: Tuple[int, ...]
    f: Tuple[int, ...]
    e: Tuple[int, ...]
    coedge_verts: Tuple[Tuple[int, int], ...]  # (start_vert, end_vert) per coedge
    face_coedges: Tuple[Tuple[int, ...], ...]  # coedge ids per face, in loop order
    num_faces: int
    num_edges: int
    boundary_coedges: Tuple[int, ...]

    # ---- construction -------------------------------------------------
    @classmethod
    def from_faces(cls, faces: Sequence[Sequence[int]]) -> "CoedgeTopology":
        """Build the topology from faces given as ordered vertex-index loops."""
        n_list: List[int] = []
        p_list: List[int] = []
        f_list: List[int] = []
        cverts: List[Tuple[int, int]] = []
        face_coedges: List[Tuple[int, ...]] = []

        for fi, loop in enumerate(faces):
            k = len(loop)
            if k < 2:
                raise ValueError(f"face {fi} has fewer than 2 vertices")
            base = len(n_list)
            ids = tuple(range(base, base + k))
            for j in range(k):
                a = loop[j]
                b = loop[(j + 1) % k]
                cverts.append((a, b))
                f_list.append(fi)
                n_list.append(base + (j + 1) % k)
                p_list.append(base + (j - 1) % k)
            face_coedges.append(ids)

        # undirected edge ids + mate pairing
        edge_key_to_id: Dict[Tuple[int, int], int] = {}
        e_list: List[int] = [0] * len(cverts)
        edge_to_coedges: Dict[int, List[int]] = {}
        for ci, (a, b) in enumerate(cverts):
            key = (a, b) if a <= b else (b, a)
            eid = edge_key_to_id.get(key)
            if eid is None:
                eid = len(edge_key_to_id)
                edge_key_to_id[key] = eid
            e_list[ci] = eid
            edge_to_coedges.setdefault(eid, []).append(ci)

        m_list: List[int] = list(range(len(cverts)))  # self by default
        boundary: List[int] = []
        for eid, cs in edge_to_coedges.items():
            if len(cs) == 2:
                a, b = cs
                m_list[a] = b
                m_list[b] = a
            elif len(cs) == 1:
                boundary.append(cs[0])
            else:
                # non-manifold edge: pair opposite-direction coedges greedily,
                # deterministically by coedge id
                cs_sorted = sorted(cs)
                for a in cs_sorted:
                    if m_list[a] != a:
                        continue
                    for b in cs_sorted:
                        if b != a and m_list[b] == b and \
                                cverts[b] == (cverts[a][1], cverts[a][0]):
                            m_list[a] = b
                            m_list[b] = a
                            break
                    if m_list[a] == a:
                        boundary.append(a)

        return cls(
            n=tuple(n_list), p=tuple(p_list), m=tuple(m_list),
            f=tuple(f_list), e=tuple(e_list),
            coedge_verts=tuple(cverts),
            face_coedges=tuple(face_coedges),
            num_faces=len(faces),
            num_edges=len(edge_key_to_id),
            boundary_coedges=tuple(sorted(boundary)),
        )

    @property
    def num_coedges(self) -> int:
        return len(self.n)

    # ---- walks --------------------------------------------------------
    def walk(self, coedge: int, instructions: str) -> int:
        """Resolve a topological walk from ``coedge``.

        Instructions are applied left to right.  ``n``/``p``/``m`` stay in the
        coedge domain; ``f`` and ``e`` map to a face/edge id and *must* be the
        final instruction (a face/edge is not a coedge, so nothing may follow).
        Returns a coedge id (walk of only n/p/m), or a face/edge id if the walk
        ends in ``f``/``e``.  An empty walk returns ``coedge`` itself.
        """
        c = coedge
        for idx, ins in enumerate(instructions):
            last = idx == len(instructions) - 1
            if ins == "n":
                c = self.n[c]
            elif ins == "p":
                c = self.p[c]
            elif ins == "m":
                c = self.m[c]
            elif ins == "f":
                if not last:
                    raise ValueError("'f' must be the final instruction")
                return self.f[c]
            elif ins == "e":
                if not last:
                    raise ValueError("'e' must be the final instruction")
                return self.e[c]
            else:
                raise ValueError(f"unknown walk instruction {ins!r}")
        return c

    def kernel_neighbourhood(
        self, coedge: int, kernel: Dict[str, List[str]]
    ) -> Dict[str, List[int]]:
        """Apply a kernel to one coedge, returning face/edge/coedge id lists."""
        return {
            "faces": [self.walk(coedge, w if w.endswith("f") else w + "f")
                      for w in kernel["faces"]],
            "edges": [self.walk(coedge, w if w.endswith("e") else w + "e")
                      for w in kernel["edges"]],
            "coedges": [self.walk(coedge, w) for w in kernel["coedges"]],
        }

    # ---- topological fingerprints (WL refinement) ---------------------
    def coedge_signature(self, rounds: int = 3) -> List[int]:
        """Weisfeiler-Lehman colour per coedge, invariant under relabelling.

        Round 0 colours every coedge by its local degree signature (whether it
        has a distinct mate and its face's loop length).  Each round refines a
        coedge's colour from the multiset of its ``n``/``p``/``m`` neighbours'
        colours.  Colours are canonicalised to small ints each round so the
        output depends only on topology, not on construction order.
        """
        nc = self.num_coedges
        loop_len = [len(self.face_coedges[self.f[c]]) for c in range(nc)]
        colour = [
            (loop_len[c], 0 if self.m[c] == c else 1)
            for c in range(nc)
        ]
        colour = _canonicalise(colour)
        for _ in range(max(0, rounds)):
            nxt = []
            for c in range(nc):
                nxt.append((
                    colour[c],
                    colour[self.n[c]],
                    colour[self.p[c]],
                    colour[self.m[c]],
                ))
            colour = _canonicalise(nxt)
        return colour

    def face_signature(self, rounds: int = 3) -> List[Tuple[int, ...]]:
        """A relabelling-invariant fingerprint per face.

        The face's signature is the sorted multiset of the WL colours of the
        coedges on its boundary loop -- a persistent topological name that is
        stable under any renumbering of faces/edges/coedges.
        """
        cc = self.coedge_signature(rounds=rounds)
        out: List[Tuple[int, ...]] = []
        for fi in range(self.num_faces):
            out.append(tuple(sorted(cc[c] for c in self.face_coedges[fi])))
        return out

    # ---- canonical ordering (AutoBrep BFS serialisation) --------------
    def face_adjacency(self) -> Dict[int, List[int]]:
        """Undirected face adjacency via mate coedges (edge-sharing faces)."""
        adj: Dict[int, set] = {fi: set() for fi in range(self.num_faces)}
        for c in range(self.num_coedges):
            mc = self.m[c]
            if mc == c:
                continue
            fa, fb = self.f[c], self.f[mc]
            if fa != fb:
                adj[fa].add(fb)
                adj[fb].add(fa)
        return {fi: sorted(s) for fi, s in adj.items()}

    def canonical_bfs_face_order(self) -> List[int]:
        """Canonical BFS order of faces over mate-adjacency (AutoBrep serialisation).

        The seed is the face with the lexicographically smallest signature, ties
        broken by smallest face index; BFS then expands neighbours ordered by
        (signature, index).  Disconnected shells are appended by the same rule.
        The result is a canonical relabelling of faces independent of input order.
        """
        adj = self.face_adjacency()
        sig = self.face_signature()
        order: List[int] = []
        seen = [False] * self.num_faces

        def rank(fi: int) -> Tuple:
            return (sig[fi], fi)

        remaining = sorted(range(self.num_faces), key=rank)
        for seed in remaining:
            if seen[seed]:
                continue
            seen[seed] = True
            queue = [seed]
            while queue:
                # pop the best-ranked frontier face (deterministic)
                queue.sort(key=rank)
                cur = queue.pop(0)
                order.append(cur)
                for nb in sorted(adj[cur], key=rank):
                    if not seen[nb]:
                        seen[nb] = True
                        queue.append(nb)
        return order


def _canonicalise(labels: Sequence) -> List[int]:
    """Map arbitrary hashable labels to dense ints 0..k-1 by sorted order."""
    uniq = sorted(set(labels))
    index = {lab: i for i, lab in enumerate(uniq)}
    return [index[lab] for lab in labels]
