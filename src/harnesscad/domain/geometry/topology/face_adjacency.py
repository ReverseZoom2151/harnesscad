"""Face-adjacency segmentation over a B-rep face graph (deterministic, stdlib-only).

Ported from QueryCAD's ``CADFaceUtils.prune_non_adj_faces`` and
``CADPartUtils.add_neighboring_faces_to_part``
(``src/cad_service/utils/``). After QueryCAD tags a set of candidate faces as
belonging to a queried machining feature (e.g. "all faces of a pocket"), those
tags are a flat *set* -- they do not yet distinguish one physical pocket from a
second identical pocket elsewhere on the part. QueryCAD resolves that purely from
topology: it walks the face-adjacency graph and keeps only faces that are
edge-connected to a chosen seed, so each connected component of the candidate set
becomes one physical part instance.

The kernel supplies only the adjacency relation (which faces share an edge); the
graph reasoning is pure and reimplemented here:

* :meth:`FaceAdjacencyGraph.connected_component` -- QueryCAD's ``__prune_faces``
  DFS: from a seed face, collect every face reachable through adjacency arcs that
  also lies in a *whitelist* (the same-feature candidate set). This is how a flat
  feature tag is split into concrete instances.
* :meth:`FaceAdjacencyGraph.partition` -- run the above over every seed to break
  a candidate set into its connected components (distinct part instances).
* :meth:`FaceAdjacencyGraph.dilate` -- QueryCAD's ``add_neighboring_faces``: grow
  a face set by one (or more) rings of edge-adjacent faces, the morphological
  dilation used to include the fillet/wall faces surrounding a feature.

Unlike ``geometry/cascade_entity_selector`` and ``geometry/cq_selector_algebra``
(which select faces by *geometric predicate* -- parallel, largest, within-box),
this module selects by *topological connectivity*, which those modules do not
model at all.

All traversals are iterative (no recursion-depth limit) and visit faces in
first-seen order for deterministic output.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Sequence, Set


@dataclass
class FaceAdjacencyGraph:
    """Undirected face graph: nodes are face ids, arcs are shared edges.

    Face ids are arbitrary hashables (typically ``int`` face indices). Adjacency
    is stored as an insertion-ordered list per node so traversals are
    reproducible.
    """

    _adj: Dict[int, List[int]] = field(default_factory=dict)

    def add_face(self, face: int) -> None:
        if face not in self._adj:
            self._adj[face] = []

    def add_adjacency(self, face_a: int, face_b: int) -> None:
        """Record that ``face_a`` and ``face_b`` share an edge."""
        if face_a == face_b:
            return
        self.add_face(face_a)
        self.add_face(face_b)
        if face_b not in self._adj[face_a]:
            self._adj[face_a].append(face_b)
        if face_a not in self._adj[face_b]:
            self._adj[face_b].append(face_a)

    @classmethod
    def from_edges(cls, edges: Sequence[Sequence[int]]) -> "FaceAdjacencyGraph":
        """Build from a list of ``(face_a, face_b)`` shared-edge pairs."""
        g = cls()
        for a, b in edges:
            g.add_adjacency(int(a), int(b))
        return g

    @property
    def faces(self) -> List[int]:
        return list(self._adj.keys())

    def neighbors(self, face: int) -> List[int]:
        return list(self._adj.get(face, ()))

    def connected_component(
        self, seed: int, whitelist: Iterable[int] | None = None
    ) -> List[int]:
        """Faces reachable from ``seed`` through adjacency, restricted to ``whitelist``.

        Reimplements QueryCAD's ``prune_non_adj_faces``: only faces that are in
        ``whitelist`` (default: the whole graph) are traversed, so the result is
        the connected component of ``seed`` *within* the candidate set. Iterative
        DFS; output is in first-seen order.
        """
        allowed: Set[int] | None = None if whitelist is None else set(whitelist)
        if allowed is not None and seed not in allowed:
            return []
        if seed not in self._adj:
            return [seed] if (allowed is None or seed in allowed) else []

        visited: Set[int] = set()
        order: List[int] = []
        stack = [seed]
        while stack:
            node = stack.pop()
            if node in visited:
                continue
            visited.add(node)
            order.append(node)
            # Push neighbours in reverse so first-listed is expanded first.
            for nb in reversed(self._adj.get(node, ())):
                if nb in visited:
                    continue
                if allowed is not None and nb not in allowed:
                    continue
                stack.append(nb)
        return order

    def partition(self, faces: Iterable[int]) -> List[List[int]]:
        """Split a face set into connected components (distinct part instances).

        Each returned component is edge-connected within ``faces``; components are
        ordered by the first-seen order of their seeds, and each component's faces
        are in traversal order. Deterministic.
        """
        pool = list(dict.fromkeys(faces))  # de-dup, preserve order
        allowed = set(pool)
        remaining = set(pool)
        components: List[List[int]] = []
        for seed in pool:
            if seed not in remaining:
                continue
            comp = self.connected_component(seed, whitelist=allowed)
            comp = [f for f in comp if f in remaining]
            for f in comp:
                remaining.discard(f)
            components.append(comp)
        return components

    def dilate(self, faces: Iterable[int], rings: int = 1) -> List[int]:
        """Grow a face set by ``rings`` of edge-adjacent faces (morphological dilation).

        Reimplements ``add_neighboring_faces_to_part``: for each face already in
        the set, add every edge-adjacent face. Repeats ``rings`` times. Output
        preserves the original faces first, then newly added faces in first-seen
        order; deterministic.
        """
        if rings < 0:
            raise ValueError("rings must be non-negative")
        result: List[int] = list(dict.fromkeys(faces))
        present: Set[int] = set(result)
        frontier = list(result)
        for _ in range(rings):
            new_frontier: List[int] = []
            for face in frontier:
                for nb in self._adj.get(face, ()):
                    if nb not in present:
                        present.add(nb)
                        result.append(nb)
                        new_frontier.append(nb)
            if not new_frontier:
                break
            frontier = new_frontier
        return result

    def boundary_faces(self, faces: Iterable[int]) -> List[int]:
        """Faces in the set that have at least one neighbour outside the set.

        Useful for isolating the walls of a feature region from its interior
        faces. Deterministic first-seen order.
        """
        member = set(faces)
        out: List[int] = []
        for face in dict.fromkeys(faces):
            for nb in self._adj.get(face, ()):
                if nb not in member:
                    out.append(face)
                    break
        return out
