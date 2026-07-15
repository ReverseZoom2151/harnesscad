"""Flat sketch representation via symmetric difference (HistCAD, Dong et al. 2026).

HistCAD ("A Constraint-Aware Parametric History-Based CAD Representation, Dataset,
and Benchmark with Industrial Complexity", Dong et al. 2026, Sec. 3.2) replaces the
usual *nested* sketch encoding -- faces bounded by loops, each loop a list of
sub-primitives -- with a compact **flat set** of atomic sub-primitives. Prior formats
serialise every face boundary, so an interior edge shared by two neighbouring faces is
written twice; that is redundant.

The paper's key construction (Sec. 3.2, and the boundary-equivalence proposition in
Appendix A) is::

    P_flat = symmetric_difference over all selected face boundaries of their edge sets

i.e. an atomic sub-primitive ``e`` is *retained* iff it appears on an **odd** number of
selected face boundaries. Under the stated assumptions (each atomic sub-primitive is
incident to at most two faces, and no face boundary traverses the same sub-primitive
more than once) this exactly recovers the geometric boundary ``P_hier = boundary(U)``
of the selected region ``U`` = union of the faces:

    * a shared *interior* edge appears on two face boundaries -> cancels (even count),
    * an outer-contour or hole-boundary edge appears once -> retained (odd count).

This module implements that flattening deterministically over hashable, orientation-
independent edge keys, plus the inverse *loop recovery* (reassemble loops from the flat
primitive connectivity) and collinear-fragment merging mentioned in Sec. 3.3.

Pure stdlib, deterministic (sorted outputs, no wall clock, no randomness).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Hashable, Iterable, List, Sequence, Tuple

__all__ = [
    "SubPrimitive",
    "flatten_faces",
    "flatten_edge_keys",
    "is_boundary_valid",
    "recover_loops",
    "merge_collinear",
]

# A sub-primitive is identified by an orientation-independent *key*. Endpoints are
# the two vertices it connects (used for loop recovery and collinear merging).
Vertex = Tuple[float, float]


@dataclass(frozen=True)
class SubPrimitive:
    """One atomic sketch sub-primitive (a line/arc fragment between two endpoints).

    ``key`` is any hashable identity for the underlying geometry that is *the same*
    for the fragment regardless of which face lists it (so a shared interior edge has
    one key). If ``key`` is ``None`` a canonical key is derived from the (unordered)
    endpoint pair, rounded to ``ndigits`` to absorb quantisation noise.
    """

    a: Vertex
    b: Vertex
    key: Hashable = None
    ndigits: int = 6

    def canonical_key(self) -> Hashable:
        if self.key is not None:
            return self.key
        pa = (round(self.a[0], self.ndigits), round(self.a[1], self.ndigits))
        pb = (round(self.b[0], self.ndigits), round(self.b[1], self.ndigits))
        return (pa, pb) if pa <= pb else (pb, pa)


def flatten_edge_keys(face_boundaries: Sequence[Iterable[Hashable]]) -> List[Hashable]:
    """Symmetric difference over faces given as iterables of edge *keys*.

    Returns the sorted list of keys that appear on an **odd** number of face
    boundaries -- HistCAD Eq. in Sec. 3.2. Duplicate keys *within* one face boundary
    are ignored (the proposition assumes a boundary never traverses an edge twice),
    so each face contributes a set.
    """
    counts: Dict[Hashable, int] = {}
    for boundary in face_boundaries:
        for k in set(boundary):
            counts[k] = counts.get(k, 0) + 1
    retained = [k for k, c in counts.items() if c % 2 == 1]
    return sorted(retained, key=_sort_key)


def flatten_faces(faces: Sequence[Sequence[SubPrimitive]]) -> List[SubPrimitive]:
    """Flatten hierarchical faces of :class:`SubPrimitive` into the flat boundary set.

    Interior edges shared by two faces (same canonical key) cancel; boundary/hole
    edges are retained. Returns the retained sub-primitives, de-duplicated by key and
    sorted for determinism.
    """
    by_key: Dict[Hashable, SubPrimitive] = {}
    counts: Dict[Hashable, int] = {}
    for face in faces:
        seen = set()
        for sp in face:
            k = sp.canonical_key()
            if k in seen:
                continue
            seen.add(k)
            counts[k] = counts.get(k, 0) + 1
            by_key.setdefault(k, sp)
    retained = [by_key[k] for k, c in counts.items() if c % 2 == 1]
    return sorted(retained, key=lambda sp: _sort_key(sp.canonical_key()))


def is_boundary_valid(faces: Sequence[Sequence[SubPrimitive]]) -> bool:
    """Check the proposition's precondition: no atomic edge is incident to >2 faces.

    HistCAD's boundary equivalence assumes each sub-primitive lies on at most two
    selected face boundaries. If an edge appears three+ times the symmetric-difference
    flattening is not guaranteed to equal the geometric boundary; this predicate lets a
    caller detect that.
    """
    counts: Dict[Hashable, int] = {}
    for face in faces:
        for sp in set(sp.canonical_key() for sp in face):
            counts[sp] = counts.get(sp, 0) + 1
    return all(c <= 2 for c in counts.values())


def recover_loops(primitives: Sequence[SubPrimitive], ndigits: int = 6) -> List[List[SubPrimitive]]:
    """Reassemble closed loops from a flat primitive set via endpoint connectivity.

    HistCAD stores primitives *unordered*; downstream 3D ops need loops. This walks the
    endpoint graph, emitting each connected closed cycle. Determinism: adjacency lists
    and start vertices are visited in sorted order. Assumes every vertex has even degree
    (a valid boundary); dangling/odd-degree fragments are skipped.
    """
    def vkey(v: Vertex) -> Vertex:
        return (round(v[0], ndigits), round(v[1], ndigits))

    adj: Dict[Vertex, List[Tuple[Vertex, int]]] = {}
    for i, sp in enumerate(primitives):
        a, b = vkey(sp.a), vkey(sp.b)
        adj.setdefault(a, []).append((b, i))
        adj.setdefault(b, []).append((a, i))

    used = [False] * len(primitives)
    loops: List[List[SubPrimitive]] = []
    for start in sorted(adj):
        for nxt, ei in sorted(adj[start], key=lambda t: (_sort_key(t[0]), t[1])):
            if used[ei]:
                continue
            loop: List[SubPrimitive] = []
            cur, came = start, ei
            # Walk until we return to start.
            v = start
            e = ei
            # Emit first edge
            while True:
                used[e] = True
                loop.append(primitives[e])
                # Move to the other endpoint of e
                sp = primitives[e]
                other = vkey(sp.b) if vkey(sp.a) == v else vkey(sp.a)
                v = other
                if v == start:
                    break
                # Pick next unused edge at v
                cand = [(w, idx) for (w, idx) in sorted(adj.get(v, []), key=lambda t: (_sort_key(t[0]), t[1])) if not used[idx]]
                if not cand:
                    break
                e = cand[0][1]
            if loop and v == start:
                loops.append(loop)
    return loops


def merge_collinear(loop: Sequence[SubPrimitive], ndigits: int = 6) -> List[SubPrimitive]:
    """Merge consecutive collinear line fragments in an ordered loop (HistCAD Sec. 3.3).

    Two consecutive sub-primitives that share an endpoint and are collinear are fused
    into one straight segment. Arcs (marked by ``key`` starting with ``('arc',``) are
    never merged. Deterministic, closed-form cross-product collinearity test.
    """
    if len(loop) < 2:
        return list(loop)

    def is_line(sp: SubPrimitive) -> bool:
        k = sp.key
        return not (isinstance(k, tuple) and k and k[0] == "arc")

    out: List[SubPrimitive] = list(loop)
    changed = True
    while changed and len(out) > 1:
        changed = False
        merged: List[SubPrimitive] = []
        i = 0
        while i < len(out):
            if i + 1 < len(out):
                s1, s2 = out[i], out[i + 1]
                shared = _shared_endpoint(s1, s2, ndigits)
                if is_line(s1) and is_line(s2) and shared is not None:
                    p = _other(s1, shared, ndigits)
                    q = _other(s2, shared, ndigits)
                    if _collinear(p, shared, q):
                        merged.append(SubPrimitive(p, q, ndigits=ndigits))
                        i += 2
                        changed = True
                        continue
            merged.append(out[i])
            i += 1
        out = merged
    return out


# --- helpers ---------------------------------------------------------------
def _sort_key(k: Hashable):
    return (0, k) if isinstance(k, (int, float, str)) else (1, repr(k))


def _shared_endpoint(s1: SubPrimitive, s2: SubPrimitive, nd: int):
    def r(v):
        return (round(v[0], nd), round(v[1], nd))
    e1 = {r(s1.a), r(s1.b)}
    e2 = {r(s2.a), r(s2.b)}
    common = e1 & e2
    return next(iter(common)) if len(common) == 1 else None


def _other(sp: SubPrimitive, shared, nd: int):
    def r(v):
        return (round(v[0], nd), round(v[1], nd))
    return r(sp.b) if r(sp.a) == shared else r(sp.a)


def _collinear(p, q, r, tol: float = 1e-9) -> bool:
    cross = (q[0] - p[0]) * (r[1] - p[1]) - (q[1] - p[1]) * (r[0] - p[0])
    return abs(cross) <= tol
