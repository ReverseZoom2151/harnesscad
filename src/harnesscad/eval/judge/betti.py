"""Betti numbers of a tessellated boundary, computed from a triangle mesh.

Mined from CADGenBench's topology-match pipeline (docs/metrics/topo_match.md).
Given a *watertight, manifold, orientation-consistent* triangle surface, the
three Betti numbers of the enclosed solid are:

* ``b0`` -- connected solid components (pieces of material),
* ``b1`` -- independent through-handles (through-holes),
* ``b2`` -- enclosed internal voids (cavities).

The upstream splits outer shells (``b0``) from void shells (``b2``) by ray-cast
containment, which needs a geometry kernel. This module extracts the two purely
combinatorial ingredients that need no kernel:

* :func:`surface_components` -- number of connected triangle-surface shells,
  by union-find over shared vertices,
* :func:`euler_characteristic` -- ``chi = V - E + F`` over the whole mesh,

and combines them via the manifold identity ``chi = 2(b0 - b1 + b2)`` into a
full Betti triple (:func:`betti_from_mesh`) once the caller supplies the void
count (``n_voids``, default 0 for parts with no internal cavities). A shell is
either an outer boundary of a solid component (``b0``) or the boundary of a void
(``b2``), so ``shells = b0 + b2`` and ``b1 = shells - chi/2``.

Everything is stdlib-only and deterministic. A ``face`` is a triple of integer
vertex indices.
"""

from __future__ import annotations

from typing import Dict, List, Sequence, Tuple

__all__ = [
    "euler_characteristic",
    "surface_components",
    "betti_from_mesh",
]

Face = Sequence[int]


class _UnionFind:
    def __init__(self) -> None:
        self._parent: Dict[int, int] = {}

    def find(self, x: int) -> int:
        p = self._parent.setdefault(x, x)
        while p != x:
            self._parent[x] = self._parent[p]  # path halving
            x, p = p, self._parent[p]
        return x

    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self._parent[ra] = rb

    def roots(self) -> int:
        return len({self.find(x) for x in self._parent})


def _unique_edges(faces: Sequence[Face]) -> int:
    edges = set()
    for f in faces:
        if len(f) != 3:
            raise ValueError("every face must be a triangle (3 vertex indices)")
        a, b, c = f
        edges.add((a, b) if a < b else (b, a))
        edges.add((b, c) if b < c else (c, b))
        edges.add((a, c) if a < c else (c, a))
    return len(edges)


def euler_characteristic(faces: Sequence[Face]) -> int:
    """Euler characteristic ``chi = V - E + F`` of a triangle mesh.

    ``V`` is the number of distinct referenced vertex indices, ``E`` the number
    of distinct undirected edges, ``F`` the triangle count. For a closed
    orientable genus-g surface this equals ``2 - 2g``.
    """
    if not faces:
        return 0
    verts = set()
    for f in faces:
        verts.update(f)
    V = len(verts)
    E = _unique_edges(faces)
    F = len(faces)
    return V - E + F


def surface_components(faces: Sequence[Face]) -> int:
    """Number of connected surface shells (union-find over shared vertices)."""
    if not faces:
        return 0
    uf = _UnionFind()
    for f in faces:
        a, b, c = f
        uf.union(a, b)
        uf.union(b, c)
    return uf.roots()


def betti_from_mesh(faces: Sequence[Face], *, n_voids: int = 0) -> Tuple[int, int, int]:
    """Full Betti triple ``(b0, b1, b2)`` of the solid bounded by *faces*.

    ``n_voids`` is the number of shells that are void boundaries rather than
    outer solid boundaries (0 for parts with no internal cavities). With
    ``shells = surface_components`` and ``chi = euler_characteristic``:

        b2 = n_voids
        b0 = shells - n_voids
        b1 = b0 + b2 - chi // 2   ==   shells - chi // 2

    Raises ``ValueError`` if ``n_voids`` exceeds the shell count.
    """
    shells = surface_components(faces)
    if n_voids < 0 or n_voids > shells:
        raise ValueError(f"n_voids={n_voids} inconsistent with {shells} shells")
    chi = euler_characteristic(faces)
    if chi % 2 != 0:
        # chi is always even for a closed orientable manifold; an odd value
        # means the mesh is not a clean closed surface.
        raise ValueError(f"odd Euler characteristic {chi}: mesh is not closed/manifold")
    b2 = n_voids
    b0 = shells - n_voids
    b1 = shells - chi // 2
    return (b0, b1, b2)
