"""Cubical-complex Betti numbers of a 3D voxel shape (Topology-Aware LDM).

Hu, Fei et al., "Topology-Aware Latent Diffusion for 3D Shape Generation"
(2024), Sec. 3, represent a 3D shape as an implicit field and build a *cubical
complex* from it to perform topological analysis.  Each occupied voxel is a unit
3-cell (``c3``) whose faces (``c2``), edges (``c1``) and vertices (``c0``) are
also in the complex (the closure condition of Sec. 3, "Cubical complex").  The
``d``-th Betti number ``beta_d = rank(H_d)`` counts:

  * ``beta_0`` -- connected components,
  * ``beta_1`` -- independent loops / tunnels / handles,
  * ``beta_2`` -- enclosed voids (cavities).

This is *distinct* from ``bench/evocad_topology_metrics.py`` and
``bench/cadmium_mesh_metrics.py``, which compute the Euler characteristic
``chi = V - E + F`` of a *surface mesh* (2-complex).  Here we work directly from
a solid *voxel occupancy grid* (a 3-complex), recover the full Betti vector
``(beta_0, beta_1, beta_2)`` -- not just ``chi`` -- and separate handles
(``beta_1``) from cavities (``beta_2``), which a single scalar ``chi`` cannot.

Method (all exact, deterministic, stdlib only):

  * ``chi`` of the solid cubical complex is counted exactly from its cells:
    ``chi = |c0| - |c1| + |c2| - |c3|`` over the *union* of cells of all occupied
    voxels (shared cells counted once).
  * ``beta_0`` -- 6-connected components of the occupied voxels (two cubes are
    adjacent iff they share a 2-face).
  * ``beta_2`` -- 6-connected components of the *background* inside a 1-voxel
    padded bounding box that do **not** touch the padded boundary; these are the
    enclosed cavities.
  * ``beta_1 = beta_0 + beta_2 - chi`` (rearranging
    ``chi = beta_0 - beta_1 + beta_2`` for a solid, where ``beta_d = 0`` for
    ``d >= 3``).

A voxel shape is a set of integer ``(x, y, z)`` cells.  Genus (handles) of a
single connected, cavity-free solid equals ``beta_1``.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Sequence, Set, Tuple

Voxel = Tuple[int, int, int]

# 6-connectivity face-neighbour offsets.
_NEIGHBORS_6 = (
    (1, 0, 0), (-1, 0, 0),
    (0, 1, 0), (0, -1, 0),
    (0, 0, 1), (0, 0, -1),
)


def _as_voxel_set(voxels: Iterable[Voxel]) -> Set[Voxel]:
    out: Set[Voxel] = set()
    for v in voxels:
        x, y, z = v
        out.add((int(x), int(y), int(z)))
    return out


def voxels_from_grid(grid: Sequence[Sequence[Sequence[int]]]) -> Set[Voxel]:
    """Occupied ``(x, y, z)`` cells from a nested ``grid[x][y][z]`` (truthy=solid)."""
    out: Set[Voxel] = set()
    for x, plane in enumerate(grid):
        for y, row in enumerate(plane):
            for z, val in enumerate(row):
                if val:
                    out.add((x, y, z))
    return out


def cubical_cell_counts(voxels: Iterable[Voxel]) -> Tuple[int, int, int, int]:
    """Return ``(|c0|, |c1|, |c2|, |c3|)`` of the solid cubical complex.

    Each occupied voxel ``(x, y, z)`` is the unit cube spanning
    ``[x, x+1] x [y, y+1] x [z, z+1]``; its 8 vertices, 12 edges and 6 square
    faces are shared with neighbours and counted once via set membership.
    """
    vox = _as_voxel_set(voxels)
    verts: Set[Tuple[int, int, int]] = set()
    # An edge is identified by its two endpoint lattice vertices (ordered).
    edges: Set[Tuple[Tuple[int, int, int], Tuple[int, int, int]]] = set()
    # A face is identified by the frozenset of its 4 corner vertices.
    faces: Set[frozenset] = set()
    for (x, y, z) in vox:
        corners = [
            (x + dx, y + dy, z + dz)
            for dx in (0, 1) for dy in (0, 1) for dz in (0, 1)
        ]
        verts.update(corners)
        # 12 edges: pairs of corners differing in exactly one axis by 1.
        for i in range(len(corners)):
            for j in range(i + 1, len(corners)):
                a, b = corners[i], corners[j]
                diff = sum(1 for k in range(3) if a[k] != b[k])
                if diff == 1:
                    edges.add((a, b))
        # 6 faces: fix one axis at its low/high value.
        for axis in range(3):
            for fixed in (0, 1):
                face = frozenset(
                    c for c in corners if c[axis] - (x, y, z)[axis] == fixed
                )
                faces.add(face)
    return len(verts), len(edges), len(faces), len(vox)


def cubical_euler_characteristic(voxels: Iterable[Voxel]) -> int:
    """``chi = |c0| - |c1| + |c2| - |c3|`` of the solid cubical complex."""
    c0, c1, c2, c3 = cubical_cell_counts(voxels)
    return c0 - c1 + c2 - c3


def connected_components(voxels: Iterable[Voxel]) -> int:
    """Number of 6-connected components of the occupied voxels (``beta_0``)."""
    vox = _as_voxel_set(voxels)
    seen: Set[Voxel] = set()
    count = 0
    for start in vox:
        if start in seen:
            continue
        count += 1
        stack = [start]
        seen.add(start)
        while stack:
            cx, cy, cz = stack.pop()
            for dx, dy, dz in _NEIGHBORS_6:
                nb = (cx + dx, cy + dy, cz + dz)
                if nb in vox and nb not in seen:
                    seen.add(nb)
                    stack.append(nb)
    return count


def cavity_count(voxels: Iterable[Voxel]) -> int:
    """Number of enclosed voids (``beta_2``): background components not reaching
    the boundary of a 1-voxel padded bounding box."""
    vox = _as_voxel_set(voxels)
    if not vox:
        return 0
    xs = [v[0] for v in vox]
    ys = [v[1] for v in vox]
    zs = [v[2] for v in vox]
    lo = (min(xs) - 1, min(ys) - 1, min(zs) - 1)
    hi = (max(xs) + 1, max(ys) + 1, max(zs) + 1)

    def in_box(p: Voxel) -> bool:
        return all(lo[k] <= p[k] <= hi[k] for k in range(3))

    # Flood the exterior background starting from a padded corner.
    exterior: Set[Voxel] = set()
    start = lo
    stack = [start]
    exterior.add(start)
    while stack:
        cx, cy, cz = stack.pop()
        for dx, dy, dz in _NEIGHBORS_6:
            nb = (cx + dx, cy + dy, cz + dz)
            if in_box(nb) and nb not in vox and nb not in exterior:
                exterior.add(nb)
                stack.append(nb)

    # Any background cell in the box not reached from outside is a cavity cell;
    # count its connected components.
    seen: Set[Voxel] = set()
    cavities = 0
    for x in range(lo[0], hi[0] + 1):
        for y in range(lo[1], hi[1] + 1):
            for z in range(lo[2], hi[2] + 1):
                p = (x, y, z)
                if p in vox or p in exterior or p in seen:
                    continue
                cavities += 1
                stack = [p]
                seen.add(p)
                while stack:
                    cx, cy, cz = stack.pop()
                    for dx, dy, dz in _NEIGHBORS_6:
                        nb = (cx + dx, cy + dy, cz + dz)
                        if (in_box(nb) and nb not in vox
                                and nb not in exterior and nb not in seen):
                            seen.add(nb)
                            stack.append(nb)
    return cavities


@dataclass(frozen=True)
class BettiNumbers:
    """Betti vector of a voxel shape plus derived invariants."""

    b0: int  # connected components
    b1: int  # loops / tunnels / handles
    b2: int  # enclosed voids (cavities)
    euler: int

    @property
    def vector(self) -> Tuple[int, int, int]:
        return (self.b0, self.b1, self.b2)

    @property
    def genus(self) -> int:
        """Handles of a single connected, cavity-free solid (``= beta_1``)."""
        return self.b1


def betti_numbers(voxels: Iterable[Voxel]) -> BettiNumbers:
    """Compute ``(beta_0, beta_1, beta_2)`` and ``chi`` of a voxel shape."""
    vox = _as_voxel_set(voxels)
    if not vox:
        return BettiNumbers(0, 0, 0, 0)
    chi = cubical_euler_characteristic(vox)
    b0 = connected_components(vox)
    b2 = cavity_count(vox)
    # chi = b0 - b1 + b2  ->  b1 = b0 + b2 - chi.
    b1 = b0 + b2 - chi
    return BettiNumbers(b0, b1, b2, chi)


def genus(voxels: Iterable[Voxel]) -> int:
    """Total handle count ``sum(beta_1)`` of the shape (== ``beta_1``)."""
    return betti_numbers(voxels).b1
