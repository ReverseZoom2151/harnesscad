"""Deterministic voxel part-decomposition and assembly for TAR3D.

From "TAR3D: Creating High-Quality 3D Assets via Next-Part Prediction" (Zhang et
al., 2024). TAR3D's central idea is that "the composition of 3D geometries can be
modeled part by part" and objects are "created by autoregressively generating
discrete geometric parts". The learned codebook that names the parts is external,
but *decomposing an asset into an ordered list of parts* and *assembling parts
back into the asset* are deterministic operations, implemented here.

Given a voxel occupancy grid, a **part** is a 6-connected component of occupied
voxels (a connected sub-shape with a bounding region -- the "geometric part" the
paper composes). Parts are placed into the **canonical part order** the paper's
sequence builder needs: raster-scan over each part's minimum corner (z, y, x),
tie-broken by size then sorted voxel content, so the ordering is total and
seed-free. Assembly is the inverse (union of parts) with validity checks:
disjointness (no two parts overlap) and coverage (parts reproduce the asset).

This is voxel-native and distinct from ``geometry.jointsdf_mesh_segments``
(face-label connected components on a *mesh*) and ``geometry.octfusion_octree``
(a space-partitioning tree, not shape components). Stdlib-only, deterministic.
"""

from __future__ import annotations

from typing import Dict, Iterable, List, Sequence, Set, Tuple

Voxel = Tuple[int, int, int]

# 6-connectivity offsets (face neighbours).
_NEIGHBORS_6 = (
    (1, 0, 0), (-1, 0, 0),
    (0, 1, 0), (0, -1, 0),
    (0, 0, 1), (0, 0, -1),
)


class VoxelPart:
    """One connected sub-shape: its voxels plus a cached axis-aligned box."""

    def __init__(self, voxels: Iterable[Voxel]):
        vs = frozenset(voxels)
        if not vs:
            raise ValueError("a part must have at least one voxel")
        self.voxels = vs

    def __len__(self) -> int:
        return len(self.voxels)

    def __eq__(self, other) -> bool:
        return isinstance(other, VoxelPart) and other.voxels == self.voxels

    def __hash__(self) -> int:
        return hash(self.voxels)

    def bounds(self) -> Tuple[Voxel, Voxel]:
        """Return ``(min_corner, max_corner)`` of the part's bounding box."""
        xs = [v[0] for v in self.voxels]
        ys = [v[1] for v in self.voxels]
        zs = [v[2] for v in self.voxels]
        return (min(xs), min(ys), min(zs)), (max(xs), max(ys), max(zs))

    def min_corner(self) -> Voxel:
        return self.bounds()[0]

    def _sort_key(self) -> Tuple:
        # Raster order over the min corner in (z, y, x), then size, then content.
        x, y, z = self.min_corner()
        return (z, y, x, len(self.voxels), tuple(sorted(self.voxels)))


def connected_parts(occupied: Iterable[Voxel]) -> List[VoxelPart]:
    """Split ``occupied`` into 6-connected components as :class:`VoxelPart`s.

    The returned list is in **canonical part order** (see module docstring), so
    two runs over the same shape always emit the same sequence of parts.
    """
    remaining: Set[Voxel] = set(occupied)
    parts: List[VoxelPart] = []
    # Deterministic seeding: always start a flood from the smallest unvisited
    # voxel, so component discovery order does not depend on set iteration.
    while remaining:
        seed = min(remaining)
        stack = [seed]
        remaining.discard(seed)
        comp: List[Voxel] = []
        while stack:
            v = stack.pop()
            comp.append(v)
            x, y, z = v
            for dx, dy, dz in _NEIGHBORS_6:
                n = (x + dx, y + dy, z + dz)
                if n in remaining:
                    remaining.discard(n)
                    stack.append(n)
        parts.append(VoxelPart(comp))
    parts.sort(key=lambda p: p._sort_key())
    return parts


def part_order_key(part: VoxelPart) -> Tuple:
    """Expose the canonical ordering key (raster over min corner, size, content)."""
    return part._sort_key()


def assemble(parts: Sequence[VoxelPart]) -> Set[Voxel]:
    """Union the parts back into a single occupied-voxel set."""
    out: Set[Voxel] = set()
    for p in parts:
        out |= set(p.voxels)
    return out


def parts_disjoint(parts: Sequence[VoxelPart]) -> bool:
    """True iff no two parts share a voxel (a valid decomposition)."""
    seen: Set[Voxel] = set()
    for p in parts:
        for v in p.voxels:
            if v in seen:
                return False
            seen.add(v)
    return True


def covers(parts: Sequence[VoxelPart], occupied: Iterable[Voxel]) -> bool:
    """True iff assembling ``parts`` reproduces ``occupied`` exactly."""
    return assemble(parts) == set(occupied)


def is_valid_decomposition(parts: Sequence[VoxelPart],
                           occupied: Iterable[Voxel]) -> bool:
    """A decomposition is valid iff parts are disjoint and cover the asset."""
    return parts_disjoint(parts) and covers(parts, occupied)
