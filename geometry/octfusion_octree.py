"""Volumetric octree over the unit cube for OctFusion's shape representation.

From "OctFusion: Octree-based Diffusion Models for 3D Shape Generation"
(Xiong, Wei, Zheng, Cao, Lian & Wang, 2024), Section 3.1. The learned VAE /
diffusion parts are external, but the *octree itself* is a purely deterministic
data structure, and the paper's description of it is precise:

  * "Given a mesh or a point cloud, we convert it to an octree by recursive
    subdividing nonempty octree nodes until the maximum depth is reached."
  * "All leaf nodes of the octree form an adaptive partition of the 3D volume."
  * "denote the i-th leaf node as v_i, with its center as o_i, its cell size as
    r_i" (used later by the MPU blend, see ``geometry.octfusion_mpu``).
  * depth <-> resolution: "octree nodes with depth 4 ... equivalent to full
    voxels with resolution 16^3", i.e. resolution = 2**depth per axis.

This module implements the classic *region octree*: a node is subdivided iff it
contains at least one sample point and its depth is below ``max_depth``; empty
regions remain large leaves and occupied regions refine down to ``max_depth``.
The leaves therefore tile the cube without overlap ("complete coverage of the
3D bounding volume" -- the paper's "Completeness" property). Child ordering is
Morton/Z order: child ``c`` has x-bit ``c & 1``, y-bit ``(c >> 1) & 1``, z-bit
``(c >> 2) & 1``.

Distinct from ``geometry.lasdiff_sparse_subdivision`` (which is flat coarse->fine
voxel book-keeping, no tree) and from ``geometry.meshdiff_tet_grid`` (a tet
lattice). Stdlib-only, deterministic.
"""

from __future__ import annotations

from typing import Dict, Iterator, List, Optional, Sequence, Set, Tuple

Point = Tuple[float, float, float]
Coord = Tuple[int, int, int]


class OctreeNode:
    """A single cubic cell of an octree.

    ``children`` is ``None`` for a leaf, else a length-8 list in Morton order.
    ``occupied`` records whether any sample point fell inside the cell.
    """

    __slots__ = ("depth", "ix", "iy", "iz", "children", "occupied")

    def __init__(self, depth: int, ix: int, iy: int, iz: int) -> None:
        self.depth = depth
        self.ix = ix
        self.iy = iy
        self.iz = iz
        self.children: Optional[List["OctreeNode"]] = None
        self.occupied: bool = False

    @property
    def is_leaf(self) -> bool:
        return self.children is None

    def resolution(self) -> int:
        """Voxels per axis at this node's depth (``2**depth``)."""
        return 1 << self.depth

    def cell_size(self, size: float = 1.0) -> float:
        """Edge length of this cell within a cube of side ``size``."""
        return size / (1 << self.depth)

    def bounds(self, origin: Point = (0.0, 0.0, 0.0), size: float = 1.0) -> Tuple[Point, Point]:
        """Lower and upper corner of the cell in world coordinates."""
        h = self.cell_size(size)
        lo = (origin[0] + self.ix * h, origin[1] + self.iy * h, origin[2] + self.iz * h)
        hi = (lo[0] + h, lo[1] + h, lo[2] + h)
        return lo, hi

    def center(self, origin: Point = (0.0, 0.0, 0.0), size: float = 1.0) -> Point:
        """Cell center ``o_i`` in world coordinates."""
        h = self.cell_size(size)
        return (
            origin[0] + (self.ix + 0.5) * h,
            origin[1] + (self.iy + 0.5) * h,
            origin[2] + (self.iz + 0.5) * h,
        )

    def key(self) -> Tuple[int, int, int, int]:
        """Stable identity ``(depth, ix, iy, iz)``."""
        return (self.depth, self.ix, self.iy, self.iz)


def _child_index(bx: int, by: int, bz: int) -> int:
    return (bz << 2) | (by << 1) | bx


class Octree:
    """A region octree over the axis-aligned cube ``[origin, origin+size]^3``."""

    def __init__(self, root: OctreeNode, max_depth: int, origin: Point, size: float) -> None:
        self.root = root
        self.max_depth = max_depth
        self.origin = origin
        self.size = size

    # -- construction ------------------------------------------------------

    @classmethod
    def from_points(
        cls,
        points: Sequence[Point],
        max_depth: int,
        origin: Point = (0.0, 0.0, 0.0),
        size: float = 1.0,
    ) -> "Octree":
        """Build by recursively subdividing nonempty nodes to ``max_depth``.

        Points outside the cube are ignored. A node is a leaf when it is empty
        or reaches ``max_depth``; it is ``occupied`` when it holds >=1 point.
        """
        if max_depth < 0:
            raise ValueError("max_depth must be >= 0")
        if size <= 0.0:
            raise ValueError("size must be positive")
        inside = [p for p in points if _in_cube(p, origin, size)]
        root = cls._build(inside, 0, 0, 0, 0, max_depth, origin, size)
        return cls(root, max_depth, origin, size)

    @staticmethod
    def _build(
        pts: List[Point],
        depth: int,
        ix: int,
        iy: int,
        iz: int,
        max_depth: int,
        origin: Point,
        size: float,
    ) -> OctreeNode:
        node = OctreeNode(depth, ix, iy, iz)
        node.occupied = len(pts) > 0
        if depth >= max_depth or not pts:
            return node
        h = size / (1 << depth)
        mx = origin[0] + (ix + 0.5) * h
        my = origin[1] + (iy + 0.5) * h
        mz = origin[2] + (iz + 0.5) * h
        buckets: List[List[Point]] = [[] for _ in range(8)]
        for p in pts:
            bx = 1 if p[0] >= mx else 0
            by = 1 if p[1] >= my else 0
            bz = 1 if p[2] >= mz else 0
            buckets[_child_index(bx, by, bz)].append(p)
        node.children = []
        for c in range(8):
            bx = c & 1
            by = (c >> 1) & 1
            bz = (c >> 2) & 1
            node.children.append(
                Octree._build(
                    buckets[c],
                    depth + 1,
                    2 * ix + bx,
                    2 * iy + by,
                    2 * iz + bz,
                    max_depth,
                    origin,
                    size,
                )
            )
        return node

    @classmethod
    def from_voxels(
        cls,
        occupied: Sequence[Coord],
        max_depth: int,
        origin: Point = (0.0, 0.0, 0.0),
        size: float = 1.0,
    ) -> "Octree":
        """Build an octree from an occupied voxel set at resolution ``2**max_depth``.

        A node is subdivided iff any occupied voxel lies within it; occupied
        voxels refine down to ``max_depth``. This is the voxel->octree direction
        (inverse of :meth:`to_voxels`).
        """
        if max_depth < 0:
            raise ValueError("max_depth must be >= 0")
        r = 1 << max_depth
        occ: Set[Coord] = set()
        for (x, y, z) in occupied:
            if 0 <= x < r and 0 <= y < r and 0 <= z < r:
                occ.add((x, y, z))
        root = cls._build_from_voxels(occ, 0, 0, 0, 0, max_depth)
        return cls(root, max_depth, origin, size)

    @staticmethod
    def _build_from_voxels(
        occ: Set[Coord], depth: int, ix: int, iy: int, iz: int, max_depth: int
    ) -> OctreeNode:
        node = OctreeNode(depth, ix, iy, iz)
        node.occupied = len(occ) > 0
        if depth >= max_depth or not occ:
            return node
        # shift factor mapping this node's occupied voxels to child octants
        shift = max_depth - depth - 1
        buckets: List[Set[Coord]] = [set() for _ in range(8)]
        for (x, y, z) in occ:
            bx = (x >> shift) & 1
            by = (y >> shift) & 1
            bz = (z >> shift) & 1
            buckets[_child_index(bx, by, bz)].add((x, y, z))
        node.children = []
        for c in range(8):
            bx = c & 1
            by = (c >> 1) & 1
            bz = (c >> 2) & 1
            node.children.append(
                Octree._build_from_voxels(
                    buckets[c], depth + 1, 2 * ix + bx, 2 * iy + by, 2 * iz + bz, max_depth
                )
            )
        return node

    # -- traversal ---------------------------------------------------------

    def leaves(self) -> Iterator[OctreeNode]:
        """Yield all leaf nodes in deterministic depth-first Morton order."""
        stack = [self.root]
        while stack:
            node = stack.pop()
            if node.is_leaf:
                yield node
            else:
                # push in reverse so children pop in ascending Morton order
                for c in range(7, -1, -1):
                    stack.append(node.children[c])  # type: ignore[index]

    def occupied_leaves(self) -> Iterator[OctreeNode]:
        """Yield only surface/occupied leaves ("voxels intersecting the shape")."""
        for leaf in self.leaves():
            if leaf.occupied:
                yield leaf

    def nodes(self) -> Iterator[OctreeNode]:
        """Yield every node (internal and leaf) in Morton DFS order."""
        stack = [self.root]
        while stack:
            node = stack.pop()
            yield node
            if not node.is_leaf:
                for c in range(7, -1, -1):
                    stack.append(node.children[c])  # type: ignore[index]

    def leaf_count(self) -> int:
        return sum(1 for _ in self.leaves())

    def node_count(self) -> int:
        return sum(1 for _ in self.nodes())

    def occupied_leaf_count(self) -> int:
        return sum(1 for _ in self.occupied_leaves())

    def depth_reached(self) -> int:
        """Deepest depth of any node (may be < ``max_depth`` if data is coarse)."""
        return max(node.depth for node in self.nodes())

    # -- queries -----------------------------------------------------------

    def find_leaf(self, point: Point) -> Optional[OctreeNode]:
        """Return the leaf whose cell contains ``point`` (point-location query).

        Returns ``None`` if the point is outside the cube.
        """
        if not _in_cube(point, self.origin, self.size):
            return None
        node = self.root
        while not node.is_leaf:
            _, hi = _midpoint(node, self.origin, self.size)
            bx = 1 if point[0] >= hi[0] else 0
            by = 1 if point[1] >= hi[1] else 0
            bz = 1 if point[2] >= hi[2] else 0
            node = node.children[_child_index(bx, by, bz)]  # type: ignore[index]
        return node

    def face_neighbor(self, node: OctreeNode, axis: int, sign: int) -> Optional[OctreeNode]:
        """Leaf adjacent to ``node`` across the face on ``axis`` in ``sign`` dir.

        ``axis`` is 0/1/2 (x/y/z), ``sign`` is +1 or -1. Returns the leaf whose
        cell contains the center of the neighbouring same-depth cell, or ``None``
        if that cell falls outside the cube. When the neighbour is a coarser
        leaf, that (larger) leaf is returned; when finer, the leaf containing the
        probe point is returned.
        """
        if axis not in (0, 1, 2):
            raise ValueError("axis must be 0, 1 or 2")
        if sign not in (1, -1):
            raise ValueError("sign must be +1 or -1")
        h = node.cell_size(self.size)
        cx, cy, cz = node.center(self.origin, self.size)
        probe = [cx, cy, cz]
        probe[axis] += sign * h
        return self.find_leaf((probe[0], probe[1], probe[2]))

    # -- conversion --------------------------------------------------------

    def to_voxels(self, depth: Optional[int] = None) -> Set[Coord]:
        """Rasterise occupied leaves to a dense voxel set at ``depth``.

        ``depth`` defaults to ``max_depth``. An occupied leaf at depth ``d``
        covers ``(2**(depth-d))**3`` voxels. Leaves deeper than ``depth`` are
        collapsed to the single voxel containing their cell.
        """
        target = self.max_depth if depth is None else depth
        if target < 0:
            raise ValueError("depth must be >= 0")
        out: Set[Coord] = set()
        for leaf in self.occupied_leaves():
            if leaf.depth <= target:
                f = 1 << (target - leaf.depth)
                bx, by, bz = leaf.ix * f, leaf.iy * f, leaf.iz * f
                for a in range(f):
                    for b in range(f):
                        for c in range(f):
                            out.add((bx + a, by + b, bz + c))
            else:
                shift = leaf.depth - target
                out.add((leaf.ix >> shift, leaf.iy >> shift, leaf.iz >> shift))
        return out


def _in_cube(p: Point, origin: Point, size: float) -> bool:
    return all(origin[i] <= p[i] <= origin[i] + size for i in range(3))


def _midpoint(node: OctreeNode, origin: Point, size: float) -> Tuple[Point, Point]:
    lo, _ = node.bounds(origin, size)
    h = node.cell_size(size)
    mid = (lo[0] + h / 2.0, lo[1] + h / 2.0, lo[2] + h / 2.0)
    return lo, mid
