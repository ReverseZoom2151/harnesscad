"""3D bounding-volume hierarchy for broad-phase collision (Manifold ``Collider``).

Manifold's boolean does broad-phase overlap detection with a bounding-volume
hierarchy (``src/collider.h``): each triangle gets an axis-aligned bounding box
(``Box``) and a 30-bit Morton code from its centroid; a binary radix tree is
built over the Morton-sorted leaves (Karras' parallel LBVH construction) and
internal-node boxes are the union of their children.  Overlap queries do a
stackful depth-first traversal, recording every leaf whose box overlaps the
query box; a ``selfCollision`` flag skips the ``i == i`` self report.

This module reimplements the transferable algorithm in stdlib Python:

* :class:`AABB` axis-aligned box with :meth:`overlaps`, :meth:`union`,
  :meth:`contains_point` and :meth:`center`;
* :func:`morton3` -- the exact ``SpreadBits3`` bit-interleave of a 10-bit-per-
  axis quantised centroid used by Manifold to order leaves;
* :class:`BVH` built by a deterministic top-down median split on the Morton
  order (equivalently the longest-axis midpoint; both are provided), storing
  union boxes at internal nodes;
* :meth:`BVH.query` returning every leaf index whose box overlaps a query box
  via the same stack DFS as ``Collider::Collisions``;
* :meth:`BVH.self_collisions` returning all overlapping leaf *pairs* (broad
  phase for self-intersection), skipping ``i == i``.

The harness had **no 3D spatial index**: only ``geometry.arcs_quadtree_space``
(a 2D point quadtree), ``geometry.octfusion_octree`` (a fixed-grid occupancy
octree, not a BVH over arbitrary boxes) and ``ingest.spatial_order.morton2`` (a
2D Morton key).  A 3D BVH over triangle boxes with an overlap-pair query is new
and is the broad-phase every mesh boolean / self-intersection test needs.

Pure stdlib, deterministic (median split; ties broken by leaf index).
"""

from __future__ import annotations

import math
from typing import Callable, List, Optional, Sequence, Tuple

__all__ = [
    "AABB",
    "morton3",
    "spread_bits3",
    "BVH",
    "boxes_of_triangles",
    "brute_force_pairs",
]

Vec3 = Tuple[float, float, float]


class AABB:
    """Axis-aligned bounding box.  An empty box has ``min > max``."""

    __slots__ = ("min", "max")

    def __init__(self, lo: Sequence[float], hi: Sequence[float]):
        self.min: Vec3 = (float(lo[0]), float(lo[1]), float(lo[2]))
        self.max: Vec3 = (float(hi[0]), float(hi[1]), float(hi[2]))

    @classmethod
    def empty(cls) -> "AABB":
        inf = float("inf")
        return cls((inf, inf, inf), (-inf, -inf, -inf))

    @classmethod
    def of_points(cls, pts: Sequence[Sequence[float]]) -> "AABB":
        box = cls.empty()
        for p in pts:
            box = box.expanded(p)
        return box

    def is_empty(self) -> bool:
        return (self.min[0] > self.max[0] or self.min[1] > self.max[1]
                or self.min[2] > self.max[2])

    def expanded(self, p: Sequence[float]) -> "AABB":
        return AABB(
            (min(self.min[0], p[0]), min(self.min[1], p[1]), min(self.min[2], p[2])),
            (max(self.max[0], p[0]), max(self.max[1], p[1]), max(self.max[2], p[2])),
        )

    def union(self, other: "AABB") -> "AABB":
        return AABB(
            (min(self.min[0], other.min[0]), min(self.min[1], other.min[1]),
             min(self.min[2], other.min[2])),
            (max(self.max[0], other.max[0]), max(self.max[1], other.max[1]),
             max(self.max[2], other.max[2])),
        )

    def overlaps(self, other: "AABB") -> bool:
        """Closed-interval AABB overlap test (touching boxes overlap)."""
        return (self.min[0] <= other.max[0] and self.max[0] >= other.min[0]
                and self.min[1] <= other.max[1] and self.max[1] >= other.min[1]
                and self.min[2] <= other.max[2] and self.max[2] >= other.min[2])

    def contains_point(self, p: Sequence[float]) -> bool:
        return (self.min[0] <= p[0] <= self.max[0]
                and self.min[1] <= p[1] <= self.max[1]
                and self.min[2] <= p[2] <= self.max[2])

    def center(self) -> Vec3:
        return (0.5 * (self.min[0] + self.max[0]),
                0.5 * (self.min[1] + self.max[1]),
                0.5 * (self.min[2] + self.max[2]))

    def __repr__(self):
        return "AABB(%r, %r)" % (self.min, self.max)


def spread_bits3(v: int) -> int:
    """Insert two zero bits after each of the low 10 bits (Manifold ``SpreadBits3``)."""
    v &= 0x3FF
    v = (v * 0x00010001) & 0xFF0000FF
    v = (v * 0x00000101) & 0x0F00F00F
    v = (v * 0x00000011) & 0xC30C30C3
    v = (v * 0x00000005) & 0x49249249
    return v


def morton3(position: Sequence[float], box: AABB) -> int:
    """30-bit Morton code of ``position`` normalised inside ``box``.

    Matches ``Collider::MortonCode``: each axis is scaled to [0, 1023] and the
    bits are interleaved x-high, then y, then z.
    """
    def norm(lo, hi, p):
        if hi <= lo:
            return 0
        t = 1024.0 * (p - lo) / (hi - lo)
        return int(min(1023.0, max(0.0, t)))

    x = spread_bits3(norm(box.min[0], box.max[0], position[0]))
    y = spread_bits3(norm(box.min[1], box.max[1], position[1]))
    z = spread_bits3(norm(box.min[2], box.max[2], position[2]))
    return x * 4 + y * 2 + z


class _Node:
    __slots__ = ("box", "left", "right", "leaf")

    def __init__(self):
        self.box: Optional[AABB] = None
        self.left: int = -1
        self.right: int = -1
        self.leaf: int = -1  # original leaf index if this is a leaf, else -1


class BVH:
    """Bounding-volume hierarchy over a list of leaf boxes.

    Built top-down: the leaves are Morton-sorted (matching Manifold's leaf
    order), then recursively split at the median so the tree is balanced and
    the build is deterministic.  Internal-node boxes are the union of their
    children, exactly as ``Collider::UpdateBoxes`` accumulates them.
    """

    __slots__ = ("nodes", "root", "leaf_boxes", "num_leaves")

    def __init__(self, leaf_boxes: Sequence[AABB]):
        self.leaf_boxes: List[AABB] = list(leaf_boxes)
        self.num_leaves = len(self.leaf_boxes)
        self.nodes: List[_Node] = []
        self.root = -1
        if self.num_leaves == 0:
            return
        world = AABB.empty()
        for b in self.leaf_boxes:
            world = world.union(b)
        order = list(range(self.num_leaves))
        # Morton-sort leaves by centroid; ties broken by index for determinism.
        keyed = sorted(order, key=lambda i: (morton3(self.leaf_boxes[i].center(), world), i))
        self.root = self._build(keyed)

    def _build(self, idxs: List[int]) -> int:
        node = _Node()
        ni = len(self.nodes)
        self.nodes.append(node)
        if len(idxs) == 1:
            node.leaf = idxs[0]
            node.box = self.leaf_boxes[idxs[0]]
            return ni
        mid = len(idxs) // 2
        left = self._build(idxs[:mid])
        right = self._build(idxs[mid:])
        node.left = left
        node.right = right
        node.box = self.nodes[left].box.union(self.nodes[right].box)
        return ni

    def bounding_box(self) -> AABB:
        if self.root < 0:
            return AABB.empty()
        return self.nodes[self.root].box

    def query(self, q: AABB) -> List[int]:
        """Leaf indices whose box overlaps ``q``, in ascending order.

        Stack-based DFS matching ``Collider::Collisions``: descend only into
        nodes whose box overlaps the query.
        """
        out: List[int] = []
        if self.root < 0:
            return out
        stack = [self.root]
        nodes = self.nodes
        while stack:
            ni = stack.pop()
            node = nodes[ni]
            if not node.box.overlaps(q):
                continue
            if node.leaf >= 0:
                out.append(node.leaf)
            else:
                stack.append(node.left)
                stack.append(node.right)
        out.sort()
        return out

    def query_point(self, p: Sequence[float]) -> List[int]:
        """Leaf indices whose box contains point ``p``."""
        pt = AABB(p, p)
        return [i for i in self.query(pt) if self.leaf_boxes[i].contains_point(p)]

    def self_collisions(self) -> List[Tuple[int, int]]:
        """All overlapping leaf pairs ``(i, j)`` with ``i < j`` (broad phase)."""
        pairs: List[Tuple[int, int]] = []
        for i in range(self.num_leaves):
            for j in self.query(self.leaf_boxes[i]):
                if j > i:
                    pairs.append((i, j))
        pairs.sort()
        return pairs


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------


def boxes_of_triangles(vertices: Sequence[Sequence[float]],
                       tris: Sequence[Sequence[int]]) -> List[AABB]:
    """One AABB per triangle."""
    boxes = []
    for (i, j, k) in tris:
        boxes.append(AABB.of_points([vertices[i], vertices[j], vertices[k]]))
    return boxes


def brute_force_pairs(boxes: Sequence[AABB]) -> List[Tuple[int, int]]:
    """Reference O(n^2) overlapping-pair enumeration for validation."""
    out = []
    n = len(boxes)
    for i in range(n):
        for j in range(i + 1, n):
            if boxes[i].overlaps(boxes[j]):
                out.append((i, j))
    return out
