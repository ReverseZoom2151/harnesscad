"""AABB quadtree spatial index (the ``Space`` resource of the ``arcs`` CAD system).

``arcs/src/components/spatial_entity.rs`` keeps a global ``Space`` resource: a
quadtree of ``(bounding_box, entity)`` pairs that answers "which entities live
at this point / in this region", is incrementally updated whenever an entity's
bounding box changes (``modify``), and *grows* itself when an inserted box falls
outside the current world bounds. That is what makes picking, rubber-band
selection and hit-testing cheap in a drawing with many entities.

The harness had no spatial index at all -- only ``ingest.spatial_order.morton2``
(a z-order key, no queries). This module supplies the index, and fixes the bug
in the Rust source along the way: ``BoundingBox::intersects_with`` there is a
stub that delegates to ``fully_contains`` (``// FIXME: Actually implement this``),
so overlapping-but-not-contained boxes are missed. :func:`bbox_intersects` is the
real separating-axis test.

Determinism: node capacity/depth are fixed, items keep insertion order inside a
node, and every query returns keys in insertion order, so results never depend on
hash iteration order.

Bounding boxes are ``(min_x, min_y, max_x, max_y)`` float tuples.
"""

from __future__ import annotations

from typing import Dict, Iterable, Iterator, List, Optional, Sequence, Tuple

Point = Tuple[float, float]
BBox = Tuple[float, float, float, float]

DEFAULT_MAX_CHILDREN = 16
DEFAULT_MAX_DEPTH = 8
DEFAULT_WORLD_RADIUS = 1.0e6

__all__ = [
    "DEFAULT_MAX_CHILDREN",
    "DEFAULT_MAX_DEPTH",
    "DEFAULT_WORLD_RADIUS",
    "QuadTreeSpace",
    "bbox_area",
    "bbox_centre",
    "bbox_contains_point",
    "bbox_fully_contains",
    "bbox_intersects",
    "bbox_merge",
    "bbox_around_points",
    "bbox_new",
]


def bbox_new(first: Point, second: Point) -> BBox:
    """Normalised bounding box around two corners."""
    return (
        min(first[0], second[0]),
        min(first[1], second[1]),
        max(first[0], second[0]),
        max(first[1], second[1]),
    )


def bbox_around_points(points: Iterable[Point]) -> Optional[BBox]:
    """Bounding box around a point cloud, or ``None`` when empty."""
    result: Optional[BBox] = None
    for p in points:
        box = (p[0], p[1], p[0], p[1])
        result = box if result is None else bbox_merge(result, box)
    return result


def bbox_merge(left: BBox, right: BBox) -> BBox:
    return (
        min(left[0], right[0]),
        min(left[1], right[1]),
        max(left[2], right[2]),
        max(left[3], right[3]),
    )


def bbox_area(box: BBox) -> float:
    return (box[2] - box[0]) * (box[3] - box[1])


def bbox_centre(box: BBox) -> Point:
    return ((box[0] + box[2]) / 2.0, (box[1] + box[3]) / 2.0)


def bbox_fully_contains(outer: BBox, inner: BBox) -> bool:
    return (
        outer[0] <= inner[0]
        and inner[2] <= outer[2]
        and outer[1] <= inner[1]
        and inner[3] <= outer[3]
    )


def bbox_intersects(left: BBox, right: BBox) -> bool:
    """Do the boxes overlap (touching counts)? Separating-axis test."""
    return not (
        left[2] < right[0]
        or right[2] < left[0]
        or left[3] < right[1]
        or right[3] < left[1]
    )


def bbox_contains_point(box: BBox, point: Point) -> bool:
    return (
        box[0] <= point[0] <= box[2] and box[1] <= point[1] <= box[3]
    )


class _Node:
    __slots__ = ("bounds", "depth", "items", "children")

    def __init__(self, bounds: BBox, depth: int) -> None:
        self.bounds = bounds
        self.depth = depth
        self.items: List[object] = []  # keys held at this level
        self.children: Optional[List["_Node"]] = None

    def quadrants(self) -> List[BBox]:
        min_x, min_y, max_x, max_y = self.bounds
        cx, cy = (min_x + max_x) / 2.0, (min_y + max_y) / 2.0
        return [
            (min_x, min_y, cx, cy),  # SW
            (cx, min_y, max_x, cy),  # SE
            (min_x, cy, cx, max_y),  # NW
            (cx, cy, max_x, max_y),  # NE
        ]


class QuadTreeSpace:
    """A quadtree of axis-aligned bounding boxes keyed by arbitrary hashables."""

    def __init__(
        self,
        world: Optional[BBox] = None,
        max_children: int = DEFAULT_MAX_CHILDREN,
        max_depth: int = DEFAULT_MAX_DEPTH,
    ) -> None:
        if max_children < 1:
            raise ValueError("max_children must be >= 1")
        if max_depth < 0:
            raise ValueError("max_depth must be >= 0")

        self.max_children = max_children
        self.max_depth = max_depth
        if world is None:
            r = DEFAULT_WORLD_RADIUS
            world = (-r, -r, r, r)
        self._root = _Node(world, 0)
        self._boxes: Dict[object, BBox] = {}
        self._order: List[object] = []  # insertion order, for determinism

    # -- introspection ----------------------------------------------------
    @property
    def world(self) -> BBox:
        return self._root.bounds

    def __len__(self) -> int:
        return len(self._boxes)

    def __contains__(self, key: object) -> bool:
        return key in self._boxes

    def keys(self) -> List[object]:
        """Keys in insertion order."""
        return list(self._order)

    def bounds_of(self, key: object) -> BBox:
        return self._boxes[key]

    def items(self) -> Iterator[Tuple[object, BBox]]:
        for key in self._order:
            yield key, self._boxes[key]

    def clear(self) -> None:
        self._root = _Node(self._root.bounds, 0)
        self._boxes.clear()
        self._order.clear()

    # -- mutation ---------------------------------------------------------
    def insert(self, key: object, box: BBox) -> None:
        """Insert (or replace) ``key`` with the bounding box ``box``."""
        box = _normalise(box)
        if key in self._boxes:
            self.remove(key)

        if not bbox_fully_contains(self._root.bounds, box):
            self._grow_to_fit(box)

        self._boxes[key] = box
        self._order.append(key)
        self._insert_into(self._root, key, box)

    #: ``arcs`` calls this ``modify``: insert-or-update in one shot.
    modify = insert

    def remove(self, key: object) -> bool:
        """Remove ``key``; returns ``False`` when it was not present."""
        if key not in self._boxes:
            return False
        box = self._boxes.pop(key)
        self._order.remove(key)
        self._remove_from(self._root, key, box)
        return True

    # -- queries ----------------------------------------------------------
    def query_region(self, box: BBox) -> List[object]:
        """Keys whose bounding box intersects ``box``, in insertion order."""
        box = _normalise(box)
        found: List[object] = []
        self._query(self._root, box, found)
        seen = set(found)
        return [key for key in self._order if key in seen]

    def query_point(self, point: Point, radius: float = 0.0) -> List[object]:
        """Keys whose bounding box is within ``radius`` of ``point``."""
        if radius < 0.0:
            raise ValueError("radius must be >= 0")
        box = (
            point[0] - radius,
            point[1] - radius,
            point[0] + radius,
            point[1] + radius,
        )
        return self.query_region(box)

    def total_bounds(self) -> Optional[BBox]:
        """Bounding box around every indexed item (``None`` when empty)."""
        result: Optional[BBox] = None
        for _, box in self.items():
            result = box if result is None else bbox_merge(result, box)
        return result

    # -- internals --------------------------------------------------------
    def _insert_into(self, node: _Node, key: object, box: BBox) -> None:
        while True:
            if node.children is None:
                node.items.append(key)
                if (
                    len(node.items) > self.max_children
                    and node.depth < self.max_depth
                ):
                    self._subdivide(node)
                return

            target = self._child_for(node, box)
            if target is None:
                node.items.append(key)
                return
            node = target

    def _child_for(self, node: _Node, box: BBox) -> Optional[_Node]:
        assert node.children is not None
        for child in node.children:
            if bbox_fully_contains(child.bounds, box):
                return child
        return None

    def _subdivide(self, node: _Node) -> None:
        node.children = [
            _Node(bounds, node.depth + 1) for bounds in node.quadrants()
        ]
        staying: List[object] = []
        for key in node.items:
            box = self._boxes[key]
            child = self._child_for(node, box)
            if child is None:
                staying.append(key)
            else:
                self._insert_into(child, key, box)
        node.items = staying

    def _remove_from(self, node: _Node, key: object, box: BBox) -> bool:
        if key in node.items:
            node.items.remove(key)
            return True
        if node.children is None:
            return False
        for child in node.children:
            if bbox_intersects(child.bounds, box) and self._remove_from(
                child, key, box
            ):
                return True
        return False

    def _query(self, node: _Node, box: BBox, found: List[object]) -> None:
        if not bbox_intersects(node.bounds, box):
            return
        for key in node.items:
            if bbox_intersects(self._boxes[key], box):
                found.append(key)
        if node.children is not None:
            for child in node.children:
                self._query(child, box, found)

    def _grow_to_fit(self, box: BBox) -> None:
        """Rebuild the tree around a world large enough to hold ``box``."""
        world = bbox_merge(self._root.bounds, box)
        # keep the world square so quadrant splits stay well conditioned
        cx, cy = bbox_centre(world)
        half = max(world[2] - world[0], world[3] - world[1]) / 2.0
        world = (cx - half, cy - half, cx + half, cy + half)

        self._root = _Node(world, 0)
        for key in self._order:
            self._insert_into(self._root, key, self._boxes[key])


def _normalise(box: Sequence[float]) -> BBox:
    if len(box) != 4:
        raise ValueError("a bounding box needs 4 values")
    min_x, min_y, max_x, max_y = (float(v) for v in box)
    if min_x > max_x or min_y > max_y:
        raise ValueError("bounding box is inverted")
    return (min_x, min_y, max_x, max_y)
