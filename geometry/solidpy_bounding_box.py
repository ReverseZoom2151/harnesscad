"""Bounding boxes, planar model splitting with dowel holes, and grid layout.

Reimplementation of the layout/splitting utilities in SolidPython's
``solid/utils.py``: ``bounding_box``, the ``BoundingBox`` class,
``split_body_planar``, ``section_cut_xz`` and ``distribute_in_grid``.

The interesting piece is :func:`split_body_planar`, the fabrication trick for a
model that does not fit the print bed: the model's bounding box is cut in two
along an axis, each half is *intersected* with its own box to yield a piece, and
(optionally) a pair of dowel cylinders is subtracted from both pieces at the cut
face, so the halves can be pinned back together and cannot be glued at the wrong
rotation.  ``add_wall_thickness`` grows each half's box away from the cut so a
snug shell can be printed around it.

The bounding boxes are centre+size (not min/max) because that is what an
OpenSCAD ``cube(center=true)`` consumes directly.  A box computed from a point
list is exact; a box for a CSG result is only an upper bound -- honest, and the
same caveat SolidPython documents.

Pure stdlib, deterministic.  Geometry helpers return numbers; the ``*_scad``
functions return ``programs.solidpy_scad_emit`` nodes.
"""

from __future__ import annotations

import math
from typing import List, Optional, Sequence, Tuple

from programs.solidpy_scad_emit import (
    ScadNode,
    cube,
    cylinder,
    difference,
    intersection,
    rotate,
    translate,
    union,
)

__all__ = [
    "EPSILON",
    "X",
    "Y",
    "Z",
    "bounding_box",
    "BoundingBox",
    "split_body_planar",
    "section_cut",
    "distribute_in_grid",
    "grid_positions",
]

EPSILON = 1e-5

X, Y, Z = 0, 1, 2

Vec3 = Tuple[float, float, float]


def _axis(name_or_index) -> int:
    if isinstance(name_or_index, int):
        if name_or_index not in (X, Y, Z):
            raise ValueError("axis must be 0, 1 or 2")
        return name_or_index
    key = str(name_or_index).lower()
    if key not in ("x", "y", "z"):
        raise ValueError("axis must be one of 'x', 'y', 'z' or 0, 1, 2")
    return {"x": X, "y": Y, "z": Z}[key]


def bounding_box(points: Sequence[Sequence[float]]) -> Tuple[Vec3, Vec3]:
    """(min_corner, max_corner) of a point list; 2D points are treated as z = 0."""
    if not points:
        raise ValueError("bounding_box() of an empty point list")
    xs, ys, zs = [], [], []
    for p in points:
        xs.append(float(p[0]))
        ys.append(float(p[1]))
        zs.append(float(p[2]) if len(p) > 2 else 0.0)
    return ((min(xs), min(ys), min(zs)), (max(xs), max(ys), max(zs)))


class BoundingBox:
    """An axis-aligned box: a size and the position of its centre."""

    def __init__(self, size: Sequence[float],
                 position: Optional[Sequence[float]] = None) -> None:
        if len(size) != 3:
            raise ValueError("size must have 3 elements")
        if any(s < 0 for s in size):
            raise ValueError("size must be non-negative")
        self.size: List[float] = [float(s) for s in size]
        position = position if position is not None else (0.0, 0.0, 0.0)
        if len(position) != 3:
            raise ValueError("position must have 3 elements")
        self.position: List[float] = [float(p) for p in position]

    @classmethod
    def from_points(cls, points: Sequence[Sequence[float]]) -> "BoundingBox":
        lo, hi = bounding_box(points)
        size = [hi[i] - lo[i] for i in range(3)]
        center = [(hi[i] + lo[i]) / 2.0 for i in range(3)]
        return cls(size, center)

    # -- accessors ---------------------------------------------------------
    def min_corner(self) -> Vec3:
        return tuple(self.position[i] - self.size[i] / 2.0 for i in range(3))

    def max_corner(self) -> Vec3:
        return tuple(self.position[i] + self.size[i] / 2.0 for i in range(3))

    def volume(self) -> float:
        return self.size[0] * self.size[1] * self.size[2]

    def contains(self, point: Sequence[float]) -> bool:
        lo, hi = self.min_corner(), self.max_corner()
        return all(lo[i] - EPSILON <= point[i] <= hi[i] + EPSILON
                   for i in range(3))

    def intersects(self, other: "BoundingBox") -> bool:
        a_lo, a_hi = self.min_corner(), self.max_corner()
        b_lo, b_hi = other.min_corner(), other.max_corner()
        return all(a_lo[i] <= b_hi[i] + EPSILON and b_lo[i] <= a_hi[i] + EPSILON
                   for i in range(3))

    def union(self, other: "BoundingBox") -> "BoundingBox":
        return BoundingBox.from_points([self.min_corner(), self.max_corner(),
                                        other.min_corner(), other.max_corner()])

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, BoundingBox):
            return NotImplemented
        return self.size == other.size and self.position == other.position

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return "BoundingBox(size=%r, position=%r)" % (self.size, self.position)

    # -- operations --------------------------------------------------------
    def split_planar(self, axis=Z, cut_proportion: float = 0.5,
                     add_wall_thickness: float = 0.0) -> List["BoundingBox"]:
        """Cut this box in two along ``axis``; the first gets ``cut_proportion``."""
        index = _axis(axis)
        if not 0.0 < cut_proportion < 1.0:
            raise ValueError("cut_proportion must be strictly between 0 and 1")

        dim = self.size[index]
        dim_min = self.position[index] - dim / 2.0

        boxes: List[BoundingBox] = []
        consumed = 0.0
        for i, part in enumerate((cut_proportion, 1.0 - cut_proportion)):
            size = list(self.size)
            size[index] = size[index] * part
            pos = list(self.position)
            pos[index] = dim_min + consumed + dim * (part / 2.0)

            if add_wall_thickness:
                for j in (X, Y, Z):
                    size[j] += 2 * add_wall_thickness
                # do not grow into the cut, only away from it
                size[index] -= add_wall_thickness
                pos[index] += -add_wall_thickness / 2.0 + i * add_wall_thickness

            boxes.append(BoundingBox(size, pos))
            consumed += part * dim
        return boxes

    def cube(self, larger: bool = False) -> ScadNode:
        """This box as a centred OpenSCAD ``cube()``."""
        size = self.size if not larger else [s + 2 * EPSILON for s in self.size]
        return translate(self.position)(cube(list(size), center=True))


def split_body_planar(body: ScadNode, body_bb: BoundingBox, axis=Z,
                      cut_proportion: float = 0.5,
                      dowel_holes: bool = False,
                      dowel_rad: float = 4.5,
                      hole_depth: float = 15.0,
                      add_wall_thickness: float = 0.0,
                      ) -> Tuple[ScadNode, BoundingBox, ScadNode, BoundingBox]:
    """Cut ``body`` in two along ``axis``; optionally pin the halves with dowels.

    Returns ``(piece_a, box_a, piece_b, box_b)``.  The boxes describe which part
    of the original bounding box each piece came from -- they are not tight
    boxes of the pieces themselves.
    """
    index = _axis(axis)
    boxes = body_bb.split_planar(index, cut_proportion,
                                 add_wall_thickness=add_wall_thickness)
    pieces = [intersection()(body.copy(), box.cube()) for box in boxes]

    if dowel_holes:
        dowel: ScadNode = cylinder(r=dowel_rad, h=hole_depth * 2, center=True)
        if index != Z:
            rot_vec = (1, 0, 0) if index == Y else (0, 1, 0)
            dowel = rotate(a=90, v=rot_vec)(dowel)

        cut_point = boxes[0].position[index] + boxes[0].size[index] / 2.0
        trans_a = list(boxes[0].position)
        trans_a[index] = cut_point
        # offset the two dowels along another axis so the halves cannot be
        # reassembled rotated
        separation_index = {X: Y, Y: Z, Z: X}[index]
        trans_a[separation_index] -= 2 * dowel_rad
        trans_b = list(trans_a)
        trans_b[separation_index] += 4 * dowel_rad

        dowels = union()(translate(trans_a)(dowel.copy()),
                         translate(trans_b)(dowel.copy()))
        pieces = [difference()(p, dowels.copy()) for p in pieces]

    return pieces[0], boxes[0], pieces[1], boxes[1]


def section_cut(body: ScadNode, axis=Y, cut_point: float = 0.0,
                thickness: float = 2.0, extent: float = 10000.0) -> ScadNode:
    """A thin slice through ``body`` at ``cut_point`` along ``axis`` (a section view)."""
    index = _axis(axis)
    size = [extent, extent, extent]
    size[index] = thickness
    offset = [0.0, 0.0, 0.0]
    offset[index] = cut_point - thickness / 2.0
    return intersection()(translate(offset)(cube(size, center=True)), body)


def grid_positions(count: int, cell: Sequence[float],
                   rows_and_cols: Optional[Tuple[int, int]] = None
                   ) -> List[Vec3]:
    """Row-major XY grid translations for ``count`` items in cells of size ``cell``."""
    if count < 0:
        raise ValueError("count must be non-negative")
    if isinstance(cell, (int, float)):
        x_step = y_step = float(cell)
    else:
        x_step, y_step = float(cell[0]), float(cell[1])

    if rows_and_cols:
        rows, cols = rows_and_cols
        if rows * cols < count:
            raise ValueError("rows_and_cols cannot hold %d objects" % count)
    else:
        cols = rows = int(math.ceil(math.sqrt(count))) if count else 0

    out: List[Vec3] = []
    placed = 0
    for row in range(rows):
        for col in range(cols):
            if placed >= count:
                return out
            out.append((col * x_step, row * y_step, 0.0))
            placed += 1
    return out


def distribute_in_grid(objects: Sequence[ScadNode], cell: Sequence[float],
                       rows_and_cols: Optional[Tuple[int, int]] = None) -> ScadNode:
    """Lay ``objects`` out in a grid in the XY plane, one per cell."""
    positions = grid_positions(len(objects), cell, rows_and_cols)
    return union()(*[translate(p)(obj) for p, obj in zip(positions, objects)])
