"""Kernel-free bounding-box propagation over a typed CSG tree (AngelCAD get_box()).

Every AngelCAD shape implements ``bbox3d get_box() const``, and the boxes compose
*without* evaluating any boolean: ``cone::get_box`` takes the larger of the two
radii, ``linear_extrude::get_box`` takes the 2D box of the profile and adds the
third dimension, and every shape ends with ``return get_transform()*box`` --
i.e. the accumulated 4x4 matrix is applied to the box by re-enclosing its eight
corners.  A script can therefore ask a model for its size before any geometry
exists, which is what AngelCAD's ``boundingbox`` script type is for
(``dx() dy() dz() diagonal() center() p1() p2()``).

This module gives the harness the same capability on
``programs.angelcad_typed_csg`` trees:

* :class:`BBox3` -- AngelCAD's ``boundingbox``: ``enclose``, ``dx/dy/dz``,
  ``diagonal``, ``center``, ``p1/p2``, ``is_empty``, plus ``transformed`` (8
  corners re-enclosed), ``united``, ``intersected`` and ``minkowski_sum``;
* :func:`bounding_box` -- the propagation itself, with one rule per operator:
  union/hull enclose their children, difference is bounded by its *first* child,
  intersection is the overlap (and can be provably empty -- a free
  "these two solids cannot possibly touch" test), minkowski adds the boxes,
  offset2d grows by ``|delta|``, projection2d flattens z, the extrudes lift a 2D
  box into 3D and sweep bounds the swept profile along its path;
* :func:`fits_within` / :func:`is_provably_empty` -- the two cheap checks the
  box enables: does the model fit a build volume, and is an intersection void.

Boxes are *upper bounds* for unions/hulls/differences and exact for primitives
and transforms -- the same honest contract AngelCAD ships.

Distinct from ``geometry.solidpy_bounding_box`` (which measures OpenSCAD point
lists and splits a model for a print bed): this one walks a typed CSG tree and
has a rule per boolean operator.

Pure stdlib, deterministic.
"""

from __future__ import annotations

import math
from typing import List, Optional, Sequence, Tuple

from harnesscad.domain.programs.ast.angelcad_typed_csg import Node, TMatrix

__all__ = [
    "BBox3",
    "BoundingBoxError",
    "bounding_box",
    "fits_within",
    "is_provably_empty",
]

Point = Tuple[float, float, float]


class BoundingBoxError(Exception):
    """The tree cannot be bounded (unknown operator, missing parameter)."""


class BBox3:
    """An axis-aligned box that starts empty and grows by :meth:`enclose`."""

    __slots__ = ("_lo", "_hi")

    def __init__(
        self, lo: Optional[Sequence[float]] = None, hi: Optional[Sequence[float]] = None
    ) -> None:
        self._lo: Optional[Point] = None
        self._hi: Optional[Point] = None
        if lo is not None:
            self.enclose(lo)
        if hi is not None:
            self.enclose(hi)

    # -- construction -----------------------------------------------------
    def enclose(self, p: Sequence[float]) -> "BBox3":
        pt = (float(p[0]), float(p[1]), float(p[2]) if len(p) > 2 else 0.0)
        if self._lo is None:
            self._lo = pt
            self._hi = pt
        else:
            self._lo = tuple(min(self._lo[i], pt[i]) for i in range(3))  # type: ignore[assignment]
            self._hi = tuple(max(self._hi[i], pt[i]) for i in range(3))  # type: ignore[assignment]
        return self

    @classmethod
    def from_points(cls, points: Sequence[Sequence[float]]) -> "BBox3":
        box = cls()
        for p in points:
            box.enclose(p)
        return box

    # -- accessors --------------------------------------------------------
    def is_empty(self) -> bool:
        return self._lo is None

    def _require(self) -> Tuple[Point, Point]:
        if self._lo is None or self._hi is None:
            raise BoundingBoxError("bounding box is empty")
        return self._lo, self._hi

    def p1(self) -> Point:
        return self._require()[0]

    def p2(self) -> Point:
        return self._require()[1]

    def dx(self) -> float:
        lo, hi = self._require()
        return hi[0] - lo[0]

    def dy(self) -> float:
        lo, hi = self._require()
        return hi[1] - lo[1]

    def dz(self) -> float:
        lo, hi = self._require()
        return hi[2] - lo[2]

    def size(self) -> Point:
        return (self.dx(), self.dy(), self.dz())

    def diagonal(self) -> float:
        return math.sqrt(self.dx() ** 2 + self.dy() ** 2 + self.dz() ** 2)

    def center(self) -> Point:
        lo, hi = self._require()
        return tuple((lo[i] + hi[i]) * 0.5 for i in range(3))  # type: ignore[return-value]

    def corners(self) -> List[Point]:
        lo, hi = self._require()
        return [
            (lo[0] if i & 1 == 0 else hi[0],
             lo[1] if i & 2 == 0 else hi[1],
             lo[2] if i & 4 == 0 else hi[2])
            for i in range(8)
        ]

    # -- algebra ----------------------------------------------------------
    def transformed(self, m: TMatrix) -> "BBox3":
        """Re-enclose the eight transformed corners (AngelCAD ``HTmatrix*bbox3d``)."""
        if self.is_empty():
            return BBox3()
        out = BBox3()
        for c in self.corners():
            out.enclose(m.apply_pos(c))
        return out

    def united(self, other: "BBox3") -> "BBox3":
        out = BBox3()
        for box in (self, other):
            if not box.is_empty():
                out.enclose(box.p1())
                out.enclose(box.p2())
        return out

    def intersected(self, other: "BBox3") -> "BBox3":
        if self.is_empty() or other.is_empty():
            return BBox3()
        a1, a2 = self.p1(), self.p2()
        b1, b2 = other.p1(), other.p2()
        lo = tuple(max(a1[i], b1[i]) for i in range(3))
        hi = tuple(min(a2[i], b2[i]) for i in range(3))
        if any(lo[i] > hi[i] for i in range(3)):
            return BBox3()
        return BBox3(lo, hi)

    def minkowski_sum(self, other: "BBox3") -> "BBox3":
        if self.is_empty() or other.is_empty():
            return BBox3()
        a1, a2 = self.p1(), self.p2()
        b1, b2 = other.p1(), other.p2()
        return BBox3(
            tuple(a1[i] + b1[i] for i in range(3)),
            tuple(a2[i] + b2[i] for i in range(3)),
        )

    def grown(self, d: float, dz: float = 0.0) -> "BBox3":
        lo, hi = self._require()
        return BBox3(
            (lo[0] - d, lo[1] - d, lo[2] - dz),
            (hi[0] + d, hi[1] + d, hi[2] + dz),
        )

    def __eq__(self, other: object) -> bool:
        return isinstance(other, BBox3) and other._lo == self._lo and other._hi == self._hi

    def __repr__(self) -> str:  # pragma: no cover - debug aid
        if self.is_empty():
            return "BBox3(empty)"
        return "BBox3(%r, %r)" % (self._lo, self._hi)


def _p(node: Node, name: str) -> float:
    if name not in node.params:
        raise BoundingBoxError("%s is missing parameter %r" % (node.op, name))
    v = node.params[name]
    if isinstance(v, bool) or not isinstance(v, (int, float)):
        raise BoundingBoxError("%s.%s is not a number" % (node.op, name))
    return float(v)


def _centered(node: Node) -> bool:
    return bool(node.params.get("center", False))


def _radial(node: Node, r: float, h: float) -> BBox3:
    z1 = -h * 0.5 if _centered(node) else 0.0
    z2 = h * 0.5 if _centered(node) else h
    return BBox3((-r, -r, z1), (r, r, z2))


def bounding_box(node: Node) -> BBox3:
    """Bounding box of a typed CSG tree, computed without any geometry kernel."""
    op = node.op

    # ---- transforms
    if op == "transform":
        m = node.params.get("matrix")
        if not isinstance(m, TMatrix):
            raise BoundingBoxError("transform node has no matrix")
        if not node.children:
            raise BoundingBoxError("transform node has no child")
        return bounding_box(node.children[0]).transformed(m)

    # ---- 2d primitives (z == 0)
    if op == "circle":
        r = _p(node, "r")
        return BBox3((-r, -r, 0.0), (r, r, 0.0))
    if op == "square":
        s = _p(node, "size")
        lo = -s * 0.5 if _centered(node) else 0.0
        hi = s * 0.5 if _centered(node) else s
        return BBox3((lo, lo, 0.0), (hi, hi, 0.0))
    if op == "rectangle":
        dx, dy = _p(node, "dx"), _p(node, "dy")
        if _centered(node):
            return BBox3((-dx * 0.5, -dy * 0.5, 0.0), (dx * 0.5, dy * 0.5, 0.0))
        return BBox3((0.0, 0.0, 0.0), (dx, dy, 0.0))
    if op == "polygon":
        pts = node.params.get("points") or ()
        if not pts:
            raise BoundingBoxError("polygon has no points")
        return BBox3.from_points([(p[0], p[1], 0.0) for p in pts])

    # ---- 3d primitives
    if op == "sphere":
        r = _p(node, "r")
        return BBox3((-r, -r, -r), (r, r, r))
    if op == "cube":
        s = _p(node, "size")
        lo = -s * 0.5 if _centered(node) else 0.0
        hi = s * 0.5 if _centered(node) else s
        return BBox3((lo, lo, lo), (hi, hi, hi))
    if op == "cuboid":
        dx, dy, dz = _p(node, "dx"), _p(node, "dy"), _p(node, "dz")
        if _centered(node):
            return BBox3(
                (-dx * 0.5, -dy * 0.5, -dz * 0.5), (dx * 0.5, dy * 0.5, dz * 0.5)
            )
        return BBox3((0.0, 0.0, 0.0), (dx, dy, dz))
    if op == "cylinder":
        return _radial(node, _p(node, "r"), _p(node, "h"))
    if op == "cone":
        return _radial(node, max(_p(node, "r1"), _p(node, "r2")), _p(node, "h"))
    if op == "polyhedron":
        pts = node.params.get("points") or ()
        if not pts:
            raise BoundingBoxError("polyhedron has no points")
        return BBox3.from_points(pts)

    # ---- children-driven operators
    if not node.children:
        raise BoundingBoxError("%s has no children" % op)
    boxes = [bounding_box(c) for c in node.children]

    if op in ("union2d", "union3d", "hull2d", "hull3d", "fill2d"):
        out = BBox3()
        for b in boxes:
            out = out.united(b)
        return out
    if op in ("difference2d", "difference3d"):
        # the result is contained in the positive operand
        return boxes[0]
    if op in ("intersection2d", "intersection3d"):
        out = boxes[0]
        for b in boxes[1:]:
            out = out.intersected(b)
        return out
    if op in ("minkowski2d", "minkowski3d"):
        out = boxes[0]
        for b in boxes[1:]:
            out = out.minkowski_sum(b)
        return out
    if op == "offset2d":
        d = abs(_p(node, "delta"))
        return boxes[0].grown(d)
    if op == "projection2d":
        b = boxes[0]
        if b.is_empty():
            return BBox3()
        return BBox3((b.p1()[0], b.p1()[1], 0.0), (b.p2()[0], b.p2()[1], 0.0))
    if op == "linear_extrude":
        b = boxes[0]
        dz = _p(node, "dz")
        return BBox3((b.p1()[0], b.p1()[1], 0.0), (b.p2()[0], b.p2()[1], dz))
    if op == "rotate_extrude":
        b = boxes[0]
        # the profile lies in the xy plane and is revolved about the z axis:
        # x becomes the radius, y becomes the height
        r = max(abs(b.p1()[0]), abs(b.p2()[0]))
        return BBox3((-r, -r, b.p1()[1]), (r, r, b.p2()[1]))
    if op == "transform_extrude":
        out = BBox3()
        for b in boxes:
            out = out.united(b)
        return out
    if op == "sweep":
        path = node.params.get("path") or ()
        if not path:
            raise BoundingBoxError("sweep has no path")
        b = boxes[0]
        radius = max(
            abs(b.p1()[0]), abs(b.p2()[0]), abs(b.p1()[1]), abs(b.p2()[1])
        )
        out = BBox3()
        for p in path:
            out.enclose((p[0] - radius, p[1] - radius, p[2] - radius))
            out.enclose((p[0] + radius, p[1] + radius, p[2] + radius))
        return out

    raise BoundingBoxError("no bounding-box rule for operator %r" % op)


def fits_within(node: Node, build_volume: Sequence[float]) -> bool:
    """Does the model fit inside a ``(dx, dy, dz)`` build volume (any orientation)?"""
    box = bounding_box(node)
    if box.is_empty():
        return True
    return all(
        a <= b + 1e-12
        for a, b in zip(sorted(box.size()), sorted(float(v) for v in build_volume))
    )


def is_provably_empty(node: Node) -> bool:
    """True when the tree's box collapses -- e.g. an intersection that cannot overlap."""
    return bounding_box(node).is_empty()
