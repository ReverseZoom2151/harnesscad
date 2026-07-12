"""Programmatic CadQuery selector algebra: composable ``Selector`` objects.

CadQuery (``cadquery/selectors.py``) exposes selection two ways: a *string*
mini-language (``">Z"``, ``"%CIRCLE"``, ``"|Z and >Y"``) and a *programmatic*
class hierarchy of composable ``Selector`` objects that back it.  The harness
already has :mod:`geometry.cqcontrib_selector_dsl`, a recursive-descent parser
for the *string* form -- but it does **not** provide the object algebra, and it
diverges from the real grammar in several ways (see notes below).

This module builds the deterministic, stdlib-only *object* algebra:

* :class:`Selector` base with the ``&`` / ``+`` / ``-`` / unary ``-`` operator
  overloads that make selectors a composable set algebra
  (``AndSelector`` / ``SumSelector`` / ``SubtractSelector`` / ``InverseSelector``).
* Geometric primitives: :class:`NearestToPointSelector`, :class:`BoxSelector`
  (with the exact XOR containment test and bounding-box mode),
  :class:`DirectionSelector` / :class:`ParallelDirSelector` /
  :class:`PerpendicularDirSelector`, :class:`TypeSelector`.
* The Nth-with-tolerance-clustering family built on :class:`_NthSelector`:
  :class:`RadiusNthSelector`, :class:`LengthNthSelector`,
  :class:`AreaNthSelector`, :class:`CenterNthSelector`,
  :class:`DirectionMinMaxSelector`, :class:`DirectionNthSelector`.

Shapes are abstracted as :class:`Shape` (an OCCT-free stand-in carrying the
handful of scalar/vector attributes the pure-Python selectors need).  Every
filter is deterministic and preserves input order; set-combining selectors use
object identity (not value equality + ``set()`` as the real code does), which
keeps results order-stable rather than hash-ordered.

The complementary grammar-correct string compiler lives in
:mod:`geometry.cq_selector_grammar`; it targets these objects and fixes the
divergences documented there.
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Tuple

__all__ = [
    "Shape",
    "SelectorError",
    "Selector",
    "NearestToPointSelector",
    "BoxSelector",
    "BaseDirSelector",
    "ParallelDirSelector",
    "DirectionSelector",
    "PerpendicularDirSelector",
    "TypeSelector",
    "RadiusNthSelector",
    "LengthNthSelector",
    "AreaNthSelector",
    "CenterNthSelector",
    "DirectionMinMaxSelector",
    "DirectionNthSelector",
    "BinarySelector",
    "AndSelector",
    "SumSelector",
    "SubtractSelector",
    "InverseSelector",
]

Vec = Tuple[float, float, float]


class SelectorError(ValueError):
    """Raised for invalid selector use (e.g. Nth of an empty list)."""


@dataclass(frozen=True)
class Shape:
    """An OCCT-free stand-in for a CadQuery subshape.

    ``shape_type`` is one of ``Vertex``/``Edge``/``Wire``/``Face``/``Shell``/
    ``Solid``.  ``center`` is the centre of mass (or the point, for a vertex).
    ``axis`` is the face normal or edge tangent.  ``radius``/``length``/``area``
    are ``None`` when not applicable (a straight edge has no radius, etc.).
    ``bbox`` is ``((xmin, ymin, zmin), (xmax, ymax, zmax))`` when known.
    """

    shape_type: str = "Face"
    center: Vec = (0.0, 0.0, 0.0)
    axis: Vec = (0.0, 0.0, 0.0)
    geom_type: str = ""
    radius: Optional[float] = None
    length: Optional[float] = None
    area: Optional[float] = None
    bbox: Optional[Tuple[Vec, Vec]] = None
    name: str = ""


# --------------------------------------------------------------------------
# vector helpers
# --------------------------------------------------------------------------

def _sub(a: Sequence[float], b: Sequence[float]) -> Vec:
    return (a[0] - b[0], a[1] - b[1], a[2] - b[2])


def _dot(a: Sequence[float], b: Sequence[float]) -> float:
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


def _cross(a: Sequence[float], b: Sequence[float]) -> Vec:
    return (
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    )


def _norm(a: Sequence[float]) -> float:
    return math.sqrt(_dot(a, a))


def _angle(a: Sequence[float], b: Sequence[float]) -> float:
    """Angle between two vectors in radians (matches Vector.getAngle)."""
    na, nb = _norm(a), _norm(b)
    if na == 0.0 or nb == 0.0:
        raise SelectorError("zero-length vector in angle")
    c = _dot(a, b) / (na * nb)
    c = max(-1.0, min(1.0, c))
    return math.acos(c)


# --------------------------------------------------------------------------
# base
# --------------------------------------------------------------------------

class Selector(object):
    """Filters a list of shapes; the default keeps everything.

    Supports the CadQuery composition algebra: ``a & b`` (intersection),
    ``a + b`` (union), ``a - b`` (difference), ``-a`` (complement).
    """

    def filter(self, objectList: Sequence[Shape]) -> List[Shape]:
        return list(objectList)

    def __and__(self, other: "Selector") -> "AndSelector":
        return AndSelector(self, other)

    def __add__(self, other: "Selector") -> "SumSelector":
        return SumSelector(self, other)

    def __sub__(self, other: "Selector") -> "SubtractSelector":
        return SubtractSelector(self, other)

    def __neg__(self) -> "InverseSelector":
        return InverseSelector(self)


class NearestToPointSelector(Selector):
    """Selects the single shape whose centre is nearest ``pnt``."""

    def __init__(self, pnt: Sequence[float]):
        self.pnt: Vec = (float(pnt[0]), float(pnt[1]), float(pnt[2]))

    def filter(self, objectList: Sequence[Shape]) -> List[Shape]:
        if not objectList:
            return []
        return [min(objectList, key=lambda s: _norm(_sub(s.center, self.pnt)))]


class BoxSelector(Selector):
    """Selects shapes inside the axis-aligned box defined by two corner points.

    With ``boundingbox=True`` a shape is kept only when its whole bounding box
    lies inside; otherwise only its centre is tested.  The containment test uses
    the exact XOR trick of the reference implementation, so the two corners may
    be given in either order.
    """

    def __init__(self, point0, point1, boundingbox: bool = False):
        self.p0: Vec = (float(point0[0]), float(point0[1]), float(point0[2]))
        self.p1: Vec = (float(point1[0]), float(point1[1]), float(point1[2]))
        self.test_boundingbox = boundingbox

    def _inside(self, p: Sequence[float]) -> bool:
        x0, y0, z0 = self.p0
        x1, y1, z1 = self.p1
        return (
            ((p[0] < x0) ^ (p[0] < x1))
            and ((p[1] < y0) ^ (p[1] < y1))
            and ((p[2] < z0) ^ (p[2] < z1))
        )

    def filter(self, objectList: Sequence[Shape]) -> List[Shape]:
        result: List[Shape] = []
        for o in objectList:
            if self.test_boundingbox:
                if o.bbox is None:
                    continue
                lo, hi = o.bbox
                if self._inside(lo) and self._inside(hi):
                    result.append(o)
            else:
                if self._inside(o.center):
                    result.append(o)
        return result


class BaseDirSelector(Selector):
    """Selection on the basis of a single direction vector.

    Only planar faces (tested by their normal) and linear edges (tested by
    their tangent) participate; every other shape is dropped, exactly as in the
    reference ``BaseDirSelector``.
    """

    def __init__(self, vector: Sequence[float], tolerance: float = 1e-4):
        self.direction: Vec = (float(vector[0]), float(vector[1]), float(vector[2]))
        self.tolerance = tolerance

    def test(self, vec: Vec) -> bool:
        return True

    def filter(self, objectList: Sequence[Shape]) -> List[Shape]:
        r: List[Shape] = []
        for o in objectList:
            if o.shape_type == "Face" and o.geom_type == "PLANE":
                test_vector = o.axis
            elif o.shape_type == "Edge" and o.geom_type == "LINE":
                test_vector = o.axis
            else:
                continue
            if _norm(test_vector) == 0.0:
                continue
            if self.test(test_vector):
                r.append(o)
        return r


class ParallelDirSelector(BaseDirSelector):
    """Keeps shapes whose axis is parallel to ``direction`` (either sense)."""

    def test(self, vec: Vec) -> bool:
        return _norm(_cross(self.direction, vec)) < self.tolerance


class DirectionSelector(BaseDirSelector):
    """Keeps shapes whose axis points along ``direction`` (same sense)."""

    def test(self, vec: Vec) -> bool:
        return _angle(self.direction, vec) < self.tolerance


class PerpendicularDirSelector(BaseDirSelector):
    """Keeps shapes whose axis is perpendicular to ``direction``."""

    def test(self, vec: Vec) -> bool:
        return abs(_angle(self.direction, vec) - math.pi / 2) < self.tolerance


class TypeSelector(Selector):
    """Keeps shapes whose ``geom_type`` matches (case-insensitive)."""

    def __init__(self, typeString: str):
        self.typeString = typeString.upper()

    def filter(self, objectList: Sequence[Shape]) -> List[Shape]:
        return [o for o in objectList if o.geom_type.upper() == self.typeString]


class _NthSelector(Selector, ABC):
    """Selects the Nth cluster of an ordered list, grouping within tolerance.

    Shapes are sorted by :meth:`key`; consecutive keys within ``tolerance`` are
    merged into one cluster; the Nth cluster (negative allowed) is returned as a
    list.  Shapes whose key raises :class:`SelectorError`/``ValueError`` (e.g. a
    straight edge queried for radius) are silently dropped, matching CadQuery.
    """

    def __init__(self, n: int, directionMax: bool = True, tolerance: float = 1e-4):
        self.n = n
        self.directionMax = directionMax
        self.tolerance = tolerance

    @abstractmethod
    def key(self, obj: Shape) -> float:
        raise NotImplementedError

    def cluster(self, objectlist: Sequence[Shape]) -> List[List[Shape]]:
        key_and_obj: List[Tuple[float, Shape]] = []
        for obj in objectlist:
            try:
                k = self.key(obj)
            except (ValueError, SelectorError):
                continue
            key_and_obj.append((k, obj))
        if not key_and_obj:
            return []
        key_and_obj.sort(key=lambda x: x[0])
        clustered: List[List[Shape]] = [[]]
        start = key_and_obj[0][0]
        for k, obj in key_and_obj:
            if abs(k - start) <= self.tolerance:
                clustered[-1].append(obj)
            else:
                clustered.append([obj])
                start = k
        return clustered

    def filter(self, objectlist: Sequence[Shape]) -> List[Shape]:
        if len(objectlist) == 0:
            raise SelectorError("Can not return the Nth element of an empty list")
        clustered = self.cluster(objectlist)
        if not clustered:
            return []
        if not self.directionMax:
            clustered.reverse()
        try:
            return clustered[self.n]
        except IndexError:
            raise IndexError(
                f"Attempted to access index {self.n} of a list with length "
                f"{len(clustered)}"
            )


class RadiusNthSelector(_NthSelector):
    """Selects the shape(s) with the Nth radius (edges / wires only)."""

    def key(self, obj: Shape) -> float:
        if obj.shape_type in ("Edge", "Wire") and obj.radius is not None:
            return obj.radius
        raise SelectorError("Can not get a radius from this object")


class LengthNthSelector(_NthSelector):
    """Selects the shape(s) with the Nth length (edges / wires only)."""

    def key(self, obj: Shape) -> float:
        if obj.shape_type in ("Edge", "Wire") and obj.length is not None:
            return obj.length
        raise SelectorError(
            f"LengthNthSelector supports only Edges and Wires, not {obj.shape_type}"
        )


class AreaNthSelector(_NthSelector):
    """Selects the shape(s) with the Nth area (faces / shells / solids / wires)."""

    def key(self, obj: Shape) -> float:
        if obj.shape_type in ("Face", "Shell", "Solid", "Wire") and obj.area is not None:
            return abs(obj.area)
        raise SelectorError(
            "AreaNthSelector supports only Wires, Faces, Shells and Solids, "
            f"not {obj.shape_type}"
        )


class CenterNthSelector(_NthSelector):
    """Orders shapes by their centre projected onto ``direction``; keeps the Nth."""

    def __init__(
        self,
        vector: Sequence[float],
        n: int,
        directionMax: bool = True,
        tolerance: float = 1e-4,
    ):
        super().__init__(n, directionMax, tolerance)
        self.direction: Vec = (float(vector[0]), float(vector[1]), float(vector[2]))

    def key(self, obj: Shape) -> float:
        return _dot(obj.center, self.direction)


class DirectionMinMaxSelector(CenterNthSelector):
    """Selects the shape(s) closest / farthest along ``direction`` (the ``>``/``<`` string ops)."""

    def __init__(
        self, vector: Sequence[float], directionMax: bool = True, tolerance: float = 1e-4
    ):
        super().__init__(vector, n=-1, directionMax=directionMax, tolerance=tolerance)


class DirectionNthSelector(_NthSelector):
    """Filters to shapes parallel to ``direction`` then returns the Nth by centre.

    Mirrors the reference class, which multiply-inherits
    ``ParallelDirSelector`` + ``CenterNthSelector``: first the parallel filter,
    then the centre-projection Nth clustering.
    """

    def __init__(
        self,
        vector: Sequence[float],
        n: int,
        directionMax: bool = True,
        tolerance: float = 1e-4,
    ):
        super().__init__(n, directionMax, tolerance)
        self.direction: Vec = (float(vector[0]), float(vector[1]), float(vector[2]))
        self._parallel = ParallelDirSelector(self.direction, tolerance)

    def key(self, obj: Shape) -> float:
        return _dot(obj.center, self.direction)

    def filter(self, objectlist: Sequence[Shape]) -> List[Shape]:
        parallel = self._parallel.filter(objectlist)
        return _NthSelector.filter(self, parallel)


# --------------------------------------------------------------------------
# set-combining selectors (identity-based, order-preserving)
# --------------------------------------------------------------------------

class BinarySelector(Selector):
    """Base for selectors combining the results of two sub-selectors."""

    def __init__(self, left: Selector, right: Selector):
        self.left = left
        self.right = right

    def filter(self, objectList: Sequence[Shape]) -> List[Shape]:
        return self.filterResults(
            self.left.filter(objectList), self.right.filter(objectList)
        )

    def filterResults(self, r_left, r_right):
        raise NotImplementedError


class AndSelector(BinarySelector):
    """Intersection: shapes selected by both, in left order."""

    def filterResults(self, r_left, r_right):
        right_ids = {id(o) for o in r_right}
        return [o for o in r_left if id(o) in right_ids]


class SumSelector(BinarySelector):
    """Union: left results, then right results not already present."""

    def filterResults(self, r_left, r_right):
        seen = {id(o) for o in r_left}
        out = list(r_left)
        for o in r_right:
            if id(o) not in seen:
                out.append(o)
                seen.add(id(o))
        return out


class SubtractSelector(BinarySelector):
    """Difference: left results with right results removed, in left order."""

    def filterResults(self, r_left, r_right):
        drop = {id(o) for o in r_right}
        return [o for o in r_left if id(o) not in drop]


class InverseSelector(Selector):
    """Complement: every shape not selected by the wrapped selector."""

    def __init__(self, selector: Selector):
        self.selector = selector

    def filter(self, objectList: Sequence[Shape]) -> List[Shape]:
        keep = {id(o) for o in self.selector.filter(objectList)}
        return [o for o in objectList if id(o) not in keep]
