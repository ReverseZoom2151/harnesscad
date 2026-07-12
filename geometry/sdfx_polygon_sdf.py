"""Exact signed distance field for an arbitrary 2D polygon (sdfx).

Reimplementation of the winding-number polygon SDF from deadsy/sdfx
(``sdf/mesh2.go`` -- ``MeshSDF2Slow`` / ``Polygon2D``).  Unlike the *regular*
polygon field in :mod:`geometry.curv_sdf_primitives` (mitred n-gon), this
handles any simple polygon -- convex or concave, arbitrary vertex list -- and
returns an *exact* Euclidean distance:

* the unsigned distance is the minimum distance from the query point to every
  edge segment (projection clamped to the segment, so vertices are handled
  correctly);
* the sign comes from the **winding number** computed with the standard
  upward/downward crossing rule (Dan Sunday's algorithm): a non-zero winding
  number means the point is inside, so the distance is negated.

The winding rule is robust for concave polygons and correctly signs points for
self-consistent orientation without assuming convexity.  This is the exact 2D
field sdfx sweeps (extrude / revolve) into manufacturing parts, and the field a
polygon-based 2D CAD layer needs.

Pure stdlib, deterministic.
"""

from __future__ import annotations

import math
from typing import List, Sequence, Tuple

__all__ = [
    "Edge",
    "prepare_edges",
    "polygon_distance",
    "polygon_winding",
    "polygon_sdf",
    "polygon_area",
    "polygon_centroid",
    "point_in_polygon",
]

Vec2 = Tuple[float, float]


class Edge:
    """A polygon edge with pre-computed unit vector and length."""

    __slots__ = ("ax", "ay", "bx", "by", "ux", "uy", "length")

    def __init__(self, a: Vec2, b: Vec2) -> None:
        self.ax, self.ay = float(a[0]), float(a[1])
        self.bx, self.by = float(b[0]), float(b[1])
        vx = self.bx - self.ax
        vy = self.by - self.ay
        length = math.hypot(vx, vy)
        self.length = length
        if length == 0.0:
            self.ux = 0.0
            self.uy = 0.0
        else:
            self.ux = vx / length
            self.uy = vy / length

    def min_distance2(self, px: float, py: float) -> float:
        """Minimum squared distance from (px, py) to this segment."""
        pax = px - self.ax
        pay = py - self.ay
        # t-parameter of the projection of p onto the (unit) line direction.
        t = pax * self.ux + pay * self.uy
        if t < 0.0:
            # nearest point is vertex a
            return pax * pax + pay * pay
        if t > self.length:
            # nearest point is vertex b
            dx = px - self.bx
            dy = py - self.by
            return dx * dx + dy * dy
        # nearest point is the foot of the perpendicular; distance is the
        # component of pa along the edge normal (uy, -ux).
        dn = pax * self.uy - pay * self.ux
        return dn * dn

    def winding(self, px: float, py: float) -> int:
        """Winding-number increment for a ray cast from p in +x.

        Upward crossings count +1, downward crossings -1 (Sunday's rule).
        """
        # signed distance along the edge normal (uy, -ux): >0 means p is to the
        # right of the directed edge a->b.
        dn = (px - self.ax) * self.uy - (py - self.ay) * self.ux
        if self.ay <= py:
            if self.by > py and dn < 0.0:  # upward crossing, p left of edge
                return 1
        else:
            if self.by <= py and dn > 0.0:  # downward crossing, p right of edge
                return -1
        return 0


def prepare_edges(vertices: Sequence[Vec2]) -> List[Edge]:
    """Build the closed edge list for a polygon (last vertex wraps to first)."""
    n = len(vertices)
    if n < 3:
        raise ValueError("polygon needs at least 3 vertices")
    edges: List[Edge] = []
    for i in range(n):
        a = vertices[i]
        b = vertices[(i + 1) % n]
        e = Edge(a, b)
        if e.length == 0.0:
            continue  # skip degenerate (repeated) vertices
        edges.append(e)
    if len(edges) < 3:
        raise ValueError("polygon has fewer than 3 non-degenerate edges")
    return edges


def polygon_distance(px: float, py: float, edges: Sequence[Edge]) -> float:
    """Unsigned minimum distance from a point to the polygon boundary."""
    d2 = math.inf
    for e in edges:
        d = e.min_distance2(px, py)
        if d < d2:
            d2 = d
    return math.sqrt(d2)


def polygon_winding(px: float, py: float, edges: Sequence[Edge]) -> int:
    """Winding number of the polygon around a point (0 == outside)."""
    wn = 0
    for e in edges:
        wn += e.winding(px, py)
    return wn


def polygon_sdf(point: Vec2, vertices: Sequence[Vec2]) -> float:
    """Exact signed distance from ``point`` to the polygon ``vertices``.

    Negative inside, positive outside.  ``vertices`` is an open list (the
    closing edge is implied).
    """
    edges = prepare_edges(vertices)
    px, py = float(point[0]), float(point[1])
    d = polygon_distance(px, py, edges)
    if polygon_winding(px, py, edges) != 0:
        return -d
    return d


def point_in_polygon(point: Vec2, vertices: Sequence[Vec2]) -> bool:
    """True if ``point`` lies inside the polygon (winding number != 0)."""
    edges = prepare_edges(vertices)
    return polygon_winding(float(point[0]), float(point[1]), edges) != 0


def polygon_area(vertices: Sequence[Vec2]) -> float:
    """Signed area (shoelace); positive for counter-clockwise winding."""
    n = len(vertices)
    if n < 3:
        return 0.0
    s = 0.0
    for i in range(n):
        x0, y0 = vertices[i]
        x1, y1 = vertices[(i + 1) % n]
        s += x0 * y1 - x1 * y0
    return 0.5 * s


def polygon_centroid(vertices: Sequence[Vec2]) -> Vec2:
    """Area centroid of a simple polygon."""
    n = len(vertices)
    if n < 3:
        raise ValueError("polygon needs at least 3 vertices")
    a = polygon_area(vertices)
    if a == 0.0:
        # degenerate area: fall back to vertex average
        cx = sum(v[0] for v in vertices) / n
        cy = sum(v[1] for v in vertices) / n
        return (cx, cy)
    cx = 0.0
    cy = 0.0
    for i in range(n):
        x0, y0 = vertices[i]
        x1, y1 = vertices[(i + 1) % n]
        cross = x0 * y1 - x1 * y0
        cx += (x0 + x1) * cross
        cy += (y0 + y1) * cross
    scale = 1.0 / (6.0 * a)
    return (cx * scale, cy * scale)
