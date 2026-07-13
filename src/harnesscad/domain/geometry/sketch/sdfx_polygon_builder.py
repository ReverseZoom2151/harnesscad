"""Fluent 2D polygon sketch builder with arcs, fillets and chamfers (sdfx).

Reimplementation of the polygon construction layer from deadsy/sdfx
(``sdf/poly.go``).  This is a *2D CAD sketch* primitive: you add vertices --
optionally in relative or polar coordinates -- and tag individual vertices for

* **smoothing** (fillet): replace a sharp corner with a tangent circular arc of
  a given radius approximated by ``facets`` segments;
* **chamfer**: a 1-facet smoothing (a cut corner);
* **arc**: replace the *line segment* leading into a vertex with a circular arc
  bulging to one side (sign of the radius picks the side).

Calling :meth:`Polygon.vertices` runs the fixup pipeline (relative->absolute,
create arcs, smooth vertices) and returns the resolved absolute vertex list,
which can be fed to :func:`geometry.sdfx_polygon_sdf.polygon_sdf` or extruded.

This differs from the *field* primitives already in the harness (curv / libfive
regular polygon SDFs): here we generate the boundary *geometry* of an arbitrary
manufacturing profile.  :func:`nagon` provides the regular n-gon vertex ring.

Pure stdlib, deterministic.
"""

from __future__ import annotations

import math
from typing import List, Optional, Sequence, Tuple

__all__ = [
    "PolygonVertex",
    "Polygon",
    "nagon",
]

Vec2 = Tuple[float, float]

_SQRT_HALF = math.sqrt(0.5)
_TAU = 2.0 * math.pi

_NORMAL = 0
_SMOOTH = 1
_ARC = 2


def _rotate(theta: float):
    """Return a rotation function for angle theta (radians, CCW)."""
    c = math.cos(theta)
    s = math.sin(theta)

    def rot(v: Vec2) -> Vec2:
        return (v[0] * c - v[1] * s, v[0] * s + v[1] * c)

    return rot


def _sign(x: float) -> float:
    if x > 0:
        return 1.0
    if x < 0:
        return -1.0
    return 0.0


class PolygonVertex:
    """A single polygon vertex, with an optional fillet/chamfer/arc tag."""

    __slots__ = ("x", "y", "relative", "vtype", "facets", "radius")

    def __init__(self, x: float, y: float) -> None:
        self.x = float(x)
        self.y = float(y)
        self.relative = False
        self.vtype = _NORMAL
        self.facets = 0
        self.radius = 0.0

    # --- fluent modifiers (return self for chaining) ---

    def rel(self) -> "PolygonVertex":
        """Interpret this vertex position as relative to the previous one."""
        self.relative = True
        return self

    def polar(self) -> "PolygonVertex":
        """Interpret (x, y) as polar (r, theta) and convert to cartesian."""
        r, theta = self.x, self.y
        self.x = r * math.cos(theta)
        self.y = r * math.sin(theta)
        return self

    def smooth(self, radius: float, facets: int) -> "PolygonVertex":
        """Fillet this corner with a tangent arc of the given radius."""
        if radius != 0 and facets != 0:
            self.radius = radius
            self.facets = facets
            self.vtype = _SMOOTH
        return self

    def chamfer(self, size: float) -> "PolygonVertex":
        """Chamfer (cut) this corner. Exact for 90-degree corners."""
        if size != 0:
            self.radius = size * _SQRT_HALF
            self.facets = 1
            self.vtype = _SMOOTH
        return self

    def arc(self, radius: float, facets: int) -> "PolygonVertex":
        """Replace the segment leading into this vertex with a circular arc.

        Sign of ``radius`` selects which side of the chord the arc bulges to.
        """
        if radius != 0 and facets != 0:
            self.radius = radius
            self.facets = facets
            self.vtype = _ARC
        return self


class Polygon:
    """Builder collecting :class:`PolygonVertex` objects into a profile."""

    def __init__(self) -> None:
        self._vlist: List[PolygonVertex] = []
        self.closed = True
        self.reverse = False

    # --- construction ---

    def add(self, x: float, y: float) -> PolygonVertex:
        """Add an (x, y) vertex and return it (for fluent tagging)."""
        v = PolygonVertex(x, y)
        self._vlist.append(v)
        return v

    def add_set(self, points: Sequence[Vec2]) -> "Polygon":
        for p in points:
            self.add(p[0], p[1])
        return self

    def drop(self) -> "Polygon":
        """Remove the last vertex."""
        if self._vlist:
            self._vlist.pop()
        return self

    def set_open(self) -> "Polygon":
        self.closed = False
        return self

    def set_reverse(self) -> "Polygon":
        self.reverse = True
        return self

    # --- neighbour access ---

    def _next(self, i: int) -> Optional[PolygonVertex]:
        if i == len(self._vlist) - 1:
            return self._vlist[0] if self.closed else None
        return self._vlist[i + 1]

    def _prev(self, i: int) -> Optional[PolygonVertex]:
        if i == 0:
            return self._vlist[-1] if self.closed else None
        return self._vlist[i - 1]

    # --- fixup pipeline ---

    def _rel_to_abs(self) -> None:
        for i, v in enumerate(self._vlist):
            if v.relative:
                pv = self._prev(i)
                if pv is None or pv.relative:
                    raise ValueError("relative vertex needs an absolute reference")
                v.x += pv.x
                v.y += pv.y
                v.relative = False

    def _arc_vertex(self, i: int) -> bool:
        v = self._vlist[i]
        if v.vtype != _ARC:
            return False
        v.vtype = _NORMAL
        pv = self._prev(i)
        if pv is None:
            return False
        side = _sign(v.radius)
        radius = abs(v.radius)
        ax, ay = pv.x, pv.y
        bx, by = v.x, v.y
        # unit chord vector
        dx, dy = bx - ax, by - ay
        clen = math.hypot(dx, dy)
        ux, uy = dx / clen, dy / clen
        # chord normal, oriented by side
        nx, ny = uy * side, -ux * side
        # midpoint of chord
        mx, my = (ax + bx) * 0.5, (ay + by) * 0.5
        d_mid = math.hypot(mx - ax, my - ay)
        d_center = math.sqrt(max(0.0, radius * radius - d_mid * d_mid))
        cx, cy = mx + nx * d_center, my + ny * d_center
        # angle subtended between the endpoints as seen from center
        acx, acy = ax - cx, ay - cy
        bcx, bcy = bx - cx, by - cy
        aclen = math.hypot(acx, acy)
        bclen = math.hypot(bcx, bcy)
        dot = (acx * bcx + acy * bcy) / (aclen * bclen)
        dot = max(-1.0, min(1.0, dot))
        dtheta = -side * math.acos(dot) / v.facets
        rot = _rotate(dtheta)
        rvx, rvy = ax - cx, ay - cy
        new_verts: List[PolygonVertex] = []
        for _ in range(v.facets - 1):
            new_verts.append(PolygonVertex(cx + rvx, cy + rvy))
            rvx, rvy = rot((rvx, rvy))
        # insert the new vertices before vertex i
        self._vlist[i:i] = new_verts
        return True

    def _create_arcs(self) -> None:
        done = False
        while not done:
            done = True
            i = 0
            while i < len(self._vlist):
                if self._vlist[i].vtype == _ARC and self._arc_vertex(i):
                    done = False
                i += 1

    def _smooth_vertex(self, i: int) -> bool:
        v = self._vlist[i]
        if v.vtype != _SMOOTH:
            return False
        vn = self._next(i)
        vp = self._prev(i)
        if vn is None or vp is None:
            return False
        # unit vectors from v to its neighbours
        v0x, v0y = vp.x - v.x, vp.y - v.y
        l0 = math.hypot(v0x, v0y)
        v0x, v0y = v0x / l0, v0y / l0
        v1x, v1y = vn.x - v.x, vn.y - v.y
        l1 = math.hypot(v1x, v1y)
        v1x, v1y = v1x / l1, v1y / l1
        dot = max(-1.0, min(1.0, v0x * v1x + v0y * v1y))
        theta = math.acos(dot)
        # distance from vertex to the tangent points
        d1 = v.radius / math.tan(theta / 2.0)
        if d1 > l0 or d1 > l1:
            return False  # radius too large to fit
        # first tangent point along the previous edge
        p0x, p0y = v.x + v0x * d1, v.y + v0y * d1
        # distance from vertex to arc center
        d2 = v.radius / math.sin(theta / 2.0)
        vcx, vcy = v0x + v1x, v0y + v1y
        vcl = math.hypot(vcx, vcy)
        vcx, vcy = vcx / vcl, vcy / vcl
        cx, cy = v.x + vcx * d2, v.y + vcy * d2
        cross = v1x * v0y - v1y * v0x
        dtheta = _sign(cross) * (math.pi - theta) / v.facets
        rot = _rotate(dtheta)
        rvx, rvy = p0x - cx, p0y - cy
        new_verts: List[PolygonVertex] = []
        for _ in range(v.facets + 1):
            new_verts.append(PolygonVertex(cx + rvx, cy + rvy))
            rvx, rvy = rot((rvx, rvy))
        # replace vertex i with the arc points
        self._vlist[i:i + 1] = new_verts
        return True

    def _smooth_vertices(self) -> None:
        done = False
        while not done:
            done = True
            i = 0
            while i < len(self._vlist):
                if self._vlist[i].vtype == _SMOOTH and self._smooth_vertex(i):
                    done = False
                i += 1

    def _fixups(self) -> None:
        self._rel_to_abs()
        self._create_arcs()
        self._smooth_vertices()

    def vertices(self) -> List[Vec2]:
        """Resolve the builder into an absolute (x, y) vertex list."""
        if not self._vlist:
            return []
        self._fixups()
        pts = [(v.x, v.y) for v in self._vlist]
        if self.reverse:
            pts.reverse()
        return pts


def nagon(n: int, radius: float) -> List[Vec2]:
    """Vertices of a regular ``n``-sided polygon inscribed in ``radius``."""
    if n < 3:
        raise ValueError("nagon needs n >= 3")
    rot = _rotate(_TAU / n)
    pts: List[Vec2] = []
    p = (radius, 0.0)
    for _ in range(n):
        pts.append(p)
        p = rot(p)
    return pts
