"""Triangle-triangle and segment intersection substrate for the mesh boolean.

Manifold's boolean (``src/boolean3.cpp`` and the ``boolean2_predicates.cpp``
leaf primitives) is, at its combinatorial core, a robust triangle-triangle
intersection engine: for every candidate triangle pair reported by the BVH it
must decide *whether* they intersect and *where* (the intersection segment),
with the decision driven by sign predicates so the resulting arrangement stays
manifold.  The full retriangulating boolean is enormous, but the transferable
substrate -- the part the harness lacked -- is:

* the sidedness of each triangle's vertices against the other triangle's plane,
  decided by the exact :func:`numeric.manifold_predicates.orient3d` so a vertex
  exactly on a plane is reported as *on* it (sign 0), not spuriously above or
  below;
* the Moller "interval overlap on the line of plane intersection" test that
  turns those signs into a yes/no coplanar-robust intersection answer;
* the actual intersection segment (two 3D points) when the triangles cross;
* the supporting segment-plane and segment-segment (2D, via
  :func:`numeric.manifold_predicates.orient2d`) intersection points used when
  retriangulating.

This is exactly the substrate the harness did not have: it had SDF booleans
(``geometry.curv_sdf_combinators``, ``geometry.dontmesh_halfspace_csg``) and a
2D segment-crossing helper (``geometry.euclid_validity``), but **no
triangle-triangle intersection and no plane-sidedness classification** -- the
first thing any mesh boolean needs.

Pure stdlib, deterministic; all sidedness decisions use the exact predicates.
"""

from __future__ import annotations

import math
from typing import List, Optional, Sequence, Tuple

from harnesscad.domain.numeric.exact_predicates import orient3d, orient2d

__all__ = [
    "triangles_intersect",
    "triangle_triangle_segment",
    "segment_plane_intersection",
    "segment_segment_2d",
    "plane_of",
    "point_side_of_plane",
]

Vec3 = Tuple[float, float, float]
Tri = Tuple[Vec3, Vec3, Vec3]


def _sub(a, b):
    return (a[0] - b[0], a[1] - b[1], a[2] - b[2])


def _cross(a, b):
    return (a[1] * b[2] - a[2] * b[1],
            a[2] * b[0] - a[0] * b[2],
            a[0] * b[1] - a[1] * b[0])


def _dot(a, b):
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


def _add(a, b):
    return (a[0] + b[0], a[1] + b[1], a[2] + b[2])


def _scale(a, s):
    return (a[0] * s, a[1] * s, a[2] * s)


def plane_of(tri: Tri) -> Tuple[Vec3, float]:
    """Return ``(normal, d)`` for the plane ``dot(n, x) = d`` through ``tri``."""
    n = _cross(_sub(tri[1], tri[0]), _sub(tri[2], tri[0]))
    return n, _dot(n, tri[0])


def point_side_of_plane(p: Vec3, tri: Tri) -> int:
    """Exact sign of which side of ``tri``'s plane ``p`` lies on (orient3d)."""
    return orient3d(tri[0], tri[1], tri[2], p)


def segment_plane_intersection(p: Vec3, q: Vec3, tri: Tri) -> Optional[Vec3]:
    """Intersection point of segment ``p-q`` with ``tri``'s plane, or ``None``.

    Returns ``None`` when the segment is parallel to the plane (including lying
    in it).  The parameter is clamped to the segment; an endpoint exactly on the
    plane returns that endpoint.
    """
    n, d = plane_of(tri)
    dp = _dot(n, p) - d
    dq = _dot(n, q) - d
    denom = dp - dq
    if denom == 0.0:
        return None
    t = dp / denom
    if t < 0.0 or t > 1.0:
        return None
    return _add(p, _scale(_sub(q, p), t))


def segment_segment_2d(p0, p1, q0, q1) -> Optional[Tuple[float, float]]:
    """Proper intersection point of two 2D segments, or ``None``.

    Uses the exact :func:`orient2d` straddle test: a proper crossing requires
    each segment to strictly separate the other's endpoints.  Collinear or
    endpoint-touching configurations return ``None`` (they are degeneracies
    resolved elsewhere).
    """
    d1 = orient2d(q0, q1, p0)
    d2 = orient2d(q0, q1, p1)
    d3 = orient2d(p0, p1, q0)
    d4 = orient2d(p0, p1, q1)
    if d1 != d2 and d3 != d4 and d1 * d2 < 0 and d3 * d4 < 0:
        # both straddle: solve the crossing with floats (sign already certain)
        r = (p1[0] - p0[0], p1[1] - p0[1])
        s = (q1[0] - q0[0], q1[1] - q0[1])
        denom = r[0] * s[1] - r[1] * s[0]
        if denom == 0.0:
            return None
        qp = (q0[0] - p0[0], q0[1] - p0[1])
        t = (qp[0] * s[1] - qp[1] * s[0]) / denom
        return (p0[0] + t * r[0], p0[1] + t * r[1])
    return None


def _project_axes(n: Vec3) -> Tuple[int, int]:
    """The two axes to drop the dominant normal component onto (for 2D work)."""
    ax, ay, az = abs(n[0]), abs(n[1]), abs(n[2])
    if ax >= ay and ax >= az:
        return 1, 2
    if ay >= az:
        return 0, 2
    return 0, 1


def triangles_intersect(t1: Tri, t2: Tri) -> bool:
    """Moller coplanar-robust triangle-triangle intersection test.

    Returns ``True`` iff the two triangles intersect (share at least one point),
    including edge-on-edge and coplanar-overlap contact.  All sidedness
    decisions use the exact :func:`orient3d`, so a shared vertex or an edge
    lying in the other triangle's plane is detected reliably.
    """
    # Signs of t2's vertices against t1's plane, and vice versa.
    s2 = [orient3d(t1[0], t1[1], t1[2], v) for v in t2]
    if s2[0] == s2[1] == s2[2] and s2[0] != 0:
        return False  # t2 entirely on one side of t1's plane
    s1 = [orient3d(t2[0], t2[1], t2[2], v) for v in t1]
    if s1[0] == s1[1] == s1[2] and s1[0] != 0:
        return False

    coplanar = all(s == 0 for s in s2)
    if coplanar:
        return _coplanar_intersect(t1, t2)

    n1, _ = plane_of(t1)
    n2, _ = plane_of(t2)
    direction = _cross(n1, n2)
    if direction == (0.0, 0.0, 0.0):
        return False
    # Both intervals are parametrised by the SAME global coordinate
    # dot(direction, X), so they are directly comparable.
    i1 = _plane_interval(t1, t2, s2, direction)  # where t2 crosses t1's plane
    i2 = _plane_interval(t2, t1, s1, direction)  # where t1 crosses t2's plane
    if i1 is None or i2 is None:
        return False
    lo = max(i1[0], i2[0])
    hi = min(i1[1], i2[1])
    return lo <= hi


def _param(direction: Vec3, p: Vec3) -> float:
    return _dot(direction, p)


def _plane_interval(host: Tri, other: Tri, signs_of_other,
                    direction: Vec3) -> Optional[Tuple[float, float]]:
    """Interval where ``other`` meets ``host``'s plane, parametrised by
    ``dot(direction, .)``.  A single-vertex touch yields a degenerate interval."""
    params: List[float] = []
    verts = list(other)
    for i in range(3):
        a, b = i, (i + 1) % 3
        sa, sb = signs_of_other[a], signs_of_other[b]
        if sa == 0:
            params.append(_param(direction, verts[a]))
        elif sb != 0 and (sa > 0) != (sb > 0):
            hit = segment_plane_intersection(verts[a], verts[b], host)
            if hit is not None:
                params.append(_param(direction, hit))
    if not params:
        return None
    return (min(params), max(params))


def _coplanar_intersect(t1: Tri, t2: Tri) -> bool:
    """2D overlap test for two coplanar triangles."""
    n, _ = plane_of(t1)
    u, v = _project_axes(n)
    a = [(p[u], p[v]) for p in t1]
    b = [(p[u], p[v]) for p in t2]

    def edges_cross():
        for i in range(3):
            for j in range(3):
                if segment_segment_2d(a[i], a[(i + 1) % 3], b[j], b[(j + 1) % 3]):
                    return True
        return False

    if edges_cross():
        return True
    # containment: a vertex of one inside the other
    if _point_in_tri_2d(a[0], b):
        return True
    if _point_in_tri_2d(b[0], a):
        return True
    return False


def _point_in_tri_2d(p, tri) -> bool:
    d0 = orient2d(tri[0], tri[1], p)
    d1 = orient2d(tri[1], tri[2], p)
    d2 = orient2d(tri[2], tri[0], p)
    has_neg = d0 < 0 or d1 < 0 or d2 < 0
    has_pos = d0 > 0 or d1 > 0 or d2 > 0
    return not (has_neg and has_pos)


def triangle_triangle_segment(t1: Tri, t2: Tri) -> Optional[Tuple[Vec3, Vec3]]:
    """The intersection segment of two non-coplanar crossing triangles.

    Returns the two endpoints of the shared segment, or ``None`` if the
    triangles do not cross transversally (no intersection, or coplanar).  The
    segment is the overlap of each triangle's crossing chord on the line where
    the two planes meet.
    """
    s2 = [orient3d(t1[0], t1[1], t1[2], v) for v in t2]
    s1 = [orient3d(t2[0], t2[1], t2[2], v) for v in t1]
    if all(s == 0 for s in s2):
        return None  # coplanar handled separately
    if s2[0] == s2[1] == s2[2] and s2[0] != 0:
        return None
    if s1[0] == s1[1] == s1[2] and s1[0] != 0:
        return None

    n1, _ = plane_of(t1)
    n2, _ = plane_of(t2)
    direction = _cross(n1, n2)
    if direction == (0.0, 0.0, 0.0):
        return None

    def chord(host, other, signs):
        pts = []
        verts = list(other)
        for i in range(3):
            a, b = i, (i + 1) % 3
            sa, sb = signs[a], signs[b]
            if sa == 0:
                pts.append(verts[a])
            elif (sa > 0) != (sb > 0) and sb != 0:
                hit = segment_plane_intersection(verts[a], verts[b], host)
                if hit is not None:
                    pts.append(hit)
        return pts

    c1 = chord(t2, t1, s1)  # t1's crossing of t2's plane
    c2 = chord(t1, t2, s2)  # t2's crossing of t1's plane
    if len(c1) < 2 or len(c2) < 2:
        return None

    def interval(pts):
        params = sorted((_param(direction, p), p) for p in pts)
        return params[0], params[-1]

    (lo1, plo1), (hi1, phi1) = interval(c1)
    (lo2, plo2), (hi2, phi2) = interval(c2)
    # All four points are collinear on the plane-intersection line; select the
    # actual endpoints bounding the overlap so no reconstruction error creeps in.
    if lo1 >= lo2:
        lo, lo_pt = lo1, plo1
    else:
        lo, lo_pt = lo2, plo2
    if hi1 <= hi2:
        hi, hi_pt = hi1, phi1
    else:
        hi, hi_pt = hi2, phi2
    if lo > hi:
        return None
    return lo_pt, hi_pt
