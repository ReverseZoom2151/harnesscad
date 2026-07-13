"""Catmull-Rom splines, lofted patches and prisms (SolidPython ``splines.py``).

The harness already has Bezier/NURBS machinery (``geometry.nurbgen_curve``,
``geometry.dreamcad_rational_bezier``, ``numeric.nurbs_basis``), but those are
all *approximating* bases: the curve does not pass through its control points.
SolidPython's ``solid/splines.py`` supplies the missing *interpolating* spline:
uniform Catmull-Rom, where the curve passes exactly through every control point
and only the tangents are inferred.  That is the right primitive for
"draw a smooth outline through these points", which is how a text-to-CAD
generator most naturally specifies a profile.

Implemented here, PyEuclid-free:

  * :func:`catmull_rom_points` -- a smooth curve through the control points,
    open (with optional explicit end tangents) or closed into a ring;
  * :func:`catmull_rom_patch` -- a surface lofted between two Catmull-Rom
    curves, linearly in the cross direction (vertices + triangles);
  * :func:`catmull_rom_prism` -- a closed solid swept around a ring of vertical
    control curves, with optional centroid end caps; ``smooth_edges`` also
    interpolates *around* the ring with Catmull-Rom rather than linearly, so
    the horizontal cross-sections are smooth too.

Deviation from SolidPython: a closed curve here does **not** repeat its first
point at the end (n * subdivisions points for n controls), so the result is
directly usable as a polygon/ring; and the prism's vertex indices are built
from an explicit column grid rather than SolidPython's shared-index arithmetic.

Pure stdlib, deterministic.
"""

from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

from harnesscad.domain.geometry.solidpy_extrude_along_path import centroid_endcap, face_strip_list
from harnesscad.domain.programs.solidpy_scad_emit import ScadNode, polygon, polyhedron

__all__ = [
    "affine_combination",
    "centroid",
    "catmull_rom_segment",
    "catmull_rom_points",
    "catmull_rom_polygon",
    "catmull_rom_patch",
    "catmull_rom_prism",
    "catmull_rom_prism_scad",
]

Vec3 = Tuple[float, float, float]
Face = Tuple[int, ...]

DEFAULT_SUBDIVISIONS = 10


def _as3(p: Sequence[float]) -> Vec3:
    if len(p) == 2:
        return (float(p[0]), float(p[1]), 0.0)
    return (float(p[0]), float(p[1]), float(p[2]))


def affine_combination(a: Sequence[float], b: Sequence[float],
                       fraction: float) -> Vec3:
    """A point between ``a`` and ``b``; 0 -> a, 1 -> b."""
    a3, b3 = _as3(a), _as3(b)
    f = float(fraction)
    return (a3[0] * (1 - f) + b3[0] * f,
            a3[1] * (1 - f) + b3[1] * f,
            a3[2] * (1 - f) + b3[2] * f)


def centroid(points: Sequence[Sequence[float]]) -> Vec3:
    if not points:
        raise ValueError("centroid() of an empty point list")
    pts = [_as3(p) for p in points]
    n = len(pts)
    return (sum(p[0] for p in pts) / n,
            sum(p[1] for p in pts) / n,
            sum(p[2] for p in pts) / n)


def catmull_rom_segment(controls: Sequence[Sequence[float]], subdivisions: int,
                        include_last: bool = False) -> List[Vec3]:
    """Points between the 2nd and 3rd of four controls, on the uniform CR curve."""
    if len(controls) != 4:
        raise ValueError("catmull_rom_segment() needs exactly 4 control points")
    if subdivisions < 1:
        raise ValueError("subdivisions must be >= 1")
    p0, p1, p2, p3 = [_as3(p) for p in controls]

    out: List[Vec3] = []
    count = subdivisions + 1 if include_last else subdivisions
    for i in range(count):
        t = i / subdivisions
        t2 = t * t
        t3 = t2 * t
        point = []
        for axis in range(3):
            a = 2 * p1[axis]
            b = p2[axis] - p0[axis]
            c = 2 * p0[axis] - 5 * p1[axis] + 4 * p2[axis] - p3[axis]
            d = -p0[axis] + 3 * p1[axis] - 3 * p2[axis] + p3[axis]
            point.append(0.5 * (a + b * t + c * t2 + d * t3))
        out.append((point[0], point[1], point[2]))
    return out


def catmull_rom_points(points: Sequence[Sequence[float]],
                       subdivisions: int = DEFAULT_SUBDIVISIONS,
                       close_loop: bool = False,
                       start_tangent: Optional[Sequence[float]] = None,
                       end_tangent: Optional[Sequence[float]] = None) -> List[Vec3]:
    """A smooth curve passing through every point of ``points``."""
    pts = [_as3(p) for p in points]
    if len(pts) < 2:
        raise ValueError("catmull_rom_points() needs at least 2 points")
    if subdivisions < 1:
        raise ValueError("subdivisions must be >= 1")

    if close_loop:
        if len(pts) < 3:
            raise ValueError("a closed Catmull-Rom loop needs at least 3 points")
        controls = [pts[-1]] + pts + pts[0:2]
        last = len(controls) - 3
        include_last_index = -1  # a ring never repeats its first point
    else:
        if start_tangent is None:
            start_t = (pts[1][0] - pts[0][0], pts[1][1] - pts[0][1],
                       pts[1][2] - pts[0][2])
        else:
            start_t = _as3(start_tangent)
        if end_tangent is None:
            end_t = (pts[-2][0] - pts[-1][0], pts[-2][1] - pts[-1][1],
                     pts[-2][2] - pts[-1][2])
        else:
            end_t = _as3(end_tangent)
        first = (pts[0][0] + start_t[0], pts[0][1] + start_t[1], pts[0][2] + start_t[2])
        final = (pts[-1][0] + end_t[0], pts[-1][1] + end_t[1], pts[-1][2] + end_t[2])
        controls = [first] + pts + [final]
        last = len(controls) - 3
        include_last_index = last - 1

    out: List[Vec3] = []
    for i in range(last):
        out.extend(catmull_rom_segment(controls[i:i + 4], subdivisions,
                                       include_last=(i == include_last_index)))
    return out


def catmull_rom_polygon(points: Sequence[Sequence[float]],
                        subdivisions: int = DEFAULT_SUBDIVISIONS) -> ScadNode:
    """A closed OpenSCAD ``polygon()`` through all of ``points`` (2D)."""
    ring = catmull_rom_points(points, subdivisions, close_loop=True)
    return polygon([(p[0], p[1]) for p in ring])


def catmull_rom_patch(curve_a: Sequence[Sequence[float]],
                      curve_b: Sequence[Sequence[float]],
                      subdivisions: int = DEFAULT_SUBDIVISIONS,
                      index_start: int = 0) -> Tuple[List[Vec3], List[Face]]:
    """Loft a surface between the Catmull-Rom curves through ``curve_a``/``curve_b``."""
    a_pts = catmull_rom_points(curve_a, subdivisions)
    b_pts = catmull_rom_points(curve_b, subdivisions)
    if len(a_pts) != len(b_pts):
        raise ValueError("both control curves must have the same number of points")
    strip_length = len(a_pts)

    verts: List[Vec3] = []
    faces: List[Face] = []
    for i in range(subdivisions + 1):
        fraction = i / subdivisions
        verts.extend(affine_combination(a, b, fraction)
                     for a, b in zip(a_pts, b_pts))
        if i < subdivisions:
            a_start = index_start + i * strip_length
            faces.extend(face_strip_list(a_start, a_start + strip_length,
                                         strip_length))
    return verts, faces


def _prism_columns(control_curves: Sequence[Sequence[Sequence[float]]],
                   subdivisions: int, closed_ring: bool,
                   smooth_edges: bool) -> List[List[Vec3]]:
    curves = [[_as3(p) for p in c] for c in control_curves]
    if len(curves) < 2:
        raise ValueError("catmull_rom_prism() needs at least 2 control curves")
    lengths = {len(c) for c in curves}
    if len(lengths) != 1:
        raise ValueError("all control curves must have the same length")

    expanded = [catmull_rom_points(c, subdivisions) for c in curves]
    height = len(expanded[0])

    if smooth_edges:
        contours = []
        for i in range(height):
            ring_controls = [c[i] for c in expanded]
            contours.append(catmull_rom_points(ring_controls, subdivisions,
                                               close_loop=closed_ring))
        width = len(contours[0])
        return [[contours[i][k] for i in range(height)] for k in range(width)]

    ring = expanded + [expanded[0]] if closed_ring else expanded
    columns: List[List[Vec3]] = []
    for a, b in zip(ring[:-1], ring[1:]):
        for j in range(subdivisions):
            fraction = j / subdivisions
            columns.append([affine_combination(pa, pb, fraction)
                            for pa, pb in zip(a, b)])
    if not closed_ring:
        columns.append(ring[-1])
    return columns


def catmull_rom_prism(control_curves: Sequence[Sequence[Sequence[float]]],
                      subdivisions: int = DEFAULT_SUBDIVISIONS,
                      closed_ring: bool = True,
                      add_caps: bool = True,
                      smooth_edges: bool = False) -> Tuple[List[Vec3], List[Face]]:
    """A prism swept through a ring of vertical control curves."""
    columns = _prism_columns(control_curves, subdivisions, closed_ring, smooth_edges)
    height = len(columns[0])
    width = len(columns)

    verts: List[Vec3] = []
    for col in columns:
        verts.extend(col)

    faces: List[Face] = []
    last = width if closed_ring else width - 1
    for k in range(last):
        a_start = k * height
        b_start = ((k + 1) % width) * height
        faces.extend(face_strip_list(a_start, b_start, height))

    if closed_ring and add_caps:
        bottom = [k * height for k in range(width)]
        top = [k * height + height - 1 for k in range(width)]
        bot_centroid, bot_faces = centroid_endcap(verts, bottom)
        verts.append(bot_centroid)
        faces.extend(bot_faces)
        top_centroid, top_faces = centroid_endcap(verts, top, invert=True)
        verts.append(top_centroid)
        faces.extend(top_faces)

    return verts, faces


def catmull_rom_prism_scad(control_curves: Sequence[Sequence[Sequence[float]]],
                           convexity: int = 3, **kwargs) -> ScadNode:
    """:func:`catmull_rom_prism`, wrapped in an OpenSCAD ``polyhedron()``."""
    verts, faces = catmull_rom_prism(control_curves, **kwargs)
    return polyhedron(points=verts, faces=faces, convexity=convexity)
