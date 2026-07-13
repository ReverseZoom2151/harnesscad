"""Sweep a 2D cross-section along a 3D path into a closed polyhedron.

Reimplementation of SolidPython's ``solid/extrude_along_path.py`` (plus the
``transform_to_point`` look-at frame from ``solid/utils.py`` and the face-index
helpers from ``solid/splines.py``), with the PyEuclid dependency removed.

The algorithm: for each path point, build a local frame from the path *tangent*
(central difference of the neighbouring points, wrapped for closed paths) and a
reference up-vector, place a copy of the cross-section loop in that frame, and
stitch consecutive loops with two triangles per quad.  Per-loop uniform or
differential scaling, per-loop Z-rotation (or a single rotation swept smoothly
along the path), and an arbitrary per-point transform callable
``f(point, path_fraction, loop_fraction)`` are supported.

Ends are handled in one of two ways:

  * ``connect_ends`` -- the last loop is stitched back onto the first, giving a
    torus (also selected automatically when the first and last path points
    coincide);
  * ``cap_ends`` -- a centroid vertex is appended for each end loop and the
    loop is fanned to it, giving a watertight (edge-manifold) triangle mesh.
    SolidPython instead emits the raw n-gons and lets OpenSCAD triangulate;
    doing it here keeps the result usable as a mesh without a kernel.

Returns plain ``(points, faces)`` so the mesh is usable directly; a companion
:func:`extrude_along_path_scad` wraps it in a ``polyhedron()`` node from
``programs.solidpy_scad_emit``.

Pure stdlib, deterministic.
"""

from __future__ import annotations

import math
from typing import Callable, List, Optional, Sequence, Tuple, Union

from harnesscad.domain.programs.solidpy_scad_emit import ScadNode, polyhedron

__all__ = [
    "EPSILON",
    "look_at_frame",
    "transform_points_to_frame",
    "face_strip_list",
    "fan_endcap_list",
    "centroid_endcap",
    "extrude_along_path",
    "extrude_along_path_scad",
]

EPSILON = 1e-9

Vec3 = Tuple[float, float, float]
Face = Tuple[int, ...]
PointTransform = Callable[[Vec3, float, float], Sequence[float]]


def _as3(p: Sequence[float]) -> Vec3:
    if len(p) == 2:
        return (float(p[0]), float(p[1]), 0.0)
    return (float(p[0]), float(p[1]), float(p[2]))


def _sub(a: Sequence[float], b: Sequence[float]) -> Vec3:
    return (a[0] - b[0], a[1] - b[1], a[2] - b[2])


def _add(a: Sequence[float], b: Sequence[float]) -> Vec3:
    return (a[0] + b[0], a[1] + b[1], a[2] + b[2])


def _cross(a: Sequence[float], b: Sequence[float]) -> Vec3:
    return (
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    )


def _norm(a: Sequence[float]) -> float:
    return math.sqrt(a[0] * a[0] + a[1] * a[1] + a[2] * a[2])


def _unit(a: Sequence[float]) -> Vec3:
    n = _norm(a)
    if n < EPSILON:
        raise ValueError("cannot normalise a zero-length vector")
    return (a[0] / n, a[1] / n, a[2] / n)


def look_at_frame(eye: Sequence[float], normal: Sequence[float],
                  up: Sequence[float] = (0.0, 0.0, 1.0)
                  ) -> Tuple[Vec3, Vec3, Vec3, Vec3]:
    """Return the frame ``(x_axis, y_axis, z_axis, origin)`` looking along ``normal``.

    This is PyEuclid's pre-2015 ``Matrix4.new_look_at``, the behaviour
    SolidPython's ``transform_to_point`` depends on: ``z`` points *back* along
    the view direction, ``x = up cross z``, ``y = z cross x``.  When ``normal``
    is parallel to ``up`` the frame would collapse, so a fallback up-vector is
    chosen exactly as SolidPython does.
    """
    z = _unit((-normal[0], -normal[1], -normal[2]))
    up3 = _as3(up)
    if _norm(_cross(z, up3)) < EPSILON:
        # normal parallel to up: pick another reference
        if _norm(_cross(up3, (0.0, 0.0, 1.0))) < EPSILON:
            up3 = (0.0, 1.0, 0.0)
        else:
            up3 = (0.0, 0.0, 1.0)
    x = _unit(_cross(up3, z))
    y = _cross(z, x)
    return x, y, z, _as3(eye)


def transform_points_to_frame(points: Sequence[Sequence[float]],
                              eye: Sequence[float],
                              normal: Sequence[float],
                              up: Sequence[float] = (0.0, 0.0, 1.0)
                              ) -> List[Vec3]:
    """Map ``points`` (in the XY plane) into the look-at frame at ``eye``."""
    x, y, z, o = look_at_frame(eye, normal, up)
    out: List[Vec3] = []
    for p in points:
        p3 = _as3(p)
        out.append((
            o[0] + x[0] * p3[0] + y[0] * p3[1] + z[0] * p3[2],
            o[1] + x[1] * p3[0] + y[1] * p3[1] + z[1] * p3[2],
            o[2] + x[2] * p3[0] + y[2] * p3[1] + z[2] * p3[2],
        ))
    return out


def face_strip_list(a_start: int, b_start: int, length: int,
                    close_loop: bool = False) -> List[Face]:
    """Triangles stitching a row of ``length`` vertices to the next row."""
    faces: List[Face] = []
    loop = length - 1
    for i in range(loop):
        a = a_start + i
        b = b_start + i
        faces.append((a, b + 1, b))
        faces.append((a, a + 1, b + 1))
    if close_loop and length > 1:
        a_last = a_start + loop
        b_last = b_start + loop
        faces.append((a_last, b_start, b_last))
        faces.append((a_last, a_start, b_start))
    return faces


def fan_endcap_list(cap_points: int = 3, index_start: int = 0) -> List[Face]:
    """Triangle fan from the first vertex of a (convex) ring to all others."""
    return [(index_start, i, i + 1)
            for i in range(index_start + 1, index_start + cap_points - 1)]


def centroid_endcap(points: Sequence[Sequence[float]], indices: Sequence[int],
                    invert: bool = False) -> Tuple[Vec3, List[Face]]:
    """Centroid vertex plus a fan of faces closing the ring ``indices``."""
    ring = [_as3(points[i]) for i in indices]
    n = len(ring)
    center = (sum(p[0] for p in ring) / n,
              sum(p[1] for p in ring) / n,
              sum(p[2] for p in ring) / n)
    centroid_index = len(points)
    faces: List[Face] = []
    for a, b in zip(indices[:-1], indices[1:]):
        faces.append((centroid_index, a, b))
    faces.append((centroid_index, indices[-1], indices[0]))
    if invert:
        faces = [tuple(reversed(f)) for f in faces]
    return center, faces


def _loop_facet_indices(loop_start: int, loop_pt_count: int,
                        next_loop_start: Optional[int] = None) -> List[Face]:
    if next_loop_start is None:
        next_loop_start = loop_start + loop_pt_count
    faces: List[Face] = []
    for i in range(loop_pt_count):
        j = (i + 1) % loop_pt_count
        a = loop_start + i
        b = loop_start + j
        c = next_loop_start + i
        d = next_loop_start + j
        faces.append((a, b, c))
        faces.append((b, d, c))
    return faces


def _scale_loop(points: Sequence[Vec3],
                scale: Union[None, float, Sequence[float]]) -> List[Vec3]:
    if scale is None:
        return list(points)
    if isinstance(scale, (int, float)):
        sx = sy = float(scale)
    else:
        sx, sy = float(scale[0]), float(scale[1])
    return [(p[0] * sx, p[1] * sy, p[2]) for p in points]


def _rotate_loop(points: Sequence[Vec3], degrees: Optional[float]) -> List[Vec3]:
    if degrees is None:
        return list(points)
    rads = math.radians(degrees)
    c, s = math.cos(rads), math.sin(rads)
    return [(p[0] * c - p[1] * s, p[0] * s + p[1] * c, p[2]) for p in points]


def _transform_loop(points: Sequence[Vec3],
                    func: Optional[PointTransform],
                    path_fraction: float) -> List[Vec3]:
    if func is None:
        return list(points)
    n = len(points)
    out: List[Vec3] = []
    for i, p in enumerate(points):
        loop_fraction = i / (n - 1) if n > 1 else 0.0
        out.append(_as3(func(p, path_fraction, loop_fraction)))
    return out


def extrude_along_path(shape_pts: Sequence[Sequence[float]],
                       path_pts: Sequence[Sequence[float]],
                       scales: Optional[Sequence[Union[float, Sequence[float]]]] = None,
                       rotations: Optional[Sequence[float]] = None,
                       transforms: Optional[Sequence[PointTransform]] = None,
                       connect_ends: bool = False,
                       cap_ends: bool = True,
                       up: Sequence[float] = (0.0, 0.0, 1.0),
                       ) -> Tuple[List[Vec3], List[Face]]:
    """Sweep ``shape_pts`` (planar, in XY) along ``path_pts``; return (points, faces)."""
    shape = [_as3(p) for p in shape_pts]
    path = [_as3(p) for p in path_pts]
    if len(shape) < 3:
        raise ValueError("shape_pts needs at least 3 points")
    if len(path) < 2:
        raise ValueError("path_pts needs at least 2 points")

    # A path whose ends coincide is implicitly a loop
    if _norm(_sub(path[0], path[-1])) < EPSILON:
        connect_ends = True
        path = path[:-1]
        if len(path) < 2:
            raise ValueError("degenerate closed path")

    if scales is not None and len(scales) != len(path):
        raise ValueError("len(scales) must equal len(path_pts)")
    if transforms is not None and len(transforms) not in (1, len(path)):
        raise ValueError("len(transforms) must be 1 or len(path_pts)")
    if rotations is not None and len(rotations) not in (1, len(path)):
        raise ValueError("len(rotations) must be 1 or len(path_pts)")

    # Central-difference tangents, wrapped or extrapolated at the ends
    if connect_ends:
        padded = [path[-1]] + path + [path[0]]
    else:
        first = _sub(path[0], _sub(path[1], path[0]))
        last = _sub(path[-1], _sub(path[-2], path[-1]))
        padded = [first] + path + [last]
    tangents = [_sub(padded[i + 2], padded[i]) for i in range(len(path))]

    shape_n = len(shape)
    points: List[Vec3] = []
    faces: List[Face] = []

    for i in range(len(path)):
        path_fraction = i / (len(path) - 1) if len(path) > 1 else 0.0

        scale = scales[i] if scales else None
        rotate_deg = None
        if rotations:
            rotate_deg = (rotations[i] if len(rotations) > 1
                          else rotations[0] * path_fraction)
        func = None
        if transforms:
            func = transforms[i] if len(transforms) > 1 else transforms[0]

        loop = _scale_loop(shape, scale)
        loop = _rotate_loop(loop, rotate_deg)
        loop = _transform_loop(loop, func, path_fraction)
        loop = transform_points_to_frame(loop, path[i], tangents[i], up)

        if i < len(path) - 1:
            faces.extend(_loop_facet_indices(i * shape_n, shape_n))
        points.extend(loop)

    if connect_ends:
        faces.extend(_loop_facet_indices(len(points) - shape_n, shape_n, 0))
    elif cap_ends:
        last_start = len(points) - shape_n
        start_indices = list(range(shape_n))
        end_indices = list(range(last_start, last_start + shape_n))
        center, cap = centroid_endcap(points, start_indices, invert=False)
        points.append(center)
        faces.extend(cap)
        center, cap = centroid_endcap(points, end_indices, invert=True)
        points.append(center)
        faces.extend(cap)

    return points, faces


def extrude_along_path_scad(shape_pts: Sequence[Sequence[float]],
                            path_pts: Sequence[Sequence[float]],
                            convexity: int = 2,
                            **kwargs) -> ScadNode:
    """:func:`extrude_along_path`, wrapped in an OpenSCAD ``polyhedron()``."""
    points, faces = extrude_along_path(shape_pts, path_pts, **kwargs)
    return polyhedron(points=points, faces=faces, convexity=convexity)
