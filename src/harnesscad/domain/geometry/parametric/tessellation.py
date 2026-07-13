"""Differentiable tessellation of rational Bezier patches (DreamCAD, Fig. 2).

DreamCAD renders each patch by uniformly sampling the (u, v) domain on an
r x r grid, forming quadrilateral cells between adjacent grid points and
splitting each quad into two triangles, then merging neighbouring patches
along shared boundaries to obtain a C0-continuous mesh.  The evaluation is
purely a function of the control points and weights, so it is deterministic
here (the "differentiable" property in the paper refers to gradients w.r.t.
those parameters, which is out of scope for this stdlib module).

This module provides:

  * ``tessellate_patch`` -- sample one patch into (vertices, triangles).
  * ``tessellate_patches`` -- tessellate many patches, optionally welding
    coincident boundary vertices into a single shared vertex.
  * ``enforce_c0_shared_points`` -- the paper's structural C0 fix: for each
    group of control-net slots that must coincide across patches, replace
    them (and their weights) by the group mean.
  * ``mesh_vertices``/``mesh_area`` helpers for downstream metrics.
"""

from __future__ import annotations

from math import sqrt

from harnesscad.domain.geometry.parametric.bezier import bezier_surface_point


def _grid_uv(resolution):
    if resolution < 2:
        raise ValueError("resolution must be at least 2")
    return [i / (resolution - 1) for i in range(resolution)]


def tessellate_patch(grid, weights, resolution=4):
    """Tessellate a single patch into (vertices, triangles).

    Returns ``(vertices, triangles)`` where ``vertices`` is a flat list of
    point tuples in row-major (i over u, j over v) order and ``triangles`` is
    a list of index triples.  Each interior quad is split along its (i, j) ->
    (i+1, j+1) diagonal into two consistently wound triangles.
    """
    coords = _grid_uv(resolution)
    vertices = []
    for u in coords:
        for v in coords:
            vertices.append(bezier_surface_point(grid, weights, u, v))

    def idx(i, j):
        return i * resolution + j

    triangles = []
    for i in range(resolution - 1):
        for j in range(resolution - 1):
            a = idx(i, j)
            b = idx(i + 1, j)
            c = idx(i + 1, j + 1)
            d = idx(i, j + 1)
            triangles.append((a, b, c))
            triangles.append((a, c, d))
    return vertices, triangles


def _key(point, tolerance):
    if tolerance <= 0:
        return tuple(point)
    return tuple(round(x / tolerance) for x in point)


def tessellate_patches(patches, resolution=4, *, weld=False, tolerance=1e-9):
    """Tessellate a list of ``(grid, weights)`` patches into one mesh.

    With ``weld=True`` vertices closer than ``tolerance`` (after quantisation)
    are merged, so patches that already share a boundary yield a seamless,
    C0-connected mesh with no duplicated boundary vertices.  Returns
    ``(vertices, triangles)`` with global indices.
    """
    vertices = []
    triangles = []
    lookup = {}
    for grid, weights in patches:
        local_verts, local_tris = tessellate_patch(grid, weights, resolution)
        remap = []
        for point in local_verts:
            if weld:
                key = _key(point, tolerance)
                if key in lookup:
                    remap.append(lookup[key])
                    continue
                index = len(vertices)
                lookup[key] = index
                vertices.append(point)
                remap.append(index)
            else:
                remap.append(len(vertices))
                vertices.append(point)
        for a, b, c in local_tris:
            triangles.append((remap[a], remap[b], remap[c]))
    return vertices, triangles


def enforce_c0_shared_points(grids, weights, groups):
    """Structurally enforce C0 continuity by averaging shared control slots.

    ``groups`` is an iterable of groups, each a list of ``(patch, i, j)``
    control-net slots that must coincide (adjacent patches sharing a boundary
    point).  Every slot in a group is replaced by the group's mean control
    point and mean weight -- the deterministic counterpart of the paper's
    "uniformly averaging the predicted deformations and weight updates from
    all patches sharing that point".  Returns new ``(grids, weights)`` lists;
    inputs are not mutated.
    """
    new_grids = [[list(row) for row in grid] for grid in grids]
    new_weights = [[list(row) for row in w] for w in weights]
    for group in groups:
        slots = list(group)
        if not slots:
            continue
        dim = len(new_grids[slots[0][0]][slots[0][1]][slots[0][2]])
        mean_pt = [0.0] * dim
        mean_w = 0.0
        for patch, i, j in slots:
            point = new_grids[patch][i][j]
            if len(point) != dim:
                raise ValueError("mixed point dimensions in group")
            for d in range(dim):
                mean_pt[d] += point[d]
            mean_w += new_weights[patch][i][j]
        count = len(slots)
        averaged = tuple(x / count for x in mean_pt)
        avg_w = mean_w / count
        for patch, i, j in slots:
            new_grids[patch][i][j] = averaged
            new_weights[patch][i][j] = avg_w
    return new_grids, new_weights


def mesh_vertices(patches, resolution=4, *, weld=True, tolerance=1e-9):
    """Convenience: return just the welded vertex list for many patches."""
    vertices, _ = tessellate_patches(
        patches, resolution, weld=weld, tolerance=tolerance)
    return vertices


def _tri_area(p, q, r):
    u = tuple(q[d] - p[d] for d in range(3))
    v = tuple(r[d] - p[d] for d in range(3))
    cx = u[1] * v[2] - u[2] * v[1]
    cy = u[2] * v[0] - u[0] * v[2]
    cz = u[0] * v[1] - u[1] * v[0]
    return 0.5 * sqrt(cx * cx + cy * cy + cz * cz)


def mesh_area(vertices, triangles):
    """Total surface area of a tessellated (3-D) mesh."""
    return sum(_tri_area(vertices[a], vertices[b], vertices[c])
               for a, b, c in triangles)
