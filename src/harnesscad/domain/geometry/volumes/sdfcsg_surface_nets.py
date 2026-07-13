"""Naive Surface Nets isosurface extraction over a sampled SDF grid.

This is the mesh generator used by wwwtyro/sdf-csg (``src/isosurface.ts`` and
``src/sdf.ts``).  It is deliberately *different* from the isosurface algorithms
the harness already has:

* :mod:`geometry.libfive_dual_contour` places one vertex per cell and solves a
  **QEF** using surface normals (Hermite data) -- it is 2-D only in the harness.
* :mod:`geometry.meshdiff_marching_tets` splits each cube into tetrahedra and
  emits one triangle strip per tet (Marching Tetrahedra).
* :mod:`geometry.sdfcsg_marching_cubes` uses the Lorensen-Cline 256-case
  lookup tables.

**Naive Surface Nets** (Gibson 1998 / mikolalysenko) also places one vertex per
sign-changing cell, but locates it as the *plain average of the cube's edge
crossings* -- no normals, no QEF, no lookup tables.  It produces a quad mesh
whose vertices float smoothly on the surface, and is what sdf-csg ships.

This module provides:

* :func:`sample_sdf_grid`  -- tile a scalar field over a regular grid (the
  ``generateGrid`` step);
* :class:`ScalarGrid`      -- the sampled volume with ``get`` / ``shape``;
* :func:`surface_nets`     -- extract ``(positions, faces)`` as triangles;
* :func:`interpolate_attribute` -- sdf-csg's per-vertex "user data" blending;
* :func:`mesh_to_stl`      -- ASCII STL serialisation of the triangle mesh.

Everything is pure, deterministic and stdlib-only.
"""

from __future__ import annotations

import math
from typing import Callable, Dict, List, Sequence, Tuple

Vec3 = Tuple[float, float, float]

# Corners of the unit cube, indexed 0..7.
_CUBE_CORNERS: Tuple[Vec3, ...] = (
    (0, 0, 0),
    (1, 0, 0),
    (0, 1, 0),
    (1, 1, 0),
    (0, 0, 1),
    (1, 0, 1),
    (0, 1, 1),
    (1, 1, 1),
)

# The 12 edges of the cube as ordered corner-index pairs.
_CUBE_EDGES: Tuple[Tuple[int, int], ...] = (
    (0, 1), (0, 2), (0, 4), (1, 3), (1, 5), (2, 3),
    (2, 6), (3, 7), (4, 5), (4, 6), (5, 7), (6, 7),
)


class ScalarGrid:
    """A regularly sampled scalar field with world-space coordinates.

    ``values`` is a flat list in x-fastest, then y, then z order.  ``shape`` is
    ``(nx, ny, nz)`` sample counts.  ``origin`` is the world position of sample
    ``(0, 0, 0)`` and ``spacing`` the per-axis step.
    """

    __slots__ = ("values", "shape", "origin", "spacing")

    def __init__(
        self,
        values: List[float],
        shape: Tuple[int, int, int],
        origin: Vec3,
        spacing: Vec3,
    ) -> None:
        self.values = values
        self.shape = shape
        self.origin = origin
        self.spacing = spacing

    def get(self, i: int, j: int, k: int) -> float:
        nx, ny, _ = self.shape
        return self.values[i + nx * (j + ny * k)]

    def world(self, i: float, j: float, k: float) -> Vec3:
        ox, oy, oz = self.origin
        sx, sy, sz = self.spacing
        return (ox + i * sx, oy + j * sy, oz + k * sz)


def sample_sdf_grid(
    field: Callable[[Sequence[float]], float],
    bounds_min: Sequence[float],
    bounds_max: Sequence[float],
    resolution: Sequence[int],
    padding: float = 0.0,
) -> ScalarGrid:
    """Evaluate ``field`` on a regular grid spanning the (padded) bounds.

    ``resolution`` is the number of *cells* per axis; the grid therefore has
    ``resolution[i] + 1`` samples per axis.  Mirrors sdf-csg's ``generateGrid``.
    """
    rx, ry, rz = int(resolution[0]), int(resolution[1]), int(resolution[2])
    if rx < 1 or ry < 1 or rz < 1:
        raise ValueError("resolution must be >= 1 on every axis")
    mn = (bounds_min[0] - padding, bounds_min[1] - padding, bounds_min[2] - padding)
    mx = (bounds_max[0] + padding, bounds_max[1] + padding, bounds_max[2] + padding)
    spacing = (
        (mx[0] - mn[0]) / rx,
        (mx[1] - mn[1]) / ry,
        (mx[2] - mn[2]) / rz,
    )
    nx, ny, nz = rx + 1, ry + 1, rz + 1
    values = [0.0] * (nx * ny * nz)
    idx = 0
    for k in range(nz):
        z = mn[2] + k * spacing[2]
        for j in range(ny):
            y = mn[1] + j * spacing[1]
            for i in range(nx):
                x = mn[0] + i * spacing[0]
                values[idx] = float(field((x, y, z)))
                idx += 1
    return ScalarGrid(values, (nx, ny, nz), mn, spacing)


def _cell_vertex(grid: ScalarGrid, i: int, j: int, k: int, level: float):
    """Feature point (world space) for a cell, or ``None`` if no sign change.

    The point is the average of the linearly-interpolated crossings on every
    edge of the cube -- the "naive surface nets" rule.
    """
    corner = [
        grid.get(i + c[0], j + c[1], k + c[2]) - level for c in _CUBE_CORNERS
    ]
    inside = [v < 0.0 for v in corner]
    if all(inside) or not any(inside):
        return None
    sx = sy = sz = 0.0
    count = 0
    for a, b in _CUBE_EDGES:
        va, vb = corner[a], corner[b]
        if (va < 0.0) == (vb < 0.0):
            continue
        t = va / (va - vb)  # crossing fraction along edge a->b
        ca, cb = _CUBE_CORNERS[a], _CUBE_CORNERS[b]
        sx += ca[0] + t * (cb[0] - ca[0])
        sy += ca[1] + t * (cb[1] - ca[1])
        sz += ca[2] + t * (cb[2] - ca[2])
        count += 1
    inv = 1.0 / count
    return grid.world(i + sx * inv, j + sy * inv, k + sz * inv)


def surface_nets(
    grid: ScalarGrid, level: float = 0.0
) -> Tuple[List[Vec3], List[Tuple[int, int, int]]]:
    """Extract a triangle mesh from ``grid`` at iso-value ``level``.

    Returns ``(positions, faces)`` where ``positions`` is a list of 3-tuples and
    ``faces`` a list of triangle index triples.  Quads (one per minimal edge
    with a sign change) are split into two triangles with consistent winding.
    """
    nx, ny, nz = grid.shape
    verts: List[Vec3] = []
    vindex: Dict[Tuple[int, int, int], int] = {}

    def vertex_id(i: int, j: int, k: int):
        key = (i, j, k)
        got = vindex.get(key, -1)
        if got != -1:
            return got
        p = _cell_vertex(grid, i, j, k, level)
        if p is None:
            vindex[key] = None  # type: ignore[assignment]
            return None
        vid = len(verts)
        verts.append(p)
        vindex[key] = vid
        return vid

    faces: List[Tuple[int, int, int]] = []

    def emit_quad(a, b, c, d, flip: bool) -> None:
        if a is None or b is None or c is None or d is None:
            return
        if flip:
            faces.append((a, b, c))
            faces.append((a, c, d))
        else:
            faces.append((a, c, b))
            faces.append((a, d, c))

    for k in range(nz - 1):
        for j in range(ny - 1):
            for i in range(nx - 1):
                v0 = grid.get(i, j, k) - level < 0.0
                # Edge along +x: quad in the y-z plane around the edge.
                if i + 1 < nx and j > 0 and k > 0:
                    vx = grid.get(i + 1, j, k) - level < 0.0
                    if v0 != vx:
                        emit_quad(
                            vertex_id(i, j - 1, k - 1),
                            vertex_id(i, j, k - 1),
                            vertex_id(i, j, k),
                            vertex_id(i, j - 1, k),
                            v0 and not vx,
                        )
                # Edge along +y: quad in the x-z plane.
                if j + 1 < ny and i > 0 and k > 0:
                    vy = grid.get(i, j + 1, k) - level < 0.0
                    if v0 != vy:
                        emit_quad(
                            vertex_id(i - 1, j, k - 1),
                            vertex_id(i, j, k - 1),
                            vertex_id(i, j, k),
                            vertex_id(i - 1, j, k),
                            not (v0 and not vy),
                        )
                # Edge along +z: quad in the x-y plane.
                if k + 1 < nz and i > 0 and j > 0:
                    vz = grid.get(i, j, k + 1) - level < 0.0
                    if v0 != vz:
                        emit_quad(
                            vertex_id(i - 1, j - 1, k),
                            vertex_id(i, j - 1, k),
                            vertex_id(i, j, k),
                            vertex_id(i - 1, j, k),
                            v0 and not vz,
                        )
    return verts, faces


def interpolate_attribute(
    field_a: Callable[[Sequence[float]], float],
    attr_a: Sequence[float],
    field_b: Callable[[Sequence[float]], float],
    attr_b: Sequence[float],
    point: Sequence[float],
) -> List[float]:
    """Blend two per-primitive attribute vectors at a surface ``point``.

    Reproduces sdf-csg's user-data interpolation: the blend weight is the
    normalised inverse distance to each contributing primitive, so the value
    closest to a primitive's own surface is weighted toward that primitive.
    """
    if len(attr_a) != len(attr_b):
        raise ValueError("attribute vectors must have equal length")
    d1 = abs(field_a(point))
    d2 = abs(field_b(point))
    denom = d1 + d2
    frac = 0.5 if denom == 0.0 else d1 / denom
    return [attr_a[i] + frac * (attr_b[i] - attr_a[i]) for i in range(len(attr_a))]


def _tri_normal(a: Vec3, b: Vec3, c: Vec3) -> Vec3:
    ux, uy, uz = b[0] - a[0], b[1] - a[1], b[2] - a[2]
    vx, vy, vz = c[0] - a[0], c[1] - a[1], c[2] - a[2]
    nx = uy * vz - uz * vy
    ny = uz * vx - ux * vz
    nz = ux * vy - uy * vx
    m = math.sqrt(nx * nx + ny * ny + nz * nz)
    if m == 0.0:
        return (0.0, 0.0, 0.0)
    return (nx / m, ny / m, nz / m)


def mesh_to_stl(
    positions: Sequence[Vec3],
    faces: Sequence[Tuple[int, int, int]],
    name: str = "sdfcsg",
) -> str:
    """Serialise a triangle mesh to ASCII STL with per-facet normals."""
    out = ["solid %s" % name]
    for f in faces:
        a, b, c = positions[f[0]], positions[f[1]], positions[f[2]]
        n = _tri_normal(a, b, c)
        out.append("  facet normal %.6e %.6e %.6e" % n)
        out.append("    outer loop")
        for v in (a, b, c):
            out.append("      vertex %.6e %.6e %.6e" % (v[0], v[1], v[2]))
        out.append("    endloop")
        out.append("  endfacet")
    out.append("endsolid %s" % name)
    return "\n".join(out) + "\n"
