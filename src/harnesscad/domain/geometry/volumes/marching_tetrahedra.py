"""Marching tetrahedra: deterministic iso-surface extraction on a tet grid.

Given per-vertex signed-distance (SDF) values on a tetrahedral grid, marching
tetrahedra (Doi & Koide, 1991; used by DMTet [Shen et al. 2021] and MeshDiffusion
[Liu et al. 2023]) extracts the zero level-set as a triangle mesh.  For each
tetrahedron the four vertex SDF signs give one of 16 cases; the surface is the set
of points where the *piecewise-linear* interpolation of the SDF is zero.

Sign convention (matching DMTet / the paper appendix, "positive values represent
outside-ness"): ``sdf > 0`` is *outside*, ``sdf <= 0`` is *inside*.  A tetrahedron
contributes triangles only when its vertices are not all of the same sign.

For a mesh-generating edge ``(a, b)`` with SDF values ``sa, sb`` the surface vertex
is the exact linear interpolation from the paper (Section 3):

    vp = (va * sb - vb * sa) / (sb - sa)

Surface vertices are *welded* across tetrahedra by the shared grid-edge key, so a
closed SDF (e.g. a sphere) yields a watertight, edge-manifold triangle mesh.  The
triangle table is the standard DMTet table, which orders each triangle so the
face normal points toward the positive (outside) region.
"""

from __future__ import annotations

# Local tetrahedron edges, indexed 0..5, matching the triangle table below.
_EDGES = ((0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3))

# Standard marching-tetrahedra (DMTet) triangle table, indexed by the 4-bit
# occupancy code (bit v set when sdf[v] > 0).  Entries are edge indices into
# ``_EDGES``; ``-1`` pads unused slots.  Cases 0 and 15 (all inside / all outside)
# emit nothing.
_TRI_TABLE = (
    (-1, -1, -1, -1, -1, -1),
    (1, 0, 2, -1, -1, -1),
    (4, 0, 3, -1, -1, -1),
    (1, 4, 2, 1, 3, 4),
    (3, 1, 5, -1, -1, -1),
    (2, 3, 0, 2, 5, 3),
    (1, 4, 0, 1, 5, 4),
    (4, 2, 5, -1, -1, -1),
    (4, 5, 2, -1, -1, -1),
    (4, 1, 0, 4, 5, 1),
    (3, 2, 0, 3, 5, 2),
    (1, 3, 5, -1, -1, -1),
    (4, 1, 2, 4, 3, 1),
    (3, 0, 4, -1, -1, -1),
    (2, 0, 1, -1, -1, -1),
    (-1, -1, -1, -1, -1, -1),
)


def _interp(pa, pb, sa, sb):
    """Zero-crossing point on edge (pa, pb): (pa*sb - pb*sa) / (sb - sa)."""
    denom = sb - sa
    return tuple((pa[i] * sb - pb[i] * sa) / denom for i in range(3))


def marching_tets(vertices, tets, sdf):
    """Extract the SDF zero-surface as a welded triangle mesh.

    Args:
        vertices: list of ``(x, y, z)`` grid-vertex positions.
        tets: list of ``(a, b, c, d)`` vertex-index tuples.
        sdf: sequence of per-vertex SDF floats (``> 0`` outside).

    Returns:
        ``(out_vertices, triangles)`` where ``out_vertices`` is a list of
        ``(x, y, z)`` surface points and ``triangles`` a list of ``(i, j, k)``
        index triples into ``out_vertices``.
    """
    if len(sdf) != len(vertices):
        raise ValueError("sdf must have one value per vertex")

    edge_to_out = {}
    out_vertices = []
    triangles = []

    for tet in tets:
        code = 0
        for local in range(4):
            if sdf[tet[local]] > 0:
                code |= 1 << local
        if code == 0 or code == 15:
            continue
        row = _TRI_TABLE[code]
        for base in (0, 3):
            e0 = row[base]
            if e0 == -1:
                break
            tri = []
            for e in row[base:base + 3]:
                la, lb = _EDGES[e]
                ga, gb = tet[la], tet[lb]
                key = (ga, gb) if ga < gb else (gb, ga)
                out_idx = edge_to_out.get(key)
                if out_idx is None:
                    a, b = key
                    p = _interp(vertices[a], vertices[b], sdf[a], sdf[b])
                    out_idx = len(out_vertices)
                    out_vertices.append(p)
                    edge_to_out[key] = out_idx
                tri.append(out_idx)
            triangles.append((tri[0], tri[1], tri[2]))

    return out_vertices, triangles


def edge_manifold_stats(triangles):
    """Return ``(boundary_edges, nonmanifold_edges)`` counts for a triangle set.

    A closed (watertight) surface has every undirected edge shared by exactly two
    triangles: zero boundary edges (shared once) and zero non-manifold edges
    (shared 3+ times).
    """
    counts = {}
    for (i, j, k) in triangles:
        for a, b in ((i, j), (j, k), (k, i)):
            key = (a, b) if a < b else (b, a)
            counts[key] = counts.get(key, 0) + 1
    boundary = sum(1 for c in counts.values() if c == 1)
    nonmanifold = sum(1 for c in counts.values() if c > 2)
    return boundary, nonmanifold


def is_watertight(triangles):
    """True when every edge is shared by exactly two triangles (closed surface)."""
    if not triangles:
        return False
    boundary, nonmanifold = edge_manifold_stats(triangles)
    return boundary == 0 and nonmanifold == 0


def signed_volume(vertices, triangles):
    """Signed volume enclosed by a triangle mesh (sum of tetra to origin).

    Positive for a closed mesh with outward-facing winding, which is what the
    DMTet triangle table produces (normals point to the outside/positive region).
    """
    total = 0.0
    for (i, j, k) in triangles:
        a, b, c = vertices[i], vertices[j], vertices[k]
        cross = (
            b[1] * c[2] - b[2] * c[1],
            b[2] * c[0] - b[0] * c[2],
            b[0] * c[1] - b[1] * c[0],
        )
        total += a[0] * cross[0] + a[1] * cross[1] + a[2] * cross[2]
    return total / 6.0
