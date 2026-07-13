"""Uniform (regular) tetrahedral grid over a cube.

MeshDiffusion (Liu et al., ICLR 2023) parameterises 3D meshes with a *deformable
tetrahedral grid*: the cube ``[-1, 1]^3`` is discretised into a uniform lattice of
tetrahedra, each grid vertex carries an SDF value, and the zero-surface is
extracted by marching tetrahedra.  The paper initialises the lattice with a
body-centred-cubic (BCC) tiling; here we build the equally-uniform and
translation-symmetric *Freudenthal / Kuhn* subdivision, in which every unit cube
of a regular ``(n+1)^3`` grid is split into 6 tetrahedra that all share the main
diagonal.  This subdivision is *conforming* (the shared cube-face diagonals match
between neighbouring cubes) so the resulting complex is a valid, watertight
tetrahedralisation of the cube -- exactly the ``vertices + tetrahedra + adjacency``
structure the diffusion model consumes.

Everything here is deterministic: vertices are enumerated in ``(i, j, k)`` raster
order and tetrahedra in cube-raster / permutation order.
"""

from __future__ import annotations

# The six axis permutations of (0, 1, 2).  Each defines one tetrahedron of the
# Kuhn subdivision: start at cube corner (0,0,0), flip the axes in the permuted
# order one at a time, ending at (1,1,1).  All six share the 000--111 diagonal.
_PERMS = (
    (0, 1, 2),
    (0, 2, 1),
    (1, 0, 2),
    (1, 2, 0),
    (2, 0, 1),
    (2, 1, 0),
)


def _signed_volume(a, b, c, d):
    ab = (b[0] - a[0], b[1] - a[1], b[2] - a[2])
    ac = (c[0] - a[0], c[1] - a[1], c[2] - a[2])
    ad = (d[0] - a[0], d[1] - a[1], d[2] - a[2])
    cross = (
        ac[1] * ad[2] - ac[2] * ad[1],
        ac[2] * ad[0] - ac[0] * ad[2],
        ac[0] * ad[1] - ac[1] * ad[0],
    )
    return (ab[0] * cross[0] + ab[1] * cross[1] + ab[2] * cross[2]) / 6.0


def _corner_path(perm):
    """Return the 4 corner offsets (each a 3-tuple of 0/1) for one Kuhn tet."""
    cur = [0, 0, 0]
    corners = [tuple(cur)]
    for axis in perm:
        cur = list(cur)
        cur[axis] = 1
        corners.append(tuple(cur))
    return corners  # 4 corners: 000, ..., 111


class TetGrid:
    """A uniform tetrahedral lattice over an axis-aligned cube.

    Attributes:
        resolution: number of cells per axis (``n``); the grid has ``(n+1)^3``
            vertices.
        lo, hi: the cube spans ``[lo, hi]`` on every axis.
        vertices: list of ``(x, y, z)`` float tuples in ``(i, j, k)`` raster order.
        tets: list of ``(a, b, c, d)`` vertex-index tuples (6 per cube cell).
    """

    def __init__(self, resolution, lo=-1.0, hi=1.0):
        if resolution < 1:
            raise ValueError("resolution must be >= 1")
        if not hi > lo:
            raise ValueError("hi must be greater than lo")
        self.resolution = int(resolution)
        self.lo = float(lo)
        self.hi = float(hi)
        self._build()

    # -- construction --------------------------------------------------------
    def _index(self, i, j, k):
        n1 = self.resolution + 1
        return (i * n1 + j) * n1 + k

    def _build(self):
        n = self.resolution
        n1 = n + 1
        step = (self.hi - self.lo) / n
        verts = []
        for i in range(n1):
            x = self.lo + step * i
            for j in range(n1):
                y = self.lo + step * j
                for k in range(n1):
                    z = self.lo + step * k
                    verts.append((x, y, z))
        self.vertices = verts

        paths = [_corner_path(p) for p in _PERMS]
        tets = []
        for i in range(n):
            for j in range(n):
                for k in range(n):
                    for path in paths:
                        tet = tuple(
                            self._index(i + dx, j + dy, k + dz)
                            for (dx, dy, dz) in path
                        )
                        tets.append(self._orient(tet))
        self.tets = tets

    def _orient(self, tet):
        # Canonically orient a tetrahedron to positive signed volume so that
        # marching-tetrahedra produces globally consistent (outward) winding.
        a, b, c, d = (self.vertices[v] for v in tet)
        if _signed_volume(a, b, c, d) < 0.0:
            return (tet[0], tet[1], tet[3], tet[2])
        return tet

    # -- derived structure ---------------------------------------------------
    @property
    def num_vertices(self):
        return len(self.vertices)

    @property
    def num_tets(self):
        return len(self.tets)

    def edges(self):
        """Return the sorted list of unique undirected edges ``(i, j)`` (i < j).

        Every one of the 6 edges of every tetrahedron is included exactly once.
        """
        seen = set()
        for tet in self.tets:
            for a_idx in range(4):
                for b_idx in range(a_idx + 1, 4):
                    a, b = tet[a_idx], tet[b_idx]
                    if a > b:
                        a, b = b, a
                    seen.add((a, b))
        return sorted(seen)

    def tet_adjacency(self):
        """Face-adjacency between tetrahedra.

        Returns a list ``adj`` where ``adj[t]`` is the sorted list of tet indices
        sharing a triangular face (3 common vertices) with tet ``t``.
        """
        face_map = {}
        for t, tet in enumerate(self.tets):
            s = sorted(tet)
            # the 4 triangular faces = drop one vertex
            for drop in range(4):
                face = tuple(v for idx, v in enumerate(s) if idx != drop)
                face_map.setdefault(face, []).append(t)
        adj = [set() for _ in self.tets]
        for owners in face_map.values():
            if len(owners) == 2:
                a, b = owners
                adj[a].add(b)
                adj[b].add(a)
        return [sorted(s) for s in adj]

    def bounds(self):
        """Return ``((lo, lo, lo), (hi, hi, hi))``."""
        lo = (self.lo, self.lo, self.lo)
        hi = (self.hi, self.hi, self.hi)
        return lo, hi
