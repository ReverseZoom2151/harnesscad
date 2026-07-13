"""DMTet: deformable tetrahedral grid encoding of a mesh.

Deep Marching Tetrahedra (DMTet, Shen et al. 2021), the parametrisation used by
MeshDiffusion, augments a uniform tetrahedral grid with two per-vertex attributes:

  * an SDF value ``s`` (sign encodes inside/outside), and
  * a 3D deformation vector ``d`` that moves the vertex from its lattice position.

The surface is extracted by marching tetrahedra on the *deformed* grid.  Because
the deformation only relocates vertices (it does not change connectivity), the
grid remains a valid tetrahedral complex as long as the deformation is clipped
small enough not to invert any tetrahedron -- which is exactly why MeshDiffusion
(Appendix A.2) clips the deformation vectors after every update.

This module provides:
  * :class:`DMTet` -- the encoding (base grid + sdf + deformation) with surface
    extraction, deformation clipping and inversion counting;
  * SDF sign-normalisation to +/-1 (Section 4.3 two-pass scheme), which preserves
    topology while forcing ``|sb - sa| = 2`` on every mesh-generating edge;
  * barycentric SDF interpolation inside a tetrahedron (the DMTet SDF query).

Imports the grid and extractor by full path to stay self-contained.
"""

from __future__ import annotations

from harnesscad.domain.geometry.volumes.meshdiff_tet_grid import TetGrid, _signed_volume
from harnesscad.domain.geometry.volumes.meshdiff_marching_tets import marching_tets


def _clip(value, bound):
    if value > bound:
        return bound
    if value < -bound:
        return -bound
    return value


class DMTet:
    """A deformable tetrahedral grid with per-vertex SDF and deformation."""

    def __init__(self, grid, sdf, deformation=None, clip=None):
        if not isinstance(grid, TetGrid):
            raise TypeError("grid must be a TetGrid")
        if len(sdf) != grid.num_vertices:
            raise ValueError("sdf must have one value per grid vertex")
        self.grid = grid
        self.sdf = [float(s) for s in sdf]
        cell = (grid.hi - grid.lo) / grid.resolution
        # Default clip bound: half a cell edge, so a vertex cannot cross past a
        # neighbour and invert its tetrahedra.
        self.clip = 0.5 * cell if clip is None else float(clip)
        if deformation is None:
            self.deformation = [(0.0, 0.0, 0.0)] * grid.num_vertices
        else:
            if len(deformation) != grid.num_vertices:
                raise ValueError("deformation must have one vector per vertex")
            self.deformation = [self._clip_vec(d) for d in deformation]

    def _clip_vec(self, d):
        b = self.clip
        return (_clip(d[0], b), _clip(d[1], b), _clip(d[2], b))

    # -- geometry ------------------------------------------------------------
    def deformed_vertices(self):
        """Base lattice positions plus the (clipped) deformation vectors."""
        out = []
        for base, d in zip(self.grid.vertices, self.deformation):
            out.append((base[0] + d[0], base[1] + d[1], base[2] + d[2]))
        return out

    def extract_surface(self):
        """Marching-tetrahedra surface of the deformed grid.

        Returns ``(vertices, triangles)``.
        """
        return marching_tets(self.deformed_vertices(), self.grid.tets, self.sdf)

    def inverted_tet_count(self):
        """Number of tetrahedra whose signed volume flipped sign after deforming.

        Zero means the deformation preserved the orientation of every tetrahedron
        (a valid, non-self-inverting deformable grid).
        """
        dv = self.deformed_vertices()
        count = 0
        for tet in self.grid.tets:
            a, b, c, d = (dv[i] for i in tet)
            if _signed_volume(a, b, c, d) <= 0.0:
                count += 1
        return count

    def is_valid(self):
        """True when no tetrahedron is inverted or degenerate by the deformation."""
        return self.inverted_tet_count() == 0

    # -- SDF normalisation (Section 4.3) ------------------------------------
    def normalized(self):
        """Return a copy with SDF replaced by its signs (+1 outside, -1 inside).

        Topology is preserved exactly: every vertex keeps its occupancy
        (``s > 0``), so the set of mesh-generating edges is unchanged, while every
        mesh-generating edge now has ``|sb - sa| = 2`` -- removing the arbitrary
        SDF scale that makes naive marching tetrahedra noise-sensitive.
        """
        norm = [1.0 if s > 0 else -1.0 for s in self.sdf]
        return DMTet(self.grid, norm, self.deformation, clip=self.clip)


def barycentric_coords(tet_pts, point):
    """Barycentric coordinates ``(a0, a1, a2, a3)`` of ``point`` in a tetrahedron.

    ``tet_pts`` is a 4-tuple of ``(x, y, z)`` corners.  Solves the linear system so
    that ``point = sum(ai * corner_i)`` and ``sum(ai) = 1``.
    """
    p0, p1, p2, p3 = tet_pts
    # Columns of the 3x3 matrix [p1-p0, p2-p0, p3-p0].
    m = [
        [p1[r] - p0[r], p2[r] - p0[r], p3[r] - p0[r]]
        for r in range(3)
    ]
    rhs = [point[r] - p0[r] for r in range(3)]
    det = _det3(m)
    if abs(det) < 1e-18:
        raise ValueError("degenerate tetrahedron")
    b1 = _det3(_replace_col(m, 0, rhs)) / det
    b2 = _det3(_replace_col(m, 1, rhs)) / det
    b3 = _det3(_replace_col(m, 2, rhs)) / det
    b0 = 1.0 - b1 - b2 - b3
    return (b0, b1, b2, b3)


def interpolate_sdf_in_tet(tet_pts, tet_sdf, point):
    """DMTet SDF query: barycentric interpolation ``sq = sum(ai * si)``."""
    a = barycentric_coords(tet_pts, point)
    return sum(a[i] * tet_sdf[i] for i in range(4))


def _det3(m):
    return (
        m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1])
        - m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0])
        + m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0])
    )


def _replace_col(m, col, vec):
    return [[vec[r] if c == col else m[r][c] for c in range(3)] for r in range(3)]
