"""Triplane 3D representation for TAR3D.

From "TAR3D: Creating High-Quality 3D Assets via Next-Part Prediction" (Zhang,
Liu, Li, Zhang, et al., 2024). TAR3D's learned VQ-VAE / GPT are external, but the
paper's *representation* is a purely deterministic, axis-aligned construction and
is described precisely (Section 3, Fig. 2):

  * "we propose to represent the 3D shape information of meshes with triplane
    latent representations whose feature maps are associated with three-axis
    planes, i.e., XY, YZ, and XZ."
  * the triplane is "three 2D feature maps with a fixed size", each of height ``h``
    and width ``w`` (the paper uses ``h = w = 32``).
  * "the indices within each plane are placed in a raster scan order and the
    indices at the same positions of the three planes in an adjacent order."

This module implements the *geometric* half: projecting a voxel occupancy grid
onto the three axis-aligned planes (the deterministic 3D-aware encoding), and
back-projecting the three planes into a visual-hull voxel set (the deterministic
decoding, before the learned occupancy MLP). A "part cell" is one (row, col)
location shared across the three planes -- the unit the next-part sequence orders.

The index-sequence ordering and TriPE positional encoding live in
``reconstruction.tar3d_part_sequence`` and ``geometry.tar3d_tripe``; this module
holds only the plane geometry. Distinct from ``reconstruction.gaussiancad_visual_hull``
(image-silhouette carving from cameras) -- here the three silhouettes are the
axis projections of a voxel grid, exactly TAR3D's XY/YZ/XZ planes. Stdlib-only,
deterministic.
"""

from __future__ import annotations

from typing import Dict, Iterable, List, Sequence, Set, Tuple

Voxel = Tuple[int, int, int]
Cell = Tuple[int, int]

# The three axis planes, in TAR3D order.
PLANES = ("XY", "YZ", "XZ")

# For each plane, the two voxel axes it keeps (row axis, col axis) and the axis
# it projects away. Axes: 0 = x, 1 = y, 2 = z.
_PLANE_AXES = {
    "XY": (0, 1, 2),
    "YZ": (1, 2, 0),
    "XZ": (0, 2, 1),
}


class TriplaneGrid:
    """Three axis-aligned occupancy planes over a voxel grid of size ``dims``.

    ``dims`` is ``(nx, ny, nz)``. Each plane is stored as a dict mapping an
    occupied ``(row, col)`` cell to the number of voxels that projected onto it
    (the silhouette "thickness"), so the planes double as projection histograms.
    """

    def __init__(self, dims: Tuple[int, int, int]):
        nx, ny, nz = dims
        if nx <= 0 or ny <= 0 or nz <= 0:
            raise ValueError("dims must be positive")
        self.dims = (nx, ny, nz)
        self.planes: Dict[str, Dict[Cell, int]] = {p: {} for p in PLANES}

    # -- plane geometry -------------------------------------------------------

    def plane_shape(self, plane: str) -> Tuple[int, int]:
        """Return the ``(height, width)`` of ``plane`` from the voxel dims."""
        ra, ca, _ = _PLANE_AXES[plane]
        return (self.dims[ra], self.dims[ca])

    def _cell_of(self, plane: str, voxel: Voxel) -> Cell:
        ra, ca, _ = _PLANE_AXES[plane]
        return (voxel[ra], voxel[ca])

    # -- encoding (3D -> triplane) -------------------------------------------

    @classmethod
    def from_voxels(cls, occupied: Iterable[Voxel],
                    dims: Tuple[int, int, int]) -> "TriplaneGrid":
        """Project an occupied-voxel set onto the three XY/YZ/XZ planes."""
        grid = cls(dims)
        nx, ny, nz = dims
        for v in occupied:
            x, y, z = v
            if not (0 <= x < nx and 0 <= y < ny and 0 <= z < nz):
                raise ValueError("voxel %r outside dims %r" % (v, dims))
            for p in PLANES:
                cell = grid._cell_of(p, v)
                grid.planes[p][cell] = grid.planes[p].get(cell, 0) + 1
        return grid

    def occupied_cells(self, plane: str) -> List[Cell]:
        """Occupied cells of ``plane`` in raster-scan (row-major) order."""
        return sorted(self.planes[plane].keys())

    def occupancy(self, plane: str, cell: Cell) -> int:
        """Silhouette thickness (voxel count) at ``cell`` of ``plane``."""
        return self.planes[plane].get(cell, 0)

    # -- decoding (triplane -> 3D visual hull) --------------------------------

    def visual_hull(self) -> Set[Voxel]:
        """Back-project the three planes into a voxel set (their intersection).

        A voxel ``(x, y, z)`` survives iff all three of its axis projections are
        occupied -- the classic space-carving visual hull. This is a superset of
        any original shape whose projections produced these planes (see
        ``carves_superset``), the deterministic reconstruction TAR3D's occupancy
        decoder later refines.
        """
        nx, ny, nz = self.dims
        xy = self.planes["XY"]
        yz = self.planes["YZ"]
        xz = self.planes["XZ"]
        out: Set[Voxel] = set()
        for (x, y) in xy:
            for z in range(nz):
                if (y, z) in yz and (x, z) in xz:
                    out.add((x, y, z))
        return out

    def carves_superset(self, occupied: Iterable[Voxel]) -> bool:
        """True if the visual hull contains every voxel in ``occupied``."""
        hull = self.visual_hull()
        return all(v in hull for v in occupied)


def raster_index(cell: Cell, width: int) -> int:
    """Row-major raster-scan index of ``cell = (row, col)`` in a width-``w`` map."""
    row, col = cell
    if not (0 <= col < width):
        raise ValueError("col %d out of width %d" % (col, width))
    return row * width + col


def raster_cell(index: int, width: int) -> Cell:
    """Inverse of :func:`raster_index`."""
    if width <= 0:
        raise ValueError("width must be positive")
    if index < 0:
        raise ValueError("index must be non-negative")
    return (index // width, index % width)
