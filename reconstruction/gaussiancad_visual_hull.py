"""Deterministic visual-hull carving from silhouette masks (GaussianCAD init).

GaussianCAD initialises its 3D Gaussians from the **visual hull** built from the
masked orthographic views (Sec. 3.4 / Fig. 2): "we initialize the 3D Gaussians by
constructing the visual hull using the masked views and corresponding camera
parameters." The subsequent Gaussian optimisation is learned, but constructing
the visual hull itself is classical, deterministic space carving — a voxel is
kept iff its projection falls inside the foreground mask of **every** view.

This module implements space carving over a regular voxel grid for a set of
silhouettes, where each silhouette is a binary foreground mask sampled on a 2D
grid plus a projection from world space to that mask's pixel coordinates. It is
generic over the projection function, so it works with the orthographic
projections of ``geometry.gaussiancad_camera`` / ``geometry.gaussiancad_splatting``
or any callable ``world_point -> (u, v)``.

Pure stdlib, deterministic (no randomness, no wall clock).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List, Sequence, Tuple

Vec3 = Tuple[float, float, float]


@dataclass(frozen=True)
class Silhouette:
    """A binary foreground mask on a regular pixel grid plus a world projection.

    ``mask[row][col]`` is truthy where the pixel is foreground. ``width`` and
    ``height`` are the pixel-grid dimensions. ``project`` maps a world point to
    continuous ``(u, v)`` pixel coordinates (u = column, v = row).
    """

    mask: Tuple[Tuple[int, ...], ...]
    width: int
    height: int
    project: Callable[[Vec3], Tuple[float, float]]

    def contains(self, point: Vec3) -> bool:
        """True iff ``point`` projects onto a foreground pixel of this mask."""
        u, v = self.project(point)
        col = int(round(u))
        row = int(round(v))
        if col < 0 or col >= self.width or row < 0 or row >= self.height:
            return False
        return bool(self.mask[row][col])


def make_silhouette(mask: Sequence[Sequence[int]],
                    project: Callable[[Vec3], Tuple[float, float]]) -> Silhouette:
    """Build a :class:`Silhouette` from a 2D mask and a projection callable."""
    rows = tuple(tuple(int(bool(c)) for c in row) for row in mask)
    height = len(rows)
    width = len(rows[0]) if rows else 0
    if any(len(r) != width for r in rows):
        raise ValueError("mask rows must all have the same width")
    return Silhouette(mask=rows, width=width, height=height, project=project)


@dataclass(frozen=True)
class VoxelGrid:
    """A regular axis-aligned voxel grid over ``[origin, origin + n*spacing]``."""

    origin: Vec3
    spacing: float
    nx: int
    ny: int
    nz: int

    def center(self, i: int, j: int, k: int) -> Vec3:
        """World-space center of voxel ``(i, j, k)``."""
        return (
            self.origin[0] + (i + 0.5) * self.spacing,
            self.origin[1] + (j + 0.5) * self.spacing,
            self.origin[2] + (k + 0.5) * self.spacing,
        )

    def voxel_volume(self) -> float:
        return self.spacing ** 3


def carve_visual_hull(grid: VoxelGrid,
                      silhouettes: Sequence[Silhouette]) -> List[Vec3]:
    """Return the world-space centers of voxels inside the visual hull.

    A voxel is in the hull iff its center projects into the foreground of *every*
    silhouette (logical AND of the back-projected masks). With no silhouettes the
    hull is empty (nothing constrains it). Order is deterministic: i (x) outer,
    then j (y), then k (z).
    """
    if not silhouettes:
        return []
    kept: List[Vec3] = []
    for i in range(grid.nx):
        for j in range(grid.ny):
            for k in range(grid.nz):
                c = grid.center(i, j, k)
                if all(s.contains(c) for s in silhouettes):
                    kept.append(c)
    return kept


def hull_occupancy(grid: VoxelGrid, silhouettes: Sequence[Silhouette]) -> float:
    """Fraction of grid voxels retained by the visual hull, in ``[0, 1]``."""
    total = grid.nx * grid.ny * grid.nz
    if total == 0:
        return 0.0
    return len(carve_visual_hull(grid, silhouettes)) / total


def hull_bounding_box(points: Sequence[Vec3]) -> Dict[str, Vec3]:
    """Axis-aligned ``{"min": (..), "max": (..)}`` of the carved hull points."""
    if not points:
        raise ValueError("cannot bound an empty hull")
    lo = tuple(min(p[d] for p in points) for d in range(3))
    hi = tuple(max(p[d] for p in points) for d in range(3))
    return {"min": lo, "max": hi}
