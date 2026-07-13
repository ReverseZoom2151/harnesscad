"""UV-Net grid normalisation: mask-aware bounding box, unit-box scaling, 90-degree frames.

``datasets/util.py`` of UV-Net (Jayaraman et al., CVPR 2021) normalises every
solid *through its UV-grids* rather than through its mesh:

  * ``bounding_box_uvgrid`` takes the bounding box of only the grid points whose
    *trimming mask channel* is 1 -- points outside the trimmed face are sampled
    but must not influence the box;
  * ``center_and_scale_uvgrid`` recentres on the box centre and scales by
    ``2 / max(diag)`` so the solid lands in ``[-1, 1]^3``;
  * ``get_random_rotation`` / ``rotate_uvgrid`` apply an axis-aligned quarter
    turn to *both* the point channels and the normal/tangent channels.

This module rebuilds all of it deterministically (the random draw becomes an
explicit, indexable enumeration of the 12 axis-aligned quarter turns, so a
dataset augmentation is reproducible):

* :func:`bounding_box`, :func:`bounding_box_uvgrid` (mask-aware),
  :func:`bounding_box_uvgrids` (a whole solid).
* :func:`center_and_scale_uvgrid` / :func:`center_and_scale_solid` -- returns the
  transformed grids plus the ``(center, scale)`` actually applied, so the
  transform can be inverted (:func:`apply_center_scale`, :func:`invert_point`).
* :func:`quarter_turns`, :func:`rotation_matrix`, :func:`rotate_grid` --
  point + direction channels rotated together; normals stay unit because the
  matrices are orthonormal.
* :func:`grid_bounds_check` -- the post-condition ``all |coordinate| <= 1``.

Grids are the 7-channel face grids of :mod:`geometry.uvnet_uv_grid` (nested
``num_u x num_v``) or the 6-channel edge grids of
:mod:`geometry.uvnet_u_grid` (flat ``num_u``); every routine accepts both,
detecting the shape from the channel count.
"""

from __future__ import annotations

import math
from typing import Sequence, Tuple

Point = Tuple[float, float, float]

_EPS = 1e-12

FACE_CHANNELS = 7
EDGE_CHANNELS = 6


# --------------------------------------------------------------------------- #
# shape helpers -- a face grid is nested, an edge grid is flat
# --------------------------------------------------------------------------- #
def _is_face_grid(grid) -> bool:
    if not grid:
        raise ValueError("empty grid")
    first = grid[0]
    return isinstance(first, (list, tuple)) and first and isinstance(
        first[0], (list, tuple))


def _cells(grid):
    """Iterate the feature cells of a face grid or an edge grid."""
    if _is_face_grid(grid):
        for row in grid:
            for cell in row:
                yield cell
    else:
        for cell in grid:
            yield cell


def _rebuild(grid, cells):
    """Reassemble a grid of the same shape from a flat cell iterator."""
    it = iter(cells)
    if _is_face_grid(grid):
        return [[next(it) for _ in row] for row in grid]
    return [next(it) for _ in grid]


# --------------------------------------------------------------------------- #
# bounding boxes
# --------------------------------------------------------------------------- #
def bounding_box(points: Sequence[Point]):
    """``((xmin, ymin, zmin), (xmax, ymax, zmax))`` of a point list."""
    if not points:
        raise ValueError("bounding_box of an empty point list")
    lo = [min(p[i] for p in points) for i in range(3)]
    hi = [max(p[i] for p in points) for i in range(3)]
    return (tuple(lo), tuple(hi))


def grid_points(grid, masked: bool = True) -> list:
    """The 3D points of a grid; for 7-channel grids optionally mask-filtered."""
    pts = []
    for cell in _cells(grid):
        if masked and len(cell) >= FACE_CHANNELS and cell[6] < 0.5:
            continue
        pts.append((cell[0], cell[1], cell[2]))
    return pts


def bounding_box_uvgrid(grid, masked: bool = True):
    """Bounding box of a single grid, ignoring trimmed-away nodes (UV-Net)."""
    return bounding_box(grid_points(grid, masked=masked))


def bounding_box_uvgrids(grids: Sequence, masked: bool = True):
    """Bounding box of a whole solid's worth of grids."""
    pts = []
    for grid in grids:
        pts.extend(grid_points(grid, masked=masked))
    return bounding_box(pts)


def box_diagonal(box) -> Tuple[float, float, float]:
    lo, hi = box
    return tuple(hi[i] - lo[i] for i in range(3))


def box_center(box) -> Point:
    lo, hi = box
    return tuple(0.5 * (lo[i] + hi[i]) for i in range(3))


# --------------------------------------------------------------------------- #
# centre + scale
# --------------------------------------------------------------------------- #
def center_scale_from_box(box, extent: float = 2.0):
    """``(center, scale)`` mapping ``box`` into a cube of side ``extent``.

    UV-Net uses ``scale = 2 / max(diag)`` -- an isotropic scale keyed on the
    longest side, so the shape's aspect ratio survives.
    """
    diag = box_diagonal(box)
    longest = max(diag)
    if longest < _EPS:
        raise ValueError("degenerate bounding box: zero extent")
    return box_center(box), extent / longest


def apply_center_scale(grid, center: Point, scale: float):
    """Recentre + scale the *point* channels of a grid; directions untouched.

    Translation does not affect normals/tangents and a positive uniform scale
    does not change their direction, so channels 3..5 are copied verbatim --
    the same in-place behaviour as ``center_and_scale_uvgrid``.
    """
    out = []
    for cell in _cells(grid):
        p = tuple((cell[i] - center[i]) * scale for i in range(3))
        out.append(tuple(p) + tuple(cell[3:]))
    return _rebuild(grid, out)


def center_and_scale_uvgrid(grid, extent: float = 2.0, masked: bool = True):
    """Normalise one grid to ``[-1, 1]^3``; returns ``(grid, center, scale)``."""
    center, scale = center_scale_from_box(
        bounding_box_uvgrid(grid, masked=masked), extent)
    return apply_center_scale(grid, center, scale), center, scale


def center_and_scale_solid(face_grids: Sequence, edge_grids: Sequence = (),
                           extent: float = 2.0, masked: bool = True):
    """Normalise a whole solid with ONE transform derived from the face grids.

    Returns ``(face_grids, edge_grids, center, scale)``.  The edge U-grids are
    carried through the *same* transform -- deriving a separate one per grid
    would break the graph's geometric consistency.
    """
    center, scale = center_scale_from_box(
        bounding_box_uvgrids(face_grids, masked=masked), extent)
    faces = [apply_center_scale(g, center, scale) for g in face_grids]
    edges = [apply_center_scale(g, center, scale) for g in edge_grids]
    return faces, edges, center, scale


def invert_point(point: Point, center: Point, scale: float) -> Point:
    """Undo :func:`apply_center_scale` for a single point."""
    return tuple(point[i] / scale + center[i] for i in range(3))


def grid_bounds_check(grids: Sequence, extent: float = 2.0,
                      tol: float = 1e-9, masked: bool = True) -> bool:
    """True when every (masked) point lies in the cube of side ``extent``."""
    half = extent / 2.0 + tol
    for grid in grids:
        for p in grid_points(grid, masked=masked):
            if any(abs(c) > half for c in p):
                return False
    return True


# --------------------------------------------------------------------------- #
# axis-aligned quarter-turn rotations (deterministic replacement of the
# random augmentation in datasets/util.py)
# --------------------------------------------------------------------------- #
_AXES = ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0))


def rotation_matrix(axis_index: int, quarter_turns: int):
    """Rotation about axis ``0|1|2`` by ``quarter_turns * 90`` degrees.

    Entries are exactly 0/+1/-1 (built from integer sin/cos of multiples of
    90 degrees), so repeated application never drifts.
    """
    if axis_index not in (0, 1, 2):
        raise ValueError("axis_index must be 0, 1 or 2")
    k = quarter_turns % 4
    c = (1, 0, -1, 0)[k]
    s = (0, 1, 0, -1)[k]
    x, y, z = _AXES[axis_index]
    # Rodrigues with integer c, s and a canonical axis.
    return (
        (c + x * x * (1 - c), x * y * (1 - c) - z * s, x * z * (1 - c) + y * s),
        (y * x * (1 - c) + z * s, c + y * y * (1 - c), y * z * (1 - c) - x * s),
        (z * x * (1 - c) - y * s, z * y * (1 - c) + x * s, c + z * z * (1 - c)),
    )


def quarter_turns() -> list:
    """The 12 (axis, turns) augmentations UV-Net draws from, in a fixed order."""
    return [(a, k) for a in (0, 1, 2) for k in (0, 1, 2, 3)]


def rotate_vector(vec, matrix) -> Point:
    return tuple(sum(matrix[r][c] * vec[c] for c in range(3)) for r in range(3))


def rotate_grid(grid, matrix):
    """Rotate points (channels 0..2) *and* directions (channels 3..5) of a grid."""
    out = []
    for cell in _cells(grid):
        p = rotate_vector(cell[:3], matrix)
        d = rotate_vector(cell[3:6], matrix)
        out.append(tuple(p) + tuple(d) + tuple(cell[6:]))
    return _rebuild(grid, out)


def rotate_grids(grids: Sequence, matrix) -> list:
    return [rotate_grid(g, matrix) for g in grids]


def matrix_is_orthonormal(matrix, tol: float = 1e-9) -> bool:
    for i in range(3):
        for j in range(3):
            dot = sum(matrix[r][i] * matrix[r][j] for r in range(3))
            expected = 1.0 if i == j else 0.0
            if abs(dot - expected) > tol:
                return False
    det = (matrix[0][0] * (matrix[1][1] * matrix[2][2] - matrix[1][2] * matrix[2][1])
           - matrix[0][1] * (matrix[1][0] * matrix[2][2] - matrix[1][2] * matrix[2][0])
           + matrix[0][2] * (matrix[1][0] * matrix[2][1] - matrix[1][1] * matrix[2][0]))
    return abs(det - 1.0) <= tol


def vector_length(vec) -> float:
    return math.sqrt(sum(c * c for c in vec))
