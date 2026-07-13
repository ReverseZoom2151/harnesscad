"""Tests for geometry.dreamcad_tessellation."""

import unittest

from harnesscad.domain.geometry.dreamcad_rational_bezier import unit_weight_grid
from harnesscad.domain.geometry.dreamcad_tessellation import (
    enforce_c0_shared_points,
    mesh_area,
    tessellate_patch,
    tessellate_patches,
)


def _flat_grid(x0=0.0, x1=1.0, y0=0.0, y1=1.0):
    return [[(x0 + (x1 - x0) * i / 3, y0 + (y1 - y0) * j / 3, 0.0)
             for j in range(4)] for i in range(4)]


class TestTessellatePatch(unittest.TestCase):
    def test_vertex_and_triangle_counts(self):
        grid = _flat_grid()
        w = unit_weight_grid()
        verts, tris = tessellate_patch(grid, w, resolution=4)
        self.assertEqual(len(verts), 16)
        self.assertEqual(len(tris), 2 * 9)

    def test_flat_patch_area(self):
        grid = _flat_grid()
        w = unit_weight_grid()
        verts, tris = tessellate_patch(grid, w, resolution=5)
        self.assertAlmostEqual(mesh_area(verts, tris), 1.0, places=6)

    def test_resolution_minimum(self):
        with self.assertRaises(ValueError):
            tessellate_patch(_flat_grid(), unit_weight_grid(), resolution=1)


class TestTessellatePatches(unittest.TestCase):
    def test_weld_merges_shared_boundary(self):
        left = _flat_grid(x0=0.0, x1=1.0)
        right = _flat_grid(x0=1.0, x1=2.0)
        w = unit_weight_grid()
        patches = [(left, w), (right, w)]
        unwelded, _ = tessellate_patches(patches, 4, weld=False)
        welded, tris = tessellate_patches(patches, 4, weld=True)
        self.assertEqual(len(unwelded), 32)
        # 4 shared boundary vertices should be merged away.
        self.assertEqual(len(welded), 28)
        self.assertAlmostEqual(mesh_area(welded, tris), 2.0, places=6)


class TestC0Continuity(unittest.TestCase):
    def test_shared_points_averaged(self):
        # two patches whose boundary control slots disagree slightly
        g0 = _flat_grid(x0=0.0, x1=1.0)
        g1 = _flat_grid(x0=1.0, x1=2.0)
        # perturb the shared edge (i=3 of g0, i=0 of g1) so they differ
        g0[3][0] = (1.0, 0.0, 0.2)
        g1[0][0] = (1.0, 0.0, -0.2)
        w = unit_weight_grid()
        groups = [[(0, 3, 0), (1, 0, 0)]]
        grids, weights = enforce_c0_shared_points([g0, g1], [w, w], groups)
        self.assertEqual(grids[0][3][0], grids[1][0][0])
        self.assertAlmostEqual(grids[0][3][0][2], 0.0)
        # originals not mutated
        self.assertEqual(g0[3][0][2], 0.2)

    def test_weight_averaging(self):
        g0 = _flat_grid()
        g1 = _flat_grid()
        w0 = unit_weight_grid()
        w1 = unit_weight_grid()
        w0[0][0] = 2.0
        w1[0][0] = 4.0
        _, weights = enforce_c0_shared_points(
            [g0, g1], [w0, w1], [[(0, 0, 0), (1, 0, 0)]])
        self.assertAlmostEqual(weights[0][0][0], 3.0)
        self.assertAlmostEqual(weights[1][0][0], 3.0)


if __name__ == "__main__":
    unittest.main()
