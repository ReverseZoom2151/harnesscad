"""Tests for geometry.tar3d_triplane_grid."""

import unittest

from harnesscad.domain.geometry.tar3d_triplane_grid import (
    PLANES,
    TriplaneGrid,
    raster_cell,
    raster_index,
)


class TestPlaneShape(unittest.TestCase):
    def test_shapes_match_axes(self):
        g = TriplaneGrid((3, 4, 5))
        self.assertEqual(g.plane_shape("XY"), (3, 4))
        self.assertEqual(g.plane_shape("YZ"), (4, 5))
        self.assertEqual(g.plane_shape("XZ"), (3, 5))

    def test_all_three_planes(self):
        self.assertEqual(PLANES, ("XY", "YZ", "XZ"))


class TestEncoding(unittest.TestCase):
    def test_single_voxel_projects_to_one_cell_each(self):
        g = TriplaneGrid.from_voxels([(1, 2, 3)], (4, 4, 4))
        self.assertEqual(g.occupied_cells("XY"), [(1, 2)])
        self.assertEqual(g.occupied_cells("YZ"), [(2, 3)])
        self.assertEqual(g.occupied_cells("XZ"), [(1, 3)])

    def test_thickness_counts_projected_voxels(self):
        # A column of 3 voxels along z all share the same XY cell.
        vox = [(0, 0, 0), (0, 0, 1), (0, 0, 2)]
        g = TriplaneGrid.from_voxels(vox, (2, 2, 3))
        self.assertEqual(g.occupancy("XY", (0, 0)), 3)
        # ... but map to distinct YZ cells.
        self.assertEqual(sorted(g.planes["YZ"].keys()), [(0, 0), (0, 1), (0, 2)])

    def test_raster_order(self):
        vox = [(1, 1, 0), (0, 0, 0), (0, 1, 0)]
        g = TriplaneGrid.from_voxels(vox, (2, 2, 1))
        self.assertEqual(g.occupied_cells("XY"), [(0, 0), (0, 1), (1, 1)])

    def test_out_of_range_rejected(self):
        with self.assertRaises(ValueError):
            TriplaneGrid.from_voxels([(5, 0, 0)], (2, 2, 2))


class TestVisualHull(unittest.TestCase):
    def test_hull_is_superset_of_original(self):
        vox = {(0, 0, 0), (1, 1, 1), (0, 1, 0)}
        g = TriplaneGrid.from_voxels(vox, (2, 2, 2))
        self.assertTrue(g.carves_superset(vox))

    def test_hull_of_single_voxel_is_itself(self):
        g = TriplaneGrid.from_voxels([(1, 0, 1)], (2, 2, 2))
        self.assertEqual(g.visual_hull(), {(1, 0, 1)})

    def test_hull_may_add_ghost_voxels(self):
        # Four voxels whose projections saturate all three planes: the hull
        # carves back the entire 2x2x2 cube -- classic space-carving phantom.
        vox = {(0, 0, 0), (0, 1, 1), (1, 0, 1), (1, 1, 0)}
        g = TriplaneGrid.from_voxels(vox, (2, 2, 2))
        hull = g.visual_hull()
        self.assertTrue(vox.issubset(hull))
        self.assertEqual(len(hull), 8)

    def test_full_grid_round_trips(self):
        vox = {(x, y, z) for x in range(2) for y in range(2) for z in range(2)}
        g = TriplaneGrid.from_voxels(vox, (2, 2, 2))
        self.assertEqual(g.visual_hull(), vox)


class TestRaster(unittest.TestCase):
    def test_index_and_inverse(self):
        for row in range(3):
            for col in range(4):
                idx = raster_index((row, col), 4)
                self.assertEqual(raster_cell(idx, 4), (row, col))

    def test_index_value(self):
        self.assertEqual(raster_index((2, 1), 4), 9)

    def test_bad_col(self):
        with self.assertRaises(ValueError):
            raster_index((0, 4), 4)


if __name__ == "__main__":
    unittest.main()
