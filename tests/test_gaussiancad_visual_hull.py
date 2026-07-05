"""Tests for reconstruction.gaussiancad_visual_hull."""

from __future__ import annotations

import unittest

from reconstruction import gaussiancad_visual_hull as vh


def _ortho_xy(point):
    # project (x,y,z) onto x=col, y=row (drop z); grid indexed directly
    return (point[0], point[1])


def _ortho_xz(point):
    return (point[0], point[2])


class TestSilhouette(unittest.TestCase):
    def test_contains_foreground(self):
        mask = [[0, 0, 0], [0, 1, 0], [0, 0, 0]]
        s = vh.make_silhouette(mask, _ortho_xy)
        self.assertTrue(s.contains((1.0, 1.0, 0.0)))
        self.assertFalse(s.contains((0.0, 0.0, 0.0)))

    def test_out_of_bounds_false(self):
        mask = [[1, 1], [1, 1]]
        s = vh.make_silhouette(mask, _ortho_xy)
        self.assertFalse(s.contains((5.0, 5.0, 0.0)))
        self.assertFalse(s.contains((-1.0, 0.0, 0.0)))

    def test_ragged_mask_raises(self):
        with self.assertRaises(ValueError):
            vh.make_silhouette([[1, 1], [1]], _ortho_xy)

    def test_dimensions(self):
        s = vh.make_silhouette([[1, 0, 1]], _ortho_xy)
        self.assertEqual(s.width, 3)
        self.assertEqual(s.height, 1)


class TestVoxelGrid(unittest.TestCase):
    def test_center(self):
        g = vh.VoxelGrid((0.0, 0.0, 0.0), 1.0, 2, 2, 2)
        self.assertEqual(g.center(0, 0, 0), (0.5, 0.5, 0.5))
        self.assertEqual(g.center(1, 1, 1), (1.5, 1.5, 1.5))

    def test_voxel_volume(self):
        g = vh.VoxelGrid((0.0, 0.0, 0.0), 2.0, 1, 1, 1)
        self.assertEqual(g.voxel_volume(), 8.0)


class TestCarve(unittest.TestCase):
    def _cube_masks(self):
        # 4x4 fully-foreground masks so full grid survives both projections
        full = [[1, 1, 1, 1] for _ in range(4)]
        return [vh.make_silhouette(full, _ortho_xy),
                vh.make_silhouette(full, _ortho_xz)]

    def test_empty_silhouettes_empty_hull(self):
        g = vh.VoxelGrid((0.0, 0.0, 0.0), 1.0, 2, 2, 2)
        self.assertEqual(vh.carve_visual_hull(g, []), [])

    def test_full_mask_keeps_all(self):
        # origin -0.5 so voxel centers land on integers 0,1,2,3 in a 4-wide mask
        g = vh.VoxelGrid((-0.5, -0.5, -0.5), 1.0, 4, 4, 4)
        kept = vh.carve_visual_hull(g, self._cube_masks())
        self.assertEqual(len(kept), 64)

    def test_masks_carve_shape(self):
        # xy-mask keeps only a central column; hull is the intersection
        xy = [[0, 0, 0, 0],
              [0, 1, 1, 0],
              [0, 1, 1, 0],
              [0, 0, 0, 0]]
        full = [[1, 1, 1, 1] for _ in range(4)]
        g = vh.VoxelGrid((-0.5, -0.5, -0.5), 1.0, 4, 4, 4)
        sils = [vh.make_silhouette(xy, _ortho_xy),
                vh.make_silhouette(full, _ortho_xz)]
        kept = vh.carve_visual_hull(g, sils)
        # only x in {1,2}, y in {1,2}, any z (4) -> 2*2*4 = 16
        self.assertEqual(len(kept), 16)
        for (x, y, z) in kept:
            self.assertIn(round(x), (1, 2))
            self.assertIn(round(y), (1, 2))

    def test_deterministic_order(self):
        g = vh.VoxelGrid((-0.5, -0.5, -0.5), 1.0, 4, 4, 4)
        a = vh.carve_visual_hull(g, self._cube_masks())
        b = vh.carve_visual_hull(g, self._cube_masks())
        self.assertEqual(a, b)

    def test_occupancy(self):
        g = vh.VoxelGrid((-0.5, -0.5, -0.5), 1.0, 4, 4, 4)
        occ = vh.hull_occupancy(g, self._cube_masks())
        self.assertAlmostEqual(occ, 1.0)

    def test_bounding_box(self):
        pts = [(0.5, 0.5, 0.5), (3.5, 2.5, 1.5)]
        bb = vh.hull_bounding_box(pts)
        self.assertEqual(bb["min"], (0.5, 0.5, 0.5))
        self.assertEqual(bb["max"], (3.5, 2.5, 1.5))

    def test_bounding_box_empty_raises(self):
        with self.assertRaises(ValueError):
            vh.hull_bounding_box([])


if __name__ == "__main__":
    unittest.main()
