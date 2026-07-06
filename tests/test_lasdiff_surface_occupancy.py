import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from geometry.lasdiff_surface_occupancy import (
    coarsen_occupancy,
    occupancy_iou,
    occupied_cells,
    shell_thickness,
    subvoxels_of,
    surface_occupancy,
)


class TestSurfaceOccupancy(unittest.TestCase):
    def test_threshold_rule(self):
        sdf = {(0, 0, 0): 0.0, (1, 0, 0): 0.05, (2, 0, 0): 0.2, (3, 0, 0): -0.1}
        occ = surface_occupancy(sdf, delta=0.1)
        self.assertEqual(occ[(0, 0, 0)], 1)
        self.assertEqual(occ[(1, 0, 0)], 1)
        self.assertEqual(occ[(2, 0, 0)], 0)
        self.assertEqual(occ[(3, 0, 0)], 1)  # boundary |−0.1| <= 0.1

    def test_occupied_cells_and_thickness(self):
        sdf = {(0, 0, 0): 0.0, (1, 0, 0): 0.5, (2, 0, 0): -0.02}
        cells = occupied_cells(sdf, delta=0.1)
        self.assertEqual(cells, {(0, 0, 0), (2, 0, 0)})
        self.assertEqual(shell_thickness(sdf, 0.1), 2)

    def test_symmetric_shell(self):
        # positive and negative distances treated symmetrically
        self.assertEqual(occupied_cells({(0, 0, 0): -0.03}, 0.05),
                         occupied_cells({(0, 0, 0): 0.03}, 0.05))

    def test_delta_must_be_positive(self):
        with self.assertRaises(ValueError):
            surface_occupancy({(0, 0, 0): 0.0}, delta=0.0)
        with self.assertRaises(ValueError):
            occupied_cells({(0, 0, 0): 0.0}, delta=-1.0)

    def test_coarsen_any_subvoxel_rule(self):
        # coarse cell (0,0,0) covers fine (0..1)^3; one near-surface subvoxel
        fine = {
            (0, 0, 0): 0.9,          # far
            (1, 1, 1): 1.0 / 64.0,   # within 1/32 -> makes coarse cell occupied
            (2, 0, 0): 0.9,          # coarse cell (1,0,0), far
        }
        occ = coarsen_occupancy(fine, threshold=1.0 / 32.0, factor=2)
        self.assertIn((0, 0, 0), occ)
        self.assertNotIn((1, 0, 0), occ)

    def test_coarsen_boundary_threshold(self):
        fine = {(0, 0, 0): 1.0 / 32.0}
        self.assertEqual(coarsen_occupancy(fine), {(0, 0, 0)})
        fine2 = {(0, 0, 0): 1.0 / 32.0 + 1e-9}
        self.assertEqual(coarsen_occupancy(fine2), set())

    def test_subvoxels_roundtrip(self):
        subs = set(subvoxels_of((3, 4, 5), factor=2))
        self.assertEqual(len(subs), 8)
        # every subvoxel maps back to its coarse cell
        for (i, j, k) in subs:
            self.assertEqual((i // 2, j // 2, k // 2), (3, 4, 5))

    def test_subvoxels_factor_one(self):
        self.assertEqual(list(subvoxels_of((2, 2, 2), factor=1)), [(2, 2, 2)])

    def test_iou(self):
        self.assertEqual(occupancy_iou(set(), set()), 1.0)
        a = {(0, 0, 0), (1, 0, 0)}
        b = {(1, 0, 0), (2, 0, 0)}
        self.assertAlmostEqual(occupancy_iou(a, b), 1.0 / 3.0)

    def test_invalid_factor(self):
        with self.assertRaises(ValueError):
            coarsen_occupancy({(0, 0, 0): 0.0}, factor=0)


if __name__ == "__main__":
    unittest.main()
