import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from geometry.lasdiff_sparse_subdivision import (
    fill_gaussian_noise,
    mask_sdf_to_shell,
    reserve_occupied,
    subdivide,
    subdivision_ratio,
)


class TestSparseSubdivision(unittest.TestCase):
    def test_reserve_strict_threshold(self):
        probs = {(0, 0, 0): 0.9, (1, 0, 0): 0.5, (2, 0, 0): 0.51, (3, 0, 0): 0.1}
        res = reserve_occupied(probs, 0.5)
        self.assertEqual(res, {(0, 0, 0), (2, 0, 0)})  # 0.5 excluded (strict >)

    def test_subdivide_count(self):
        active = subdivide({(0, 0, 0)}, factor=2)
        self.assertEqual(len(active), 8)
        self.assertIn((0, 0, 0), active)
        self.assertIn((1, 1, 1), active)
        self.assertNotIn((2, 0, 0), active)

    def test_subdivide_coords(self):
        active = subdivide({(1, 0, 0)}, factor=2)
        # coarse (1,0,0) -> fine i in {2,3}
        for (i, j, k) in active:
            self.assertIn(i, (2, 3))
            self.assertIn(j, (0, 1))
            self.assertIn(k, (0, 1))

    def test_mask_to_shell(self):
        fine = {(0, 0, 0): 0.1, (1, 1, 1): -0.2, (5, 5, 5): 0.3}
        active = subdivide({(0, 0, 0)}, factor=2)
        masked = mask_sdf_to_shell(fine, active)
        self.assertIn((0, 0, 0), masked)
        self.assertIn((1, 1, 1), masked)
        self.assertNotIn((5, 5, 5), masked)  # outside shell dropped

    def test_noise_deterministic(self):
        active = subdivide({(0, 0, 0)}, factor=2)
        a = fill_gaussian_noise(active, seed=7)
        b = fill_gaussian_noise(active, seed=7)
        self.assertEqual(a, b)
        self.assertEqual(set(a.keys()), active)

    def test_noise_seed_varies(self):
        active = subdivide({(0, 0, 0)}, factor=2)
        a = fill_gaussian_noise(active, seed=1)
        b = fill_gaussian_noise(active, seed=2)
        self.assertNotEqual(a, b)

    def test_noise_order_independent(self):
        # same coordinate set from different iteration orders -> same result
        coords1 = [(0, 0, 0), (1, 0, 0), (0, 1, 0)]
        coords2 = [(0, 1, 0), (0, 0, 0), (1, 0, 0)]
        self.assertEqual(fill_gaussian_noise(coords1, seed=3),
                         fill_gaussian_noise(coords2, seed=3))

    def test_ratio(self):
        self.assertAlmostEqual(subdivision_ratio({(0, 0, 0), (1, 0, 0)}, 8), 0.25)

    def test_ratio_errors(self):
        with self.assertRaises(ValueError):
            subdivision_ratio({(0, 0, 0)}, 0)
        with self.assertRaises(ValueError):
            subdivision_ratio({(0, 0, 0), (1, 0, 0)}, 1)

    def test_bad_factor(self):
        with self.assertRaises(ValueError):
            subdivide({(0, 0, 0)}, factor=0)


if __name__ == "__main__":
    unittest.main()
