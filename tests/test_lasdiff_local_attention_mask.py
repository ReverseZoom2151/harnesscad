import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from geometry.lasdiff_local_attention_mask import (
    attention_mask,
    default_d_delta,
    local_neighborhood,
    mask_matrix,
    neighborhood_stability,
    patch_centers,
    patch_grid_dim,
    pinhole_project,
)


class TestLocalAttentionMask(unittest.TestCase):
    def test_grid_dim(self):
        self.assertEqual(patch_grid_dim(224, 14), 16)

    def test_grid_dim_bad(self):
        with self.assertRaises(ValueError):
            patch_grid_dim(224, 15)
        with self.assertRaises(ValueError):
            patch_grid_dim(0, 14)

    def test_patch_centers(self):
        centers = patch_centers(224, 14)
        self.assertEqual(len(centers), 16 * 16)
        self.assertEqual(centers[(0, 0)], (7.0, 7.0))
        self.assertEqual(centers[(0, 1)], (21.0, 7.0))
        self.assertEqual(centers[(1, 0)], (7.0, 21.0))

    def test_pinhole_project(self):
        # point on optical axis maps to principal point
        px, py = pinhole_project((0.0, 0.0, 0.0), focal=100.0, cx=112.0, cy=112.0)
        self.assertAlmostEqual(px, 112.0)
        self.assertAlmostEqual(py, 112.0)

    def test_pinhole_behind_camera(self):
        with self.assertRaises(ValueError):
            pinhole_project((0.0, 0.0, 3.0), focal=100.0, cx=0.0, cy=0.0, cam_z=3.0)

    def test_default_d_delta(self):
        self.assertEqual(default_d_delta(14), 56.0)

    def test_local_neighborhood_radius(self):
        centers = patch_centers(224, 14)
        # pixel at patch (0,0) centre; small radius -> only nearby patches
        nbr = local_neighborhood((7.0, 7.0), centers, d_delta=15.0)
        self.assertIn((0, 0), nbr)
        self.assertIn((0, 1), nbr)   # centre (21,7): dist 14 < 15
        self.assertIn((1, 0), nbr)
        self.assertNotIn((5, 5), nbr)

    def test_local_vs_view_agnostic(self):
        centers = patch_centers(28, 14)  # 2x2 patches
        vox = {"v0": (7.0, 7.0)}
        local = attention_mask(vox, centers, d_delta=10.0, mode="local")
        agn = attention_mask(vox, centers, d_delta=10.0, mode="view_agnostic")
        self.assertEqual(agn["v0"], set(centers.keys()))
        self.assertTrue(local["v0"].issubset(agn["v0"]))
        self.assertLess(len(local["v0"]), len(agn["v0"]))

    def test_attention_mask_bad_mode(self):
        with self.assertRaises(ValueError):
            attention_mask({}, {}, 1.0, mode="global")

    def test_mask_matrix(self):
        centers = patch_centers(28, 14)  # patches (0,0),(0,1),(1,0),(1,1)
        patch_order = sorted(centers.keys())
        vox = {"v0": (7.0, 7.0)}
        mask = attention_mask(vox, centers, d_delta=10.0, mode="local")
        mat = mask_matrix(mask, ["v0"], patch_order)
        self.assertEqual(len(mat), 1)
        self.assertEqual(len(mat[0]), 4)
        # (0,0) is within 10; (1,1) centre (21,21) dist ~19.8 is not
        self.assertTrue(mat[0][patch_order.index((0, 0))])
        self.assertFalse(mat[0][patch_order.index((1, 1))])

    def test_stability_under_small_perturbation(self):
        centers = patch_centers(224, 14)
        d = default_d_delta(14)
        base = local_neighborhood((112.0, 112.0), centers, d)
        shifted = local_neighborhood((115.0, 113.0), centers, d)
        # small pixel shift keeps most of the neighbourhood
        self.assertGreater(neighborhood_stability(base, shifted), 0.7)

    def test_stability_empty(self):
        self.assertEqual(neighborhood_stability(set(), set()), 1.0)


if __name__ == "__main__":
    unittest.main()
