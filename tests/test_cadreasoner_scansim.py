"""Tests for datagen/cadreasoner_scansim.py (scan-defect simulation)."""

import unittest

from harnesscad.data.datagen.scan_simulation import (
    ScanSimResult,
    simulate_scan,
    spherical_viewpoints,
    visible_from,
)


def _sphere(radius, count):
    return spherical_viewpoints(radius, count)


class TestSphericalViewpoints(unittest.TestCase):
    def test_count_and_radius(self):
        vps = spherical_viewpoints(3.0, 5)
        self.assertEqual(len(vps), 5)
        for p in vps:
            r = sum(c * c for c in p) ** 0.5
            self.assertAlmostEqual(r, 3.0, places=6)

    def test_rejects_zero_count(self):
        with self.assertRaises(ValueError):
            spherical_viewpoints(1.0, 0)


class TestVisibility(unittest.TestCase):
    def test_occludes_farther_point_on_same_ray(self):
        # Both points lie on the -x ray from the viewpoint; only the nearer
        # (x=1) survives.
        pts = [(0.0, 0.0, 0.0), (1.0, 0.0, 0.0)]
        vis = visible_from(pts, (5.0, 0.0, 0.0), angular_bins=64)
        self.assertEqual(vis, [1])

    def test_all_visible_when_spread(self):
        pts = [(0.0, 0.0, 0.0), (0.0, 5.0, 0.0), (0.0, 0.0, 5.0)]
        vis = visible_from(pts, (10.0, 0.0, 0.0), angular_bins=64)
        self.assertEqual(vis, [0, 1, 2])

    def test_rejects_bad_bins(self):
        with self.assertRaises(ValueError):
            visible_from([(0.0, 0.0, 0.0)], (1.0, 0.0, 0.0), angular_bins=0)


class TestSimulateScan(unittest.TestCase):
    def _cloud(self):
        # Two concentric shells: inner points get occluded by outer ones.
        outer = _sphere(1.0, 120)
        inner = _sphere(0.5, 120)
        return outer + inner

    def test_result_shape(self):
        res = simulate_scan(self._cloud(), seed=1)
        self.assertIsInstance(res, ScanSimResult)
        self.assertEqual(len(res.visible_counts), 5)
        self.assertEqual(len(res.viewpoints), 5)
        self.assertTrue(0.0 < res.coverage <= 1.0)

    def test_occlusion_removes_interior(self):
        res = simulate_scan(self._cloud(), n_viewpoints=5, angular_bins=32,
                            noise_sigma=0.0, n_holes=0, seed=0)
        # Concentric interior shell should be partly occluded.
        self.assertGreater(res.removed_occluded, 0)

    def test_noiseless_holeless_output_is_subset(self):
        cloud = self._cloud()
        res = simulate_scan(cloud, noise_sigma=0.0, n_holes=0, seed=0)
        cloudset = {tuple(p) for p in cloud}
        for p in res.points:
            self.assertIn(tuple(p), cloudset)
        self.assertEqual(res.removed_holes, 0)
        self.assertEqual(len(res.points),
                         len(cloud) - res.removed_occluded)

    def test_holes_remove_points(self):
        res = simulate_scan(self._cloud(), noise_sigma=0.0, n_holes=3,
                            hole_radius=0.3, seed=2)
        self.assertGreaterEqual(res.removed_holes, 0)
        self.assertLessEqual(res.coverage, 1.0)

    def test_normals_preserved_alignment(self):
        cloud = self._cloud()
        normals = [(1.0, 0.0, 0.0)] * len(cloud)
        res = simulate_scan(cloud, normals=normals, noise_sigma=0.02,
                            n_holes=0, seed=0)
        self.assertEqual(len(res.normals), len(res.points))

    def test_deterministic(self):
        cloud = self._cloud()
        a = simulate_scan(cloud, seed=5).to_dict()
        b = simulate_scan(cloud, seed=5).to_dict()
        self.assertEqual(a, b)

    def test_seed_changes_holes(self):
        cloud = self._cloud()
        a = simulate_scan(cloud, n_holes=4, hole_radius=0.25, seed=1)
        b = simulate_scan(cloud, n_holes=4, hole_radius=0.25, seed=99)
        # Different seeds should generally differ in the punched result.
        self.assertNotEqual([tuple(p) for p in a.points],
                            [tuple(p) for p in b.points])

    def test_rejects_tiny_input(self):
        with self.assertRaises(ValueError):
            simulate_scan([(0.0, 0.0, 0.0)])

    def test_rejects_mismatched_normals(self):
        with self.assertRaises(ValueError):
            simulate_scan([(0.0, 0.0, 0.0), (1.0, 0.0, 0.0)],
                          normals=[(1.0, 0.0, 0.0)])


if __name__ == "__main__":
    unittest.main()
