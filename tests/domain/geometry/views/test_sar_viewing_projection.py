"""Tests for domain.geometry.views.sar_viewing_projection."""

import math
import unittest

from harnesscad.domain.geometry.views.sar_viewing_projection import (
    GECM,
    build_gecm,
    camera_basis,
    dbscan,
    pose_skeleton,
    principal_axis_2d,
    project_points,
    scattering_centers,
    viewing_direction,
)


class ViewingDirectionTest(unittest.TestCase):
    def test_unit_length(self):
        v = viewing_direction(37.0, 22.0)
        n = math.sqrt(sum(c * c for c in v))
        self.assertAlmostEqual(n, 1.0)

    def test_depression_points_down(self):
        v = viewing_direction(0.0, 30.0)
        self.assertLess(v[2], 0.0)  # looking downward

    def test_basis_orthonormal(self):
        right, up, fwd = camera_basis(45.0, 20.0)
        for a in (right, up, fwd):
            self.assertAlmostEqual(math.sqrt(sum(c * c for c in a)), 1.0)
        self.assertAlmostEqual(sum(a * b for a, b in zip(right, up)), 0.0, places=6)
        self.assertAlmostEqual(sum(a * b for a, b in zip(right, fwd)), 0.0, places=6)


class ProjectionTest(unittest.TestCase):
    def test_projection_count(self):
        pts = [(0, 0, 0), (1, 0, 0), (0, 1, 0)]
        proj = project_points(pts, 10.0, 15.0)
        self.assertEqual(len(proj), 3)

    def test_determinism(self):
        pts = [(1.0, 2.0, 3.0), (-1.0, 0.5, 2.0)]
        self.assertEqual(
            project_points(pts, 33.0, 12.0), project_points(pts, 33.0, 12.0)
        )


class PrincipalAxisTest(unittest.TestCase):
    def test_horizontal_line_axis_zero(self):
        pts = [(-2.0, 0.0), (-1.0, 0.0), (1.0, 0.0), (2.0, 0.0)]
        (cx, cy), angle, (lam1, lam2) = principal_axis_2d(pts)
        self.assertAlmostEqual(cx, 0.0)
        self.assertAlmostEqual(cy, 0.0)
        self.assertAlmostEqual(math.sin(angle), 0.0, places=6)
        self.assertGreater(lam1, lam2)

    def test_vertical_line_axis_ninety(self):
        pts = [(0.0, -2.0), (0.0, -1.0), (0.0, 1.0), (0.0, 2.0)]
        _, angle, _ = principal_axis_2d(pts)
        self.assertAlmostEqual(abs(math.cos(angle)), 0.0, places=6)

    def test_pose_skeleton_length(self):
        pts = [(0.0, 0.0), (3.0, 0.0), (6.0, 0.0)]
        sk = pose_skeleton(pts)
        self.assertAlmostEqual(sk["length"], 6.0)
        self.assertEqual(len(sk["keypoints"]), 3)


class DBSCANTest(unittest.TestCase):
    def test_two_clusters(self):
        pts = [(0, 0), (0.1, 0), (0, 0.1), (10, 10), (10.1, 10), (10, 10.1)]
        labels = dbscan(pts, eps=0.5, min_samples=2)
        self.assertEqual(labels[0], labels[1])
        self.assertEqual(labels[3], labels[4])
        self.assertNotEqual(labels[0], labels[3])

    def test_noise_labelled_minus_one(self):
        pts = [(0, 0), (0.1, 0), (0.05, 0.05), (50, 50)]
        labels = dbscan(pts, eps=0.5, min_samples=2)
        self.assertEqual(labels[3], -1)

    def test_determinism(self):
        pts = [(0, 0), (0.2, 0), (5, 5), (5.2, 5)]
        self.assertEqual(dbscan(pts, 0.5, 2), dbscan(pts, 0.5, 2))

    def test_invalid_eps(self):
        with self.assertRaises(ValueError):
            dbscan([(0, 0)], eps=0.0, min_samples=1)


class ScatteringCentersTest(unittest.TestCase):
    def test_threshold_and_centroid(self):
        pts = [(0, 0), (0.1, 0), (9, 9)]
        inten = [1.0, 1.0, 0.1]
        centres = scattering_centers(pts, inten, 0.5, eps=0.5, min_samples=2)
        self.assertEqual(len(centres), 1)
        self.assertAlmostEqual(centres[0][0], 0.05)

    def test_length_mismatch(self):
        with self.assertRaises(ValueError):
            scattering_centers([(0, 0)], [1.0, 2.0], 0.5, 0.5, 1)


class BuildGECMTest(unittest.TestCase):
    def test_build(self):
        cube = [
            (0, 0, 0), (1, 0, 0), (0, 1, 0), (1, 1, 0),
            (0, 0, 1), (1, 0, 1), (0, 1, 1), (1, 1, 1),
        ]
        gecm = build_gecm(cube, 30.0, 25.0, polarization="VV",
                          intensity_threshold=0.5, eps=2.0, min_samples=1)
        self.assertIsInstance(gecm, GECM)
        self.assertEqual(gecm.polarization, "VV")
        self.assertIn("keypoints", gecm.pose)
        self.assertGreaterEqual(len(gecm.scatterers), 1)

    def test_determinism(self):
        pts = [(0, 0, 0), (2, 1, 0), (1, 3, 1)]
        a = build_gecm(pts, 15.0, 10.0)
        b = build_gecm(pts, 15.0, 10.0)
        self.assertEqual(a.pose["length"], b.pose["length"])
        self.assertEqual(a.scatterers, b.scatterers)


if __name__ == "__main__":
    unittest.main()
