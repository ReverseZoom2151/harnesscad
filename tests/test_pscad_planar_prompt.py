"""Tests for PS-CAD planar prompt detection and encoding."""

import unittest

from harnesscad.domain.geometry.views.pscad_planar_prompt import (
    DetectedPlane,
    encode_planar_prompt,
    extract_prompts,
    fit_plane,
    point_plane_distance,
    project_to_plane,
    ransac_planes,
)


def _grid(z, n=6, step=1.0):
    return [(i * step, j * step, z) for i in range(n) for j in range(n)]


class FitPlaneTest(unittest.TestCase):
    def test_z_plane_normal(self):
        plane = fit_plane((0, 0, 3), (1, 0, 3), (0, 1, 3))
        normal, offset = plane
        self.assertEqual(normal, (0.0, 0.0, 1.0))
        self.assertAlmostEqual(offset, 3.0)

    def test_canonical_sign_independent_of_order(self):
        a = fit_plane((0, 0, 3), (1, 0, 3), (0, 1, 3))
        b = fit_plane((0, 1, 3), (1, 0, 3), (0, 0, 3))
        self.assertEqual(a[0], b[0])
        self.assertAlmostEqual(a[1], b[1])

    def test_collinear_returns_none(self):
        self.assertIsNone(fit_plane((0, 0, 0), (1, 1, 1), (2, 2, 2)))

    def test_point_plane_distance(self):
        plane = ((0.0, 0.0, 1.0), 3.0)
        self.assertAlmostEqual(point_plane_distance((5, 5, 3), plane), 0.0)
        self.assertAlmostEqual(point_plane_distance((5, 5, 5), plane), 2.0)


class RansacPlanesTest(unittest.TestCase):
    def test_detects_two_parallel_planes(self):
        pts = _grid(0.0) + _grid(2.0)
        planes = ransac_planes(pts, threshold=0.05, min_inliers=9,
                               iterations=100, seed=1)
        self.assertGreaterEqual(len(planes), 2)
        # both planes have z-normal
        for plane in planes[:2]:
            self.assertEqual(plane.normal, (0.0, 0.0, 1.0))
        offsets = sorted(round(p.offset, 3) for p in planes[:2])
        self.assertEqual(offsets, [0.0, 2.0])

    def test_deterministic_across_runs(self):
        pts = _grid(0.0) + _grid(2.0)
        a = ransac_planes(pts, threshold=0.05, min_inliers=9, seed=7)
        b = ransac_planes(pts, threshold=0.05, min_inliers=9, seed=7)
        self.assertEqual([(p.normal, p.offset, p.inlier_indices) for p in a],
                         [(p.normal, p.offset, p.inlier_indices) for p in b])

    def test_inliers_removed_greedily(self):
        pts = _grid(0.0) + _grid(2.0)
        planes = ransac_planes(pts, threshold=0.05, min_inliers=9, seed=3)
        seen = set()
        for plane in planes:
            self.assertTrue(seen.isdisjoint(plane.inlier_indices))
            seen.update(plane.inlier_indices)

    def test_bad_threshold_rejected(self):
        with self.assertRaises(ValueError):
            ransac_planes(_grid(0.0), threshold=0.0, min_inliers=9)

    def test_min_inliers_floor(self):
        with self.assertRaises(ValueError):
            ransac_planes(_grid(0.0), threshold=0.1, min_inliers=2)


class ProjectAndPromptTest(unittest.TestCase):
    def test_projection_is_planar(self):
        plane = ((0.0, 0.0, 1.0), 0.0)
        origin = (0.0, 0.0, 0.0)
        xy = project_to_plane((3.0, 4.0, 0.0), plane, origin)
        # distance in-plane is preserved for a z-plane
        self.assertAlmostEqual((xy[0] ** 2 + xy[1] ** 2) ** 0.5, 5.0)

    def test_encode_prompt_fixed_size_and_boundary(self):
        pts = _grid(0.0)  # 36 points on z=0
        plane = DetectedPlane((0.0, 0.0, 1.0), 0.0, tuple(range(len(pts))))
        prompt = encode_planar_prompt(pts, plane, sample_count=16, seed=0)
        self.assertEqual(prompt.surface_type, "plane")
        self.assertEqual(prompt.sample_size, 16)
        # convex hull of a 6x6 square grid has 4 corners
        self.assertEqual(len(prompt.boundary), 4)

    def test_encode_upsamples_when_few_inliers(self):
        pts = [(0, 0, 0), (1, 0, 0), (0, 1, 0)]
        plane = DetectedPlane((0.0, 0.0, 1.0), 0.0, (0, 1, 2))
        prompt = encode_planar_prompt(pts, plane, sample_count=10, seed=0)
        self.assertEqual(prompt.sample_size, 10)

    def test_encode_deterministic(self):
        pts = _grid(0.0)
        plane = DetectedPlane((0.0, 0.0, 1.0), 0.0, tuple(range(len(pts))))
        a = encode_planar_prompt(pts, plane, sample_count=16, seed=5)
        b = encode_planar_prompt(pts, plane, sample_count=16, seed=5)
        self.assertEqual(a.sample, b.sample)

    def test_encode_empty_inliers_rejected(self):
        with self.assertRaises(ValueError):
            encode_planar_prompt([(0, 0, 0)], DetectedPlane((0, 0, 1), 0.0, ()))


class ExtractPromptsTest(unittest.TestCase):
    def test_end_to_end_two_prompts(self):
        pts = _grid(0.0) + _grid(2.0)
        prompts = extract_prompts(pts, threshold=0.05, min_inliers=9,
                                  seed=1, sample_count=8)
        self.assertGreaterEqual(len(prompts), 2)
        for prompt in prompts:
            self.assertEqual(prompt.surface_type, "plane")
            self.assertEqual(prompt.sample_size, 8)


if __name__ == "__main__":
    unittest.main()
