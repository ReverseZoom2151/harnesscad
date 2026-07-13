"""Tests for bench.neurcad_metrics (developability & reconstruction metrics)."""
import math
import unittest

from harnesscad.eval.bench.geometry.curvature_developability import (
    developability_ratio, mean_abs_gaussian_curvature,
    max_abs_gaussian_curvature, gaussian_curvature_mae,
    gaussian_curvature_rmse, gauss_bonnet_integral, gauss_bonnet_defect,
)


def sphere_grad_hess(p, r):
    n = tuple(c / r for c in p)
    H = tuple(tuple(((1.0 if i == j else 0.0) - n[i] * n[j]) / r
                    for j in range(3)) for i in range(3))
    return n, H


CYL = ((1.0, 0.0, 0.0),
       ((0.0, 0.0, 0.0), (0.0, 1.0 / 3.0, 0.0), (0.0, 0.0, 0.0)))
PLANE = ((0.0, 0.0, 1.0), ((0.0, 0.0, 0.0),) * 3)


def sphere_points(r, n):
    # n points spread over the sphere (deterministic golden-spiral-free grid).
    pts = []
    for i in range(n):
        phi = math.acos(1.0 - 2.0 * (i + 0.5) / n)
        theta = 2.0 * math.pi * i * 0.61803398875
        x = r * math.sin(phi) * math.cos(theta)
        y = r * math.sin(phi) * math.sin(theta)
        z = r * math.cos(phi)
        pts.append(sphere_grad_hess((x, y, z), r))
    return pts


class RatioTests(unittest.TestCase):
    def test_all_developable(self):
        self.assertEqual(developability_ratio([CYL, PLANE]), 1.0)

    def test_none_developable(self):
        s = sphere_points(2.0, 4)
        self.assertEqual(developability_ratio(s), 0.0)

    def test_half_developable(self):
        s = [CYL, PLANE] + sphere_points(1.5, 2)
        self.assertAlmostEqual(developability_ratio(s), 0.5, places=12)

    def test_empty_raises(self):
        with self.assertRaises(ValueError):
            developability_ratio([])


class AggregateCurvatureTests(unittest.TestCase):
    def test_mean_abs_zero_on_developable(self):
        self.assertAlmostEqual(mean_abs_gaussian_curvature([CYL, PLANE]), 0.0,
                               places=12)

    def test_mean_abs_sphere(self):
        r = 2.0
        s = sphere_points(r, 6)
        self.assertAlmostEqual(mean_abs_gaussian_curvature(s), 1.0 / (r * r),
                               places=9)

    def test_max_abs(self):
        r = 2.0
        s = [CYL] + sphere_points(r, 3)
        self.assertAlmostEqual(max_abs_gaussian_curvature(s), 1.0 / (r * r),
                               places=9)


class ErrorMetricTests(unittest.TestCase):
    def test_mae_zero_when_matching(self):
        r = 3.0
        s = sphere_points(r, 5)
        ref = [1.0 / (r * r)] * len(s)
        self.assertAlmostEqual(gaussian_curvature_mae(s, ref), 0.0, places=10)
        self.assertAlmostEqual(gaussian_curvature_rmse(s, ref), 0.0, places=10)

    def test_mae_developable_ref_zero(self):
        self.assertAlmostEqual(
            gaussian_curvature_mae([CYL, PLANE], [0.0, 0.0]), 0.0, places=12)

    def test_rmse_nonzero(self):
        r = 2.0
        s = sphere_points(r, 4)
        ref = [0.0] * len(s)  # pretend reconstruction thought it developable
        self.assertAlmostEqual(gaussian_curvature_rmse(s, ref), 1.0 / (r * r),
                               places=9)

    def test_length_mismatch_raises(self):
        with self.assertRaises(ValueError):
            gaussian_curvature_mae([CYL], [0.0, 0.0])
        with self.assertRaises(ValueError):
            gaussian_curvature_rmse([CYL], [])


class GaussBonnetTests(unittest.TestCase):
    def test_sphere_integral_is_four_pi(self):
        r = 2.5
        n = 200
        s = sphere_points(r, n)
        area = 4.0 * math.pi * r * r / n  # equal-area weights
        areas = [area] * n
        self.assertAlmostEqual(gauss_bonnet_integral(s, areas), 4.0 * math.pi,
                               places=9)

    def test_sphere_defect_zero(self):
        r = 1.0
        n = 100
        s = sphere_points(r, n)
        areas = [4.0 * math.pi * r * r / n] * n
        self.assertAlmostEqual(gauss_bonnet_defect(s, areas, 2), 0.0, places=9)

    def test_defect_positive_when_field_wrong(self):
        # A developable field integrates to 0, far from 4 pi (chi=2).
        s = [CYL, PLANE]
        self.assertAlmostEqual(gauss_bonnet_defect(s, [1.0, 1.0], 2),
                               4.0 * math.pi, places=9)

    def test_length_mismatch_raises(self):
        with self.assertRaises(ValueError):
            gauss_bonnet_integral([CYL], [1.0, 2.0])

    def test_empty_raises(self):
        with self.assertRaises(ValueError):
            gauss_bonnet_integral([], [])


if __name__ == "__main__":
    unittest.main()
