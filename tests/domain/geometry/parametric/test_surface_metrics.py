"""Tests for geometry.dreamcad_metrics."""

import unittest

from harnesscad.domain.geometry.parametric.surface_metrics import (
    chamfer_distance,
    hausdorff_distance,
    one_sided_residual,
    sample_rational_bezier,
    surface_consistency,
)
from harnesscad.domain.geometry.parametric.bezier import unit_weight_grid


def _flat_grid(z=0.0):
    return [[(i / 3.0, j / 3.0, z) for j in range(4)] for i in range(4)]


class TestChamfer(unittest.TestCase):
    def test_identical_clouds_zero(self):
        cloud = [(0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0)]
        self.assertAlmostEqual(chamfer_distance(cloud, cloud), 0.0)

    def test_shifted_cloud(self):
        a = [(0.0, 0.0, 0.0)]
        b = [(3.0, 4.0, 0.0)]
        # each direction contributes distance 5
        self.assertAlmostEqual(chamfer_distance(a, b), 10.0)

    def test_empty_rejected(self):
        with self.assertRaises(ValueError):
            chamfer_distance([], [(0.0, 0.0, 0.0)])


class TestResidualAndHausdorff(unittest.TestCase):
    def test_one_sided_residual(self):
        source = [(0.0, 0.0, 0.0), (2.0, 0.0, 0.0)]
        target = [(0.0, 0.0, 0.0)]
        # distances 0 and 2 -> mean 1.0
        self.assertAlmostEqual(one_sided_residual(source, target), 1.0)

    def test_hausdorff_worst_case(self):
        a = [(0.0, 0.0, 0.0), (10.0, 0.0, 0.0)]
        b = [(0.0, 0.0, 0.0)]
        self.assertAlmostEqual(hausdorff_distance(a, b), 10.0)


class TestSurfaceConsistency(unittest.TestCase):
    def test_same_patch_zero(self):
        grid = _flat_grid()
        w = unit_weight_grid()
        d = surface_consistency((grid, w), (grid, w), resolution=5)
        self.assertAlmostEqual(d, 0.0)

    def test_offset_patches_positive(self):
        w = unit_weight_grid()
        near = surface_consistency(
            (_flat_grid(0.0), w), (_flat_grid(0.01), w), resolution=5)
        far = surface_consistency(
            (_flat_grid(0.0), w), (_flat_grid(1.0), w), resolution=5)
        self.assertGreater(far, near)
        self.assertGreater(near, 0.0)

    def test_sample_count(self):
        pts = sample_rational_bezier(_flat_grid(), unit_weight_grid(), 6)
        self.assertEqual(len(pts), 36)


if __name__ == "__main__":
    unittest.main()
