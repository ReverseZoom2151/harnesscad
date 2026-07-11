"""Tests for numeric.makeashape_compression_metric."""
import math
import random
import unittest

from numeric.makeashape_wavelet_transform import Grid3D
from numeric.makeashape_compression_metric import (
    mse, rmse, relative_l2_error, psnr,
    occupancy_from_sdf, occupancy_iou,
    compression_ratio, coefficient_reduction_fraction, streaming_reduction_fraction,
    evaluate_top_k, TopKReport,
)


def _rng_grid(dims, seed=0):
    rng = random.Random(seed)
    n = dims[0] * dims[1] * dims[2]
    return Grid3D(dims, [rng.uniform(-3.0, 3.0) for _ in range(n)])


class SignalErrorTests(unittest.TestCase):
    def test_mse_zero_for_identical(self):
        a = [1.0, 2.0, 3.0]
        self.assertEqual(mse(a, a), 0.0)
        self.assertEqual(rmse(a, a), 0.0)

    def test_mse_and_rmse_values(self):
        a = [0.0, 0.0]
        b = [3.0, 4.0]
        self.assertAlmostEqual(mse(a, b), (9 + 16) / 2)
        self.assertAlmostEqual(rmse(a, b), math.sqrt(12.5))

    def test_relative_l2(self):
        a = [3.0, 4.0]  # norm 5
        b = [3.0, 4.0]
        self.assertAlmostEqual(relative_l2_error(a, b), 0.0)
        self.assertAlmostEqual(relative_l2_error(a, [0.0, 0.0]), 1.0)

    def test_relative_l2_zero_reference(self):
        self.assertEqual(relative_l2_error([0.0, 0.0], [0.0, 0.0]), 0.0)
        self.assertEqual(relative_l2_error([0.0], [1.0]), float("inf"))

    def test_psnr_inf_for_perfect(self):
        a = [1.0, 2.0]
        self.assertEqual(psnr(a, a, value_range=2.0), float("inf"))

    def test_psnr_decreases_with_error(self):
        a = [0.0, 0.0, 0.0, 0.0]
        near = [0.1, 0.0, 0.0, 0.0]
        far = [1.0, 0.0, 0.0, 0.0]
        self.assertGreater(psnr(a, near, 2.0), psnr(a, far, 2.0))

    def test_length_mismatch_raises(self):
        with self.assertRaises(ValueError):
            mse([1.0], [1.0, 2.0])

    def test_works_with_grid3d(self):
        g = _rng_grid((2, 2, 2), seed=1)
        self.assertEqual(mse(g, g), 0.0)


class OccupancyTests(unittest.TestCase):
    def test_occupancy_from_sdf(self):
        occ = occupancy_from_sdf([-1.0, 0.0, 0.5], iso=0.0)
        self.assertEqual(occ, [True, True, False])

    def test_iou_identical(self):
        g = _rng_grid((3, 3, 3), seed=2)
        self.assertEqual(occupancy_iou(g, g), 1.0)

    def test_iou_partial(self):
        a = [-1.0, -1.0, 1.0, 1.0]   # occupied: {0,1}
        b = [-1.0, 1.0, 1.0, -1.0]   # occupied: {0,3}
        # intersection {0}=1, union {0,1,3}=3
        self.assertAlmostEqual(occupancy_iou(a, b), 1.0 / 3.0)

    def test_iou_both_empty_is_one(self):
        a = [1.0, 2.0]
        b = [3.0, 4.0]
        self.assertEqual(occupancy_iou(a, b), 1.0)


class CompactnessTests(unittest.TestCase):
    def test_compression_ratio(self):
        self.assertEqual(compression_ratio(16777216, 1129528), 16777216 / 1129528)

    def test_reduction_fraction(self):
        self.assertAlmostEqual(coefficient_reduction_fraction(100, 25), 0.75)

    def test_streaming_reduction_matches_paper(self):
        # 266ms SDF vs 184ms representation -> ~30.8% by time; here bytes analog
        frac = streaming_reduction_fraction(266.0, 184.0)
        self.assertAlmostEqual(frac, 1.0 - 184.0 / 266.0)

    def test_invalid_inputs(self):
        with self.assertRaises(ValueError):
            compression_ratio(10, 0)
        with self.assertRaises(ValueError):
            coefficient_reduction_fraction(0, 5)
        with self.assertRaises(ValueError):
            streaming_reduction_fraction(0.0, 1.0)


class EvaluateTopKTests(unittest.TestCase):
    def test_report_fields(self):
        grid = _rng_grid((8, 8, 8), seed=3)
        rep = evaluate_top_k(grid, k=4, levels=2, wavelet="haar")
        self.assertIsInstance(rep, TopKReport)
        self.assertEqual(rep.k, 4)
        self.assertGreater(rep.total_detail_coeffs, 0)
        self.assertLessEqual(rep.kept_detail_coeffs, rep.total_detail_coeffs)
        self.assertGreaterEqual(rep.reduction_fraction, 0.0)
        self.assertLessEqual(rep.occupancy_iou, 1.0)

    def test_more_k_reduces_error(self):
        grid = _rng_grid((8, 8, 8), seed=4)
        low = evaluate_top_k(grid, k=2, levels=2)
        high = evaluate_top_k(grid, k=60, levels=2)
        # keeping more coefficients cannot worsen reconstruction error
        self.assertLessEqual(high.mse, low.mse + 1e-9)
        self.assertGreaterEqual(high.kept_detail_coeffs, low.kept_detail_coeffs)

    def test_keeping_all_is_near_lossless(self):
        grid = _rng_grid((8, 8, 8), seed=5)
        # 2 levels: finest detail dims 4^3=64 locations is the max K needed
        rep = evaluate_top_k(grid, k=64, levels=2, wavelet="haar")
        self.assertLess(rep.mse, 1e-12)
        self.assertAlmostEqual(rep.occupancy_iou, 1.0)


if __name__ == "__main__":
    unittest.main()
