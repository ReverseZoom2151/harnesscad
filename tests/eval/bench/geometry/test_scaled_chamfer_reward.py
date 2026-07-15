"""Tests for the StepForge Scaled Chamfer Distance reward."""

import math
import unittest

from harnesscad.eval.bench.geometry import scaled_chamfer_reward as scr


CUBE = [
    (0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, 1),
    (1, 1, 0), (1, 0, 1), (0, 1, 1), (1, 1, 1),
]


class ChamferTest(unittest.TestCase):
    def test_identical_clouds_zero(self):
        self.assertAlmostEqual(scr.chamfer_distance(CUBE, CUBE), 0.0)

    def test_empty_is_inf(self):
        self.assertEqual(scr.chamfer_distance([], CUBE), float("inf"))

    def test_symmetric_adds_both_terms(self):
        P = [(0, 0, 0)]
        Q = [(0, 0, 0), (2, 0, 0)]
        fwd = scr.chamfer_distance(P, Q, bidirectional=False)
        both = scr.chamfer_distance(P, Q, bidirectional=True)
        self.assertGreater(both, fwd)


class ScaledChamferTest(unittest.TestCase):
    def test_translation_invariant(self):
        shifted = [(x + 100, y + 5, z - 3) for (x, y, z) in CUBE]
        self.assertAlmostEqual(scr.scaled_chamfer_distance(shifted, CUBE), 0.0, places=9)

    def test_scale_invariant(self):
        scaled = [(x * 7, y * 7, z * 7) for (x, y, z) in CUBE]
        # After centring and normalizing by GT RMS radius, a pure scale collapses.
        val = scr.scaled_chamfer_distance(scaled, CUBE)
        self.assertLess(val, 1e-6)

    def test_degenerate_gt_inf(self):
        self.assertEqual(scr.scaled_chamfer_distance(CUBE, [(0, 0, 0)]), float("inf"))


class RGeoTest(unittest.TestCase):
    def test_below_low_is_one(self):
        self.assertEqual(scr.r_geo(0.0), 1.0)

    def test_above_high_is_zero(self):
        self.assertEqual(scr.r_geo(1.0), 0.0)

    def test_midpoint_linear(self):
        # halfway between 0.01 and 0.50
        mid = (0.01 + 0.50) / 2
        self.assertAlmostEqual(scr.r_geo(mid), 0.5)


class RewardPipelineTest(unittest.TestCase):
    def test_identical_full_reward(self):
        res = scr.compute_reward(CUBE, CUBE)
        self.assertEqual(res.fail_stage, "ok")
        self.assertAlmostEqual(res.reward, 1.0)

    def test_pred_empty_zero(self):
        res = scr.compute_reward([], CUBE)
        self.assertEqual(res.fail_stage, "pred_empty")
        self.assertEqual(res.reward, 0.0)

    def test_gt_empty_is_nan(self):
        res = scr.compute_reward(CUBE, [])
        self.assertEqual(res.fail_stage, "gt_empty")
        self.assertTrue(math.isnan(res.reward))

    def test_pred_degenerate(self):
        res = scr.compute_reward([(0, 0, 0), (0, 0, 0)], CUBE)
        self.assertEqual(res.fail_stage, "pred_degenerate")
        self.assertEqual(res.reward, 0.0)

    def test_far_shape_low_reward(self):
        far = [(x * 100, y, z) for (x, y, z) in CUBE]  # stretched, high SCD
        res = scr.compute_reward(far, CUBE)
        self.assertEqual(res.fail_stage, "ok")
        self.assertLessEqual(res.reward, 1.0)


if __name__ == "__main__":
    unittest.main()
