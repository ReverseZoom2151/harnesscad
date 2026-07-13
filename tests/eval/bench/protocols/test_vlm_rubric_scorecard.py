"""Tests for bench.t2cadbench_l4_rubric."""

import unittest

from harnesscad.eval.bench.protocols.vlm_rubric_scorecard import (
    capability_dimensions,
    decoupling_leaders,
    l4_model_scorecard,
)


def _valid(q1, q2, q3, q4, q5, overall):
    return {"valid": True, "q1": q1, "q2": q2, "q3": q3, "q4": q4,
            "q5": q5, "overall": overall}


class L4ScorecardTests(unittest.TestCase):
    def test_valid_only_averaging(self):
        samples = [
            _valid(5, 4, 4, 4, 4, 8),
            _valid(3, 2, 2, 2, 2, 6),
            {"valid": False},
        ]
        r = l4_model_scorecard(samples)
        self.assertEqual(r["n_total"], 3)
        self.assertEqual(r["n_valid"], 2)
        self.assertAlmostEqual(r["ir"], 100.0 / 3)
        self.assertEqual(r["q1"], 4.0)      # (5+3)/2, invalid excluded
        self.assertEqual(r["overall"], 7.0)
        self.assertAlmostEqual(r["feature_mean"], (4 + 3 + 3 + 3 + 3) / 5)

    def test_all_invalid(self):
        r = l4_model_scorecard([{"valid": False}, {"valid": False}])
        self.assertEqual(r["ir"], 100.0)
        self.assertIsNone(r["q1"])
        self.assertIsNone(r["feature_mean"])

    def test_out_of_range_raises(self):
        with self.assertRaises(ValueError):
            l4_model_scorecard([_valid(11, 0, 0, 0, 0, 0)])

    def test_empty_raises(self):
        with self.assertRaises(ValueError):
            l4_model_scorecard([])


class CapabilityDimensionsTests(unittest.TestCase):
    def test_dimensions(self):
        sc = l4_model_scorecard([_valid(5, 4, 4, 4, 4, 8), {"valid": False}])
        cd = capability_dimensions(sc)
        self.assertEqual(cd["executability"], 50.0)   # IR 50 -> 100-50
        self.assertEqual(cd["geometric_similarity"], 8.0)
        self.assertAlmostEqual(cd["feature_design"], (5 + 4 + 4 + 4 + 4) / 5)


class DecouplingTests(unittest.TestCase):
    def test_three_way_decoupling(self):
        # gemini: high executability, low features; minimax: low exec, high
        # features; deepseek: high geometric similarity.
        gemini = l4_model_scorecard(
            [_valid(1, 1, 1, 1, 1, 5)] * 4 + [{"valid": False}])  # low IR
        minimax = l4_model_scorecard(
            [_valid(9, 9, 9, 9, 9, 5)] + [{"valid": False}] * 4)  # high IR
        deepseek = l4_model_scorecard(
            [_valid(4, 4, 4, 4, 4, 9)] + [{"valid": False}])
        res = decoupling_leaders(
            {"gemini": gemini, "minimax": minimax, "deepseek": deepseek})
        self.assertEqual(res["leaders"]["executability"], "gemini")
        self.assertEqual(res["leaders"]["feature_design"], "minimax")
        self.assertEqual(res["leaders"]["geometric_similarity"], "deepseek")
        self.assertTrue(res["decoupled"])

    def test_single_dominant_not_decoupled(self):
        best = l4_model_scorecard([_valid(9, 9, 9, 9, 9, 9)])
        worst = l4_model_scorecard([_valid(1, 1, 1, 1, 1, 1), {"valid": False}])
        res = decoupling_leaders({"best": best, "worst": worst})
        self.assertFalse(res["decoupled"])


if __name__ == "__main__":
    unittest.main()
