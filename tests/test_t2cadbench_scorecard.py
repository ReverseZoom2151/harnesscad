"""Tests for bench.t2cadbench_scorecard."""

import unittest

from harnesscad.eval.bench.t2cadbench_scorecard import (
    cell_scorecard,
    degradation_ratio,
    prompt_style_comparison,
    rank_leaderboard,
    resolve_sample,
    survivorship_flag,
    weighted_average,
)


class ResolveSampleTests(unittest.TestCase):
    def test_best_of_attempts_picks_lowest_cd(self):
        s = {"attempts": [
            {"valid": False, "cd": None, "iou": None},
            {"valid": True, "cd": 90.0, "iou": 0.2},
            {"valid": True, "cd": 60.0, "iou": 0.4},
        ]}
        r = resolve_sample(s)
        self.assertTrue(r["valid"])
        self.assertEqual(r["cd"], 60.0)
        self.assertEqual(r["iou"], 0.4)

    def test_all_fail_invalid(self):
        s = {"attempts": [{"valid": False}, {"valid": False}]}
        r = resolve_sample(s)
        self.assertFalse(r["valid"])
        self.assertIsNone(r["cd"])

    def test_single_attempt_shorthand(self):
        r = resolve_sample({"valid": True, "cd": 44.31, "iou": 0.59})
        self.assertEqual(r["cd"], 44.31)


class CellScorecardTests(unittest.TestCase):
    def test_cd_iou_over_valid_only(self):
        samples = [
            {"valid": True, "cd": 40.0, "iou": 0.6},
            {"valid": True, "cd": 60.0, "iou": 0.4},
            {"valid": False},
            {"valid": False},
        ]
        r = cell_scorecard(samples)
        self.assertEqual(r["n_total"], 4)
        self.assertEqual(r["n_valid"], 2)
        self.assertEqual(r["ir"], 50.0)
        self.assertEqual(r["cd"], 50.0)          # mean of valid only
        self.assertAlmostEqual(r["iou"], 0.5)

    def test_all_invalid_cd_none(self):
        r = cell_scorecard([{"valid": False}, {"valid": False}])
        self.assertEqual(r["ir"], 100.0)
        self.assertIsNone(r["cd"])
        self.assertIsNone(r["iou"])

    def test_empty_raises(self):
        with self.assertRaises(ValueError):
            cell_scorecard([])


class WeightedAverageTests(unittest.TestCase):
    def test_weighting(self):
        c1 = {"n_total": 100, "n_valid": 80, "ir": 20.0, "cd": 50.0, "iou": 0.5}
        c2 = {"n_total": 100, "n_valid": 40, "ir": 60.0, "cd": 100.0, "iou": 0.2}
        r = weighted_average([c1, c2])
        self.assertEqual(r["n_total"], 200)
        self.assertEqual(r["ir"], 40.0)  # IR weighted by n_total
        # CD weighted by n_valid: (50*80 + 100*40)/120
        self.assertAlmostEqual(r["cd"], (50 * 80 + 100 * 40) / 120)


class DegradationTests(unittest.TestCase):
    def test_cd_ratio_l1_to_l3(self):
        lo = {"cd": 44.31, "ir": 11.1, "iou": 0.59}
        hi = {"cd": 93.46, "ir": 68.0, "iou": 0.23}
        r = degradation_ratio(lo, hi)
        self.assertAlmostEqual(r["cd_ratio"], 93.46 / 44.31)
        self.assertAlmostEqual(r["ir_delta"], 68.0 - 11.1)
        self.assertAlmostEqual(r["iou_delta"], 0.23 - 0.59)


class SurvivorshipTests(unittest.TestCase):
    def test_flag_when_ir_up_cd_down(self):
        # Paper example: DeepSeek L1 Geo IR 13.3 -> 67.3, CD 53.15 -> 44.56.
        a = {"ir": 13.3, "cd": 53.15}
        b = {"ir": 67.3, "cd": 44.56}
        self.assertTrue(survivorship_flag(a, b))

    def test_no_flag_when_both_worse(self):
        a = {"ir": 20.0, "cd": 50.0}
        b = {"ir": 40.0, "cd": 90.0}
        self.assertFalse(survivorship_flag(a, b))


class LeaderboardTests(unittest.TestCase):
    def test_rank_by_cd_ascending(self):
        entries = [
            {"model": "gpt", "cd": 44.31},
            {"model": "claude", "cd": 52.62},
            {"model": "qwen", "cd": None},
        ]
        board = rank_leaderboard(entries, metric="cd")
        self.assertEqual(board[0]["model"], "gpt")
        self.assertEqual(board[0]["rank"], 1)
        self.assertEqual(board[-1]["model"], "qwen")  # None last

    def test_rank_by_iou_descending(self):
        entries = [{"model": "a", "iou": 0.2}, {"model": "b", "iou": 0.59}]
        board = rank_leaderboard(entries, metric="iou")
        self.assertEqual(board[0]["model"], "b")


class PromptStyleTests(unittest.TestCase):
    def test_geo_wins_on_lower_cd(self):
        geo = {"cd": 44.31, "ir": 11.1, "iou": 0.59}
        seq = {"cd": 48.73, "ir": 19.5, "iou": 0.62}
        r = prompt_style_comparison(geo, seq)
        self.assertEqual(r["cd_better"], "geo")
        self.assertEqual(r["ir_better"], "geo")
        self.assertEqual(r["iou_better"], "seq")  # seq higher IoU here


if __name__ == "__main__":
    unittest.main()
