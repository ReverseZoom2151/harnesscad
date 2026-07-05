import unittest

from bench.engdesign_topopt_metrics import (
    volume_fraction_error, mean_volume_fraction_error, floating_material_error,
    prompting_strategy_table, caption_analysis_score,
)


class VfeTest(unittest.TestCase):
    def test_percent_error(self):
        # target 0.40, predicted 0.649 -> ~62.25% error
        self.assertAlmostEqual(volume_fraction_error(0.649, 0.40), 62.25)

    def test_mean(self):
        res = mean_volume_fraction_error([(0.5, 0.4), (0.3, 0.4)])
        self.assertAlmostEqual(res["vfe_mean"], 25.0)
        self.assertEqual(res["n"], 2)

    def test_zero_true_raises(self):
        with self.assertRaises(ValueError):
            volume_fraction_error(0.5, 0.0)


class FmeTest(unittest.TestCase):
    def test_error_rate(self):
        res = floating_material_error([True, False, True, False],
                                      [True, True, False, False])
        self.assertEqual(res["fme_percent"], 50.0)
        self.assertEqual(res["errors"], 2)
        self.assertEqual(res["random_baseline"], 50.0)

    def test_length_mismatch(self):
        with self.assertRaises(ValueError):
            floating_material_error([True], [True, False])


class StrategyTableTest(unittest.TestCase):
    def test_table(self):
        strategies = {
            "w/o Expertise": {"vf_pairs": [(0.5, 0.4)],
                              "fm_pred": [True, False],
                              "fm_true": [True, True]},
        }
        out = prompting_strategy_table(strategies)
        self.assertIn("vfe_mean", out["w/o Expertise"])
        self.assertEqual(out["w/o Expertise"]["fme_percent"], 50.0)


class CaptionAnalysisTest(unittest.TestCase):
    def test_table8_averages(self):
        # Two rows: caption fields all 1; load/bc/code vary.
        rows = [
            {"nelx": 1, "nely": 1, "F": 1, "VF": 1, "R": 1, "phi": 1,
             "load": 1, "bc": 1, "code": 0},
            {"nelx": 1, "nely": 1, "F": 1, "VF": 1, "R": 1, "phi": 1,
             "load": 0, "bc": 0, "code": 0},
        ]
        res = caption_analysis_score(rows)
        self.assertEqual(res["caption_mean"], 1.0)
        self.assertEqual(res["column_avg"]["load"], 0.5)
        self.assertEqual(res["code_mean"], 0.0)

    def test_empty_raises(self):
        with self.assertRaises(ValueError):
            caption_analysis_score([])


if __name__ == "__main__":
    unittest.main()
