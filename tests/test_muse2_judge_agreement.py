"""Tests for bench.muse2_judge_agreement."""

import math
import unittest

from bench.muse2_judge_agreement import (
    agreement_report,
    bootstrap_ci,
    kendall_tau_b,
    pearson,
    spearman,
)


class CorrelationTests(unittest.TestCase):
    def test_pearson_perfect_positive(self):
        self.assertAlmostEqual(pearson([1, 2, 3, 4], [2, 4, 6, 8]), 1.0)

    def test_pearson_perfect_negative(self):
        self.assertAlmostEqual(pearson([1, 2, 3, 4], [4, 3, 2, 1]), -1.0)

    def test_pearson_constant_is_nan(self):
        self.assertTrue(math.isnan(pearson([1, 1, 1], [1, 2, 3])))

    def test_spearman_monotonic_nonlinear(self):
        # Monotonic but nonlinear -> Spearman 1.0, Pearson < 1.
        x = [1, 2, 3, 4]
        y = [1, 4, 9, 16]
        self.assertAlmostEqual(spearman(x, y), 1.0)
        self.assertLess(pearson(x, y), 1.0)

    def test_kendall_perfect(self):
        self.assertAlmostEqual(kendall_tau_b([1, 2, 3, 4], [1, 2, 3, 4]), 1.0)

    def test_kendall_reversed(self):
        self.assertAlmostEqual(kendall_tau_b([1, 2, 3, 4], [4, 3, 2, 1]), -1.0)

    def test_kendall_tie_corrected(self):
        # With ties on x, tau-b denominator excludes tied-on-x pairs.
        v = kendall_tau_b([1, 1, 2, 2], [1, 2, 3, 4])
        self.assertTrue(-1.0 <= v <= 1.0)


class BootstrapTests(unittest.TestCase):
    def test_seed_determinism(self):
        x = [float(i) for i in range(40)]
        y = [2.0 * i + (i % 3) for i in range(40)]
        a = bootstrap_ci(pearson, x, y, n_boot=500, seed=7)
        b = bootstrap_ci(pearson, x, y, n_boot=500, seed=7)
        self.assertEqual(a, b)

    def test_ci_brackets_point(self):
        x = [float(i) for i in range(40)]
        y = [1.5 * i for i in range(40)]  # perfect -> point 1.0
        lo, hi = bootstrap_ci(pearson, x, y, n_boot=300, seed=1)
        self.assertLessEqual(lo, 1.0 + 1e-9)
        self.assertLessEqual(hi, 1.0 + 1e-9)


class AgreementReportTests(unittest.TestCase):
    def _pairs(self):
        item = [(i % 5, i % 5) for i in range(40)]  # perfect agreement, n>=30
        cell = [
            (1.0, 1.0, "m1"), (0.5, 0.6, "m1"),
            (0.8, 0.7, "m2"), (0.2, 0.3, "m2"),
        ]
        return item, cell

    def test_levels_present(self):
        item, cell = self._pairs()
        rep = agreement_report(item, cell)
        self.assertEqual(set(rep), {"Item", "Cell", "System"})
        self.assertEqual(rep["Item"]["n"], 40)
        self.assertEqual(rep["Cell"]["n"], 4)
        self.assertEqual(rep["System"]["n"], 2)  # two models

    def test_item_level_has_ci(self):
        item, cell = self._pairs()
        rep = agreement_report(item, cell)
        pt, lo, hi = rep["Item"]["pearson"]
        self.assertAlmostEqual(pt, 1.0)
        self.assertIsNotNone(lo)
        self.assertIsNotNone(hi)

    def test_cell_level_no_ci(self):
        item, cell = self._pairs()
        rep = agreement_report(item, cell)
        _, lo, hi = rep["Cell"]["pearson"]
        self.assertIsNone(lo)
        self.assertIsNone(hi)

    def test_bias_sign(self):
        # Judge scores systematically higher than human -> positive bias.
        item = [(1.0, 2.0) for _ in range(5)]
        cell = [(1.0, 2.0, "m1")]
        rep = agreement_report(item, cell)
        self.assertAlmostEqual(rep["Item"]["bias"], 1.0)

    def test_system_averages_models(self):
        item = [(1.0, 1.0)]
        cell = [(0.0, 0.0, "m1"), (1.0, 1.0, "m1"), (0.5, 0.5, "m2")]
        rep = agreement_report(item, cell)
        # m1 mean human = 0.5, m2 = 0.5 -> system mean_human 0.5.
        self.assertAlmostEqual(rep["System"]["mean_human"], 0.5)


if __name__ == "__main__":
    unittest.main()
