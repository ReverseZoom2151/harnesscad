import math
import unittest

from harnesscad.eval.bench.vision.multiview_consistency_anova import (
    one_way_anova,
    f_critical,
    is_consistent,
    consistency_score,
)


class AnovaTest(unittest.TestCase):
    def test_identical_groups_zero_between(self):
        groups = [[1.0, 2.0, 3.0], [1.0, 2.0, 3.0], [1.0, 2.0, 3.0]]
        r = one_way_anova(groups)
        self.assertAlmostEqual(r.ss_between, 0.0)
        self.assertAlmostEqual(r.f_statistic, 0.0)
        self.assertEqual(r.df_between, 2)
        self.assertEqual(r.df_within, 6)

    def test_separated_groups_large_f(self):
        groups = [[0.0, 0.1, -0.1], [10.0, 10.1, 9.9]]
        r = one_way_anova(groups)
        self.assertGreater(r.f_statistic, 100.0)

    def test_needs_two_groups(self):
        with self.assertRaises(ValueError):
            one_way_anova([[1.0, 2.0]])

    def test_empty_group_raises(self):
        with self.assertRaises(ValueError):
            one_way_anova([[1.0], []])

    def test_not_enough_samples(self):
        with self.assertRaises(ValueError):
            one_way_anova([[1.0], [2.0]])  # n_total == k


class CriticalTest(unittest.TestCase):
    def test_exact_lookup(self):
        self.assertEqual(f_critical(0.01, 2, 10), 7.56)

    def test_nearest_df2_fallback(self):
        # df2=25 not tabulated for (0.01, 2); nearest is 30
        self.assertEqual(f_critical(0.01, 2, 25), 5.39)

    def test_missing_raises(self):
        with self.assertRaises(KeyError):
            f_critical(0.01, 9, 10)


class ConsistencyTest(unittest.TestCase):
    def test_consistent_when_flat(self):
        # near-identical metric values across viewpoints -> F below critical
        groups = [[0.80, 0.81], [0.80, 0.79], [0.81, 0.80]]
        r = one_way_anova(groups)
        self.assertTrue(is_consistent(r, alpha=0.01))

    def test_inconsistent_when_separated(self):
        groups = [[0.10, 0.11], [0.90, 0.89], [0.50, 0.51]]
        r = one_way_anova(groups)
        self.assertFalse(is_consistent(r, alpha=0.01))

    def test_consistency_score_flat_is_high(self):
        self.assertGreater(consistency_score([0.8, 0.8, 0.81, 0.79]), 0.95)

    def test_consistency_score_spread_is_low(self):
        self.assertLess(consistency_score([0.1, 0.9, 0.5, 0.3]), 0.6)

    def test_score_needs_two(self):
        with self.assertRaises(ValueError):
            consistency_score([0.5])


if __name__ == "__main__":
    unittest.main()
