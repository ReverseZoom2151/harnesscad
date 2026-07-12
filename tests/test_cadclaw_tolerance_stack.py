"""Tests for verifiers.cadclaw_tolerance_stack.

Deterministic, stdlib-only. Hand-checked worst-case / RSS arithmetic and
Monte-Carlo reproducibility for a fixed seed.
"""
import math
import unittest

from verifiers.cadclaw_tolerance_stack import (
    Dimension, ToleranceChain, StackResult,
)


class DimensionTest(unittest.TestCase):

    def test_symmetric_defaults(self):
        d = Dimension("x", nominal=10.0, plus=0.2, minus=0.2)
        self.assertEqual(d.bilateral, 0.2)
        self.assertEqual(d.mean, 10.0)

    def test_asymmetric_mean_shift(self):
        d = Dimension("x", nominal=10.0, plus=0.4, minus=0.0)
        self.assertAlmostEqual(d.mean, 10.2)
        self.assertAlmostEqual(d.bilateral, 0.2)

    def test_rejects_negative_tolerance(self):
        with self.assertRaises(ValueError):
            Dimension("x", nominal=1.0, plus=-0.1, minus=0.1)

    def test_rejects_bad_distribution(self):
        with self.assertRaises(ValueError):
            Dimension("x", nominal=1.0, plus=0.1, minus=0.1,
                      distribution="cauchy")

    def test_rejects_bad_direction(self):
        with self.assertRaises(ValueError):
            Dimension("x", nominal=1.0, plus=0.1, minus=0.1, direction=2.0)

    def test_uniform_sample_in_range(self):
        import random
        d = Dimension("x", nominal=5.0, plus=1.0, minus=1.0,
                      distribution="uniform")
        rng = random.Random(0)
        for _ in range(500):
            s = d.sample(rng)
            self.assertGreaterEqual(s, 4.0)
            self.assertLessEqual(s, 6.0)

    def test_zero_tolerance_normal_is_exact(self):
        import random
        d = Dimension("x", nominal=5.0, plus=0.0, minus=0.0)
        self.assertEqual(d.sample(random.Random(1)), 5.0)


class WorstCaseTest(unittest.TestCase):

    def test_worst_case_sums_tolerances(self):
        chain = (ToleranceChain("t")
                 .add("a", nominal=10.0, plus=0.1)
                 .add("b", nominal=20.0, plus=0.2)
                 .add("c", nominal=5.0, plus=0.05))
        r = chain.analyze(target=35.0, tolerance=1.0)
        self.assertAlmostEqual(r.nominal_result, 35.0)
        # worst case band is +/- (0.1+0.2+0.05) = 0.35 around nominal
        self.assertAlmostEqual(r.worst_case_max, 35.35)
        self.assertAlmostEqual(r.worst_case_min, 34.65)
        self.assertAlmostEqual(r.worst_case_range, 0.70)
        self.assertTrue(r.worst_case_passed)

    def test_worst_case_fail(self):
        chain = (ToleranceChain("t")
                 .add("a", nominal=10.0, plus=0.6)
                 .add("b", nominal=10.0, plus=0.6))
        r = chain.analyze(target=20.0, tolerance=1.0)
        # +/- 1.2 exceeds the +/- 1.0 requirement
        self.assertFalse(r.worst_case_passed)

    def test_direction_subtracts(self):
        chain = (ToleranceChain("gap")
                 .add("beam", nominal=1000.0, plus=0.5)
                 .add("offset", nominal=1000.0, plus=0.5, direction=-1.0))
        r = chain.analyze(target=0.0, tolerance=2.0)
        self.assertAlmostEqual(r.nominal_result, 0.0)
        # worst plus uses beam.plus + offset.minus = 1.0
        self.assertAlmostEqual(r.worst_case_max, 1.0)
        self.assertAlmostEqual(r.worst_case_min, -1.0)


class RSSTest(unittest.TestCase):

    def test_rss_quadrature(self):
        chain = (ToleranceChain("t")
                 .add("a", nominal=10.0, plus=0.3)
                 .add("b", nominal=10.0, plus=0.4))
        r = chain.analyze(target=20.0, tolerance=1.0)
        # sqrt(0.3^2 + 0.4^2) = 0.5
        self.assertAlmostEqual(r.rss_max - r.nominal_result, 0.5)
        self.assertAlmostEqual(r.rss_range, 1.0)
        # RSS band is tighter than worst-case band
        self.assertLess(r.rss_range, r.worst_case_range)
        self.assertTrue(r.rss_passed)


class MonteCarloTest(unittest.TestCase):

    def test_reproducible_for_seed(self):
        def build():
            return (ToleranceChain("t")
                    .add("a", nominal=10.0, plus=0.1)
                    .add("b", nominal=10.0, plus=0.1))
        r1 = build().analyze(target=20.0, tolerance=0.5, mc_samples=2000, seed=99)
        r2 = build().analyze(target=20.0, tolerance=0.5, mc_samples=2000, seed=99)
        self.assertEqual(r1.mc_mean, r2.mc_mean)
        self.assertEqual(r1.mc_yield_pct, r2.mc_yield_pct)

    def test_seed_changes_result(self):
        def build():
            return ToleranceChain("t").add("a", nominal=10.0, plus=0.3)
        r1 = build().analyze(target=10.0, tolerance=0.4, mc_samples=2000, seed=1)
        r2 = build().analyze(target=10.0, tolerance=0.4, mc_samples=2000, seed=2)
        self.assertNotEqual(r1.mc_mean, r2.mc_mean)

    def test_tight_process_high_yield_and_cpk(self):
        # 3-sigma band (0.3) well inside a +/- 0.9 requirement -> Cpk ~ 3.
        r = (ToleranceChain("t")
             .add("a", nominal=10.0, plus=0.3)
             .analyze(target=10.0, tolerance=0.9, mc_samples=20000, seed=5))
        self.assertGreater(r.mc_yield_pct, 99.9)
        self.assertTrue(r.mc_passed)
        self.assertGreater(r.cpk, 2.0)

    def test_loose_process_low_yield(self):
        # requirement narrower than the 3-sigma band -> yield well under 99.73
        r = (ToleranceChain("t")
             .add("a", nominal=10.0, plus=0.9)
             .analyze(target=10.0, tolerance=0.3, mc_samples=20000, seed=5))
        self.assertLess(r.mc_yield_pct, 99.73)
        self.assertFalse(r.mc_passed)
        self.assertLess(r.cpk, 1.0)


class VarianceDecompositionTest(unittest.TestCase):

    def test_contributions_sum_to_100(self):
        r = (ToleranceChain("t")
             .add("a", nominal=1.0, plus=0.1)
             .add("b", nominal=1.0, plus=0.2)
             .add("c", nominal=1.0, plus=0.05)
             .analyze(target=3.0, tolerance=1.0))
        total = sum(c["variance_pct"] for c in r.contributors)
        self.assertAlmostEqual(total, 100.0)

    def test_dominant_contributor(self):
        r = (ToleranceChain("t")
             .add("small", nominal=1.0, plus=0.05)
             .add("big", nominal=1.0, plus=0.5)
             .analyze(target=2.0, tolerance=1.0))
        self.assertEqual(r.dominant_contributor, "big")


class EdgeCaseTest(unittest.TestCase):

    def test_empty_chain_raises(self):
        with self.assertRaises(ValueError):
            ToleranceChain("t").analyze()

    def test_negative_requirement_raises(self):
        with self.assertRaises(ValueError):
            ToleranceChain("t").add("a", 1.0, 0.1).analyze(tolerance=-0.1)

    def test_zero_variance_infinite_cpk(self):
        r = (ToleranceChain("t")
             .add("a", nominal=5.0, plus=0.0)
             .analyze(target=5.0, tolerance=0.5, mc_samples=100))
        self.assertTrue(math.isinf(r.cpk))


if __name__ == "__main__":
    unittest.main()
