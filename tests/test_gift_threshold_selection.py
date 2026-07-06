import unittest

from dataengine.gift_threshold_selection import (  # noqa: F401
    band_mass, empirical_cdf, fraction_below, quantile, select_gift_thresholds,
)


class TestEmpiricalCdf(unittest.TestCase):
    def test_monotone_and_reaches_one(self):
        cdf = empirical_cdf([0.1, 0.2, 0.2, 0.9])
        fracs = [f for _, f in cdf]
        self.assertEqual(fracs, sorted(fracs))
        self.assertAlmostEqual(cdf[-1][1], 1.0)

    def test_dedups_values(self):
        cdf = empirical_cdf([0.5, 0.5, 0.5])
        self.assertEqual(len(cdf), 1)
        self.assertAlmostEqual(cdf[0][1], 1.0)

    def test_empty_raises(self):
        with self.assertRaises(ValueError):
            empirical_cdf([])


class TestFractionBelow(unittest.TestCase):
    def test_strict(self):
        scores = [0.1, 0.4, 0.6, 0.95]
        self.assertAlmostEqual(fraction_below(scores, 0.5), 0.5)
        self.assertAlmostEqual(fraction_below(scores, 0.1), 0.0)
        self.assertAlmostEqual(fraction_below(scores, 1.0), 1.0)


class TestQuantile(unittest.TestCase):
    def test_endpoints(self):
        scores = [0.0, 0.25, 0.5, 0.75, 1.0]
        self.assertEqual(quantile(scores, 0.0), 0.0)
        self.assertEqual(quantile(scores, 1.0), 1.0)
        self.assertEqual(quantile(scores, 0.5), 0.5)

    def test_fraction_bounds(self):
        with self.assertRaises(ValueError):
            quantile([0.1], 1.5)


class TestSelectThresholds(unittest.TestCase):
    def test_reproduces_paper_split(self):
        # 10% below 0.5 and 40% below 0.9, like the paper's distribution.
        scores = ([0.2] * 10          # bottom 10%
                  + [0.7] * 30         # next 30% (total 40% below 0.9)
                  + [0.95] * 60)       # top 60%
        thr = select_gift_thresholds(scores, low_fraction=0.10,
                                     valid_fraction=0.40)
        self.assertLessEqual(thr["tau_low"], thr["tau_valid"])
        self.assertLessEqual(thr["tau_valid"], thr["tau_match"])
        self.assertEqual(thr["tau_match"], 0.99)
        # ~10% of the pool falls below tau_low and ~40% below tau_valid.
        self.assertAlmostEqual(fraction_below(scores, thr["tau_low"]), 0.10,
                               delta=0.02)
        self.assertAlmostEqual(fraction_below(scores, thr["tau_valid"]), 0.40,
                               delta=0.02)

    def test_monotone_repair(self):
        thr = select_gift_thresholds([0.8] * 5, low_fraction=0.1,
                                     valid_fraction=0.9, tau_match=0.5)
        self.assertLessEqual(thr["tau_low"], thr["tau_valid"])
        self.assertLessEqual(thr["tau_valid"], thr["tau_match"])

    def test_valid_fraction_must_exceed_low(self):
        with self.assertRaises(ValueError):
            select_gift_thresholds([0.5], low_fraction=0.4, valid_fraction=0.1)


class TestBandMass(unittest.TestCase):
    def test_masses_sum_to_one(self):
        scores = [0.2, 0.6, 0.7, 0.92, 0.95, 0.995]
        m = band_mass(scores, 0.5, 0.9, 0.99)
        self.assertAlmostEqual(sum(m.values()), 1.0)
        self.assertGreater(m["valid"], 0)
        self.assertGreater(m["near_miss"], 0)


if __name__ == "__main__":
    unittest.main()
