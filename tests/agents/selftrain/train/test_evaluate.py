"""The statistics that keep the comparison honest are pure python and must be
tested without a GPU: a Wilson interval that is wrong, or a McNemar test that is
wrong, would launder noise into a headline. These are the numbers the whole
exercise turns on, so they get asserted against known values."""

from __future__ import annotations

import unittest

from harnesscad.agents.selftrain.train import evaluate as E


class TestWilson(unittest.TestCase):

    def test_degenerate(self):
        self.assertEqual(E.wilson(0, 0), (0.0, 0.0, 0.0))

    def test_all_success_interval_below_one(self):
        # 16/16 must not report a point estimate of 1.0 with a zero-width CI: the
        # Wilson interval's upper bound is < 1 and lower bound < 1, which is the
        # whole reason it beats the normal approximation at the boundary.
        p, lo, hi = E.wilson(16, 16)
        self.assertEqual(p, 1.0)
        self.assertLess(lo, 1.0)
        self.assertLessEqual(hi, 1.0)
        self.assertGreater(lo, 0.7)

    def test_half(self):
        p, lo, hi = E.wilson(8, 16)
        self.assertAlmostEqual(p, 0.5, places=6)
        self.assertLess(lo, 0.5)
        self.assertGreater(hi, 0.5)
        # Wilson half-interval for 8/16 is ~0.28..0.72.
        self.assertAlmostEqual(lo, 0.2799, places=2)
        self.assertAlmostEqual(hi, 0.7201, places=2)


class TestMcNemar(unittest.TestCase):

    def test_no_discordant_pairs_is_one(self):
        self.assertEqual(E.mcnemar_exact(0, 0), 1.0)

    def test_symmetric(self):
        self.assertAlmostEqual(E.mcnemar_exact(3, 1), E.mcnemar_exact(1, 3), places=9)

    def test_all_one_direction(self):
        # 5 discordant pairs all favouring one arm: exact two-sided p = 2*(1/2)^5.
        self.assertAlmostEqual(E.mcnemar_exact(5, 0), 2.0 * (0.5 ** 5), places=9)

    def test_p_never_exceeds_one(self):
        self.assertLessEqual(E.mcnemar_exact(2, 2), 1.0)


class TestCompare(unittest.TestCase):

    def test_compare_counts_wins(self):
        a = E.ArmResult(name="a", n=3, accepted=2,
                        per_brief={"x": True, "y": True, "z": False})
        b = E.ArmResult(name="b", n=3, accepted=1,
                        per_brief={"x": True, "y": False, "z": False})
        cmp = E.compare(a, b)
        self.assertEqual(cmp["a_wins"], 1)   # y: a right, b wrong
        self.assertEqual(cmp["b_wins"], 0)
        self.assertEqual(cmp["both_pass"], 1)  # x
        self.assertEqual(cmp["both_fail"], 1)  # z


if __name__ == "__main__":
    unittest.main()
