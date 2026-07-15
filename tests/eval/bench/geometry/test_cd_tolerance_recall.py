"""Tests for eval.bench.geometry.cd_tolerance_recall."""

import unittest

from harnesscad.eval.bench.geometry.cd_tolerance_recall import (
    Attempt,
    auc_tr,
    cd_tolerance_recall_curve,
    recall_at_tolerance,
)


class AttemptTest(unittest.TestCase):
    def test_executed_without_cd_raises(self):
        with self.assertRaises(ValueError):
            Attempt(True, None)

    def test_negative_cd_raises(self):
        with self.assertRaises(ValueError):
            Attempt(True, -1.0)

    def test_failure_never_satisfies(self):
        self.assertFalse(Attempt(False).satisfies(1e9))


class RecallTest(unittest.TestCase):
    def setUp(self):
        self.attempts = [
            Attempt(True, 0.1),
            Attempt(True, 0.5),
            Attempt(False),      # survivor bias would drop this
            Attempt(True, 2.0),
        ]

    def test_recall_counts_failures(self):
        # at tau=0.6, two of FOUR (not three) satisfy
        self.assertAlmostEqual(recall_at_tolerance(self.attempts, 0.6), 0.5)

    def test_recall_monotone(self):
        r_low = recall_at_tolerance(self.attempts, 0.2)
        r_high = recall_at_tolerance(self.attempts, 5.0)
        self.assertLessEqual(r_low, r_high)
        self.assertAlmostEqual(r_high, 0.75)  # failure caps recall below 1.0

    def test_negative_tau_raises(self):
        with self.assertRaises(ValueError):
            recall_at_tolerance(self.attempts, -1.0)


class CurveTest(unittest.TestCase):
    def test_curve_length(self):
        curve = cd_tolerance_recall_curve([Attempt(True, 0.5)], 1.0, steps=10)
        self.assertEqual(len(curve), 11)
        self.assertEqual(curve[0][0], 0.0)
        self.assertEqual(curve[-1][0], 1.0)

    def test_all_executing_precise_high_auc(self):
        good = [Attempt(True, 0.0) for _ in range(4)]
        self.assertAlmostEqual(auc_tr(good, 1.0), 1.0)

    def test_all_failing_zero_auc(self):
        bad = [Attempt(False) for _ in range(4)]
        self.assertAlmostEqual(auc_tr(bad, 1.0), 0.0)

    def test_auc_bounded(self):
        a = auc_tr([Attempt(True, 0.3), Attempt(False)], 1.0)
        self.assertGreaterEqual(a, 0.0)
        self.assertLessEqual(a, 1.0)


if __name__ == "__main__":
    unittest.main()
