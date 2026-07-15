"""Tests for eval.bench.sequence.soft_target_distribution."""

import unittest
from math import log

from harnesscad.eval.bench.sequence.soft_target_distribution import (
    soft_cross_entropy,
    soft_target,
)


class SoftTargetTest(unittest.TestCase):
    def test_sums_to_one(self):
        dist = soft_target(10, 20)
        self.assertAlmostEqual(sum(dist), 1.0)

    def test_peak_at_target(self):
        dist = soft_target(10, 20, delta=3, beta=2.0)
        self.assertEqual(max(range(len(dist)), key=lambda k: dist[k]), 10)

    def test_symmetric_and_windowed(self):
        # beta=2 => weight (beta-|k-t|) is positive only for |k-t| < 2,
        # so support is {t-1, t, t+1} even though delta=3 widens the window.
        dist = soft_target(10, 21, delta=3, beta=2.0)
        self.assertGreater(dist[9], 0.0)
        self.assertGreater(dist[11], 0.0)
        self.assertEqual(dist[8], 0.0)   # |k-t| == 2 -> weight 0
        self.assertEqual(dist[12], 0.0)
        self.assertEqual(dist[7], 0.0)
        self.assertEqual(dist[13], 0.0)
        # Symmetry around the target.
        self.assertAlmostEqual(dist[9], dist[11])

    def test_relative_weights(self):
        # beta - |k - t|: at t weight 2, at t+-1 weight 1 (ratio 2:1).
        dist = soft_target(5, 20, delta=3, beta=2.0)
        self.assertAlmostEqual(dist[5] / dist[4], 2.0)

    def test_edge_clipping_still_normalises(self):
        dist = soft_target(0, 10, delta=3, beta=2.0)
        self.assertAlmostEqual(sum(dist), 1.0)
        self.assertEqual(dist[9], 0.0)

    def test_invalid_args(self):
        with self.assertRaises(ValueError):
            soft_target(20, 10)
        with self.assertRaises(ValueError):
            soft_target(5, 0)
        with self.assertRaises(ValueError):
            soft_target(5, 10, beta=0.0)


class SoftCrossEntropyTest(unittest.TestCase):
    def test_matches_manual(self):
        probs = [0.1, 0.2, 0.4, 0.2, 0.1]
        tgt = soft_target(2, 5, delta=1, beta=2.0)
        expected = -sum(tgt[k] * log(probs[k]) for k in range(5) if tgt[k] > 0)
        self.assertAlmostEqual(soft_cross_entropy(probs, 2, delta=1, beta=2.0),
                               expected)

    def test_lower_when_matching_soft_target(self):
        # A prediction equal to the soft target beats a flat distribution.
        matched = soft_target(1, 5, delta=1, beta=2.0)
        flat = [0.2, 0.2, 0.2, 0.2, 0.2]
        self.assertLess(soft_cross_entropy(matched, 1, delta=1),
                        soft_cross_entropy(flat, 1, delta=1))

    def test_zero_mass_raises(self):
        with self.assertRaises(ValueError):
            soft_cross_entropy([0.0, 0.0, 0.0], 1)


if __name__ == "__main__":
    unittest.main()
