"""Tests for cfd_gradient_variance (CFD scaled gradient-variance metric)."""

import random
import unittest

import numeric.cfd_gradient_variance as gv


class TestScaledGradientVariance(unittest.TestCase):
    def test_constant_gradient_is_zero(self):
        grads = [[1.0, -2.0, 0.5] for _ in range(50)]
        self.assertAlmostEqual(gv.scaled_gradient_variance(grads), 0.0, places=6)
        self.assertAlmostEqual(gv.scaled_gradient_variance_direct(grads), 0.0, places=9)

    def test_zero_mean_noise_approaches_one(self):
        rng = random.Random(0)
        grads = [[rng.gauss(0.0, 1.0) for _ in range(8)] for _ in range(4000)]
        # Pure zero-mean noise: Var(g) ~ E[g^2], so sigma ~ 1.
        self.assertAlmostEqual(gv.scaled_gradient_variance_direct(grads), 1.0, delta=0.03)

    def test_range_is_unit_interval(self):
        rng = random.Random(3)
        grads = [[rng.gauss(0.5, 0.2) for _ in range(5)] for _ in range(500)]
        val = gv.scaled_gradient_variance_direct(grads)
        self.assertGreaterEqual(val, 0.0)
        self.assertLessEqual(val, 1.0)

    def test_more_consistent_has_lower_variance(self):
        rng = random.Random(7)
        # Signal + noise: larger signal (more consistent direction) -> lower sigma.
        def make(signal):
            return [
                [signal + rng.gauss(0.0, 1.0) for _ in range(6)]
                for _ in range(3000)
            ]

        low_signal = gv.scaled_gradient_variance_direct(make(0.5))
        high_signal = gv.scaled_gradient_variance_direct(make(5.0))
        self.assertLess(high_signal, low_signal)

    def test_ema_tracks_direct_for_stationary_stream(self):
        rng = random.Random(11)
        grads = [[2.0 + rng.gauss(0.0, 1.0) for _ in range(4)] for _ in range(6000)]
        ema = gv.scaled_gradient_variance(grads)
        direct = gv.scaled_gradient_variance_direct(grads)
        self.assertAlmostEqual(ema, direct, delta=0.12)

    def test_empty_raises(self):
        with self.assertRaises(ValueError):
            gv.scaled_gradient_variance([])


if __name__ == "__main__":
    unittest.main()
