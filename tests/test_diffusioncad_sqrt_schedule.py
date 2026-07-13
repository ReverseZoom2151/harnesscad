import math
import random
import unittest

from harnesscad.domain.numeric.diffusioncad_sqrt_schedule import (
    SqrtNoiseSchedule,
    classifier_free_mix,
    conditional_noise_seed,
    dequantize_levels,
    forward_diffuse,
    posterior_mean_coeffs,
    quantize_levels,
)


class TestSqrtSchedule(unittest.TestCase):
    def setUp(self):
        self.sched = SqrtNoiseSchedule(steps=1000)

    def test_alpha_bar_monotone_decreasing(self):
        prev = self.sched.alpha_bar(0)
        for t in range(1, self.sched.steps + 1):
            cur = self.sched.alpha_bar(t)
            self.assertLessEqual(cur, prev)
            prev = cur

    def test_alpha_bar_bounds(self):
        self.assertLessEqual(self.sched.alpha_bar(0), 1.0)
        self.assertGreaterEqual(self.sched.alpha_bar(0), 0.98)
        self.assertLess(self.sched.alpha_bar(1000), 0.01)

    def test_sqrt_form(self):
        # alpha_bar(t) = 1 - sqrt(t/T + s)
        t = 250
        expected = 1.0 - math.sqrt(t / 1000 + self.sched.offset)
        self.assertAlmostEqual(self.sched.alpha_bar(t), expected, places=9)

    def test_beta_from_ratio(self):
        t = 500
        alpha_t = self.sched.alpha_bar(t) / self.sched.alpha_bar(t - 1)
        self.assertAlmostEqual(self.sched.beta(t), 1.0 - alpha_t, places=12)
        self.assertAlmostEqual(self.sched.alpha(t), alpha_t, places=12)

    def test_beta_in_range(self):
        for t in range(1, self.sched.steps + 1, 37):
            self.assertGreaterEqual(self.sched.beta(t), 0.0)
            self.assertLessEqual(self.sched.beta(t), 1.0)

    def test_snr_decreases(self):
        self.assertGreater(self.sched.snr(1), self.sched.snr(999))

    def test_bad_steps(self):
        with self.assertRaises(ValueError):
            SqrtNoiseSchedule(steps=0)
        with self.assertRaises(IndexError):
            self.sched.alpha_bar(1001)
        with self.assertRaises(IndexError):
            self.sched.beta(0)


class TestForwardDiffusion(unittest.TestCase):
    def setUp(self):
        self.sched = SqrtNoiseSchedule(steps=1000)

    def test_deterministic_with_seed(self):
        x0 = [1.0, -2.0, 3.5, 0.0]
        a = forward_diffuse(x0, 300, self.sched, random.Random(7))
        b = forward_diffuse(x0, 300, self.sched, random.Random(7))
        self.assertEqual(a, b)

    def test_mean_matches_reparam(self):
        # Averaging many draws recovers sqrt(alpha_bar_t) * x0.
        x0 = [4.0]
        t = 100
        rng = random.Random(0)
        n = 40000
        s = sum(forward_diffuse(x0, t, self.sched, rng)[0] for _ in range(n)) / n
        self.assertAlmostEqual(s, self.sched.sqrt_alpha_bar(t) * 4.0, places=1)

    def test_length_preserved(self):
        x0 = [0.0] * 17
        self.assertEqual(len(forward_diffuse(x0, 10, self.sched, random.Random(1))), 17)

    def test_posterior_terminal(self):
        self.assertEqual(posterior_mean_coeffs(1, self.sched), (1.0, 0.0))

    def test_posterior_coeffs_positive(self):
        c0, ct = posterior_mean_coeffs(500, self.sched)
        self.assertGreater(c0, 0.0)
        self.assertGreater(ct, 0.0)


class TestConditionalSeed(unittest.TestCase):
    def setUp(self):
        self.sched = SqrtNoiseSchedule(steps=1000)

    def test_deterministic(self):
        a = conditional_noise_seed(10, self.sched, {2: 5.0}, random.Random(3))
        b = conditional_noise_seed(10, self.sched, {2: 5.0}, random.Random(3))
        self.assertEqual(a, b)

    def test_conditioned_dim_biased(self):
        # At a small t the conditioned coordinate mean is sqrt(alpha_bar_t)*e_c.
        t = 5
        e_c = 10.0
        rng = random.Random(0)
        n = 20000
        vals = [
            conditional_noise_seed(3, self.sched, {1: e_c}, rng, t=t)[1]
            for _ in range(n)
        ]
        mean = sum(vals) / n
        self.assertAlmostEqual(mean, self.sched.sqrt_alpha_bar(t) * e_c, places=1)

    def test_unconditioned_standard(self):
        rng = random.Random(0)
        n = 20000
        vals = [conditional_noise_seed(3, self.sched, {1: 9.0}, rng)[0] for _ in range(n)]
        self.assertAlmostEqual(sum(vals) / n, 0.0, places=1)

    def test_length(self):
        out = conditional_noise_seed(17, self.sched, {}, random.Random(1))
        self.assertEqual(len(out), 17)


class TestCFGandQuantize(unittest.TestCase):
    def test_cfg_endpoints(self):
        u = [0.0, 1.0]
        c = [2.0, 5.0]
        self.assertEqual(classifier_free_mix(u, c, 0.0), u)
        self.assertEqual(classifier_free_mix(u, c, 1.0), c)

    def test_cfg_extrapolate(self):
        u = [0.0]
        c = [1.0]
        self.assertEqual(classifier_free_mix(u, c, 2.0), [2.0])

    def test_cfg_length_mismatch(self):
        with self.assertRaises(ValueError):
            classifier_free_mix([1.0], [1.0, 2.0], 1.0)

    def test_quantize_roundtrip(self):
        for v in (0.0, 0.5, 1.0):
            idx = quantize_levels(v, 0.0, 1.0, 256)
            self.assertTrue(0 <= idx <= 255)
        self.assertEqual(quantize_levels(0.0, 0.0, 1.0), 0)
        self.assertEqual(quantize_levels(1.0, 0.0, 1.0), 255)

    def test_quantize_clamps(self):
        self.assertEqual(quantize_levels(-5.0, 0.0, 1.0), 0)
        self.assertEqual(quantize_levels(5.0, 0.0, 1.0), 255)

    def test_dequantize_centers(self):
        self.assertAlmostEqual(dequantize_levels(0, 0.0, 1.0), 0.0)
        self.assertAlmostEqual(dequantize_levels(255, 0.0, 1.0), 1.0)

    def test_quantize_bad_args(self):
        with self.assertRaises(ValueError):
            quantize_levels(0.5, 1.0, 0.0)
        with self.assertRaises(ValueError):
            quantize_levels(0.5, 0.0, 1.0, levels=1)


if __name__ == "__main__":
    unittest.main()
