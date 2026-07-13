"""Tests for numeric.lion_ddim_sampler (deterministic DDIM + diffuse-denoise)."""

import unittest
from math import sqrt

from harnesscad.domain.numeric.ddim_sampler import (
    ddim_sample,
    ddim_step,
    diffuse_denoise_sample,
    diffuse_denoise_steps,
    make_timesteps,
    predict_x0,
)


class LinearSchedule:
    """Toy schedule with alpha_bar(0) == 1 for exact-recovery testing."""

    def __init__(self, total=10):
        self.total = total

    def alpha_bar(self, t):
        # 1.0 at t=0, decreasing to 0.1 at t=total.
        return 1.0 - 0.9 * t / self.total


class PredictX0Test(unittest.TestCase):
    def test_recovers_true_x0(self):
        x0 = [1.0, -2.0, 3.0]
        eps = [0.5, -0.5, 0.25]
        ab = 0.36  # sqrt = 0.6, sqrt(1-ab)=0.8
        x_t = [sqrt(ab) * a + sqrt(1 - ab) * e for a, e in zip(x0, eps)]
        rec = predict_x0(x_t, eps, ab)
        for a, r in zip(x0, rec):
            self.assertAlmostEqual(a, r, places=9)

    def test_invalid_alpha_bar(self):
        with self.assertRaises(ValueError):
            predict_x0([1.0], [0.0], 0.0)
        with self.assertRaises(ValueError):
            predict_x0([1.0], [0.0], 1.5)


class DDIMStepTest(unittest.TestCase):
    def test_single_step_to_clean_recovers_x0(self):
        x0 = [2.0, -1.0]
        eps = [0.3, 0.7]
        ab_t = 0.25
        x_t = [sqrt(ab_t) * a + sqrt(1 - ab_t) * e for a, e in zip(x0, eps)]
        # ab_prev = 1.0 => fully clean target
        out = ddim_step(x_t, eps, ab_t, 1.0)
        for a, o in zip(x0, out):
            self.assertAlmostEqual(a, o, places=9)

    def test_invalid_prev(self):
        with self.assertRaises(ValueError):
            ddim_step([1.0], [0.0], 0.5, 1.5)


class TimestepsTest(unittest.TestCase):
    def test_descending_includes_total(self):
        ts = make_timesteps(1000, 25)
        self.assertEqual(ts[0], 1000)
        self.assertEqual(ts, sorted(ts, reverse=True))
        self.assertEqual(len(set(ts)), len(ts))
        self.assertTrue(all(1 <= t <= 1000 for t in ts))

    def test_full_visits_all(self):
        ts = make_timesteps(10, 10)
        self.assertEqual(ts, list(range(10, 0, -1)))

    def test_single(self):
        self.assertEqual(make_timesteps(50, 1), [50])

    def test_invalid(self):
        with self.assertRaises(ValueError):
            make_timesteps(10, 0)
        with self.assertRaises(ValueError):
            make_timesteps(10, 11)


class DDIMSampleTest(unittest.TestCase):
    def test_full_recovery_with_true_noise(self):
        sched = LinearSchedule(10)
        x0 = [1.0, -1.0, 0.5]
        eps = [0.2, -0.3, 0.4]
        ab_T = sched.alpha_bar(10)
        x_T = [sqrt(ab_T) * a + sqrt(1 - ab_T) * e for a, e in zip(x0, eps)]
        out = ddim_sample(x_T, sched, lambda x, t: eps, 10, sample_steps=10)
        for a, o in zip(x0, out):
            self.assertAlmostEqual(a, o, places=9)

    def test_subsampled_also_recovers(self):
        sched = LinearSchedule(10)
        x0 = [3.0, 0.0]
        eps = [0.1, -0.2]
        ab_T = sched.alpha_bar(10)
        x_T = [sqrt(ab_T) * a + sqrt(1 - ab_T) * e for a, e in zip(x0, eps)]
        out = ddim_sample(x_T, sched, lambda x, t: eps, 10, sample_steps=4)
        for a, o in zip(x0, out):
            self.assertAlmostEqual(a, o, places=9)

    def test_deterministic(self):
        sched = LinearSchedule(10)
        model = lambda x, t: [0.01 * t] * len(x)
        a = ddim_sample([0.5, 0.5], sched, model, 10, 5)
        b = ddim_sample([0.5, 0.5], sched, model, 10, 5)
        self.assertEqual(a, b)

    def test_eps_length_mismatch(self):
        sched = LinearSchedule(10)
        with self.assertRaises(ValueError):
            ddim_sample([1.0, 2.0], sched, lambda x, t: [0.0], 10, 2)


class DiffuseDenoiseTest(unittest.TestCase):
    def test_tau_zero_is_identity(self):
        sched = LinearSchedule(10)
        x0 = [1.0, 2.0, 3.0]
        out = diffuse_denoise_sample(x0, sched, lambda x, t: [0.0] * 3, 10, 0)
        self.assertEqual(out, x0)

    def test_recovers_with_matching_noise(self):
        sched = LinearSchedule(10)
        x0 = [1.0, -2.0]
        noise = [0.3, -0.1]
        out = diffuse_denoise_sample(
            x0, sched, lambda x, t: noise, 10, tau=5, forward_noise=noise
        )
        for a, o in zip(x0, out):
            self.assertAlmostEqual(a, o, places=9)

    def test_steps_sequence(self):
        self.assertEqual(diffuse_denoise_steps(10, 3), [3, 2, 1])
        self.assertEqual(diffuse_denoise_steps(10, 0), [])

    def test_steps_invalid(self):
        with self.assertRaises(ValueError):
            diffuse_denoise_steps(10, 11)

    def test_noise_length_mismatch(self):
        sched = LinearSchedule(10)
        with self.assertRaises(ValueError):
            diffuse_denoise_sample(
                [1.0, 2.0], sched, lambda x, t: [0.0, 0.0], 10, 5,
                forward_noise=[0.1],
            )


if __name__ == "__main__":
    unittest.main()
