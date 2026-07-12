"""Tests for numeric.regdiff_ddim_inversion (CADiffusion Sec. 3.3, Eq. 6-7)."""

import random
import unittest

from numeric.diffusioncad_sqrt_schedule import SqrtNoiseSchedule
from numeric.lion_ddim_sampler import ddim_sample, ddim_step
from numeric.regdiff_ddim_inversion import (
    ddim_invert,
    ddim_inversion_step,
    gaussian_noise,
    gaussian_perturb,
    make_inversion_timesteps,
    perturb_with_seed,
)


class TestInversionStep(unittest.TestCase):
    def test_inversion_step_inverts_reverse_step_with_fixed_eps(self):
        # With eps held fixed, inversion (z_prev -> z_t) and the DDIM reverse
        # step (z_t -> z_prev) are exact analytic inverses.
        z_prev = [0.3, -1.2, 0.5]
        eps = [0.1, 0.2, -0.05]
        ab_prev, ab_t = 0.9, 0.4
        z_t = ddim_inversion_step(z_prev, eps, ab_prev, ab_t)
        back = ddim_step(z_t, eps, ab_t, ab_prev)
        for a, b in zip(back, z_prev):
            self.assertAlmostEqual(a, b, places=10)

    def test_inversion_step_adds_noise_magnitude(self):
        # Climbing to a noisier timestep should not simply return the input.
        z_prev = [1.0, 0.0]
        eps = [0.5, 0.5]
        z_t = ddim_inversion_step(z_prev, eps, 0.99, 0.2)
        self.assertNotAlmostEqual(z_t[0], z_prev[0], places=6)

    def test_inversion_step_rejects_bad_alpha(self):
        with self.assertRaises(ValueError):
            ddim_inversion_step([1.0], [0.0], 0.0, 0.5)
        with self.assertRaises(ValueError):
            ddim_inversion_step([1.0], [0.0], 0.5, 1.5)


class TestInversionTimesteps(unittest.TestCase):
    def test_ascending_and_terminal(self):
        steps = make_inversion_timesteps(1000, 5)
        self.assertEqual(steps, sorted(steps))
        self.assertEqual(steps[-1], 1000)
        self.assertEqual(steps[0], 1)

    def test_single_step(self):
        self.assertEqual(make_inversion_timesteps(1000, 1), [1000])

    def test_full_schedule_visits_all(self):
        steps = make_inversion_timesteps(6, 6)
        self.assertEqual(steps, [1, 2, 3, 4, 5, 6])

    def test_rejects_bad_args(self):
        with self.assertRaises(ValueError):
            make_inversion_timesteps(0, 1)
        with self.assertRaises(ValueError):
            make_inversion_timesteps(10, 11)


class TestFullInversionRoundTrip(unittest.TestCase):
    def test_invert_then_sample_recovers_latent(self):
        # A constant eps model makes inversion + reverse DDIM exact inverses over
        # the whole trajectory (they visit the mirrored timestep sub-sequences).
        sched = SqrtNoiseSchedule(steps=50)
        const = [0.05, -0.1, 0.2, 0.0]

        def eps_model(z, t):
            return const

        z0 = [0.4, 0.1, -0.3, 0.7]
        z_T = ddim_invert(z0, sched, eps_model, total_steps=50, sample_steps=50)
        recon = ddim_sample(z_T, sched, eps_model, total_steps=50, sample_steps=50)
        for a, b in zip(recon, z0):
            self.assertAlmostEqual(a, b, places=6)

    def test_inversion_is_deterministic(self):
        sched = SqrtNoiseSchedule(steps=30)

        def eps_model(z, t):
            return [0.01 * (t + 1)] * len(z)

        z0 = [0.2, -0.4, 0.9]
        a = ddim_invert(z0, sched, eps_model, 30, 10)
        b = ddim_invert(z0, sched, eps_model, 30, 10)
        self.assertEqual(a, b)

    def test_eps_length_mismatch_raises(self):
        sched = SqrtNoiseSchedule(steps=10)

        def bad(z, t):
            return [0.0]  # wrong length

        with self.assertRaises(ValueError):
            ddim_invert([0.1, 0.2], sched, bad, 10, 5)


class TestGaussianPerturbation(unittest.TestCase):
    def test_perturb_convex_blend(self):
        z = [1.0, 2.0, 3.0]
        noise = [0.0, 0.0, 0.0]
        out = gaussian_perturb(z, 0.1, noise)
        for a, b in zip(out, z):
            self.assertAlmostEqual(a, 0.9 * b, places=12)

    def test_sigma_zero_is_identity(self):
        z = [0.5, -0.5]
        out = gaussian_perturb(z, 0.0, [9.0, 9.0])
        self.assertEqual(out, [0.5, -0.5])

    def test_sigma_one_is_pure_noise(self):
        out = gaussian_perturb([1.0, 1.0], 1.0, [7.0, -3.0])
        self.assertEqual(out, [7.0, -3.0])

    def test_rejects_bad_sigma_and_length(self):
        with self.assertRaises(ValueError):
            gaussian_perturb([1.0], 1.5, [0.0])
        with self.assertRaises(ValueError):
            gaussian_perturb([1.0, 2.0], 0.1, [0.0])

    def test_gaussian_noise_is_seed_reproducible(self):
        a = gaussian_noise(5, random.Random(7))
        b = gaussian_noise(5, random.Random(7))
        self.assertEqual(a, b)
        self.assertEqual(len(a), 5)

    def test_perturb_with_seed_reproducible(self):
        z = [0.1, 0.2, 0.3]
        a = perturb_with_seed(z, 0.1, seed=42)
        b = perturb_with_seed(z, 0.1, seed=42)
        self.assertEqual(a, b)
        # different seed -> generally different result
        c = perturb_with_seed(z, 0.1, seed=43)
        self.assertNotEqual(a, c)


if __name__ == "__main__":
    unittest.main()
