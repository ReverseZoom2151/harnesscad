"""Tests for cfd_clean_flow_ode (CFD clean-flow ODE / EDM Heun integrator)."""

import unittest

from harnesscad.domain.numeric import cfd_clean_flow_ode as cf


class TestAnalyticPredictions(unittest.TestCase):
    def test_sample_prediction_recovers_mean_at_high_noise(self):
        # At near-pure-noise the sample prediction collapses toward the mean mu.
        a, s = cf.cosine_schedule(0.999)
        x = s * 1.3
        pred = cf.sample_prediction(x, a, s, mu=2.0, s=0.5)
        self.assertAlmostEqual(pred, 2.0, delta=0.2)

    def test_eps_zero_variance_target(self):
        # If s->0 (delta at mu), eps_phi = (x - alpha mu)/sigma exactly.
        a, s = cf.cosine_schedule(0.5)
        x = 0.7
        got = cf.eps_gaussian(x, a, s, mu=1.0, s=1e-9)
        self.assertAlmostEqual(got, (x - a * 1.0) / s, places=6)


class TestCleanFlowODE(unittest.TestCase):
    def test_endpoint_matches_gaussian_oracle(self):
        mu, s = 1.5, 0.4
        for eps_tilde in (-1.5, -0.5, 0.0, 0.8, 2.0):
            got = cf.clean_flow_ode_endpoint(eps_tilde, mu, s, steps=400)
            want = cf.target_sample(eps_tilde, mu, s)  # mu + s * eps_tilde
            self.assertAlmostEqual(got, want, delta=0.02)

    def test_more_steps_reduce_error(self):
        mu, s, eps_tilde = 0.0, 1.0, 1.2
        want = cf.target_sample(eps_tilde, mu, s)
        err_coarse = abs(cf.clean_flow_ode_endpoint(eps_tilde, mu, s, steps=20) - want)
        err_fine = abs(cf.clean_flow_ode_endpoint(eps_tilde, mu, s, steps=400) - want)
        self.assertLess(err_fine, err_coarse)

    def test_deterministic(self):
        a = cf.clean_flow_ode_endpoint(0.7, 1.0, 0.5, steps=100)
        b = cf.clean_flow_ode_endpoint(0.7, 1.0, 0.5, steps=100)
        self.assertEqual(a, b)


class TestEdmHeunSampler(unittest.TestCase):
    def test_heun_more_accurate_than_euler(self):
        mu, s, eps_tilde = 0.5, 0.7, 1.1
        want = cf.target_sample(eps_tilde, mu, s)
        # Order-of-accuracy is asymptotic; compare in a non-degenerate step regime.
        euler_err = abs(
            cf.clean_flow_ode_endpoint(eps_tilde, mu, s, steps=50) - want
        )
        heun_err = abs(
            cf.edm_heun_sampler(mu, s, steps=50, gamma=0.0, eps_tilde=eps_tilde) - want
        )
        self.assertLess(heun_err, euler_err)

    def test_gamma_zero_is_deterministic(self):
        a = cf.edm_heun_sampler(0.0, 1.0, steps=30, gamma=0.0, eps_tilde=0.9)
        b = cf.edm_heun_sampler(0.0, 1.0, steps=30, gamma=0.0, eps_tilde=0.9)
        self.assertEqual(a, b)

    def test_stochastic_sampler_mean_matches_target(self):
        # With noise injection, average many runs -> approaches E[x0] = mu.
        mu, s = 1.0, 0.6
        vals = [
            cf.edm_heun_sampler(mu, s, steps=40, gamma=0.01, seed=i)
            for i in range(400)
        ]
        mean = sum(vals) / len(vals)
        self.assertAlmostEqual(mean, mu, delta=0.15)

    def test_invalid_gamma_raises(self):
        with self.assertRaises(ValueError):
            cf.edm_heun_sampler(0.0, 1.0, gamma=1.0)


if __name__ == "__main__":
    unittest.main()
