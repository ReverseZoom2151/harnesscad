"""Tests for numeric.vqcad_mask_and_replace (VQ-Diffusion mask-and-replace)."""

from __future__ import annotations

import random
import unittest

from harnesscad.domain.numeric.categorical_diffusion import matmul, vec_mat
from harnesscad.domain.numeric.mask_and_replace import (
    beta_from,
    converges_to_mask,
    cumulative_for,
    cumulative_parameters,
    diffuse_index,
    forward_marginal_index,
    linear_gamma_schedule,
    mask_and_replace_matrix,
    mask_index,
)


class TestTransitionMatrix(unittest.TestCase):
    def test_shape_is_k_plus_one(self):
        q = mask_and_replace_matrix(4, alpha=0.7, gamma=0.1)
        self.assertEqual(len(q), 5)
        self.assertTrue(all(len(row) == 5 for row in q))

    def test_rows_are_stochastic(self):
        q = mask_and_replace_matrix(5, alpha=0.6, gamma=0.2)
        for row in q:
            self.assertAlmostEqual(sum(row), 1.0, places=12)

    def test_mask_row_is_absorbing(self):
        k = 3
        q = mask_and_replace_matrix(k, alpha=0.5, gamma=0.3)
        mask_row = q[mask_index(k)]
        self.assertEqual(mask_row, [0.0, 0.0, 0.0, 1.0])

    def test_diagonal_and_offdiagonal_masses(self):
        k = 4
        alpha, gamma = 0.7, 0.1
        beta = beta_from(k, alpha, gamma)
        q = mask_and_replace_matrix(k, alpha, gamma)
        # real row 1
        self.assertAlmostEqual(q[1][1], alpha + beta, places=12)
        self.assertAlmostEqual(q[1][0], beta, places=12)
        self.assertAlmostEqual(q[1][2], beta, places=12)
        self.assertAlmostEqual(q[1][k], gamma, places=12)  # to mask

    def test_beta_constraint(self):
        # alpha + K*beta + gamma == 1
        k = 6
        alpha, gamma = 0.55, 0.15
        beta = beta_from(k, alpha, gamma)
        self.assertAlmostEqual(alpha + k * beta + gamma, 1.0, places=12)

    def test_infeasible_rates_raise(self):
        with self.assertRaises(ValueError):
            beta_from(3, alpha=0.8, gamma=0.5)  # sum > 1
        with self.assertRaises(ValueError):
            beta_from(3, alpha=-0.1, gamma=0.2)

    def test_pure_uniform_limit(self):
        # gamma = 0 reduces the real-block to the uniform/Multinomial matrix.
        k = 4
        q = mask_and_replace_matrix(k, alpha=0.8, gamma=0.0)
        for i in range(k):
            self.assertAlmostEqual(q[i][k], 0.0, places=12)  # no mask leakage
            self.assertAlmostEqual(sum(q[i][:k]), 1.0, places=12)

    def test_pure_absorbing_limit(self):
        # beta = 0 (alpha + gamma = 1) reduces to the absorbing/[MASK] matrix.
        k = 4
        alpha, gamma = 0.7, 0.3
        beta = beta_from(k, alpha, gamma)
        self.assertAlmostEqual(beta, 0.0, places=12)
        q = mask_and_replace_matrix(k, alpha, gamma)
        for i in range(k):
            self.assertAlmostEqual(q[i][i], alpha, places=12)
            self.assertAlmostEqual(q[i][k], gamma, places=12)


class TestCumulativeClosedForm(unittest.TestCase):
    def _explicit_cumulative(self, k, alphas, gammas):
        """Multiply the actual (K+1)x(K+1) matrices to cross-check the closed form."""
        mats = [mask_and_replace_matrix(k, a, g) for a, g in zip(alphas, gammas)]
        cum = [row[:] for row in mats[0]]
        cums = [cum]
        for m in mats[1:]:
            cum = matmul(cum, m)
            cums.append(cum)
        return cums

    def test_closed_form_matches_matrix_product(self):
        k = 4
        alphas = [0.9, 0.85, 0.8, 0.7]
        gammas = [0.05, 0.1, 0.15, 0.25]
        cum_params = cumulative_for(k, alphas, gammas)
        explicit = self._explicit_cumulative(k, alphas, gammas)
        for t in range(1, len(alphas) + 1):
            # marginal from a real x0 = 0 via closed form
            marg = forward_marginal_index(0, k, cum_params[t - 1])
            # marginal via explicit matrix: x0 one-hot @ Qbar_t
            x0 = [0.0] * (k + 1)
            x0[0] = 1.0
            ref = vec_mat(x0, explicit[t - 1])
            for a, b in zip(marg, ref):
                self.assertAlmostEqual(a, b, places=10)

    def test_cumulative_parameters_residual(self):
        # low-level helper returns residual (not divided by K)
        raw = cumulative_parameters([0.9, 0.8], [0.1, 0.2])
        alpha_bar, residual, gamma_bar = raw[-1]
        self.assertAlmostEqual(alpha_bar, 0.9 * 0.8, places=12)
        self.assertAlmostEqual(gamma_bar, 1.0 - (0.9 * 0.8), places=12)
        self.assertAlmostEqual(alpha_bar + residual + gamma_bar, 1.0, places=12)

    def test_marginal_is_distribution(self):
        k = 5
        cum_params = cumulative_for(k, [0.8, 0.7], [0.1, 0.2])
        for x0 in range(k + 1):
            marg = forward_marginal_index(x0, k, cum_params[-1])
            self.assertAlmostEqual(sum(marg), 1.0, places=12)
            self.assertTrue(all(p >= -1e-12 for p in marg))

    def test_length_mismatch_raises(self):
        with self.assertRaises(ValueError):
            cumulative_parameters([0.9], [0.1, 0.2])


class TestConvergenceAndSchedule(unittest.TestCase):
    def test_linear_schedule_converges_to_mask(self):
        gammas = linear_gamma_schedule(100, gamma_final=1.0)
        self.assertTrue(converges_to_mask(gammas))

    def test_linear_schedule_hits_target(self):
        gammas = linear_gamma_schedule(50, gamma_final=0.9)
        keep = 1.0
        for g in gammas:
            keep *= (1.0 - g)
        self.assertAlmostEqual(1.0 - keep, 0.9, places=10)

    def test_non_converging_schedule(self):
        self.assertFalse(converges_to_mask([0.01, 0.01, 0.01]))

    def test_at_full_mask_marginal_is_mask(self):
        k = 4
        gammas = linear_gamma_schedule(200, gamma_final=1.0)
        alphas = [0.0] * len(gammas)  # all mass leaves the real block
        cum_params = cumulative_for(k, alphas, gammas)
        marg = forward_marginal_index(2, k, cum_params[-1])
        self.assertAlmostEqual(marg[mask_index(k)], 1.0, places=6)


class TestDiffuseIndex(unittest.TestCase):
    def test_deterministic_given_seed(self):
        k = 5
        cum_params = cumulative_for(k, [0.8, 0.7, 0.6], [0.1, 0.15, 0.2])
        a = [diffuse_index(0, k, cum_params, 2, random.Random(7)) for _ in range(3)]
        self.assertEqual(a, [a[0]] * 3)

    def test_samples_in_range(self):
        k = 4
        cum_params = cumulative_for(k, [0.7, 0.6], [0.2, 0.3])
        rng = random.Random(1)
        for _ in range(200):
            x = diffuse_index(1, k, cum_params, 2, rng)
            self.assertTrue(0 <= x <= k)

    def test_t_out_of_range(self):
        k = 3
        cum_params = cumulative_for(k, [0.8], [0.1])
        with self.assertRaises(IndexError):
            diffuse_index(0, k, cum_params, 5, random.Random(0))


if __name__ == "__main__":
    unittest.main()
