import random
import unittest

from numeric.sketchdnn_categorical_diffusion import (
    absorbing_transition_matrix,
    argmax_decode,
    categorical_posterior,
    cumulative_matrices,
    diffuse_categorical,
    forward_marginal,
    forward_step,
    matmul,
    one_hot,
    sample_categorical,
    transpose,
    uniform_transition_matrix,
    vec_mat,
)


def _row_stochastic(m):
    return all(abs(sum(row) - 1.0) < 1e-9 for row in m)


class TestTransitionMatrices(unittest.TestCase):
    def test_uniform_row_stochastic(self):
        q = uniform_transition_matrix(4, 0.3)
        self.assertTrue(_row_stochastic(q))

    def test_uniform_beta_zero_is_identity(self):
        q = uniform_transition_matrix(3, 0.0)
        for i in range(3):
            for j in range(3):
                self.assertAlmostEqual(q[i][j], 1.0 if i == j else 0.0)

    def test_uniform_diagonal_dominant(self):
        q = uniform_transition_matrix(5, 0.2)
        for i in range(5):
            self.assertGreater(q[i][i], q[i][(i + 1) % 5])

    def test_absorbing_row_stochastic(self):
        q = absorbing_transition_matrix(4, 0.4)
        self.assertTrue(_row_stochastic(q))

    def test_absorbing_state_is_fixed(self):
        q = absorbing_transition_matrix(4, 0.4)
        # last row is identity row (absorbing)
        self.assertAlmostEqual(q[3][3], 1.0)
        self.assertAlmostEqual(sum(q[3][:3]), 0.0)

    def test_absorbing_mass_leaks_to_mask(self):
        q = absorbing_transition_matrix(4, 0.4)
        self.assertAlmostEqual(q[0][0], 0.6)
        self.assertAlmostEqual(q[0][3], 0.4)

    def test_beta_out_of_range(self):
        with self.assertRaises(ValueError):
            uniform_transition_matrix(3, 1.5)


class TestLinearAlgebra(unittest.TestCase):
    def test_matmul_identity(self):
        ident = [[1.0, 0.0], [0.0, 1.0]]
        a = [[2.0, 3.0], [4.0, 5.0]]
        self.assertEqual(matmul(a, ident), a)

    def test_vec_mat(self):
        v = [1.0, 2.0]
        m = [[1.0, 0.0], [0.0, 1.0]]
        self.assertEqual(vec_mat(v, m), [1.0, 2.0])

    def test_transpose(self):
        m = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]
        self.assertEqual(transpose(m), [[1.0, 4.0], [2.0, 5.0], [3.0, 6.0]])

    def test_cumulative_preserves_stochasticity(self):
        qs = [uniform_transition_matrix(4, b) for b in (0.1, 0.2, 0.3)]
        cum = cumulative_matrices(qs)
        self.assertEqual(len(cum), 3)
        for m in cum:
            self.assertTrue(_row_stochastic(m))
        # Qbar_1 == Q_1
        self.assertEqual(cum[0], qs[0])
        # Qbar_2 == Q_1 @ Q_2
        self.assertEqual(cum[1], matmul(qs[0], qs[1]))


class TestForwardAndPosterior(unittest.TestCase):
    def test_forward_marginal_is_probability(self):
        qs = [uniform_transition_matrix(4, 0.25) for _ in range(5)]
        cum = cumulative_matrices(qs)
        marg = forward_marginal(one_hot(1, 4), cum[-1])
        self.assertAlmostEqual(sum(marg), 1.0, places=9)
        self.assertTrue(all(p >= 0.0 for p in marg))

    def test_uniform_converges_toward_uniform(self):
        # Heavy noising drives the marginal toward 1/K each.
        qs = [uniform_transition_matrix(4, 0.9) for _ in range(50)]
        cum = cumulative_matrices(qs)
        marg = forward_marginal(one_hot(0, 4), cum[-1])
        for p in marg:
            self.assertAlmostEqual(p, 0.25, places=3)

    def test_absorbing_converges_to_mask(self):
        qs = [absorbing_transition_matrix(4, 0.5) for _ in range(40)]
        cum = cumulative_matrices(qs)
        marg = forward_marginal(one_hot(0, 4), cum[-1])
        self.assertAlmostEqual(marg[3], 1.0, places=6)

    def test_forward_step_matches_manual(self):
        q = uniform_transition_matrix(3, 0.3)
        x = one_hot(0, 3)
        self.assertEqual(forward_step(x, q), q[0])

    def test_posterior_is_probability(self):
        qs = [uniform_transition_matrix(4, 0.3) for _ in range(3)]
        cum = cumulative_matrices(qs)
        x0 = one_hot(2, 4)
        x_t = forward_marginal(x0, cum[2])
        post = categorical_posterior(x_t, x0, qs[2], cum[1])
        self.assertAlmostEqual(sum(post), 1.0, places=9)
        self.assertTrue(all(p >= 0.0 for p in post))

    def test_posterior_t1_uses_x0(self):
        # At t=1, Qbar_0 = I so the x0 factor is exactly x0: posterior
        # concentrates on the true class when x_t equals its transition row.
        q1 = uniform_transition_matrix(4, 0.2)
        x0 = one_hot(1, 4)
        x_t = forward_step(x0, q1)
        post = categorical_posterior(x_t, x0, q1, None)
        self.assertEqual(len(post), 4)
        # x0 is one-hot at index 1 so only that index survives the product.
        self.assertAlmostEqual(post[1], 1.0, places=9)

    def test_posterior_peaks_at_true_class_low_noise(self):
        qs = [uniform_transition_matrix(4, 0.05) for _ in range(3)]
        cum = cumulative_matrices(qs)
        x0 = one_hot(3, 4)
        x_t = forward_marginal(x0, cum[2])
        post = categorical_posterior(x_t, x0, qs[2], cum[1])
        self.assertEqual(argmax_decode(post), 3)


class TestSampling(unittest.TestCase):
    def test_sample_deterministic_with_seed(self):
        probs = [0.1, 0.2, 0.3, 0.4]
        a = [sample_categorical(probs, random.Random(7)) for _ in range(3)]
        b = [sample_categorical(probs, random.Random(7)) for _ in range(3)]
        self.assertEqual(a, b)

    def test_sample_respects_point_mass(self):
        probs = [0.0, 1.0, 0.0]
        self.assertEqual(sample_categorical(probs, random.Random(1)), 1)

    def test_sample_empirical_frequency(self):
        probs = [0.7, 0.3]
        rng = random.Random(123)
        counts = [0, 0]
        for _ in range(4000):
            counts[sample_categorical(probs, rng)] += 1
        self.assertAlmostEqual(counts[0] / 4000, 0.7, delta=0.03)

    def test_argmax_decode(self):
        self.assertEqual(argmax_decode([0.1, 0.5, 0.4]), 1)

    def test_diffuse_categorical_deterministic(self):
        qs = [uniform_transition_matrix(4, 0.3) for _ in range(6)]
        a = diffuse_categorical(0, 4, qs, 4, random.Random(5))
        b = diffuse_categorical(0, 4, qs, 4, random.Random(5))
        self.assertEqual(a, b)
        self.assertIn(a, range(4))

    def test_diffuse_low_noise_keeps_class(self):
        qs = [uniform_transition_matrix(4, 0.001) for _ in range(3)]
        # Almost no noise -> should nearly always stay class 2.
        rng = random.Random(0)
        stay = sum(
            diffuse_categorical(2, 4, qs, 3, rng) == 2 for _ in range(200)
        )
        self.assertGreater(stay, 190)


if __name__ == "__main__":
    unittest.main()
