import math
import random
import unittest

from harnesscad.domain.numeric.gaussian_softmax_diffusion import (
    argmax_class,
    gs_cumulative_sample,
    gs_forward_step,
    gs_posterior_logit_coeffs,
    gs_posterior_mean,
    gs_posterior_sigma,
    gs_reverse_step,
    gs_reverse_step_mean,
    label_smooth,
    safe_log,
    softmax,
)


def _is_simplex(v, places=9):
    return abs(sum(v) - 1.0) < 10 ** (-places) and all(x >= 0.0 for x in v)


class TestSoftmax(unittest.TestCase):
    def test_softmax_on_simplex(self):
        s = softmax([1.0, 2.0, 3.0])
        self.assertTrue(_is_simplex(s))

    def test_softmax_shift_invariance(self):
        a = softmax([1.0, 2.0, 3.0])
        b = softmax([1.0 + 5.0, 2.0 + 5.0, 3.0 + 5.0])
        for x, y in zip(a, b):
            self.assertAlmostEqual(x, y, places=12)

    def test_softmax_monotone(self):
        s = softmax([0.0, 1.0, 2.0])
        self.assertLess(s[0], s[1])
        self.assertLess(s[1], s[2])

    def test_softmax_uniform_input(self):
        s = softmax([3.0, 3.0, 3.0, 3.0])
        for p in s:
            self.assertAlmostEqual(p, 0.25)


class TestLabelSmooth(unittest.TestCase):
    def test_label_smooth_still_simplex(self):
        y = label_smooth([1.0, 0.0, 0.0, 0.0], k=0.99)
        self.assertTrue(_is_simplex(y))

    def test_label_smooth_keeps_argmax(self):
        y = label_smooth([0.0, 1.0, 0.0], k=0.99)
        self.assertEqual(argmax_class(y), 1)

    def test_label_smooth_no_zeros(self):
        y = label_smooth([1.0, 0.0, 0.0], k=0.99)
        self.assertTrue(all(v > 0.0 for v in y))

    def test_safe_log_finite_on_zero(self):
        vals = safe_log([0.0, 1.0])
        self.assertTrue(all(math.isfinite(v) for v in vals))

    def test_label_smooth_bad_k(self):
        with self.assertRaises(ValueError):
            label_smooth([1.0, 0.0], k=1.5)


class TestForward(unittest.TestCase):
    def test_forward_step_on_simplex(self):
        y0 = [1.0, 0.0, 0.0, 0.0]
        y1 = gs_forward_step(label_smooth(y0), 0.9, random.Random(0))
        self.assertTrue(_is_simplex(y1))

    def test_cumulative_sample_on_simplex(self):
        y0 = [0.0, 1.0, 0.0, 0.0]
        yt = gs_cumulative_sample(y0, 0.5, random.Random(1))
        self.assertTrue(_is_simplex(yt))

    def test_cumulative_deterministic(self):
        y0 = [1.0, 0.0, 0.0]
        a = gs_cumulative_sample(y0, 0.5, random.Random(3))
        b = gs_cumulative_sample(y0, 0.5, random.Random(3))
        self.assertEqual(a, b)

    def test_low_noise_preserves_label(self):
        # abar close to 1 -> argmax should almost always equal the true class.
        y0 = [0.0, 0.0, 1.0, 0.0]
        rng = random.Random(42)
        keep = sum(
            argmax_class(gs_cumulative_sample(y0, 0.999, rng)) == 2
            for _ in range(200)
        )
        self.assertGreater(keep, 190)

    def test_full_noise_argmax_roughly_uniform(self):
        # abar = 0 -> softmax(eps): argmax uniform over classes.
        rng = random.Random(7)
        counts = [0, 0, 0, 0]
        for _ in range(4000):
            yt = gs_cumulative_sample([1.0, 0.0, 0.0, 0.0], 0.0, rng)
            counts[argmax_class(yt)] += 1
        for c in counts:
            self.assertAlmostEqual(c / 4000, 0.25, delta=0.04)


class TestPosterior(unittest.TestCase):
    def test_sigma_nonnegative(self):
        s = gs_posterior_sigma(0.9, 0.5, 0.6)
        self.assertGreaterEqual(s, 0.0)

    def test_coeffs_match_ddpm_form(self):
        a_t, ab_t, ab_prev = 0.8, 0.4, 0.5
        c_t, c_0 = gs_posterior_logit_coeffs(a_t, ab_t, ab_prev)
        denom = 1.0 - ab_t
        self.assertAlmostEqual(c_t, math.sqrt(a_t) * (1 - ab_prev) / denom)
        self.assertAlmostEqual(c_0, math.sqrt(ab_prev) * (1 - a_t) / denom)

    def test_posterior_mean_is_interpolation(self):
        y_t = [0.7, 0.2, 0.1]
        y0 = [0.1, 0.8, 0.1]
        mean = gs_posterior_mean(y_t, y0, 0.8, 0.4, 0.5)
        c_t, c_0 = gs_posterior_logit_coeffs(0.8, 0.4, 0.5)
        lt, l0 = safe_log(y_t), safe_log(y0)
        for i in range(3):
            self.assertAlmostEqual(mean[i], c_t * lt[i] + c_0 * l0[i])

    def test_reverse_step_on_simplex(self):
        y_t = [0.4, 0.4, 0.2]
        y0 = [0.0, 1.0, 0.0]
        y_prev = gs_reverse_step(y_t, y0, 0.8, 0.4, 0.5, random.Random(2))
        self.assertTrue(_is_simplex(y_prev))

    def test_reverse_mean_pulls_toward_prediction(self):
        # With a confident clean prediction, the mean reverse step should
        # move argmax toward the predicted class.
        y_t = [0.34, 0.33, 0.33]
        y0 = [0.98, 0.01, 0.01]
        y_prev = gs_reverse_step_mean(y_t, y0, 0.5, 0.2, 0.6)
        self.assertEqual(argmax_class(y_prev), 0)

    def test_reverse_step_deterministic(self):
        y_t = [0.4, 0.3, 0.3]
        y0 = [0.0, 0.0, 1.0]
        a = gs_reverse_step(y_t, y0, 0.8, 0.4, 0.5, random.Random(9))
        b = gs_reverse_step(y_t, y0, 0.8, 0.4, 0.5, random.Random(9))
        self.assertEqual(a, b)


if __name__ == "__main__":
    unittest.main()
