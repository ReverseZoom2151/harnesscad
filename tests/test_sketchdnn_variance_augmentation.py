import unittest

from numeric.sketchdnn_variance_augmentation import (
    argmax_marginal,
    augmented_alpha,
    augmented_schedule,
    cosine_alpha_bar,
    gumbel_f,
    implied_discrete_keep,
    inverse_gumbel_f,
)


class TestCosine(unittest.TestCase):
    def test_cosine_starts_at_one(self):
        self.assertAlmostEqual(cosine_alpha_bar(0, 1000), 1.0, places=9)

    def test_cosine_monotone_decreasing(self):
        prev = cosine_alpha_bar(0, 100)
        for t in range(1, 101):
            cur = cosine_alpha_bar(t, 100)
            self.assertLessEqual(cur, prev + 1e-12)
            prev = cur

    def test_cosine_ends_near_zero(self):
        self.assertLess(cosine_alpha_bar(1000, 1000), 0.01)

    def test_cosine_range(self):
        with self.assertRaises(IndexError):
            cosine_alpha_bar(5, 4)


class TestGumbelF(unittest.TestCase):
    def test_f_zero_at_zero(self):
        self.assertAlmostEqual(gumbel_f(0.0, 5), 0.0)

    def test_f_negative_for_positive_x(self):
        # (1-x)/((D-1)x+1) < 1 for x>0 so log is negative.
        self.assertLess(gumbel_f(0.5, 5), 0.0)

    def test_f_needs_two_classes(self):
        with self.assertRaises(ValueError):
            gumbel_f(0.2, 1)

    def test_f_domain(self):
        with self.assertRaises(ValueError):
            gumbel_f(1.0, 5)


class TestAugmentedAlpha(unittest.TestCase):
    def test_alpha_zero_when_b_zero(self):
        self.assertAlmostEqual(augmented_alpha(0.0, 5), 0.0)

    def test_alpha_in_unit_interval(self):
        for b in (0.0, 0.1, 0.5, 0.9, 0.98):
            a = augmented_alpha(b, 5)
            self.assertGreaterEqual(a, 0.0)
            self.assertLessEqual(a, 1.0)

    def test_alpha_monotone_in_b(self):
        prev = -1.0
        for b in (0.0, 0.2, 0.4, 0.6, 0.8, 0.95):
            a = augmented_alpha(b, 5)
            self.assertGreater(a, prev)
            prev = a

    def test_alpha_at_b_equals_k(self):
        # When b_t = k, f(b_t)^2 == f(k)^2 so alpha = 1/2.
        self.assertAlmostEqual(augmented_alpha(0.99, 5, k=0.99), 0.5, places=9)

    def test_schedule_maps_all(self):
        sched = augmented_schedule([0.0, 0.5, 0.9], 5)
        self.assertEqual(len(sched), 3)
        self.assertLess(sched[0], sched[1])
        self.assertLess(sched[1], sched[2])


class TestImpliedKeep(unittest.TestCase):
    def test_symmetric_form(self):
        # b -> alpha -> b round trip via the shared closed form.
        k = 0.99
        d = 5
        b = 0.3
        a = augmented_alpha(b, d, k)
        b_back = implied_discrete_keep(a, d, k)
        self.assertAlmostEqual(b_back, b, places=6)

    def test_keep_in_unit_interval(self):
        for a in (0.0, 0.25, 0.75, 0.95):
            v = implied_discrete_keep(a, 5)
            self.assertGreaterEqual(v, 0.0)
            self.assertLessEqual(v, 1.0)

    def test_inverse_gumbel_roundtrip(self):
        for x in (0.1, 0.3, 0.6, 0.9):
            y = gumbel_f(x, 5)
            self.assertAlmostEqual(inverse_gumbel_f(y, 5), x, places=9)

    def test_keep_monotone_in_alpha(self):
        prev = -1.0
        for a in (0.0, 0.2, 0.5, 0.8, 0.95):
            v = implied_discrete_keep(a, 5)
            self.assertGreater(v, prev)
            prev = v


class TestArgmaxMarginal(unittest.TestCase):
    def test_marginal_is_simplex(self):
        m = argmax_marginal([1.0, 0.0, 0.0, 0.0], 0.4)
        self.assertAlmostEqual(sum(m), 1.0, places=9)
        self.assertTrue(all(p >= 0.0 for p in m))

    def test_b_one_is_clean(self):
        m = argmax_marginal([1.0, 0.0, 0.0, 0.0], 1.0)
        self.assertEqual(m, [1.0, 0.0, 0.0, 0.0])

    def test_b_zero_is_uniform(self):
        m = argmax_marginal([1.0, 0.0, 0.0, 0.0], 0.0)
        for p in m:
            self.assertAlmostEqual(p, 0.25)

    def test_bad_b(self):
        with self.assertRaises(ValueError):
            argmax_marginal([1.0, 0.0], 1.5)


if __name__ == "__main__":
    unittest.main()
