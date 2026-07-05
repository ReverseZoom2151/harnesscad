"""Tests for cfd_integral_noise (CFD variance-preserving noise transport)."""

import math
import random
import unittest

import numeric.cfd_integral_noise as n


class TestAggregation(unittest.TestCase):
    def test_empty_coverage_is_zero(self):
        self.assertEqual(n.aggregate_cell([]), 0.0)

    def test_sqrt_normalization(self):
        # sum / sqrt(n): four ones -> 4/2 = 2.
        self.assertAlmostEqual(n.aggregate_cell([1.0, 1.0, 1.0, 1.0]), 2.0, places=12)

    def test_preserves_unit_variance_across_coverage_sizes(self):
        ref = n.ReferenceNoise(seed=11)
        rng = random.Random(5)
        cell = 0
        # Build many query pixels, each covering a different number of cells.
        outputs = []
        for _ in range(6000):
            k = rng.randint(1, 12)
            cells = list(range(cell, cell + k))
            cell += k
            outputs.append(n.aggregate_cell(ref.values(cells)))
        var = n.sample_variance(outputs)
        self.assertAlmostEqual(var, 1.0, delta=0.06)


class TestCorrespondence(unittest.TestCase):
    def test_overlapping_coverage_is_correlated(self):
        ref = n.ReferenceNoise(seed=2)
        # Two views: pixel A and pixel B share reference cells -> correlated.
        shared = list(range(0, 8))
        cov_a = [shared[:] for _ in range(4000)]
        # Shift the shared block per sample so we get a distribution.
        outs_a, outs_b = [], []
        for i in range(4000):
            base = 100 + i * 20
            common = list(range(base, base + 6))
            a_only = list(range(base + 6, base + 8))
            b_only = list(range(base + 8, base + 10))
            outs_a.append(n.aggregate_cell(ref.values(common + a_only)))
            outs_b.append(n.aggregate_cell(ref.values(common + b_only)))
        corr = _pearson(outs_a, outs_b)
        # 6 shared of 8 -> strong positive correlation.
        self.assertGreater(corr, 0.5)

    def test_disjoint_coverage_is_uncorrelated(self):
        ref = n.ReferenceNoise(seed=9)
        outs_a, outs_b = [], []
        for i in range(4000):
            base = i * 20
            outs_a.append(n.aggregate_cell(ref.values(list(range(base, base + 6)))))
            outs_b.append(n.aggregate_cell(ref.values(list(range(base + 6, base + 12)))))
        corr = _pearson(outs_a, outs_b)
        self.assertAlmostEqual(corr, 0.0, delta=0.05)

    def test_transport_matches_per_pixel(self):
        ref = n.ReferenceNoise(seed=4)
        coverage = [[1, 2, 3], [4, 5], []]
        got = n.transport(ref, coverage)
        self.assertAlmostEqual(got[0], n.aggregate_cell(ref.values([1, 2, 3])))
        self.assertEqual(got[2], 0.0)


class TestNoiseInjection(unittest.TestCase):
    def test_gamma_zero_is_identity(self):
        rng = random.Random(0)
        eps = [0.3, -1.2, 4.0]
        self.assertEqual(n.inject_noise(eps, 0.0, rng), eps)

    def test_injection_preserves_unit_variance(self):
        rng = random.Random(123)
        dim = 4000
        eps = [rng.gauss(0.0, 1.0) for _ in range(dim)]
        for _ in range(200):
            eps = n.inject_noise(eps, 0.02, rng)
        var = n.sample_variance(eps)
        self.assertAlmostEqual(var, 1.0, delta=0.08)

    def test_invalid_gamma(self):
        with self.assertRaises(ValueError):
            n.inject_noise([1.0], 1.0, random.Random(0))


class TestGammaFormulas(unittest.TestCase):
    def test_gamma_from_beta_integral(self):
        self.assertAlmostEqual(n.gamma_from_beta_integral(0.0), 0.0, places=12)
        self.assertAlmostEqual(
            n.gamma_from_beta_integral(0.5), 1.0 - math.exp(-1.0), places=12
        )

    def test_ddpm_equivalent_gamma_matches_paper(self):
        # Paper: t~0.212, sig/alpha_t~0.60, sig/alpha_T~12.59, k=25000 -> ~0.00024.
        g = n.ddpm_equivalent_gamma(0.60, 12.59, 25000)
        self.assertAlmostEqual(g, 0.00024, delta=2e-5)

    def test_exact_and_approx_agree_for_large_k(self):
        exact = n.ddpm_equivalent_gamma(0.60, 12.59, 25000)
        approx = n.ddpm_equivalent_gamma_approx(0.60, 12.59, 25000)
        self.assertAlmostEqual(exact, approx, delta=1e-6)


def _pearson(a, b):
    na, nb = len(a), len(b)
    assert na == nb
    ma = sum(a) / na
    mb = sum(b) / nb
    cov = sum((x - ma) * (y - mb) for x, y in zip(a, b)) / na
    va = sum((x - ma) ** 2 for x in a) / na
    vb = sum((y - mb) ** 2 for y in b) / nb
    if va <= 0 or vb <= 0:
        return 0.0
    return cov / math.sqrt(va * vb)


if __name__ == "__main__":
    unittest.main()
