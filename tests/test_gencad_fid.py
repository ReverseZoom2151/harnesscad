"""Tests for bench.gencad_fid (Frechet latent distance / FID)."""

import random
import unittest

from bench.gencad_fid import (
    mean_vector,
    covariance_matrix,
    jacobi_eigen,
    symmetric_sqrt,
    frechet_gaussian_distance,
    fid_score,
)


def _matmul(a, b):
    n, k, p = len(a), len(b), len(b[0])
    return [[sum(a[i][t] * b[t][j] for t in range(k)) for j in range(p)]
            for i in range(n)]


class MeanCovTest(unittest.TestCase):
    def test_mean(self):
        self.assertEqual(mean_vector([[1.0, 2.0], [3.0, 4.0]]), [2.0, 3.0])

    def test_covariance_unbiased(self):
        # cov of [[0,0],[2,0],[0,2]] with ddof=1.
        cov = covariance_matrix([[0.0, 0.0], [2.0, 0.0], [0.0, 2.0]], ddof=1)
        # Manual: mean=(2/3,2/3); var each dim = ((4/9)+(16/9)+(4/9))/2 = 24/18 = 4/3
        self.assertAlmostEqual(cov[0][0], 4.0 / 3.0, places=9)
        self.assertAlmostEqual(cov[1][1], 4.0 / 3.0, places=9)
        self.assertAlmostEqual(cov[0][1], cov[1][0], places=12)

    def test_single_sample_zero_cov(self):
        cov = covariance_matrix([[1.0, 2.0, 3.0]], ddof=1)
        self.assertEqual(cov, [[0.0] * 3 for _ in range(3)])


class JacobiTest(unittest.TestCase):
    def test_diagonal(self):
        vals, _ = jacobi_eigen([[3.0, 0.0], [0.0, 5.0]])
        self.assertAlmostEqual(sorted(vals)[0], 3.0, places=10)
        self.assertAlmostEqual(sorted(vals)[1], 5.0, places=10)

    def test_reconstruction(self):
        a = [[2.0, 1.0, 0.0], [1.0, 2.0, 1.0], [0.0, 1.0, 2.0]]
        vals, vecs = jacobi_eigen(a)
        # Rebuild Q diag(vals) Q^T and compare to a.
        n = 3
        q = [[vecs[k][i] for k in range(n)] for i in range(n)]  # columns = eigenvectors
        qt = [[q[j][i] for j in range(n)] for i in range(n)]
        diag = [[vals[i] if i == j else 0.0 for j in range(n)] for i in range(n)]
        recon = _matmul(_matmul(q, diag), qt)
        for i in range(n):
            for j in range(n):
                self.assertAlmostEqual(recon[i][j], a[i][j], places=8)

    def test_eigenvectors_orthonormal(self):
        a = [[4.0, 1.0], [1.0, 3.0]]
        vals, vecs = jacobi_eigen(a)
        # eigenvector dot products
        dot = sum(vecs[0][i] * vecs[1][i] for i in range(2))
        self.assertAlmostEqual(dot, 0.0, places=9)
        for v in vecs:
            self.assertAlmostEqual(sum(x * x for x in v), 1.0, places=9)


class SqrtTest(unittest.TestCase):
    def test_sqrt_squares_back(self):
        a = [[2.0, 0.5, 0.1], [0.5, 3.0, 0.2], [0.1, 0.2, 1.5]]
        r = symmetric_sqrt(a)
        rr = _matmul(r, r)
        for i in range(3):
            for j in range(3):
                self.assertAlmostEqual(rr[i][j], a[i][j], places=7)

    def test_identity_sqrt(self):
        r = symmetric_sqrt([[1.0, 0.0], [0.0, 1.0]])
        self.assertAlmostEqual(r[0][0], 1.0, places=10)
        self.assertAlmostEqual(r[1][1], 1.0, places=10)
        self.assertAlmostEqual(r[0][1], 0.0, places=10)


class FrechetTest(unittest.TestCase):
    def test_identical_distributions_zero(self):
        mu = [1.0, -2.0]
        sig = [[2.0, 0.3], [0.3, 1.0]]
        self.assertAlmostEqual(
            frechet_gaussian_distance(mu, sig, mu, sig), 0.0, places=8)

    def test_mean_shift_only(self):
        # Same covariance -> FID reduces to squared mean distance.
        sig = [[1.0, 0.0], [0.0, 1.0]]
        d = frechet_gaussian_distance([0.0, 0.0], sig, [3.0, 4.0], sig)
        self.assertAlmostEqual(d, 25.0, places=6)

    def test_diagonal_covariance_closed_form(self):
        # Diagonal covariances: tr((S G)^{1/2}) = sum sqrt(s_i g_i).
        s = [[4.0, 0.0], [0.0, 9.0]]
        g = [[1.0, 0.0], [0.0, 16.0]]
        mu = [0.0, 0.0]
        # FID = 0 + (4+9) + (1+16) - 2*(sqrt(4*1)+sqrt(9*16))
        #     = 13 + 17 - 2*(2 + 12) = 30 - 28 = 2
        d = frechet_gaussian_distance(mu, s, mu, g)
        self.assertAlmostEqual(d, 2.0, places=6)

    def test_non_negative_and_symmetric(self):
        s = [[3.0, 0.5], [0.5, 2.0]]
        g = [[1.5, -0.2], [-0.2, 2.5]]
        mua, mub = [1.0, 2.0], [0.0, 1.0]
        d1 = frechet_gaussian_distance(mua, s, mub, g)
        d2 = frechet_gaussian_distance(mub, g, mua, s)
        self.assertGreaterEqual(d1, 0.0)
        self.assertAlmostEqual(d1, d2, places=8)


class FidScoreTest(unittest.TestCase):
    def test_same_samples_near_zero(self):
        rng = random.Random(7)
        pts = [[rng.gauss(0, 1) for _ in range(3)] for _ in range(50)]
        self.assertAlmostEqual(fid_score(pts, pts), 0.0, places=6)

    def test_shifted_samples_positive(self):
        rng = random.Random(11)
        real = [[rng.gauss(0, 1) for _ in range(3)] for _ in range(200)]
        gen = [[x + 5.0 for x in p] for p in real]
        d = fid_score(real, gen)
        # Pure translation by 5 in each of 3 dims -> ~ 75.
        self.assertGreater(d, 60.0)
        self.assertLess(d, 90.0)

    def test_deterministic(self):
        rng = random.Random(3)
        a = [[rng.gauss(0, 1) for _ in range(4)] for _ in range(30)]
        b = [[rng.gauss(1, 2) for _ in range(4)] for _ in range(30)]
        self.assertEqual(fid_score(a, b), fid_score(a, b))


if __name__ == "__main__":
    unittest.main()
