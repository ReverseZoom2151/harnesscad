"""Tests for geometry.gaussiancad_splatting."""

from __future__ import annotations

import unittest
from math import isclose, pi, sqrt

from harnesscad.domain.geometry import gaussiancad_splatting as gs


class TestQuaternion(unittest.TestCase):
    def test_identity_quaternion_is_identity_matrix(self):
        r = gs.quaternion_to_matrix((1.0, 0.0, 0.0, 0.0))
        for i in range(3):
            for j in range(3):
                self.assertAlmostEqual(r[i][j], 1.0 if i == j else 0.0)

    def test_normalize_unnormalized_quaternion(self):
        w, x, y, z = gs.normalize_quaternion((0.0, 0.0, 0.0, 2.0))
        self.assertAlmostEqual(sqrt(w * w + x * x + y * y + z * z), 1.0)
        self.assertAlmostEqual(z, 1.0)

    def test_zero_quaternion_raises(self):
        with self.assertRaises(ValueError):
            gs.normalize_quaternion((0.0, 0.0, 0.0, 0.0))

    def test_90deg_z_rotation(self):
        # quaternion for 90deg about z: w=cos45, z=sin45
        c = sqrt(0.5)
        r = gs.quaternion_to_matrix((c, 0.0, 0.0, c))
        v = gs.mat3_vec(r, (1.0, 0.0, 0.0))
        self.assertAlmostEqual(v[0], 0.0, places=9)
        self.assertAlmostEqual(v[1], 1.0, places=9)

    def test_rotation_is_orthonormal_det_one(self):
        r = gs.quaternion_to_matrix((0.3, 0.2, -0.5, 0.7))
        self.assertAlmostEqual(gs.mat3_det(r), 1.0, places=9)
        rt = gs.mat3_transpose(r)
        prod = gs.mat3_mul(r, rt)
        for i in range(3):
            for j in range(3):
                self.assertAlmostEqual(prod[i][j], 1.0 if i == j else 0.0, places=9)


class TestLinalg(unittest.TestCase):
    def test_inverse_roundtrip(self):
        a = ((2.0, 0.3, 0.1), (0.3, 1.5, 0.2), (0.1, 0.2, 1.0))
        inv = gs.mat3_inverse(a)
        prod = gs.mat3_mul(a, inv)
        for i in range(3):
            for j in range(3):
                self.assertAlmostEqual(prod[i][j], 1.0 if i == j else 0.0, places=9)

    def test_singular_raises(self):
        with self.assertRaises(ValueError):
            gs.mat3_inverse(((1.0, 2.0, 3.0), (2.0, 4.0, 6.0), (0.0, 0.0, 1.0)))

    def test_mat2_inverse(self):
        a = ((4.0, 1.0), (1.0, 3.0))
        inv = gs.mat2_inverse(a)
        self.assertAlmostEqual(a[0][0] * inv[0][0] + a[0][1] * inv[1][0], 1.0)


class TestCovariance(unittest.TestCase):
    def test_isotropic_identity_rotation(self):
        cov = gs.covariance_from_scale_rotation((2.0, 3.0, 4.0), (1.0, 0.0, 0.0, 0.0))
        self.assertAlmostEqual(cov[0][0], 4.0)
        self.assertAlmostEqual(cov[1][1], 9.0)
        self.assertAlmostEqual(cov[2][2], 16.0)
        self.assertAlmostEqual(cov[0][1], 0.0)

    def test_covariance_symmetric_positive_det(self):
        cov = gs.covariance_from_scale_rotation((1.0, 2.0, 0.5), (0.3, 0.2, -0.5, 0.7))
        for i in range(3):
            for j in range(3):
                self.assertAlmostEqual(cov[i][j], cov[j][i], places=9)
        self.assertGreater(gs.mat3_det(cov), 0.0)

    def test_bad_scale_raises(self):
        with self.assertRaises(ValueError):
            gs.covariance_from_scale_rotation((1.0, -1.0, 1.0), (1.0, 0.0, 0.0, 0.0))


class TestEvaluate(unittest.TestCase):
    def test_peak_at_mean(self):
        cov = gs.covariance_from_scale_rotation((1.0, 1.0, 1.0), (1.0, 0.0, 0.0, 0.0))
        self.assertAlmostEqual(gs.evaluate_gaussian_3d((0, 0, 0), (0, 0, 0), cov), 1.0)

    def test_kernel_decreases_with_distance(self):
        cov = gs.covariance_from_scale_rotation((1.0, 1.0, 1.0), (1.0, 0.0, 0.0, 0.0))
        near = gs.evaluate_gaussian_3d((0.5, 0, 0), (0, 0, 0), cov)
        far = gs.evaluate_gaussian_3d((2.0, 0, 0), (0, 0, 0), cov)
        self.assertGreater(near, far)
        self.assertGreater(1.0, near)

    def test_mahalanobis_one_sigma(self):
        cov = gs.covariance_from_scale_rotation((2.0, 2.0, 2.0), (1.0, 0.0, 0.0, 0.0))
        # one std along x -> mahalanobis distance squared = 1
        m = gs.mahalanobis_sq((2.0, 0, 0), (0, 0, 0), cov)
        self.assertAlmostEqual(m, 1.0, places=9)

    def test_normalized_integral_constant(self):
        cov = gs.covariance_from_scale_rotation((1.0, 1.0, 1.0), (1.0, 0.0, 0.0, 0.0))
        peak = gs.evaluate_gaussian_3d((0, 0, 0), (0, 0, 0), cov, normalized=True)
        self.assertAlmostEqual(peak, 1.0 / ((2 * pi) ** 1.5), places=12)


class TestProjection(unittest.TestCase):
    def test_orthographic_front_marginal(self):
        cov = gs.covariance_from_scale_rotation((2.0, 3.0, 4.0), (1.0, 0.0, 0.0, 0.0))
        mu2d, cov2d = gs.project_gaussian_orthographic((1.0, 5.0, -2.0), cov, "front")
        # front: h=X, v=Z  -> mean picks x,z ; covariance picks (xx, xz; zx, zz)
        self.assertAlmostEqual(mu2d[0], 1.0)
        self.assertAlmostEqual(mu2d[1], -2.0)
        self.assertAlmostEqual(cov2d[0][0], 4.0)
        self.assertAlmostEqual(cov2d[1][1], 16.0)

    def test_side_view_mean(self):
        cov = gs.covariance_from_scale_rotation((1.0, 1.0, 1.0), (1.0, 0.0, 0.0, 0.0))
        mu2d, _ = gs.project_gaussian_orthographic((7.0, 3.0, 9.0), cov, "side")
        self.assertAlmostEqual(mu2d[0], 3.0)  # h=Y
        self.assertAlmostEqual(mu2d[1], 9.0)  # v=Z

    def test_unknown_view_raises(self):
        cov = gs.covariance_from_scale_rotation((1.0, 1.0, 1.0), (1.0, 0.0, 0.0, 0.0))
        with self.assertRaises(ValueError):
            gs.project_gaussian_orthographic((0, 0, 0), cov, "back")

    def test_general_projection_matches_selection(self):
        cov = gs.covariance_from_scale_rotation((2.0, 3.0, 4.0), (1.0, 0.0, 0.0, 0.0))
        p = ((1.0, 0.0, 0.0), (0.0, 0.0, 1.0))
        mu, c = gs.project_gaussian((1.0, 5.0, -2.0), cov, p)
        self.assertAlmostEqual(mu[0], 1.0)
        self.assertAlmostEqual(c[1][1], 16.0)


class TestFootprint(unittest.TestCase):
    def test_eigenvalues_diagonal(self):
        lo, hi = gs.covariance_eigenvalues_2d(((9.0, 0.0), (0.0, 4.0)))
        self.assertAlmostEqual(lo, 4.0)
        self.assertAlmostEqual(hi, 9.0)

    def test_footprint_three_sigma(self):
        bb = gs.footprint_bbox((10.0, 20.0), ((4.0, 0.0), (0.0, 9.0)), sigma=3.0)
        self.assertAlmostEqual(bb[0], 10.0 - 6.0)   # 3*sqrt(4)=6
        self.assertAlmostEqual(bb[1], 20.0 - 9.0)   # 3*sqrt(9)=9
        self.assertAlmostEqual(bb[2], 16.0)
        self.assertAlmostEqual(bb[3], 29.0)

    def test_footprint_bad_sigma(self):
        with self.assertRaises(ValueError):
            gs.footprint_bbox((0, 0), ((1.0, 0.0), (0.0, 1.0)), sigma=0.0)

    def test_evaluate_gaussian_2d_peak(self):
        self.assertAlmostEqual(
            gs.evaluate_gaussian_2d((1, 1), (1, 1), ((1.0, 0.0), (0.0, 1.0))), 1.0)


if __name__ == "__main__":
    unittest.main()
