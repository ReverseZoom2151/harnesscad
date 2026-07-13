"""Tests for cfd_sphere_square_map (CFD equal-area sphere<->square map)."""

import math
import unittest

from harnesscad.domain.numeric import cfd_sphere_square_map as m


class TestRoundTrip(unittest.TestCase):
    def test_inverse_recovers_angles(self):
        for theta in (0.1, 0.7, 1.4, 2.2, 3.0):
            for phi in (0.0, 0.3, 0.9, 1.4):
                xr, yr = m.sphere_to_square(theta, phi)
                th2, ph2 = m.square_to_sphere(xr, yr)
                self.assertAlmostEqual(theta, th2, places=9)
                self.assertAlmostEqual(phi, ph2, places=9)

    def test_bounds(self):
        xr, yr = m.sphere_to_square(math.pi, 0.4)
        self.assertLessEqual(xr, math.sqrt(2.0) + 1e-12)
        self.assertLessEqual(abs(yr), xr + 1e-12)

    def test_domain_validation(self):
        with self.assertRaises(ValueError):
            m.sphere_to_square(-0.1, 0.0)
        with self.assertRaises(ValueError):
            m.sphere_to_square(1.0, math.pi)  # phi out of [0, pi/2)


class TestAreaPreservation(unittest.TestCase):
    def test_jacobian_matches_analytic(self):
        # |d(xr,yr)/d(theta,phi)| == (2/pi) sin(theta) everywhere.
        for theta in (0.2, 0.6, 1.1, 1.9, 2.7):
            for phi in (0.05, 0.5, 1.0, 1.3):
                num = m.numerical_jacobian(theta, phi)
                ana = m.area_jacobian(theta)
                self.assertAlmostEqual(num, ana, places=4)

    def test_jacobian_independent_of_phi(self):
        vals = [m.area_jacobian(1.0)]
        # analytic form has no phi dependence; numeric should agree for all phi
        for phi in (0.1, 0.6, 1.2):
            self.assertAlmostEqual(m.numerical_jacobian(1.0, phi), vals[0], places=4)


class TestUniformity(unittest.TestCase):
    def test_uniform_directions_are_uniform_on_lune(self):
        dirs = m.uniform_octant_directions(20000, seed=7)
        self.assertEqual(len(dirs), 20000)
        # z = cos(theta) should be ~ U[-1, 1]: mean 0, variance 1/3.
        zs = [d[2] for d in dirs]
        mean_z = sum(zs) / len(zs)
        var_z = sum((z - mean_z) ** 2 for z in zs) / len(zs)
        self.assertAlmostEqual(mean_z, 0.0, delta=0.03)
        self.assertAlmostEqual(var_z, 1.0 / 3.0, delta=0.02)

    def test_all_directions_unit_length(self):
        for d in m.uniform_octant_directions(500, seed=1):
            norm = math.sqrt(d[0] ** 2 + d[1] ** 2 + d[2] ** 2)
            self.assertAlmostEqual(norm, 1.0, places=9)

    def test_deterministic_seed(self):
        a = m.uniform_octant_directions(100, seed=3)
        b = m.uniform_octant_directions(100, seed=3)
        self.assertEqual(a, b)


if __name__ == "__main__":
    unittest.main()
