"""Tests for geometry.neurcad_developability (zero-Gaussian developability)."""
import math
import unittest

from harnesscad.domain.geometry.neurcad_developability import (
    point_developability_defect, developability_energy,
    developability_energy_squared,
    double_trough_coeffs, double_trough, double_trough_deriv,
    developability_energy_double_trough,
    annealing_factor, annealed_developability_weight,
    surface_projection,
)


def sphere_grad_hess(p, r):
    # SDF |x|-r: g = p/r (unit), H = (I - nn^T)/r.  Gaussian K = 1/r^2.
    n = tuple(c / r for c in p)
    H = tuple(tuple(((1.0 if i == j else 0.0) - n[i] * n[j]) / r
                    for j in range(3)) for i in range(3))
    return n, H


# Cylinder radius R about z: g=(1,0,0), one tangential curvature 1/R, axial 0.
CYL = ((1.0, 0.0, 0.0),
       ((0.0, 0.0, 0.0), (0.0, 1.0 / 3.0, 0.0), (0.0, 0.0, 0.0)))
# Cone-like developable patch: parabolic (one principal curvature 0).
CONE = ((1.0, 0.0, 0.0),
        ((0.0, 0.0, 0.0), (0.0, 0.0, 0.0), (0.0, 0.0, 0.9)))
# Plane z=0, SDF f=z.
PLANE = ((0.0, 0.0, 1.0), ((0.0, 0.0, 0.0),) * 3)


class DevelopabilityEnergyTests(unittest.TestCase):
    def test_cylinder_developable_zero(self):
        self.assertAlmostEqual(point_developability_defect(*CYL), 0.0, places=12)

    def test_cone_developable_zero(self):
        self.assertAlmostEqual(point_developability_defect(*CONE), 0.0, places=12)

    def test_plane_developable_zero(self):
        self.assertAlmostEqual(point_developability_defect(*PLANE), 0.0, places=12)

    def test_sphere_nonzero_matches_analytic(self):
        r = 2.5
        g, H = sphere_grad_hess((r, 0.0, 0.0), r)
        self.assertAlmostEqual(point_developability_defect(g, H),
                               1.0 / (r * r), places=10)

    def test_energy_zero_on_developable_batch(self):
        samples = [CYL, CONE, PLANE]
        self.assertAlmostEqual(developability_energy(samples), 0.0, places=12)
        self.assertAlmostEqual(developability_energy_squared(samples), 0.0,
                               places=12)

    def test_energy_positive_on_sphere(self):
        r = 2.0
        s = [sphere_grad_hess((r, 0, 0), r), sphere_grad_hess((0, r, 0), r)]
        self.assertAlmostEqual(developability_energy(s), 1.0 / (r * r), places=10)

    def test_empty_raises(self):
        with self.assertRaises(ValueError):
            developability_energy([])
        with self.assertRaises(ValueError):
            developability_energy_squared([])


class DoubleTroughTests(unittest.TestCase):
    def test_interpolation_conditions(self):
        # Eq. 8: DT(0)=0, DT(pi/4)=pi/4, DT'(pi/4)=0, DT(pi/2)=a, DT'(pi/2)=0.
        a = 0.25
        self.assertAlmostEqual(double_trough(0.0, a), 0.0, places=12)
        self.assertAlmostEqual(double_trough(math.pi / 4, a), math.pi / 4, places=9)
        self.assertAlmostEqual(double_trough_deriv(math.pi / 4, a), 0.0, places=9)
        self.assertAlmostEqual(double_trough(math.pi / 2, a), a, places=9)
        self.assertAlmostEqual(double_trough_deriv(math.pi / 2, a), 0.0, places=9)

    def test_peak_above_troughs(self):
        # Peak at pi/4 exceeds troughs at 0 and pi/2 (default a=1/4).
        self.assertGreater(double_trough(math.pi / 4), double_trough(0.0))
        self.assertGreater(double_trough(math.pi / 4), double_trough(math.pi / 2))

    def test_custom_trough_height(self):
        self.assertAlmostEqual(double_trough(math.pi / 2, a=0.1), 0.1, places=9)

    def test_coeffs_c0_implicit_zero(self):
        # DT built with no constant term -> DT(0) exactly 0.
        c1, c2, c3, c4 = double_trough_coeffs()
        self.assertEqual(0.0 * c1 + 0.0 * c2 + 0.0 * c3 + 0.0 * c4, 0.0)

    def test_dt_energy_tolerates_corner(self):
        # A near-pi/2 "corner" curvature gives low DT energy vs a mid-range one.
        corner = ((1.0, 0.0, 0.0),
                  ((0.0, 0.0, 0.0),
                   (0.0, math.sqrt(math.pi / 2), 0.0),
                   (0.0, 0.0, math.sqrt(math.pi / 2))))  # K = pi/2
        self.assertAlmostEqual(point_developability_defect(*corner),
                               math.pi / 2, places=9)
        e = developability_energy_double_trough([corner])
        self.assertLess(e, double_trough(math.pi / 4))

    def test_dt_energy_zero_on_developable(self):
        self.assertAlmostEqual(
            developability_energy_double_trough([CYL, PLANE]), 0.0, places=12)

    def test_dt_empty_raises(self):
        with self.assertRaises(ValueError):
            developability_energy_double_trough([])


class AnnealingTests(unittest.TestCase):
    def test_hold_phase(self):
        self.assertEqual(annealing_factor(0.0), 1.0)
        self.assertEqual(annealing_factor(0.1), 1.0)
        self.assertEqual(annealing_factor(0.2), 1.0)

    def test_decay_to_mid(self):
        self.assertAlmostEqual(annealing_factor(0.5), 1e-4, places=12)
        mid = annealing_factor(0.35)  # halfway through 0.2..0.5
        self.assertAlmostEqual(mid, (1.0 + 1e-4) / 2.0, places=9)

    def test_drop_to_zero(self):
        self.assertAlmostEqual(annealing_factor(1.0), 0.0, places=12)
        self.assertGreater(annealing_factor(0.75), 0.0)
        self.assertLess(annealing_factor(0.75), 1e-4)

    def test_monotone_non_increasing(self):
        prev = annealing_factor(0.0)
        p = 0.0
        while p <= 1.0:
            cur = annealing_factor(p)
            self.assertLessEqual(cur, prev + 1e-12)
            prev = cur
            p += 0.02

    def test_clamps_out_of_range(self):
        self.assertEqual(annealing_factor(-0.5), 1.0)
        self.assertAlmostEqual(annealing_factor(1.5), 0.0, places=12)

    def test_annealed_weight(self):
        self.assertAlmostEqual(annealed_developability_weight(0.0, 10.0), 10.0,
                               places=9)
        self.assertAlmostEqual(annealed_developability_weight(1.0, 10.0), 0.0,
                               places=9)


class SurfaceProjectionTests(unittest.TestCase):
    def test_projects_onto_surface_true_sdf(self):
        # True SDF: |grad|=1.  Point 0.3 outside plane z=0 along +z.
        x = (1.0, 2.0, 0.3)
        g = (0.0, 0.0, 1.0)
        xp = surface_projection(x, g, 0.3)
        self.assertAlmostEqual(xp[2], 0.0, places=12)
        self.assertEqual((xp[0], xp[1]), (1.0, 2.0))

    def test_projection_moves_opposite_gradient_for_positive_value(self):
        x = (5.0, 0.0, 0.0)
        g = (2.0, 0.0, 0.0)  # non-unit gradient
        xp = surface_projection(x, g, 4.0)  # value/|g| = 2 step back
        self.assertAlmostEqual(xp[0], 3.0, places=12)

    def test_zero_gradient_raises(self):
        with self.assertRaises(ValueError):
            surface_projection((0, 0, 0), (0, 0, 0), 1.0)


if __name__ == "__main__":
    unittest.main()
