"""Tests for quality.cadclaw_beam_screening.

Deterministic, stdlib-only. Section properties and beam formulas checked
against hand computations; budget checks against threshold logic.
"""
import math
import unittest

from harnesscad.eval.quality.physics.cadclaw_beam_screening import (
    rectangular_section, simply_supported_deflection,
    motor_torque_budget, belt_tension, STANDARD_GRAVITY,
)


class SectionTest(unittest.TestCase):

    def test_square_symmetry(self):
        s = rectangular_section(0.02, 0.02)
        self.assertAlmostEqual(s.area, 4e-4)
        self.assertAlmostEqual(s.Iy, s.Iz)
        self.assertAlmostEqual(s.Sy, s.Sz)

    def test_rectangle_second_moment(self):
        # Iz = b h^3 / 12 with b=0.04, h=0.08
        s = rectangular_section(0.04, 0.08)
        self.assertAlmostEqual(s.Iz, 0.04 * 0.08 ** 3 / 12.0)
        self.assertAlmostEqual(s.Iy, 0.08 * 0.04 ** 3 / 12.0)
        self.assertAlmostEqual(s.Sz, 0.04 * 0.08 ** 2 / 6.0)
        # tall bar: strong axis (z) modulus is the larger one
        self.assertEqual(s.strong_axis_modulus, s.Sz)
        self.assertGreater(s.Iz, s.Iy)

    def test_torsion_positive_and_symmetric(self):
        s1 = rectangular_section(0.04, 0.08)
        s2 = rectangular_section(0.08, 0.04)
        self.assertGreater(s1.J, 0)
        self.assertAlmostEqual(s1.J, s2.J)  # J independent of orientation

    def test_thin_bar_torsion_limit(self):
        # For a very thin bar J -> (1/3) long * short^3
        long_s, short_s = 1.0, 0.001
        s = rectangular_section(short_s, long_s)
        approx = (1.0 / 3.0) * long_s * short_s ** 3
        self.assertAlmostEqual(s.J, approx, places=6)

    def test_rejects_nonpositive(self):
        with self.assertRaises(ValueError):
            rectangular_section(0.0, 0.05)


class DeflectionTest(unittest.TestCase):

    def test_point_load_formula(self):
        # PL^3/(48EI); with self weight zero
        span, load, I, E = 1.0, 10.0, 1e-6, 70e9
        r = simply_supported_deflection(span, load, I, beam_kg_per_m=0.0,
                                        E_Pa=E, gravity=STANDARD_GRAVITY)
        P = load * STANDARD_GRAVITY
        expect_mm = (P * span ** 3) / (48 * E * I) * 1000.0
        self.assertAlmostEqual(r.point_load_mm, expect_mm)
        self.assertAlmostEqual(r.self_weight_mm, 0.0)
        self.assertAlmostEqual(r.total_mm, expect_mm)

    def test_self_weight_term(self):
        span, I, E, wkg = 2.0, 18e-8, 68.9e9, 2.45
        r = simply_supported_deflection(span, 0.0, I, wkg, E_Pa=E)
        w = wkg * STANDARD_GRAVITY
        expect_mm = (5 * w * span ** 4) / (384 * E * I) * 1000.0
        self.assertAlmostEqual(r.self_weight_mm, expect_mm)
        self.assertAlmostEqual(r.point_load_mm, 0.0)

    def test_pass_fail_limit(self):
        stiff = simply_supported_deflection(0.5, 1.0, 5e-6, 1.0, limit_mm=0.5)
        self.assertTrue(stiff.passed)
        floppy = simply_supported_deflection(3.0, 20.0, 1e-8, 3.0, limit_mm=0.5)
        self.assertFalse(floppy.passed)
        self.assertGreater(floppy.total_mm, 0.5)

    def test_longer_span_more_deflection(self):
        a = simply_supported_deflection(1.0, 5.0, 1e-6, 2.0)
        b = simply_supported_deflection(2.0, 5.0, 1e-6, 2.0)
        self.assertGreater(b.total_mm, a.total_mm)

    def test_rejects_bad_input(self):
        with self.assertRaises(ValueError):
            simply_supported_deflection(0.0, 1.0, 1e-6, 1.0)
        with self.assertRaises(ValueError):
            simply_supported_deflection(1.0, 1.0, 0.0, 1.0)


class MotorBudgetTest(unittest.TestCase):

    def test_force_components(self):
        r = motor_torque_budget(mass_kg=10.0, n_motors=1,
                                pulley_radius_m=0.006, motor_torque_Nm=1.0,
                                accel_m_s2=1.0, friction_coeff=0.02,
                                gravity_axis=False)
        self.assertAlmostEqual(r.force_accel_N, 10.0)
        self.assertAlmostEqual(r.force_friction_N, 10.0 * STANDARD_GRAVITY * 0.02)
        self.assertAlmostEqual(r.force_gravity_N, 0.0)
        self.assertAlmostEqual(r.force_total_N,
                               r.force_accel_N + r.force_friction_N)

    def test_gravity_axis_adds_weight(self):
        flat = motor_torque_budget(5.0, 1, 0.006, 1.0, gravity_axis=False)
        vert = motor_torque_budget(5.0, 1, 0.006, 1.0, gravity_axis=True)
        self.assertGreater(vert.force_total_N, flat.force_total_N)
        self.assertAlmostEqual(vert.force_gravity_N, 5.0 * STANDARD_GRAVITY)

    def test_torque_and_safety(self):
        r = motor_torque_budget(mass_kg=2.0, n_motors=1, pulley_radius_m=0.006,
                                motor_torque_Nm=1.0, accel_m_s2=0.5,
                                friction_coeff=0.01, belt_efficiency=0.95,
                                torque_derating=0.7, min_safety=1.5)
        F_total = 2.0 * 0.5 + 2.0 * STANDARD_GRAVITY * 0.01
        T_req = F_total * 0.006 / 0.95
        self.assertAlmostEqual(r.torque_required_Nm, T_req)
        self.assertAlmostEqual(r.torque_available_Nm, 0.7)
        self.assertAlmostEqual(r.safety_factor, 0.7 / T_req)
        self.assertTrue(r.passed)

    def test_two_motors_share_load(self):
        one = motor_torque_budget(10.0, 1, 0.006, 1.0)
        two = motor_torque_budget(10.0, 2, 0.006, 1.0)
        self.assertAlmostEqual(two.torque_required_Nm,
                               one.torque_required_Nm / 2.0)

    def test_rejects_bad_input(self):
        with self.assertRaises(ValueError):
            motor_torque_budget(1.0, 0, 0.006, 1.0)


class BeltTensionTest(unittest.TestCase):

    def test_safety_factors(self):
        r = belt_tension(force_N=100.0, n_belts=1,
                         breaking_N=900.0, working_N=450.0)
        self.assertAlmostEqual(r.tension_per_belt_N, 100.0)
        self.assertAlmostEqual(r.safety_to_break, 9.0)
        self.assertAlmostEqual(r.safety_to_working, 4.5)
        self.assertTrue(r.passed)

    def test_multiple_belts_reduce_tension(self):
        r = belt_tension(force_N=200.0, n_belts=2)
        self.assertAlmostEqual(r.tension_per_belt_N, 100.0)

    def test_fail_when_overloaded(self):
        r = belt_tension(force_N=400.0, working_N=450.0, min_safety=2.0)
        self.assertFalse(r.passed)  # 450/400 = 1.125 < 2.0

    def test_zero_force_infinite_safety(self):
        r = belt_tension(force_N=0.0)
        self.assertTrue(math.isinf(r.safety_to_working))
        self.assertTrue(r.passed)

    def test_rejects_bad_input(self):
        with self.assertRaises(ValueError):
            belt_tension(100.0, n_belts=0)


if __name__ == "__main__":
    unittest.main()
