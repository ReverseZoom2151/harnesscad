"""Tests for verifiers.dlwheel_modal (paper 112 CAE modal surrogate)."""

import math
import unittest

from verifiers import dlwheel_modal as dm


class ModeTaxonomyTests(unittest.TestCase):
    def test_rigid_body_modes(self):
        for i in range(1, 7):
            self.assertTrue(dm.is_rigid_body_mode(i))
            self.assertEqual(dm.mode_label(i), "rigid body mode")
        self.assertFalse(dm.is_rigid_body_mode(7))

    def test_labels(self):
        self.assertEqual(dm.mode_label(7), "rim mode 1")
        self.assertEqual(dm.mode_label(8), "rim mode 1")
        self.assertEqual(dm.mode_label(9), "rim mode 2")
        self.assertEqual(dm.mode_label(11), "spoke lateral mode")
        self.assertEqual(dm.mode_label(14), "spoke bending mode")

    def test_lateral_mode_constant(self):
        self.assertEqual(dm.LATERAL_MODE_INDEX, 11)
        self.assertEqual(dm.mode_label(dm.LATERAL_MODE_INDEX), "spoke lateral mode")

    def test_unlabeled_elastic_mode(self):
        self.assertEqual(dm.mode_label(20), "elastic mode 20")

    def test_invalid_index(self):
        with self.assertRaises(ValueError):
            dm.mode_label(0)
        with self.assertRaises(ValueError):
            dm.is_rigid_body_mode(-1)


class FrequencyRelationTests(unittest.TestCase):
    def test_known_value(self):
        # k = (2*pi)^2, m = 1 -> f = 1
        f = dm.natural_frequency((2.0 * math.pi) ** 2, 1.0)
        self.assertAlmostEqual(f, 1.0, places=9)

    def test_roundtrip_stiffness(self):
        k = dm.stiffness_from_frequency(50.0, 2.0)
        f = dm.natural_frequency(k, 2.0)
        self.assertAlmostEqual(f, 50.0, places=9)

    def test_roundtrip_mass(self):
        m = dm.mass_from_frequency(30.0, 500.0)
        f = dm.natural_frequency(500.0, m)
        self.assertAlmostEqual(f, 30.0, places=9)

    def test_proportionality(self):
        # Doubling stiffness raises frequency by sqrt(2).
        f1 = dm.natural_frequency(100.0, 4.0)
        f2 = dm.natural_frequency(200.0, 4.0)
        self.assertAlmostEqual(f2 / f1, math.sqrt(2.0), places=9)
        # Doubling mass lowers frequency by sqrt(2).
        f3 = dm.natural_frequency(100.0, 8.0)
        self.assertAlmostEqual(f1 / f3, math.sqrt(2.0), places=9)

    def test_invalid_inputs(self):
        with self.assertRaises(ValueError):
            dm.natural_frequency(0.0, 1.0)
        with self.assertRaises(ValueError):
            dm.natural_frequency(1.0, -1.0)
        with self.assertRaises(ValueError):
            dm.stiffness_from_frequency(-1.0, 1.0)
        with self.assertRaises(ValueError):
            dm.mass_from_frequency(1.0, 0.0)


class EvaluationTests(unittest.TestCase):
    def test_evaluate_recovers_stiffness(self):
        ev = dm.evaluate_wheel(mass=3.0, frequency=40.0)
        expected_k = dm.stiffness_from_frequency(40.0, 3.0)
        self.assertAlmostEqual(ev.stiffness, expected_k, places=6)
        self.assertEqual(ev.mode_index, 11)
        self.assertEqual(ev.mode_label, "spoke lateral mode")
        self.assertIsNone(ev.meets_stiffness_floor)

    def test_stiffness_floor_pass_fail(self):
        ev = dm.evaluate_wheel(mass=3.0, frequency=40.0)
        floor_ok = ev.stiffness * 0.5
        floor_bad = ev.stiffness * 2.0
        pass_ev = dm.evaluate_wheel(3.0, 40.0, stiffness_floor=floor_ok)
        fail_ev = dm.evaluate_wheel(3.0, 40.0, stiffness_floor=floor_bad)
        self.assertTrue(pass_ev.meets_stiffness_floor)
        self.assertFalse(fail_ev.meets_stiffness_floor)

    def test_screen_and_rank(self):
        # Three concepts with increasing frequency -> increasing stiffness.
        concepts = [(2.0, 10.0), (2.0, 30.0), (2.0, 20.0)]
        floor = dm.stiffness_from_frequency(15.0, 2.0)
        evals = dm.screen_concepts(concepts, stiffness_floor=floor)
        self.assertEqual(len(evals), 3)
        passing = dm.passing_concepts(evals)
        # 30 Hz and 20 Hz pass; 10 Hz fails. Sorted stiffest-first.
        self.assertEqual(len(passing), 2)
        self.assertAlmostEqual(passing[0].frequency, 30.0)
        self.assertAlmostEqual(passing[1].frequency, 20.0)


if __name__ == "__main__":
    unittest.main()
