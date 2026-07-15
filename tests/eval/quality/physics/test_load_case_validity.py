"""Tests for eval.quality.physics.load_case_validity."""

import unittest

from harnesscad.eval.quality.physics.load_case_validity import (
    LoadCase,
    design_objective,
    functional_validity,
    in_safety_band,
)


def _lc():
    return LoadCase(
        fixed_supports=((0.0, 0.0, 0.0),),
        forces=(((1.0, 0.0, 0.0), (0.0, 0.0, -100.0)),),
        design_space=((0.0, 0.0, 0.0), (10.0, 10.0, 10.0)),
    )


class LoadCaseTest(unittest.TestCase):
    def test_envelope_volume(self):
        self.assertAlmostEqual(_lc().envelope_volume(), 1000.0)

    def test_bad_design_space(self):
        with self.assertRaises(ValueError):
            LoadCase(((0, 0, 0),), (((0, 0, 0), (0, 0, -1)),),
                     ((0, 0, 0), (0, 0, 0)))

    def test_no_forces(self):
        with self.assertRaises(ValueError):
            LoadCase(((0, 0, 0),), (), ((0, 0, 0), (1, 1, 1)))


class SafetyBandTest(unittest.TestCase):
    def test_in_band(self):
        self.assertTrue(in_safety_band(3.0))
        self.assertTrue(in_safety_band(2.0))
        self.assertTrue(in_safety_band(5.0))

    def test_out_of_band(self):
        self.assertFalse(in_safety_band(1.5))
        self.assertFalse(in_safety_band(6.0))


class ValidityTest(unittest.TestCase):
    def test_full_in_band(self):
        self.assertEqual(functional_validity(3.5), 1.0)

    def test_under_designed_decays(self):
        self.assertAlmostEqual(functional_validity(1.0), 0.5)  # 1.0/2.0

    def test_over_designed_decays(self):
        self.assertAlmostEqual(functional_validity(7.5), 0.5)  # (10-7.5)/5

    def test_failure_zero(self):
        self.assertEqual(functional_validity(0.0), 0.0)

    def test_negative_raises(self):
        with self.assertRaises(ValueError):
            functional_validity(-1.0)


class ObjectiveTest(unittest.TestCase):
    def test_invalid_gets_zero(self):
        self.assertEqual(design_objective(0.0, 100.0, _lc()), 0.0)

    def test_lighter_valid_scores_higher(self):
        light = design_objective(3.0, 100.0, _lc())
        heavy = design_objective(3.0, 900.0, _lc())
        self.assertGreater(light, heavy)

    def test_bounded(self):
        s = design_objective(3.0, 10.0, _lc())
        self.assertGreaterEqual(s, 0.0)
        self.assertLessEqual(s, 1.0)


if __name__ == "__main__":
    unittest.main()
