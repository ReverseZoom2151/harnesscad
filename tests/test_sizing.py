"""Tests for the engineering-sizing front-of-pipeline (sizing.calc)."""

import math
import unittest

from sizing.calc import SizingCalc, default_formulas


class TestShaftDiameter(unittest.TestCase):
    def test_shaft_diameter_matches_textbook(self):
        calc = SizingCalc()
        # Solid round shaft, T = 1.0e6 N*mm, allowable shear = 40 MPa.
        req = {"formula": "shaft_diameter_torsion",
               "torque": 1.0e6, "allowable_shear": 40.0}
        res = calc.size(req)
        expected = (16.0 * 1.0e6 / (math.pi * 40.0)) ** (1.0 / 3.0)
        self.assertAlmostEqual(res["value"], expected, places=6)
        self.assertAlmostEqual(res["value"], 50.31, places=2)  # ~50.31 mm
        self.assertEqual(res["formula"], "shaft_diameter_torsion")
        self.assertEqual(res["dimension"], "diameter_mm")
        self.assertEqual(res["safety_factor"], 1.0)
        self.assertIn("torque", res["inputs"])
        self.assertTrue(res["citation"])

    def test_safety_factor_scales_demand(self):
        calc = SizingCalc()
        base = calc.size({"formula": "shaft_diameter_torsion",
                          "torque": 1.0e6, "allowable_shear": 40.0})
        sf2 = calc.size({"formula": "shaft_diameter_torsion",
                         "torque": 1.0e6, "allowable_shear": 40.0,
                         "safety_factor": 2.0})
        # d scales with (T)^(1/3): doubling the effective torque -> *2^(1/3).
        self.assertAlmostEqual(sf2["value"], base["value"] * 2.0 ** (1.0 / 3.0),
                               places=6)


class TestPlateThickness(unittest.TestCase):
    def test_beam_bending_thickness_matches_textbook(self):
        calc = SizingCalc()
        # Simply-supported strip: p = 0.1 MPa, span = 100 mm, sigma = 100 MPa.
        req = {"formula": "plate_thickness_bending",
               "pressure": 0.1, "span": 100.0, "allowable_stress": 100.0}
        res = calc.size(req)
        expected = 100.0 * math.sqrt(0.75 * 0.1 / 100.0)  # t = L*sqrt(0.75 p/sigma)
        self.assertAlmostEqual(res["value"], expected, places=6)
        self.assertAlmostEqual(res["value"], 2.7386, places=3)
        self.assertEqual(res["dimension"], "thickness_mm")

    def test_width_does_not_change_result(self):
        calc = SizingCalc()
        a = calc.size({"formula": "plate_thickness_bending",
                       "pressure": 0.2, "span": 80.0, "allowable_stress": 120.0})
        b = calc.size({"formula": "plate_thickness_bending", "width": 50.0,
                       "pressure": 0.2, "span": 80.0, "allowable_stress": 120.0})
        self.assertAlmostEqual(a["value"], b["value"], places=9)


class TestBoltCount(unittest.TestCase):
    def test_bolt_count_rounds_up(self):
        calc = SizingCalc()
        # F = 50 kN, M10 bolts (d=10), tau=100 MPa. Per bolt = pi*25*100 ~ 7854 N.
        res = calc.size({"formula": "bolt_count_shear",
                         "load": 50000.0, "bolt_diameter": 10.0,
                         "allowable_shear": 100.0})
        per_bolt = (math.pi * 100.0 / 4.0) * 100.0
        self.assertEqual(res["value"], math.ceil(50000.0 / per_bolt))
        self.assertEqual(res["dimension"], "count")


class TestGearTeeth(unittest.TestCase):
    def test_gear_teeth_from_ratio(self):
        calc = SizingCalc()
        # ratio 3, C = 100 mm, module 2 -> N1 = 2*100/(2*4) = 25 teeth.
        res = calc.gear_pair(ratio=3.0, center_distance=100.0, module=2.0)
        self.assertEqual(res["pinion_teeth"], 25.0)
        self.assertEqual(res["gear_teeth"], 75.0)


class TestRegistryAndErrors(unittest.TestCase):
    def test_registry_populated(self):
        self.assertIn("shaft_diameter_torsion", default_formulas())
        self.assertGreaterEqual(len(SizingCalc().names()), 4)

    def test_unknown_formula_raises(self):
        with self.assertRaises(KeyError):
            SizingCalc().size({"formula": "does_not_exist"})

    def test_missing_input_raises(self):
        with self.assertRaises(KeyError):
            SizingCalc().size({"formula": "shaft_diameter_torsion", "torque": 1.0})

    def test_deterministic(self):
        calc = SizingCalc()
        req = {"formula": "shaft_diameter_torsion",
               "torque": 1234.0, "allowable_shear": 55.0}
        self.assertEqual(calc.size(req)["value"], calc.size(req)["value"])


if __name__ == "__main__":
    unittest.main()
