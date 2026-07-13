"""Tests for bench.muse_functionality."""

import unittest

from harnesscad.eval.bench.protocols.functionality_score import (
    muse_functionality,
    parameters_within_omega,
    score_functional,
    score_robust,
    support_polygon,
)


class OmegaTests(unittest.TestCase):
    def test_all_in_range(self):
        ok, bad = parameters_within_omega(
            {"w": 400, "h": 450}, {"w": (300, 500), "h": (400, 480)})
        self.assertTrue(ok)
        self.assertEqual(bad, ())

    def test_out_of_range(self):
        ok, bad = parameters_within_omega({"w": 600}, {"w": (300, 500)})
        self.assertFalse(ok)
        self.assertEqual(bad, ("w",))

    def test_missing_required_param(self):
        ok, bad = parameters_within_omega({}, {"w": (0, 1)})
        self.assertFalse(ok)
        self.assertIn("w", bad)

    def test_inverted_range_raises(self):
        with self.assertRaises(ValueError):
            parameters_within_omega({"w": 1}, {"w": (5, 1)})


class FunctionalTests(unittest.TestCase):
    def test_full_function(self):
        design = {
            "structures": ["seat", "leg", "backrest"],
            "must_have": ["seat", "leg"],
            "nice_to_have": ["backrest"],
            "parameters": {"seat_h": 450},
            "parameter_ranges": {"seat_h": (400, 480)},
        }
        r = score_functional(design)
        self.assertEqual(r["functional"], 1)
        self.assertAlmostEqual(r["coverage"], 1.0)

    def test_missing_must_have_fails(self):
        r = score_functional({
            "structures": ["seat"], "must_have": ["seat", "leg"]})
        self.assertEqual(r["functional"], 0)
        self.assertIn("must_have_incomplete", r["reasons"])

    def test_param_out_of_omega_fails(self):
        r = score_functional({
            "structures": ["seat"], "must_have": ["seat"],
            "parameters": {"seat_h": 900},
            "parameter_ranges": {"seat_h": (400, 480)}})
        self.assertEqual(r["functional"], 0)
        self.assertIn("param_out_of_range:seat_h", r["reasons"])

    def test_nice_to_have_missing_still_passes(self):
        # must-have fully covered, nice-to-have absent -> still functional.
        r = score_functional({
            "structures": ["seat", "leg"],
            "must_have": ["seat", "leg"],
            "nice_to_have": ["armrest"]})
        self.assertEqual(r["functional"], 1)
        self.assertAlmostEqual(r["coverage"], 0.7)


class RobustTests(unittest.TestCase):
    def test_stable_four_legged(self):
        design = {
            "ground_contacts": [(0, 0), (10, 0), (10, 10), (0, 10)],
            "center_of_mass": (5, 5),
            "load_bearing_members": [{"name": "leg", "thickness": 30.0}],
        }
        r = score_robust(design)
        self.assertEqual(r["robust"], 1)
        self.assertAlmostEqual(r["support_area"], 100.0)

    def test_com_outside_support_tips(self):
        r = score_robust({
            "ground_contacts": [(0, 0), (10, 0), (10, 10), (0, 10)],
            "center_of_mass": (50, 50)})
        self.assertEqual(r["robust"], 0)
        self.assertIn("com_outside_support", r["reasons"])

    def test_too_few_contacts(self):
        r = score_robust({
            "ground_contacts": [(0, 0), (10, 0)],
            "center_of_mass": (5, 0)})
        self.assertEqual(r["robust"], 0)
        self.assertIn("insufficient_ground_contacts", r["reasons"])

    def test_thin_and_broken_members(self):
        r = score_robust({
            "ground_contacts": [(0, 0), (10, 0), (5, 10)],
            "center_of_mass": (5, 3),
            "load_bearing_members": [
                {"name": "spindle", "thickness": 0.2},
                {"name": "beam", "thickness": 5.0, "connected": False},
            ]})
        self.assertEqual(r["robust"], 0)
        self.assertIn("thin_member:spindle", r["reasons"])
        self.assertIn("broken_load_path:beam", r["reasons"])

    def test_com_on_boundary_is_inside(self):
        r = score_robust({
            "ground_contacts": [(0, 0), (10, 0), (10, 10), (0, 10)],
            "center_of_mass": (0, 5)})
        self.assertEqual(r["robust"], 1)

    def test_support_polygon_hull(self):
        poly = support_polygon([(0, 0), (10, 0), (10, 10), (0, 10), (5, 5)])
        self.assertEqual(len(poly), 4)


class PillarTests(unittest.TestCase):
    def test_full_pillar(self):
        design = {
            "structures": ["seat", "leg"], "must_have": ["seat", "leg"],
            "ground_contacts": [(0, 0), (10, 0), (10, 10), (0, 10)],
            "center_of_mass": (5, 5),
            "load_bearing_members": [{"name": "leg", "thickness": 30.0}],
        }
        r = muse_functionality(design)
        self.assertEqual(r["average"], 1.0)

    def test_half_pillar(self):
        design = {
            "structures": ["seat", "leg"], "must_have": ["seat", "leg"],
            "ground_contacts": [(0, 0), (1, 0)],  # unstable
            "center_of_mass": (5, 5)}
        r = muse_functionality(design)
        self.assertEqual(r["functional"], 1)
        self.assertEqual(r["robust"], 0)
        self.assertEqual(r["average"], 0.5)

    def test_unknown_kwarg_raises(self):
        with self.assertRaises(TypeError):
            muse_functionality({"structures": [], "must_have": []}, bogus=1)


if __name__ == "__main__":
    unittest.main()
