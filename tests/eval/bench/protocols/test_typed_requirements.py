"""Tests for eval.bench.protocols.typed_requirements."""

import unittest

from harnesscad.eval.bench.protocols.typed_requirements import (
    REQUIREMENT_CATEGORIES,
    Requirement,
    check,
    grade,
    mean_requirement_pass,
    strict_pass,
)


class RequirementTest(unittest.TestCase):
    def test_six_categories(self):
        self.assertEqual(len(REQUIREMENT_CATEGORIES), 6)

    def test_bad_category(self):
        with self.assertRaises(ValueError):
            Requirement("x", "thermal", "<=", 1.0)

    def test_bad_op(self):
        with self.assertRaises(ValueError):
            Requirement("x", "stress", "~", 1.0)


class CheckTest(unittest.TestCase):
    def test_le(self):
        self.assertTrue(check(Requirement("s", "stress", "<=", 250.0), 200.0))
        self.assertFalse(check(Requirement("s", "stress", "<=", 250.0), 300.0))

    def test_ge(self):
        self.assertTrue(check(Requirement("sf", "buckling", ">=", 2.0), 3.0))


class GradeTest(unittest.TestCase):
    def setUp(self):
        self.reqs = [
            Requirement("max_stress", "stress", "<=", 250.0),
            Requirement("max_disp", "displacement", "<=", 1.0),
            Requirement("min_clear", "clearance", ">=", 0.5),
        ]

    def test_partial_credit(self):
        m = {"max_stress": 200.0, "max_disp": 2.0, "min_clear": 0.6}
        self.assertAlmostEqual(mean_requirement_pass(self.reqs, m), 2 / 3)

    def test_missing_measurement_fails(self):
        m = {"max_stress": 200.0}
        self.assertAlmostEqual(mean_requirement_pass(self.reqs, m), 1 / 3)

    def test_strict_pass(self):
        m = {"max_stress": 200.0, "max_disp": 0.5, "min_clear": 0.6}
        self.assertTrue(strict_pass(self.reqs, m))

    def test_strict_fail(self):
        m = {"max_stress": 300.0, "max_disp": 0.5, "min_clear": 0.6}
        self.assertFalse(strict_pass(self.reqs, m))

    def test_grade_by_category(self):
        m = {"max_stress": 200.0, "max_disp": 2.0, "min_clear": 0.6}
        g = grade(self.reqs, m)
        self.assertIn("stress", g["by_category"])
        self.assertEqual(g["by_category"]["stress"]["passed"], 1)
        self.assertNotIn("modal", g["by_category"])  # no modal reqs

    def test_empty_raises(self):
        with self.assertRaises(ValueError):
            mean_requirement_pass([], {})


if __name__ == "__main__":
    unittest.main()
