"""Tests for hypervolume indicator and constraint metrics (BikeBench)."""

import unittest

from harnesscad.eval.quality.reward import hypervolume as hv


class HypervolumeTest(unittest.TestCase):
    def test_single_point(self):
        # point (1,1), ref (2,2): dominated box is 1x1 = 1.
        self.assertAlmostEqual(hv.hypervolume([(1.0, 1.0)], (2.0, 2.0)), 1.0)

    def test_two_point_union(self):
        # points (1,2) and (2,1), ref (3,3).
        # boxes [1,3]x[2,3]=2 and [2,3]x[1,3]=2 ; overlap [2,3]x[2,3]=1 ; union=3.
        self.assertAlmostEqual(hv.hypervolume([(1, 2), (2, 1)], (3, 3)), 3.0)

    def test_dominated_point_ignored(self):
        # (2,2) is dominated by (1,1); hv same as single point.
        self.assertAlmostEqual(hv.hypervolume([(1, 1), (2, 2)], (3, 3)), 4.0)

    def test_point_outside_reference(self):
        self.assertAlmostEqual(hv.hypervolume([(5, 5)], (3, 3)), 0.0)

    def test_empty(self):
        self.assertEqual(hv.hypervolume([], (1, 1)), 0.0)

    def test_three_dimensions(self):
        # single point (0,0,0) ref (2,2,2) -> volume 8.
        self.assertAlmostEqual(hv.hypervolume([(0, 0, 0)], (2, 2, 2)), 8.0)


class NonDominatedTest(unittest.TestCase):
    def test_front(self):
        front = hv.non_dominated([(1, 2), (2, 1), (2, 2)])
        self.assertEqual(set(front), {(1.0, 2.0), (2.0, 1.0)})


class ConstraintTest(unittest.TestCase):
    def setUp(self):
        self.checks = [
            lambda d: d["w"] > 0,
            lambda d: d["h"] > 0,
            lambda d: d["w"] < 100,
        ]

    def test_mean_violation(self):
        designs = [{"w": 5, "h": 5}, {"w": -1, "h": 5}, {"w": 200, "h": -1}]
        # violations: 0, 1, 2 -> mean 1.0
        self.assertAlmostEqual(hv.mean_constraint_violation(designs, self.checks), 1.0)

    def test_satisfaction_rate(self):
        designs = [{"w": 5, "h": 5}, {"w": -1, "h": 5}]
        # satisfied pairs: 3 + 2 = 5 of 6
        self.assertAlmostEqual(hv.constraint_satisfaction_rate(designs, self.checks), 5 / 6)

    def test_feasible_subset(self):
        designs = [{"w": 5, "h": 5}, {"w": -1, "h": 5}]
        self.assertEqual(hv.feasible_designs(designs, self.checks), [{"w": 5, "h": 5}])


class SuiteTest(unittest.TestCase):
    def test_suite_reports_named_violations(self):
        suite = hv.ConstraintSuite({
            "positive_w": lambda d: d["w"] > 0,
            "triangle": lambda d: hv.triangle_inequality_ok(d["a"], d["b"], d["c"]),
        })
        good = {"w": 1, "a": 3, "b": 4, "c": 5}
        bad = {"w": -1, "a": 1, "b": 1, "c": 5}
        self.assertEqual(suite.violations(good), ())
        self.assertTrue(suite.is_feasible(good))
        self.assertEqual(set(suite.violations(bad)), {"positive_w", "triangle"})


class ClosedFormChecksTest(unittest.TestCase):
    def test_positive_dimensions(self):
        self.assertTrue(hv.positive_dimensions([1.0, 2.0, 0.0]))
        self.assertFalse(hv.positive_dimensions([1.0, -0.1]))

    def test_triangle_inequality(self):
        self.assertTrue(hv.triangle_inequality_ok(3, 4, 5))
        self.assertFalse(hv.triangle_inequality_ok(1, 1, 5))


if __name__ == "__main__":
    unittest.main()
