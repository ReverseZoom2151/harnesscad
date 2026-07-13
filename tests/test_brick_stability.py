"""Tests for verifiers.brick_stability (static-equilibrium stability LP)."""

import unittest

from harnesscad.domain.geometry.assembly.brick_structure import Brick, BrickStructure
from harnesscad.eval.verifiers.brick_stability import (
    analyze_stability,
    is_stable,
    solve_lp,
)


def struct(bricks):
    return BrickStructure.from_bricks(bricks)


class TestLPSolver(unittest.TestCase):
    def test_equality_objective(self):
        feasible, x, obj = solve_lp([1.0, 1.0], [[1.0, 1.0]], [3.0])
        self.assertTrue(feasible)
        self.assertAlmostEqual(obj, 3.0, places=5)
        self.assertAlmostEqual(x[0] + x[1], 3.0, places=5)

    def test_minimises(self):
        # min 3 x0 + x1 s.t. x0 + x1 = 3  ->  put all mass on x1
        feasible, x, obj = solve_lp([3.0, 1.0], [[1.0, 1.0]], [3.0])
        self.assertTrue(feasible)
        self.assertAlmostEqual(obj, 3.0, places=5)
        self.assertAlmostEqual(x[1], 3.0, places=5)
        self.assertAlmostEqual(x[0], 0.0, places=5)

    def test_greater_equal_via_slack(self):
        # min 2 x + 3 y s.t. x + y - s = 4, all >= 0  ->  x = 4
        feasible, x, obj = solve_lp([2.0, 3.0, 0.0], [[1.0, 1.0, -1.0]], [4.0])
        self.assertTrue(feasible)
        self.assertAlmostEqual(obj, 8.0, places=5)

    def test_feasible_simple(self):
        # x = 2 (single non-negative variable pinned by one equality)
        feasible, x, _ = solve_lp([1.0], [[1.0]], [2.0])
        self.assertTrue(feasible)
        self.assertAlmostEqual(x[0], 2.0, places=5)

    def test_infeasible(self):
        # x0 = 1 and x0 = 2 simultaneously is infeasible
        feasible, _, _ = solve_lp([0.0], [[1.0], [1.0]], [1.0, 2.0])
        self.assertFalse(feasible)


class TestStability(unittest.TestCase):
    def test_single_brick_on_baseplate(self):
        r = analyze_stability(struct([Brick(2, 2, 0, 0, 0)]))
        self.assertTrue(r.stable)
        self.assertAlmostEqual(r.min_score, 1.0, places=5)

    def test_clean_stack_is_stable(self):
        s = struct([Brick(2, 2, 0, 0, 0), Brick(2, 2, 0, 0, 1), Brick(2, 2, 0, 0, 2)])
        r = analyze_stability(s)
        self.assertTrue(r.stable)
        self.assertTrue(is_stable(s))
        for sc in r.scores:
            self.assertAlmostEqual(sc, 1.0, places=5)

    def test_floating_brick_has_no_equilibrium(self):
        s = struct([Brick(2, 2, 0, 0, 0), Brick(2, 2, 0, 0, 2)])
        r = analyze_stability(s)
        self.assertFalse(r.feasible)
        self.assertFalse(r.stable)
        self.assertIn(1, r.unstable_indices())

    def test_extreme_cantilever_is_unstable(self):
        # a long 8x1 brick held by a single 1x1 stud far from its centre of mass
        s = struct([Brick(1, 1, 0, 0, 0), Brick(8, 1, 0, 0, 1)])
        r = analyze_stability(s)
        self.assertTrue(r.feasible)  # equilibrium exists via clutch/friction ...
        self.assertFalse(r.stable)  # ... but required drag exceeds capacity
        self.assertAlmostEqual(r.min_score, 0.0, places=5)

    def test_small_overhang_supported(self):
        s = struct([Brick(4, 2, 0, 0, 0), Brick(2, 2, 3, 0, 1)])
        self.assertTrue(analyze_stability(s).stable)

    def test_friction_capacity_monotonic(self):
        # marginal overhang: score increases with friction capacity
        s = struct([Brick(2, 1, 0, 0, 0), Brick(6, 1, 0, 0, 1)])
        low = analyze_stability(s, friction_capacity=2.0)
        mid = analyze_stability(s, friction_capacity=4.0)
        high = analyze_stability(s, friction_capacity=16.0)
        self.assertFalse(low.stable)
        self.assertTrue(mid.stable)
        self.assertLess(mid.min_score, high.min_score)
        self.assertTrue(0.0 < mid.min_score < 1.0)

    def test_deterministic(self):
        s = struct([Brick(2, 1, 0, 0, 0), Brick(6, 1, 0, 0, 1)])
        self.assertEqual(analyze_stability(s).scores, analyze_stability(s).scores)

    def test_empty_structure_is_stable(self):
        r = analyze_stability(struct([]))
        self.assertTrue(r.stable)
        self.assertEqual(r.scores, ())

    def test_scores_in_unit_interval(self):
        s = struct([Brick(2, 2, 0, 0, 0), Brick(4, 2, 1, 0, 1), Brick(2, 2, 3, 0, 2)])
        for sc in analyze_stability(s).scores:
            self.assertGreaterEqual(sc, 0.0)
            self.assertLessEqual(sc, 1.0)


if __name__ == "__main__":
    unittest.main()
