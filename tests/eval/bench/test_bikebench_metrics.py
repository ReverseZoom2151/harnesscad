"""Tests for eval.bench.bikebench_metrics (BikeBench taxonomy + metrics)."""

import math
import unittest

from harnesscad.eval.bench.bikebench_metrics import (
    REQUIREMENTS,
    apply_standardiser,
    average_constraint_violation,
    average_novelty,
    conditional_names,
    constraint_names,
    constraint_violation_rate,
    dpp_diversity,
    evaluators,
    mean_constraint_violation_magnitude,
    mean_objective,
    min_objective,
    mmd_rbf,
    objective_names,
    rbf_gamma_median,
    requirements_for,
    standardiser,
)


class TestTaxonomy(unittest.TestCase):
    def test_evaluators(self):
        self.assertEqual(
            evaluators(),
            ["Aero", "FrameValidity", "Structural", "Aesthetics", "Ergonomics"],
        )

    def test_objective_vs_constraint_counts(self):
        # 1 aero + 4 structural + 1 aesthetics + 3 ergo angle-errors = 9 objectives
        self.assertEqual(len(objective_names()), 9)
        # 1 validity + 2 structural safety + 6 ergo fit = 9 constraints
        self.assertEqual(len(constraint_names()), 9)
        self.assertEqual(len(REQUIREMENTS), 18)

    def test_drag_is_conditional_objective(self):
        req = next(r for r in REQUIREMENTS if r.name == "Drag Force (N)")
        self.assertTrue(req.is_objective)
        self.assertTrue(req.conditional)

    def test_structural_group(self):
        names = [r.name for r in requirements_for("Structural")]
        self.assertIn("Mass (kg)", names)
        self.assertIn("Planar Safety Factor", names)

    def test_conditional_names_nonempty(self):
        self.assertIn("Cosine Distance To Text", conditional_names())


class TestStandardiser(unittest.TestCase):
    def test_zscore(self):
        ref = [[0.0], [2.0]]
        mean, std = standardiser(ref)
        self.assertEqual(mean, [1.0])
        self.assertEqual(std, [1.0])  # population std of {0,2} = 1
        out = apply_standardiser([[0.0], [2.0]], mean, std)
        self.assertEqual(out, [[-1.0], [1.0]])

    def test_zero_variance_column(self):
        mean, std = standardiser([[5.0], [5.0]])
        self.assertEqual(std, [1.0])

    def test_empty_raises(self):
        with self.assertRaises(ValueError):
            standardiser([])


class TestRBFandMMD(unittest.TestCase):
    def test_gamma_positive(self):
        ref = [[0.0], [1.0], [2.0]]
        g = rbf_gamma_median(ref)
        self.assertGreater(g, 0.0)

    def test_mmd_identical_sets_is_zero(self):
        s = [[0.0], [1.0], [2.0]]
        g = rbf_gamma_median(s)
        self.assertAlmostEqual(mmd_rbf(s, s, g), 0.0, places=9)

    def test_mmd_disjoint_positive(self):
        ref = [[0.0], [0.1]]
        gen = [[10.0], [10.1]]
        g = rbf_gamma_median(ref)
        self.assertGreater(mmd_rbf(gen, ref, g), 0.0)


class TestNovelty(unittest.TestCase):
    def test_zero_when_generated_in_reference(self):
        ref = [[0.0, 0.0], [1.0, 1.0]]
        self.assertAlmostEqual(average_novelty([[0.0, 0.0]], ref), 0.0)

    def test_distance(self):
        ref = [[0.0, 0.0]]
        self.assertAlmostEqual(average_novelty([[3.0, 4.0]], ref), 5.0)


class TestDPP(unittest.TestCase):
    def test_single_design_zero(self):
        mean, std = standardiser([[0.0], [1.0]])
        self.assertEqual(dpp_diversity([[0.0]], mean, std), 0.0)

    def test_more_spread_more_diverse(self):
        ref = [[0.0], [1.0], [2.0], [3.0]]
        mean, std = standardiser(ref)
        close = dpp_diversity([[0.0], [0.001], [0.002]], mean, std)
        spread = dpp_diversity([[0.0], [1.0], [2.0]], mean, std)
        # more diverse -> lower loss
        self.assertLess(spread, close)

    def test_deduplicates(self):
        mean, std = standardiser([[0.0], [1.0]])
        # duplicates collapse to a single design -> 0.0
        self.assertEqual(dpp_diversity([[0.5], [0.5]], mean, std), 0.0)


class TestConstraintMetrics(unittest.TestCase):
    def test_average_constraint_violation(self):
        cons = [[-1.0, 2.0], [3.0, 4.0]]  # 1 + 2 violated
        self.assertEqual(average_constraint_violation(cons), 1.5)

    def test_violation_rate(self):
        cons = [[-1.0, 1.0], [1.0, 1.0]]
        self.assertEqual(constraint_violation_rate(cons), [0.5, 1.0])

    def test_violation_magnitude(self):
        cons = [[-1.0, 2.0], [4.0, -3.0]]
        self.assertEqual(mean_constraint_violation_magnitude(cons), [2.0, 1.0])


class TestObjectiveMetrics(unittest.TestCase):
    def test_min_and_mean_over_feasible(self):
        objs = [[1.0], [2.0], [3.0]]
        cons = [[0.0], [0.0], [1.0]]  # third infeasible
        self.assertEqual(min_objective(objs, cons, [99.0]), [1.0])
        self.assertEqual(mean_objective(objs, cons, [99.0]), [1.5])

    def test_fallback_to_ref_point(self):
        objs = [[1.0]]
        cons = [[5.0]]  # infeasible
        self.assertEqual(min_objective(objs, cons, [99.0]), [99.0])
        self.assertEqual(mean_objective(objs, cons, [99.0]), [99.0])


if __name__ == "__main__":
    unittest.main()
