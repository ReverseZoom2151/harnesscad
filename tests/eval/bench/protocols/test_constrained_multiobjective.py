"""Tests for eval.bench.protocols.constrained_multiobjective."""

import unittest

from harnesscad.eval.bench.protocols.constrained_multiobjective import (
    aggregate_objectives,
    constraint_satisfaction,
    design_feasible,
    evaluate_population,
)


class FeasibilityTest(unittest.TestCase):
    def test_feasible(self):
        self.assertTrue(design_feasible({"g1": -1.0, "g2": 0.0}))

    def test_infeasible(self):
        self.assertFalse(design_feasible({"g1": 0.5}))

    def test_satisfaction_fraction(self):
        self.assertAlmostEqual(
            constraint_satisfaction({"a": -1, "b": 2, "c": 0}), 2 / 3
        )


class PopulationTest(unittest.TestCase):
    def test_population_metrics(self):
        designs = [
            {"a": -1, "b": -1},   # feasible
            {"a": 1, "b": -1},    # infeasible on a
            {"a": -1, "b": -1},   # feasible
        ]
        out = evaluate_population(designs)
        self.assertAlmostEqual(out["feasibility_rate"], 2 / 3)
        self.assertEqual(out["feasible_indices"], [0, 2])
        self.assertAlmostEqual(out["per_constraint"]["a"], 2 / 3)
        self.assertAlmostEqual(out["per_constraint"]["b"], 1.0)

    def test_empty_raises(self):
        with self.assertRaises(ValueError):
            evaluate_population([])


class ObjectiveTest(unittest.TestCase):
    def test_weighted_normalised_best(self):
        objs = [
            {"aero": 10.0, "mass": 5.0},
            {"aero": 20.0, "mass": 1.0},
        ]
        out = aggregate_objectives(
            objs, weights={"aero": 1.0, "mass": 1.0},
            feasible_indices=[0, 1], minimize=["mass"],
        )
        # design 1 has higher aero and lower mass -> best on both.
        self.assertEqual(out["best"], 1)
        self.assertAlmostEqual(out["scores"][1], 2.0)
        self.assertAlmostEqual(out["scores"][0], 0.0)

    def test_no_feasible(self):
        out = aggregate_objectives([{"x": 1}], {"x": 1.0}, feasible_indices=[])
        self.assertIsNone(out["best"])

    def test_single_feasible_degenerate(self):
        out = aggregate_objectives(
            [{"x": 3.0}], {"x": 1.0}, feasible_indices=[0]
        )
        self.assertEqual(out["best"], 0)
        self.assertAlmostEqual(out["scores"][0], 1.0)


if __name__ == "__main__":
    unittest.main()
