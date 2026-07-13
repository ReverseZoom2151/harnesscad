import unittest

from harnesscad.eval.quality.sketch.design_alignment import (
    ConstraintEconomy,
    SolveCondition,
    SolveSnapshot,
    StabilityCase,
    VerifiedAttempt,
    constraint_blame_trace,
    parameter_stability,
    score_intent,
    solver_verified_pass_at_k,
)


def snap(condition, x=0.0, *, solved=True, entities=("p",)):
    return SolveSnapshot(
        condition,
        {name: (x, float(index)) for index, name in enumerate(entities)},
        solved=solved,
    )


class StabilityTests(unittest.TestCase):
    def test_stable_perturbations_and_scorecard(self):
        baseline = snap(SolveCondition.FULLY_CONSTRAINED, 0.1)
        values = {
            1.0: snap(SolveCondition.FULLY_CONSTRAINED, 0.2),
            2.0: snap(SolveCondition.UNDER_CONSTRAINED, 0.3),
        }
        report = parameter_stability(
            baseline,
            [StabilityCase("small", {"width": 1.0}), StabilityCase("large", {"width": 2.0})],
            lambda params: values[params["width"]],
            spatial_bin=1.0,
        )
        self.assertTrue(report.stable)
        self.assertEqual(report.stable_fraction, 1.0)
        card = score_intent(baseline, report)
        self.assertTrue(card.fully_constrained)
        self.assertTrue(card.stable)
        self.assertEqual(card.condition, SolveCondition.FULLY_CONSTRAINED)

    def test_detects_branch_jump_and_unsolvable_case(self):
        baseline = snap(SolveCondition.FULLY_CONSTRAINED, 0.1)
        results = iter(
            [
                snap(SolveCondition.FULLY_CONSTRAINED, 2.1),
                snap(SolveCondition.UNSOLVABLE, solved=False),
            ]
        )
        report = parameter_stability(
            baseline,
            [StabilityCase("jump", {}), StabilityCase("fail", {})],
            lambda _: next(results),
            spatial_bin=1.0,
        )
        self.assertFalse(report.stable)
        self.assertFalse(report.results[0].same_spatial_bins)
        self.assertEqual(report.results[1].condition, SolveCondition.UNSOLVABLE)

    def test_entity_change_is_unstable(self):
        report = parameter_stability(
            snap(SolveCondition.FULLY_CONSTRAINED),
            [StabilityCase("topology", {})],
            lambda _: snap(SolveCondition.FULLY_CONSTRAINED, entities=("q",)),
        )
        self.assertFalse(report.results[0].same_entities)

    def test_invalid_stability_configuration(self):
        with self.assertRaises(ValueError):
            parameter_stability(
                snap(SolveCondition.FULLY_CONSTRAINED), [], lambda _: None, spatial_bin=0
            )


class EconomyTests(unittest.TestCase):
    def test_economy_metrics_and_reward_hacking(self):
        economy = ConstraintEconomy(
            dimensional=8,
            geometric=2,
            duplicate=1,
            ineffective=1,
            reference_only_dimensions=2,
        )
        self.assertEqual(economy.dimension_to_geometric_ratio, 4.0)
        self.assertEqual(economy.useful_fraction, 0.8)
        self.assertEqual(
            economy.reward_hacking_diagnostics(),
            (
                "dimension_stuffing",
                "duplicate_constraints",
                "ineffective_constraints",
                "reference_dimension_inflation",
            ),
        )

    def test_zero_denominator_is_explicit(self):
        self.assertEqual(ConstraintEconomy(0, 0).dimension_to_geometric_ratio, 0.0)
        self.assertEqual(ConstraintEconomy(1, 0).dimension_to_geometric_ratio, float("inf"))


class BlameTests(unittest.TestCase):
    def test_transition_and_drop_identify_bad_constraint(self):
        constraints = ("horizontal", "duplicate", "length")

        def evaluate(items):
            condition = (
                SolveCondition.OVER_CONSTRAINED
                if "duplicate" in items
                else SolveCondition.FULLY_CONSTRAINED
            )
            return snap(condition)

        trace = constraint_blame_trace(constraints, evaluate)
        self.assertTrue(trace[1].blamed_by_transition)
        self.assertTrue(trace[1].blamed_by_drop)
        self.assertFalse(trace[0].blamed_by_drop)

    def test_trace_is_empty_for_empty_program(self):
        self.assertEqual(
            constraint_blame_trace((), lambda _: snap(SolveCondition.UNDER_CONSTRAINED)),
            (),
        )


class PassAtKTests(unittest.TestCase):
    def test_solver_verified_pass_at_k(self):
        attempts = [
            VerifiedAttempt(True, True),
            VerifiedAttempt(True, False),
            VerifiedAttempt(True, False),
            VerifiedAttempt(False, True),
        ]
        self.assertAlmostEqual(solver_verified_pass_at_k(attempts, 2), 2 / 3)
        self.assertAlmostEqual(solver_verified_pass_at_k(attempts, 3), 1.0)

    def test_invalid_k(self):
        with self.assertRaises(ValueError):
            solver_verified_pass_at_k([], 0)
        with self.assertRaises(ValueError):
            solver_verified_pass_at_k([VerifiedAttempt(True, True)], 2)


class SnapshotTests(unittest.TestCase):
    def test_unsolvable_snapshot_must_not_claim_solved(self):
        with self.assertRaises(ValueError):
            snap(SolveCondition.UNSOLVABLE)


if __name__ == "__main__":
    unittest.main()
