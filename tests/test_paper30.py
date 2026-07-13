import unittest

from harnesscad.eval.bench.sequence.command_metrics import command_metrics
from harnesscad.eval.bench.judges.compiler_judge import (
    CompilerJudge, VerificationLevel, component_scorecard, pareto_scorecards,
)
from harnesscad.eval.bench.judges.judge_calibration import calibrate_threshold, select_threshold
from harnesscad.eval.bench.judges.judge_efficiency import judge_efficiency
from harnesscad.eval.bench.harness.morphology_report import morphology_report
from harnesscad.eval.bench.harness.review_iterations import review_iteration_report
from harnesscad.eval.bench.protocols.reward_hacking import reward_hacking_audit
from harnesscad.data.dataengine.preference.binary_preferences import (
    BinaryPreference, audit_preferences,
)
from harnesscad.data.dataengine.preference.binary_sampling import sample_binary
from harnesscad.data.dataengine.preference.kto import implied_reward, kto_row, kto_utility
from harnesscad.domain.geometry.mesh.mesh_sampling import sample_mesh, triangle_area
from harnesscad.eval.reliability.compiler_diagnostics import normalize_compiler_error
from harnesscad.governance.research.judge_ablation import judge_ablation


class CompilerJudgeTests(unittest.TestCase):
    def test_stage_aware_judge_and_cache(self):
        compiler = lambda seq: seq["points"]
        sampler = lambda shape, count, seed: tuple(shape)
        judge = CompilerJudge(compiler, sampler, threshold=.02, sample_count=2,
                              provenance={"compiler": "fake-v1"})
        candidate = {"points": ((0, 0, 0), (1.1, 0, 0))}
        reference = {"points": ((0, 0, 0), (1, 0, 0))}
        first = judge.judge(candidate, reference)
        second = judge.judge(candidate, reference)
        self.assertEqual(first.level, VerificationLevel.MORPHOLOGY)
        self.assertTrue(first.label)
        self.assertFalse(first.cache_hit)
        self.assertTrue(second.cache_hit)

    def test_compile_and_sampling_failures_have_honest_levels(self):
        def compiler(value):
            if value == "bad": raise ValueError("open loop")
            return value
        judge = CompilerJudge(compiler, lambda *args: (), threshold=1)
        failure = judge.judge("bad", "reference")
        self.assertEqual(failure.stage, "candidate_compile")
        self.assertEqual(failure.level, VerificationLevel.STRUCTURAL)
        sample_failure = judge.judge("candidate", "reference")
        self.assertEqual(sample_failure.stage, "sample")
        self.assertEqual(sample_failure.level, VerificationLevel.VALIDITY)
        self.assertFalse(sample_failure.morphology_verified)
        self.assertFalse(sample_failure.requirements_verified)

    def test_scorecard_never_upgrades_compile_only(self):
        judge = CompilerJudge(lambda value: value, lambda *args: (), threshold=1)
        result = judge.judge("candidate", "reference")
        card = component_scorecard(result, requirements_verified=True)
        self.assertFalse(card["accepted"])
        self.assertFalse(card["morphology_verified"])
        self.assertTrue(card["requirements_verified"])
        strong = dict(card, compile_valid=True, morphology_verified=True,
                      accepted=True, distance=.1, command_fidelity=.9)
        weak = dict(strong, distance=.2, command_fidelity=.8)
        self.assertEqual(pareto_scorecards((weak, strong)), (strong,))


class GeometryAndDataTests(unittest.TestCase):
    def test_area_weighted_sampling_is_seeded_and_permutation_stable(self):
        small = ((0, 0, 0), (1, 0, 0), (0, 1, 0))
        large = ((0, 0, 1), (2, 0, 1), (0, 2, 1))
        self.assertEqual(triangle_area(large), 4 * triangle_area(small))
        a = sample_mesh((small, large), 100, seed=3)
        b = sample_mesh((large, small), 100, seed=3)
        self.assertEqual(a, b)
        self.assertGreater(sum(point[2] == 1 for point in a), 65)
        with self.assertRaises(ValueError):
            sample_mesh((((0, 0, 0),)*3,), 1)

    def test_binary_records_block_reference_negative_and_conflicts(self):
        positive = BinaryPreference.create(
            "p", {"x": 1}, True, reference={"x": 1}, reason="within-threshold")
        with self.assertRaises(ValueError):
            BinaryPreference.create(
                "p", {"x": 1}, False, reference={"x": 1}, reason="paper-typo")
        negative = BinaryPreference.create(
            "p", {"x": 2}, False, reference={"x": 1}, reason="distance")
        conflict = BinaryPreference.create(
            "p", {"x": 2}, True, reference={"x": 1}, reason="override")
        self.assertEqual(audit_preferences((negative, conflict))[0][0],
                         "conflicting-label")
        sampled = sample_binary((positive, negative), 2, positive_fraction=.5, seed=1)
        self.assertEqual(sampled["selected"], 2)
        self.assertEqual(sampled["positive_fraction"], .5)

    def test_kto_math_is_monotonic_and_exportable(self):
        self.assertEqual(implied_reward(-1, -2), 1)
        self.assertGreater(kto_utility(2, 0, True), kto_utility(-2, 0, True))
        preference = BinaryPreference.create(
            "p", "candidate", True, reference="reference", reason="ok")
        row = kto_row(preference, policy_logprob=-1, reference_logprob=-2,
                      reference_point=0)
        self.assertGreater(row["utility"], .5)


class MetricsAndAuditTests(unittest.TestCase):
    def test_calibration_exact_threshold_is_positive(self):
        records = ({"distance": .1, "accepted": True},
                   {"distance": .2, "accepted": False})
        rows = calibrate_threshold(records, (.05, .1, .2))
        self.assertEqual(select_threshold(rows)["threshold"], .1)

    def test_diagnostics_are_normalized_with_raw_evidence(self):
        diagnostic = normalize_compiler_error("Profile is not closed", provider="occt")
        self.assertEqual(diagnostic.code, "open-loop")
        self.assertEqual(diagnostic.provider, "occt")
        self.assertIn("not closed", diagnostic.raw)

    def test_review_iterations_exposes_plateau_and_cost(self):
        report = review_iteration_report((
            ({"iteration": 0, "valid": False, "distance": None, "cost": 1},
             {"iteration": 1, "valid": True, "distance": .2, "cost": 1},
             {"iteration": 2, "valid": True, "distance": .2, "cost": 1}),
            ({"iteration": 0, "valid": True, "distance": .1, "cost": 1},
             {"iteration": 1, "valid": True, "distance": .1, "cost": 1},
             {"iteration": 2, "valid": True, "distance": .1, "cost": 1}),
        ))
        self.assertEqual(report["plateau"], 2)
        self.assertEqual(report["first_valid"], (1, 0))

    def test_command_metrics_and_arc_shortcut(self):
        expected = ({"type": "arc", "params": (0, 1)},)
        actual = ({"type": "line", "params": (0, 1)},
                  {"type": "circle", "params": (0, 1)})
        metrics = command_metrics(actual, expected)
        self.assertEqual(metrics["arc"]["f1"], 0)
        audit = reward_hacking_audit(actual, expected, candidate_distance=.1,
                                     baseline_distance=.2)
        self.assertIn("arc-substitution", audit["flags"])
        self.assertIn("geometry-semantic-conflict", audit["flags"])

    def test_morphology_report_keeps_invalid_denominator(self):
        report = morphology_report((
            {"valid": True, "distance": 1}, {"valid": False, "distance": None}),
            invalid_penalty=10)
        self.assertEqual(report["invalidity_ratio"], .5)
        self.assertEqual(report["mean_valid_distance"], 1)
        self.assertEqual(report["failure_penalized_distance"], 5.5)

    def test_efficiency_and_candidate_controlled_ablation(self):
        efficiency = judge_efficiency({
            "compiler": ({"latency": 1, "cost": 0, "cache_hit": True},),
            "vlm": ({"latency": 5, "cost": 2, "cache_hit": False},),
        })
        self.assertEqual(efficiency[0]["judge"], "compiler")
        rows = (
            {"method": "binary", "candidate_id": "a", "f1": .9,
             "distance": .1, "invalid": False, "time": 1, "cost": 0},
            {"method": "paired", "candidate_id": "a", "f1": .8,
             "distance": .2, "invalid": False, "time": 2, "cost": 1},
        )
        report = judge_ablation(rows)
        self.assertTrue(report["candidate_controlled"])
        self.assertEqual(len(report["reports"]), 2)


if __name__ == "__main__":
    unittest.main()
