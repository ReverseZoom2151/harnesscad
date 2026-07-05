import unittest

from quality.assembly_readiness import (
    AspectResult,
    AssemblyRequirements,
    CorrectionAttempt,
    HandoffPolicy,
    InterventionMode,
    ReadinessAspect,
    RequirementField,
    assess_readiness,
)


class RequirementProfileTests(unittest.TestCase):
    def test_dlngf_completeness_and_questions(self):
        profile = AssemblyRequirements.from_mapping(
            {
                "dimensions": ["body diameter 150 mm"],
                "layout_constraints": ["jaws every 120 degrees"],
                "element_count": ["three jaws and one body"],
            }
        )
        self.assertEqual(profile.completeness, 0.6)
        self.assertEqual(
            profile.missing,
            (RequirementField.GEOMETRY, RequirementField.FUNCTION),
        )
        self.assertEqual(len(profile.questions()), 2)

    def test_complete_profile(self):
        profile = AssemblyRequirements.from_mapping(
            {field: [field.value] for field in RequirementField}
        )
        self.assertTrue(profile.complete)


class HandoffTests(unittest.TestCase):
    def test_diminishing_prompt_gain_hands_to_code(self):
        decision = HandoffPolicy(minimum_prompt_improvement=0.05).decide(
            [CorrectionAttempt(InterventionMode.PROMPT, 0.70, 0.72)]
        )
        self.assertIs(decision.mode, InterventionMode.CODE)
        self.assertIn("diminishing", decision.reason)

    def test_repeated_code_edits_hand_to_cad(self):
        attempts = [
            CorrectionAttempt(InterventionMode.CODE, 0.7, 0.8),
            CorrectionAttempt(InterventionMode.CODE, 0.8, 0.82),
        ]
        self.assertIs(
            HandoffPolicy().decide(attempts).mode, InterventionMode.DIRECT_CAD
        )

    def test_efficient_prompting_continues(self):
        attempt = CorrectionAttempt(
            InterventionMode.PROMPT, 0.3, 0.6, effort=1.0
        )
        self.assertAlmostEqual(attempt.efficiency, 0.3)
        self.assertIs(HandoffPolicy().decide([attempt]).mode, InterventionMode.PROMPT)


class ReadinessTests(unittest.TestCase):
    def _all(self):
        return [
            AspectResult(aspect, 5, 5, evidence=("verified",))
            for aspect in ReadinessAspect
        ]

    def test_all_aspects_required_for_production_readiness(self):
        result = assess_readiness(self._all()[:-1])
        self.assertFalse(result.production_ready)
        self.assertEqual(result.blockers, ("missing:function",))

    def test_detailed_feature_failure_blocks_plausible_shape(self):
        checks = self._all()
        checks[3] = AspectResult(ReadinessAspect.DETAILED_FEATURES, 1, 5)
        result = assess_readiness(checks)
        self.assertFalse(result.production_ready)
        self.assertIn("below_threshold:detailed_features", result.blockers)
        self.assertGreater(result.score, 0.8)

    def test_complete_verified_assembly_is_ready(self):
        result = assess_readiness(self._all())
        self.assertTrue(result.production_ready)
        self.assertEqual(result.score, 1.0)

    def test_invalid_counts_rejected(self):
        with self.assertRaises(ValueError):
            AspectResult(ReadinessAspect.FUNCTION, 2, 1)


if __name__ == "__main__":
    unittest.main()
