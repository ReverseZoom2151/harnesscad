import unittest

from surfaces.mmcad_adaptive_ux import (
    InteractionStats,
    ProficiencyEstimator,
    ProficiencyStateMachine,
    ProficiencyTier,
    UxProfile,
    Verbosity,
    proficiency_score,
    recommend_ux,
    score_to_tier,
)
from surfaces.mmcad_modality_fusion import ModalityKind


class TestInteractionStats(unittest.TestCase):
    def test_rejects_negative(self):
        with self.assertRaises(ValueError):
            InteractionStats(commands_issued=-1)

    def test_distinct_cannot_exceed_total(self):
        with self.assertRaises(ValueError):
            InteractionStats(commands_issued=2, distinct_commands=3)


class TestProficiencyScore(unittest.TestCase):
    def test_zero_activity_is_zero(self):
        self.assertEqual(proficiency_score(InteractionStats()), 0.0)

    def test_high_fluency_low_friction_scores_high(self):
        s = InteractionStats(commands_issued=20, distinct_commands=18, errors=0)
        self.assertGreaterEqual(proficiency_score(s), 0.7)

    def test_friction_drags_score_down(self):
        clean = InteractionStats(commands_issued=10, distinct_commands=8)
        noisy = InteractionStats(
            commands_issued=10, distinct_commands=8, errors=5, undos=3
        )
        self.assertGreater(proficiency_score(clean), proficiency_score(noisy))

    def test_score_clamped_to_unit_interval(self):
        s = InteractionStats(
            commands_issued=5, distinct_commands=1, errors=10, undos=10,
            help_requests=10,
        )
        val = proficiency_score(s)
        self.assertGreaterEqual(val, 0.0)
        self.assertLessEqual(val, 1.0)


class TestTierMapping(unittest.TestCase):
    def test_thresholds(self):
        self.assertEqual(score_to_tier(0.0), ProficiencyTier.NOVICE)
        self.assertEqual(score_to_tier(0.39), ProficiencyTier.NOVICE)
        self.assertEqual(score_to_tier(0.4), ProficiencyTier.INTERMEDIATE)
        self.assertEqual(score_to_tier(0.69), ProficiencyTier.INTERMEDIATE)
        self.assertEqual(score_to_tier(0.7), ProficiencyTier.EXPERT)
        self.assertEqual(score_to_tier(1.0), ProficiencyTier.EXPERT)

    def test_estimator_matches_helpers(self):
        est = ProficiencyEstimator()
        s = InteractionStats(commands_issued=10, distinct_commands=9)
        self.assertEqual(est.tier(s), score_to_tier(est.score(s)))


class TestRecommendUx(unittest.TestCase):
    def test_novice_foregrounds_visual_modalities(self):
        p = recommend_ux(ProficiencyTier.NOVICE)
        self.assertEqual(p.primary_modalities[0], ModalityKind.SKETCH)
        self.assertEqual(p.verbosity, Verbosity.HIGH)
        self.assertFalse(p.show_command_palette)

    def test_expert_foregrounds_keyboard(self):
        p = recommend_ux(ProficiencyTier.EXPERT)
        self.assertEqual(p.primary_modalities[0], ModalityKind.KEYBOARD)
        self.assertEqual(p.verbosity, Verbosity.LOW)
        self.assertTrue(p.show_command_palette)
        self.assertFalse(p.autosuggest)

    def test_high_complexity_reenables_autosuggest_for_expert(self):
        p = recommend_ux(ProficiencyTier.EXPERT, project_complexity=30)
        self.assertTrue(p.autosuggest)

    def test_negative_complexity_rejected(self):
        with self.assertRaises(ValueError):
            recommend_ux(ProficiencyTier.NOVICE, project_complexity=-1)

    def test_returns_ux_profile(self):
        self.assertIsInstance(recommend_ux(ProficiencyTier.INTERMEDIATE), UxProfile)


class TestStateMachine(unittest.TestCase):
    def _expert_stats(self):
        return InteractionStats(commands_issued=20, distinct_commands=18)

    def _novice_stats(self):
        return InteractionStats(commands_issued=10, distinct_commands=2, errors=6)

    def test_starts_novice(self):
        sm = ProficiencyStateMachine()
        self.assertEqual(sm.tier, ProficiencyTier.NOVICE)

    def test_promotion_requires_patience(self):
        sm = ProficiencyStateMachine(patience=2)
        # first expert observation: not yet promoted (hysteresis)
        self.assertEqual(sm.update(self._expert_stats()), ProficiencyTier.NOVICE)
        # second consecutive: promoted
        self.assertEqual(sm.update(self._expert_stats()), ProficiencyTier.EXPERT)

    def test_single_fluke_does_not_move(self):
        sm = ProficiencyStateMachine(patience=2)
        sm.update(self._expert_stats())          # pending expert
        sm.update(self._novice_stats())          # resets pending
        self.assertEqual(sm.tier, ProficiencyTier.NOVICE)

    def test_stable_observation_clears_pending(self):
        sm = ProficiencyStateMachine(patience=3)
        sm.update(self._expert_stats())
        # a matching-current observation clears the streak
        sm.update(self._novice_stats())
        self.assertEqual(sm.tier, ProficiencyTier.NOVICE)

    def test_patience_must_be_positive(self):
        with self.assertRaises(ValueError):
            ProficiencyStateMachine(patience=0)

    def test_profile_reflects_current_tier(self):
        sm = ProficiencyStateMachine(patience=1)
        sm.update(self._expert_stats())
        self.assertEqual(sm.tier, ProficiencyTier.EXPERT)
        self.assertEqual(sm.profile().tier, ProficiencyTier.EXPERT)

    def test_determinism(self):
        a = ProficiencyStateMachine(patience=2)
        b = ProficiencyStateMachine(patience=2)
        for _ in range(3):
            ra = a.update(self._expert_stats())
            rb = b.update(self._expert_stats())
            self.assertEqual(ra, rb)


if __name__ == "__main__":
    unittest.main()
