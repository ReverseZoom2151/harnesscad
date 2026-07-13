import unittest

from harnesscad.domain.spec.clarify_ambiguity import CADSpec, Feature, audit
from harnesscad.domain.spec.clarify_dialogue import (
    ClarificationDialogue, run_dialogue, oracle_from_truth,
    ACCEPT, ASK, DONE, AWAIT_ANSWERS,
)


def truth_spec():
    return CADSpec(
        general_shape="a cylindrical rod",
        workplane="XY", origin=(-19.0, 0.0, -100.0),
        extrude_direction="positive_normal", extrude_distance=200.0,
        features=[Feature("circle", "rod", {"radius": 19.0})],
    )


class TestRound1(unittest.TestCase):
    def test_accepts_unambiguous(self):
        dlg = ClarificationDialogue(truth_spec())
        turn = dlg.step_round1()
        self.assertEqual(turn.action, ACCEPT)
        self.assertEqual(dlg.state, DONE)

    def test_asks_when_ambiguous(self):
        spec = truth_spec()
        spec.origin = None
        spec.features[0].params["radius"] = None
        dlg = ClarificationDialogue(spec)
        turn = dlg.step_round1()
        self.assertEqual(turn.action, ASK)
        self.assertEqual(dlg.state, AWAIT_ANSWERS)
        self.assertEqual(len(turn.questions), 2)


class TestTwoRound(unittest.TestCase):
    def test_recovers_missing_dims(self):
        # This mirrors the paper's cylindrical-rod case study.
        amb = truth_spec()
        amb.origin = None
        amb.features[0].params["radius"] = None
        result = run_dialogue(amb, truth_spec())
        self.assertTrue(result.is_misleading)
        self.assertEqual(result.rounds, 2)
        self.assertEqual(result.interaction_cost(), 1)
        # after clarification the spec matches the truth and is clean.
        self.assertEqual(result.corrected.origin, (-19.0, 0.0, -100.0))
        self.assertEqual(result.corrected.features[0].params["radius"], 19.0)
        self.assertFalse(audit(result.corrected).is_misleading)

    def test_conflict_resolved_to_truth(self):
        amb = truth_spec()
        amb.features[0].params["radius"] = [19.0, 25.0]
        result = run_dialogue(amb, truth_spec())
        self.assertTrue(result.is_misleading)
        self.assertEqual(result.corrected.features[0].params["radius"], 19.0)
        self.assertFalse(audit(result.corrected).is_misleading)

    def test_unambiguous_single_round(self):
        result = run_dialogue(truth_spec(), truth_spec())
        self.assertFalse(result.is_misleading)
        self.assertEqual(result.rounds, 1)
        self.assertEqual(result.interaction_cost(), 0)

    def test_does_not_mutate_input(self):
        amb = truth_spec()
        amb.origin = None
        run_dialogue(amb, truth_spec())
        self.assertIsNone(amb.origin)

    def test_ambiguous_without_user_raises(self):
        amb = truth_spec()
        amb.origin = None
        dlg = ClarificationDialogue(amb)
        with self.assertRaises(ValueError):
            dlg.run(None)

    def test_asked_keys_reported(self):
        amb = truth_spec()
        amb.origin = None
        result = run_dialogue(amb, truth_spec())
        self.assertIn("setup.origin", result.asked_keys)


if __name__ == "__main__":
    unittest.main()
