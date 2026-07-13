"""Tests for agents/idea2cad_workflow.py (paper 86: From Idea to CAD).

Exercises the four nested empty-feedback loops (Algorithms 1-4) deterministically
with heuristic role stand-ins -- no VLM, no CAD kernel.
"""

import unittest

from harnesscad.agents.agents.idea2cad_blackboard import DesignBlackboard
from harnesscad.agents.agents.idea2cad_roles import (
    RequirementsEngineer, CadEngineer, QualityAssuranceEngineer, User,
)
from harnesscad.agents.agents.idea2cad_workflow import Idea2CadWorkflow


class TestRequirementsLoop(unittest.TestCase):
    def test_converges_when_no_ambiguities(self):
        wf = Idea2CadWorkflow()
        bb = DesignBlackboard()
        bb.post_input(None, "length=10 width=5cm height=2cm")
        trace = wf.run_requirements(bb)
        self.assertTrue(trace.converged)
        self.assertEqual(trace.rounds, 1)
        self.assertIsNotNone(trace.addendum)

    def test_user_replies_resolve_ambiguity(self):
        # first spec ambiguous; the reply supplies dimensions
        wf = Idea2CadWorkflow()
        bb = DesignBlackboard()
        bb.post_input(None, "a plastic block")
        replies = iter(["length=10 width=10 height=2cm"])
        trace = wf.run_requirements(bb, user_reply=lambda amb, t: next(replies, ""))
        self.assertTrue(trace.converged)
        self.assertGreaterEqual(trace.rounds, 2)

    def test_stops_when_user_silent(self):
        wf = Idea2CadWorkflow()
        bb = DesignBlackboard()
        bb.post_input(None, "vague")           # stays ambiguous
        trace = wf.run_requirements(bb, user_reply=lambda amb, t: "")
        self.assertFalse(trace.converged)
        self.assertEqual(trace.rounds, 1)


class TestDesignLoop(unittest.TestCase):
    def test_produces_model_first_try(self):
        wf = Idea2CadWorkflow()
        bb = DesignBlackboard()
        bb.post_input(None, "len=10")
        trace = wf.run_design(bb)
        self.assertTrue(trace.produced)
        self.assertEqual(trace.attempts, 1)
        self.assertIsNotNone(bb.model)
        self.assertIsNotNone(bb.plan)      # plan produced when no feedback

    def test_retries_on_check_failure(self):
        # first generated code is invalid, then valid
        codes = iter(["not valid python (", "x = 1"])
        cad = CadEngineer(code_fn=lambda spec, hints: next(codes, "x = 1"))
        wf = Idea2CadWorkflow(cad_engineer=cad)
        bb = DesignBlackboard()
        bb.post_input(None, "len=10")
        trace = wf.run_design(bb)
        self.assertTrue(trace.produced)
        self.assertEqual(trace.check_failures, 1)
        self.assertEqual(trace.attempts, 2)

    def test_hints_used_when_feedback_present(self):
        seen = {}

        def code_fn(spec, hints):
            seen["hints"] = hints
            return "x = 1"

        cad = CadEngineer(code_fn=code_fn)
        wf = Idea2CadWorkflow(cad_engineer=cad, docs="cadquery docs")
        bb = DesignBlackboard()
        bb.post_input(None, "len=10")
        bb.post_verification_feedback(["rotate the cylinder"])
        wf.run_design(bb)
        self.assertIsNotNone(seen["hints"])
        self.assertIn("rotate the cylinder", seen["hints"])


class TestVerificationLoop(unittest.TestCase):
    def test_converges_when_qa_clean(self):
        wf = Idea2CadWorkflow()
        bb = DesignBlackboard()
        bb.post_input(None, "len=10")
        trace = wf.run_verification(bb)
        self.assertTrue(trace.converged)
        self.assertEqual(trace.round_count, 1)

    def test_loops_until_qa_clean(self):
        # QA returns an issue on the first round, then accepts
        rounds = iter([["cylinder orientation wrong"], []])
        qa = QualityAssuranceEngineer(qa_fn=lambda r, imgs: next(rounds, []))
        wf = Idea2CadWorkflow(qa_engineer=qa)
        bb = DesignBlackboard()
        bb.post_input(None, "len=10")
        trace = wf.run_verification(bb)
        self.assertTrue(trace.converged)
        self.assertEqual(trace.round_count, 2)
        self.assertEqual(trace.rounds[0].issues, ["cylinder orientation wrong"])

    def test_guard_trips_when_qa_never_clean(self):
        qa = QualityAssuranceEngineer(qa_fn=lambda r, imgs: ["always broken"])
        wf = Idea2CadWorkflow(qa_engineer=qa, max_verify_iters=3)
        bb = DesignBlackboard()
        bb.post_input(None, "len=10")
        trace = wf.run_verification(bb)
        self.assertFalse(trace.converged)
        self.assertEqual(trace.round_count, 3)


class TestValidationLoop(unittest.TestCase):
    def test_accepts_immediately(self):
        wf = Idea2CadWorkflow()
        bb = DesignBlackboard()
        bb.post_input(None, "len=10")
        rounds = wf.run_validation(bb)
        self.assertEqual(len(rounds), 1)
        self.assertTrue(rounds[-1].accepted)

    def test_user_feedback_drives_second_round(self):
        replies = iter([["make wheels half as wide"], []])
        user = User(feedback_fn=lambda r, m: next(replies, []))
        wf = Idea2CadWorkflow(user=user)
        bb = DesignBlackboard()
        bb.post_input(None, "len=12 car")
        rounds = wf.run_validation(bb)
        self.assertEqual(len(rounds), 2)
        self.assertTrue(rounds[-1].accepted)
        self.assertEqual(rounds[0].feedback, ["make wheels half as wide"])


class TestFullWorkflow(unittest.TestCase):
    def test_happy_path_accept(self):
        wf = Idea2CadWorkflow()
        result = wf.run(None, "length=10 width=5cm height=2cm")
        self.assertTrue(result.accepted)
        self.assertEqual(result.stop_reason, "accepted")
        self.assertIsNotNone(result.model)
        self.assertTrue(result.requirements.converged)
        self.assertEqual(result.validation_round_count, 1)

    def test_toy_car_three_validation_rounds(self):
        # mirrors Table 2: three user feedback steps, then accept
        car_feedback = iter([
            ["make the wheels parallel to the XZ plane"],
            ["the wheels are asymmetric"],
            ["make the wheels only half as wide"],
            [],
        ])
        user = User(feedback_fn=lambda r, m: next(car_feedback, []))
        wf = Idea2CadWorkflow(user=user)
        result = wf.run(None, "toy car length=12cm width=8cm height=6cm wheels dia=4cm")
        self.assertTrue(result.accepted)
        self.assertEqual(result.validation_round_count, 4)

    def test_verification_failure_stop_reason(self):
        qa = QualityAssuranceEngineer(qa_fn=lambda r, imgs: ["never ok"])
        wf = Idea2CadWorkflow(qa_engineer=qa, max_verify_iters=2, max_validate_iters=2)
        result = wf.run(None, "len=10")
        self.assertFalse(result.accepted)
        self.assertEqual(result.stop_reason, "verification-not-converged")

    def test_blackboard_records_phases(self):
        wf = Idea2CadWorkflow()
        bb = DesignBlackboard()
        wf.run(None, "len=10", blackboard=bb)
        phases = {r.phase for r in bb.log}
        # all four V-model phases were entered
        from harnesscad.agents.agents.idea2cad_blackboard import VPhase
        self.assertEqual(phases >= {VPhase.REQUIREMENTS, VPhase.DESIGN,
                                    VPhase.VERIFICATION, VPhase.VALIDATION}, True)

    def test_invalid_guard_raises(self):
        with self.assertRaises(ValueError):
            Idea2CadWorkflow(max_design_iters=0)


if __name__ == "__main__":
    unittest.main()
