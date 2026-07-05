"""Tests for the multi-agent supervisor layer (agents/).

Everything runs with mock/heuristic personas + a real StubBackend/HarnessSession —
no network, no API keys. Covers:
  * the Supervisor drives the full role pipeline to a verified stop on a good plan;
  * it escalates/loops then stops on a harder plan (bad first, good second);
  * the Reviewer self-prioritizes findings (ERROR before WARNING before INFO);
  * the RedTeam can veto (blocks approval, records a reason);
  * the AsyncOverseer emits HALT on a stagnating / looping event stream.
"""

import unittest

from backends.stub import StubBackend
from loop import HarnessSession
from verify import Severity
from cisp.ops import NewSketch, AddRectangle, Constrain, Extrude

from llm.base import CompletionResult
from agents.roles import (
    Designer, Modeler, Verifier, DFMCritic, Reviewer, RedTeam,
    Finding, prioritize, findings_from,
)
from agents.supervisor import Supervisor, Trajectory
from agents.overseer import AsyncOverseer, Halt

from tests.test_llm import MockLLM, plate_ops_json
from tests.test_planner import _over_constrained_plate_json


# --------------------------------------------------------------------------- #
# Role-level unit tests
# --------------------------------------------------------------------------- #
class TestDesigner(unittest.TestCase):
    def test_designer_plans_from_llm(self):
        d = Designer(llm=MockLLM([plate_ops_json()]))
        plan = d.design("make a 20x10x5 plate")
        self.assertTrue(plan.ok)
        self.assertEqual(len(plan.ops), 7)
        self.assertIsInstance(plan.ops[0], NewSketch)
        self.assertIsInstance(plan.ops[-1], Extrude)

    def test_designer_heuristic_plan_fn(self):
        ops = [NewSketch(), AddRectangle(sketch="sk1", w=20, h=10)]
        d = Designer(plan_fn=lambda b, s, diag: ops)
        plan = d.design("anything")
        self.assertTrue(plan.ok)
        self.assertEqual(plan.ops, ops)

    def test_designer_bad_output_not_ok(self):
        d = Designer(llm=MockLLM(["not json {"]))
        plan = d.design("make a plate")
        self.assertFalse(plan.ok)
        self.assertIsInstance(plan.error, str)


class TestMechanicalRoles(unittest.TestCase):
    def test_modeler_applies_and_verifier_clean(self):
        session = HarnessSession(StubBackend())
        d = Designer(llm=MockLLM([plate_ops_json()]))
        plan = d.design("make a plate")
        model = Modeler().model(session, plan)
        self.assertTrue(model.ok)
        self.assertEqual(model.applied, 7)
        outcome = Verifier().verify(session)
        self.assertTrue(outcome.ok)

    def test_dfm_critic_is_advisory_only(self):
        session = HarnessSession(StubBackend())
        d = Designer(llm=MockLLM([plate_ops_json()]))
        Modeler().model(session, d.design("make a plate"))
        dfm = DFMCritic().critique(session)
        # DFM never emits ERROR — advisory WARNING/INFO only.
        self.assertFalse(any(x.severity is Severity.ERROR for x in dfm.diagnostics))


class TestReviewerPrioritization(unittest.TestCase):
    def test_prioritize_orders_error_warning_info(self):
        findings = [
            Finding(Severity.INFO, "i", "info", "dfm-critic"),
            Finding(Severity.ERROR, "e", "err", "verifier"),
            Finding(Severity.WARNING, "w", "warn", "dfm-critic"),
        ]
        ordered = prioritize(findings)
        self.assertEqual([f.severity for f in ordered],
                         [Severity.ERROR, Severity.WARNING, Severity.INFO])

    def test_reviewer_reports_prioritized_findings_and_blocks_on_error(self):
        findings = [
            Finding(Severity.INFO, "i", "info", "dfm-critic"),
            Finding(Severity.ERROR, "over-constrained", "bad", "verifier"),
        ]
        review = Reviewer().review("brief", findings, blocking_ok=False)
        self.assertFalse(review.approved)
        self.assertEqual(review.findings[0].severity, Severity.ERROR)
        self.assertEqual(len(review.blocking), 1)

    def test_reviewer_approves_clean_round(self):
        review = Reviewer().review("brief", [], blocking_ok=True)
        self.assertTrue(review.approved)
        self.assertIn("APPROVE", review.reflection)


class TestRedTeam(unittest.TestCase):
    def test_default_probe_vetoes_nonmanufacturable_code(self):
        findings = [Finding(Severity.WARNING, "thin-envelope", "too thin", "dfm-critic")]
        result = RedTeam().attack(None, findings)
        self.assertTrue(result.veto)
        self.assertTrue(result.reasons)

    def test_custom_probe_veto(self):
        rt = RedTeam(probe=lambda session, findings: ["interference detected"])
        result = rt.attack(None, [])
        self.assertTrue(result.veto)
        self.assertEqual(result.reasons, ["interference detected"])

    def test_no_veto_when_clean(self):
        result = RedTeam().attack(None, [])
        self.assertFalse(result.veto)


# --------------------------------------------------------------------------- #
# Supervisor integration tests
# --------------------------------------------------------------------------- #
class TestSupervisor(unittest.TestCase):
    def test_good_plan_runs_full_pipeline_to_verified_stop(self):
        session = HarnessSession(StubBackend())
        sup = Supervisor(Designer(llm=MockLLM([plate_ops_json()])))
        traj = sup.run(session, "make a 20x10x5 plate")

        self.assertIsInstance(traj, Trajectory)
        self.assertTrue(traj.approved)
        self.assertEqual(traj.stop_reason, "verified-and-approved")
        self.assertEqual(traj.round_count, 1)

        # The full role pipeline actually ran this round.
        rec = traj.final
        self.assertTrue(rec.plan.ok)
        self.assertTrue(rec.model.ok)
        self.assertTrue(rec.verify.ok)
        self.assertIsNotNone(rec.dfm)
        self.assertFalse(rec.red_team.veto)
        self.assertTrue(rec.review.approved)

        # And the geometry really landed.
        self.assertTrue(session.summary()["solid_present"])

    def test_harder_plan_escalates_then_stops(self):
        # Round 1 over-constrains (ERROR -> rolled back, not ok); round 2 is good.
        session = HarnessSession(StubBackend())
        sup = Supervisor(
            Designer(llm=MockLLM([_over_constrained_plate_json(), plate_ops_json()])),
            max_rounds=5,
        )
        traj = sup.run(session, "make a 20x10x5 plate")

        self.assertTrue(traj.approved)
        self.assertEqual(traj.round_count, 2)
        # Round 1 escalated (not approved), round 2 stopped.
        self.assertFalse(traj.rounds[0].review.approved)
        self.assertTrue(traj.rounds[1].review.approved)
        # Round 1 surfaced a blocking (ERROR) finding that the reviewer prioritized.
        self.assertTrue(traj.rounds[0].review.blocking)
        self.assertTrue(session.summary()["solid_present"])

    def test_redteam_veto_blocks_approval(self):
        session = HarnessSession(StubBackend())
        # A red team that always vetoes: an otherwise-good plan can never be approved.
        sup = Supervisor(
            Designer(llm=MockLLM([plate_ops_json()] * 5)),
            red_team=RedTeam(probe=lambda s, f: ["non-manufacturable geometry"]),
            max_rounds=3,
        )
        traj = sup.run(session, "make a plate")

        self.assertFalse(traj.approved)
        self.assertEqual(traj.stop_reason, "max-rounds-exhausted")
        self.assertTrue(traj.final.red_team.veto)
        self.assertIn("veto", traj.final.review.reflection.lower())

    def test_gives_up_after_max_rounds_on_persistent_failure(self):
        session = HarnessSession(StubBackend())
        sup = Supervisor(
            Designer(llm=MockLLM([_over_constrained_plate_json()] * 10)),
            max_rounds=3,
        )
        traj = sup.run(session, "make a plate")
        self.assertFalse(traj.approved)
        self.assertEqual(traj.round_count, 3)


# --------------------------------------------------------------------------- #
# AsyncOverseer tests
# --------------------------------------------------------------------------- #
def _ev(kind, run_id="r", **data):
    return {"kind": kind, "run_id": run_id, "data": data}


class TestAsyncOverseer(unittest.TestCase):
    def test_halts_on_repeated_rejection_loop(self):
        ov = AsyncOverseer(loop_window=6, loop_threshold=3)
        bad = {"op": "extrude", "sketch": "nope", "distance": 5.0}
        halt = None
        # Same op rejected across three runs, no op_applied in between.
        for _ in range(3):
            self.assertIsNone(ov.observe(_ev("run_start", op_count=1)))
            halt = ov.observe(_ev("rejected", op=bad, reason="backend-rejected"))
            if halt is not None:
                break
            ov.observe(_ev("run_end", ok=False, digest="d0"))
        self.assertIsInstance(halt, Halt)
        self.assertEqual(halt.kind, "loop")
        self.assertTrue(ov.halted)

    def test_halts_on_digest_stagnation(self):
        ov = AsyncOverseer(stagnation_rounds=3)
        halt = None
        for _ in range(5):
            halt = ov.observe(_ev("run_end", ok=True, digest="frozen"))
            if halt is not None:
                break
        self.assertIsInstance(halt, Halt)
        self.assertEqual(halt.kind, "stagnation")

    def test_no_halt_on_healthy_progress(self):
        ov = AsyncOverseer(stagnation_rounds=3)
        for i in range(6):
            self.assertIsNone(ov.observe(_ev("op_applied", op={"op": "new_sketch", "plane": "XY"},
                                              index=i, digest=f"d{i}")))
            self.assertIsNone(ov.observe(_ev("run_end", ok=True, digest=f"d{i}")))
        self.assertFalse(ov.halted)

    def test_latches_and_records_halt(self):
        ov = AsyncOverseer(stagnation_rounds=2)
        for _ in range(3):
            ov.observe(_ev("run_end", ok=True, digest="x"))
        self.assertTrue(ov.halted)
        # Latched: a further event returns the same Halt.
        again = ov.observe(_ev("run_end", ok=True, digest="y"))
        self.assertIs(again, ov.halt)
        self.assertEqual(len(ov.halts), 1)

    def test_plugs_into_session_as_tracer(self):
        # AsyncOverseer satisfies the Tracer protocol: watch a live session run.
        ov = AsyncOverseer()
        session = HarnessSession(StubBackend(), tracer=ov)
        d = Designer(llm=MockLLM([plate_ops_json()]))
        Modeler().model(session, d.design("make a plate"))
        # A healthy single run does not halt.
        self.assertFalse(ov.halted)


if __name__ == "__main__":
    unittest.main()
