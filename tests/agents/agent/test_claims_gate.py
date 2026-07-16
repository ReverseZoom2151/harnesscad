import unittest

from harnesscad.agents.agent.claims_gate import (
    INTENT_REGISTRY,
    MUTATE_GEOMETRY,
    MUTATION_CLAIM,
    READ_ONLY,
    SOLVER_CLAIM,
    SOLVER_PREPARE,
    SOLVER_RUN,
    RunEvidence,
    ToolEvent,
    collect_evidence,
    gate_claims,
    keyword_classify,
    main,
    resolve_route_intent,
)
from harnesscad.agents.agent.termination import (
    TerminationDecision,
    gate_termination,
)

BUILD = INTENT_REGISTRY["build"]
MODIFY = INTENT_REGISTRY["modify"]
CRITIQUE = INTENT_REGISTRY["critique"]
SIMULATE = INTENT_REGISTRY["simulate"]

PREP = ToolEvent("cae.prepare_solver_run", SOLVER_PREPARE)
RUN = ToolEvent("cae.run_solver", SOLVER_RUN)
EDIT = ToolEvent("cad.execute_build123d", MUTATE_GEOMETRY)


class EvidenceTests(unittest.TestCase):
    def test_empty_log_earns_nothing(self):
        ev = collect_evidence([])
        self.assertFalse(ev.mutation_succeeded)
        self.assertFalse(ev.solver_executed)
        self.assertIsNone(ev.solver_status)

    def test_only_approved_non_error_mutation_counts(self):
        self.assertTrue(collect_evidence([EDIT]).mutation_succeeded)
        for bad in (ToolEvent("cad.x", MUTATE_GEOMETRY, status="error"),
                    ToolEvent("cad.x", MUTATE_GEOMETRY, approved=False),
                    ToolEvent("cad.x", MUTATE_GEOMETRY, status="pending")):
            self.assertFalse(collect_evidence([bad]).mutation_succeeded, bad)

    def test_read_only_never_counts_as_mutation(self):
        ev = collect_evidence([ToolEvent("cad.critique", READ_ONLY)])
        self.assertTrue(ev.read_only_result)
        self.assertFalse(ev.mutation_succeeded)

    def test_solver_run_requires_prior_successful_prepare(self):
        self.assertFalse(collect_evidence([RUN]).solver_executed)
        self.assertFalse(collect_evidence([RUN, PREP]).solver_executed)  # order
        self.assertTrue(collect_evidence([PREP, RUN]).solver_executed)

    def test_failed_prepare_does_not_enable_run(self):
        bad_prep = ToolEvent("cae.prepare_solver_run", SOLVER_PREPARE,
                             status="error")
        self.assertFalse(collect_evidence([bad_prep, RUN]).solver_executed)

    def test_denied_and_errored_runs_reported_honestly(self):
        ev = collect_evidence([PREP, ToolEvent("cae.run_solver", SOLVER_RUN,
                                               approved=False)])
        self.assertFalse(ev.solver_executed)
        self.assertEqual(ev.solver_status, "denied")
        ev = collect_evidence([PREP, ToolEvent("cae.run_solver", SOLVER_RUN,
                                               status="error")])
        self.assertFalse(ev.solver_executed)
        self.assertEqual(ev.solver_status, "error")
        self.assertIn("cae.run_solver:error", ev.denied_or_failed)

    def test_last_solver_status_wins(self):
        ev = collect_evidence([PREP, RUN,
                               ToolEvent("cae.run_solver", SOLVER_RUN,
                                         status="error")])
        self.assertEqual(ev.solver_status, "error")


class MutationGuardTests(unittest.TestCase):
    def test_bare_final_rejected_on_build_intent(self):
        v = gate_claims("complete", intent=BUILD, evidence=collect_evidence([]))
        self.assertFalse(v.accepted)
        self.assertIn("without-successful-mutation", v.reason)

    def test_read_only_result_does_not_satisfy_modify(self):
        ev = collect_evidence([ToolEvent("cad.critique", READ_ONLY)])
        self.assertFalse(gate_claims("complete", intent=MODIFY,
                                     evidence=ev).accepted)

    def test_successful_mutation_opens_the_final(self):
        ev = collect_evidence([EDIT])
        self.assertTrue(gate_claims("complete", intent=BUILD, evidence=ev).accepted)

    def test_honest_failure_always_allowed(self):
        ev = collect_evidence([])
        for state in ("blocked", "continue"):
            self.assertTrue(gate_claims(state, intent=BUILD, evidence=ev).accepted)

    def test_read_only_and_simulation_suppress_guard(self):
        ev = collect_evidence([ToolEvent("cad.get_source", READ_ONLY)])
        self.assertTrue(CRITIQUE.suppresses_mutation_guard)
        self.assertTrue(SIMULATE.suppresses_mutation_guard)
        self.assertTrue(gate_claims("complete", intent=CRITIQUE,
                                    evidence=ev).accepted)
        self.assertTrue(gate_claims("complete", intent=SIMULATE,
                                    evidence=ev).accepted)


class SolverHonestyTests(unittest.TestCase):
    def test_solver_claim_without_run_rejected(self):
        v = gate_claims("complete", [SOLVER_CLAIM], intent=SIMULATE,
                        evidence=collect_evidence([PREP]))
        self.assertFalse(v.accepted)
        self.assertIn("without-executed-run", v.reason)
        self.assertIn("deck-prepared", v.reason)

    def test_solver_claim_after_denied_run_rejected(self):
        ev = collect_evidence([PREP, ToolEvent("cae.run_solver", SOLVER_RUN,
                                               approved=False)])
        v = gate_claims("complete", [SOLVER_CLAIM], evidence=ev)
        self.assertFalse(v.accepted)
        self.assertIn("denied", v.reason)

    def test_solver_claim_after_real_run_accepted_and_tiered(self):
        v = gate_claims("complete", [SOLVER_CLAIM], intent=SIMULATE,
                        evidence=collect_evidence([PREP, RUN]))
        self.assertTrue(v.accepted)
        self.assertEqual(v.credibility_tier, "executed_solver_result")

    def test_no_solver_claim_is_not_tiered_as_solver(self):
        v = gate_claims("complete", intent=SIMULATE,
                        evidence=collect_evidence([PREP, RUN]))
        self.assertTrue(v.accepted)
        self.assertNotEqual(v.credibility_tier, "executed_solver_result")

    def test_declared_mutation_checked_on_read_only_route(self):
        ev = collect_evidence([ToolEvent("cad.critique", READ_ONLY)])
        v = gate_claims("complete", [MUTATION_CLAIM], intent=CRITIQUE,
                        evidence=ev)
        self.assertFalse(v.accepted)
        self.assertIn("without-evidence", v.reason)

    def test_unknown_claims_are_ignored_not_guessed(self):
        v = gate_claims("complete", ["it_is_beautiful"], evidence=RunEvidence())
        self.assertTrue(v.accepted)
        self.assertEqual(v.checked_claims, ())


class IntentResolutionTests(unittest.TestCase):
    def test_explicit_slash_command_wins(self):
        r = resolve_route_intent("/build a bracket and explain it")
        self.assertEqual(r.command, "build")
        self.assertEqual(r.source, "explicit")
        self.assertEqual(r.confidence, 1.0)

    def test_explicit_beats_classifier(self):
        r = resolve_route_intent("/critique it", classifier=lambda t: ("build", 1.0))
        self.assertEqual(r.command, "critique")
        self.assertEqual(r.source, "explicit")

    def test_classifier_beats_keyword(self):
        r = resolve_route_intent("build a bracket",
                                 classifier=lambda t: ("simulate", 0.95))
        self.assertEqual(r.command, "simulate")
        self.assertEqual(r.source, "classifier")

    def test_classifier_degrades_silently(self):
        def boom(_text):
            raise RuntimeError("no api key")

        for clf in (boom, lambda t: (None, 0.0), lambda t: ("nonsense", 1.0)):
            r = resolve_route_intent("build a bracket", classifier=clf)
            self.assertEqual(r.source, "keyword")
            self.assertEqual(r.command, "build")

    def test_low_confidence_classifier_abstains(self):
        r = resolve_route_intent("bracket", classifier=lambda t: ("build", 0.1))
        self.assertTrue(r.abstain)
        self.assertIsNone(r.command)
        self.assertTrue(r.clarify_instruction)

    def test_ambiguous_keywords_abstain(self):
        r = resolve_route_intent("review the plate and change the wall thickness")
        self.assertTrue(r.abstain)
        self.assertIsNone(r.command)
        self.assertGreaterEqual(set(r.candidates), {"modify", "critique"})

    def test_no_actionable_intent_left_untouched(self):
        r = resolve_route_intent("hello there")
        self.assertIsNone(r.intent)
        self.assertEqual(r.source, "none")
        self.assertFalse(r.abstain)

    def test_keyword_mutation_precedence(self):
        best, _conf, _cands = keyword_classify("build")
        self.assertEqual(best, "build")

    def test_abstain_forces_no_guard_but_still_checks_declarations(self):
        r = resolve_route_intent("review the plate and change the wall thickness")
        ev = collect_evidence([])
        self.assertTrue(gate_claims("complete", intent=r.intent,
                                    evidence=ev).accepted)
        self.assertFalse(gate_claims("complete", [MUTATION_CLAIM],
                                     intent=r.intent, evidence=ev).accepted)

    def test_deterministic(self):
        text = "build a 40mm bracket"
        self.assertEqual(resolve_route_intent(text).to_dict(),
                         resolve_route_intent(text).to_dict())


class TerminationWiringTests(unittest.TestCase):
    def test_legacy_two_arg_behaviour_unchanged(self):
        self.assertFalse(gate_termination(TerminationDecision("complete"),
                                          False).accepted)
        self.assertTrue(gate_termination(TerminationDecision("complete"),
                                         True).terminal)
        self.assertEqual(gate_termination(TerminationDecision("continue"),
                                          False).state, "continue")
        self.assertTrue(gate_termination(TerminationDecision("blocked"),
                                         False).terminal)

    def test_verifier_still_runs_first(self):
        res = gate_termination(TerminationDecision("complete"), False,
                               intent=BUILD, evidence=collect_evidence([EDIT]))
        self.assertEqual(res.diagnostic, "premature-completion")

    def test_claim_gate_sends_agent_back_to_continue(self):
        res = gate_termination(TerminationDecision("complete"), True,
                               intent=BUILD, evidence=collect_evidence([]))
        self.assertFalse(res.accepted)
        self.assertFalse(res.terminal)
        self.assertEqual(res.state, "continue")
        self.assertIn("without-successful-mutation", res.diagnostic)

    def test_declared_solver_claim_gated_through_termination(self):
        dec = TerminationDecision("complete", claims=(SOLVER_CLAIM,))
        res = gate_termination(dec, True, intent=SIMULATE,
                               evidence=collect_evidence([PREP]))
        self.assertFalse(res.accepted)
        res = gate_termination(dec, True, intent=SIMULATE,
                               evidence=collect_evidence([PREP, RUN]))
        self.assertTrue(res.accepted)
        self.assertTrue(res.terminal)
        self.assertEqual(res.credibility_tier, "executed_solver_result")

    def test_blocked_final_never_claim_gated(self):
        res = gate_termination(TerminationDecision("blocked", "no CAD model"),
                               False, intent=BUILD,
                               evidence=collect_evidence([]))
        self.assertTrue(res.accepted)
        self.assertTrue(res.terminal)


class SelfcheckTests(unittest.TestCase):
    def test_selfcheck_exits_zero(self):
        self.assertEqual(main(["--selfcheck"]), 0)


if __name__ == "__main__":
    unittest.main()
