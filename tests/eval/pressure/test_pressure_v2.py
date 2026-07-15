"""v2 of the pressure experiment: the plumbing bugs, and the arm we never ran.

Every test here corresponds to one defect in the v1 experiment. They are grouped
by the defect, and each names it.
"""

from __future__ import annotations

import unittest
from unittest import mock

from harnesscad.eval.pressure import (loops, oracle, prompts, session, shape,
                                      stats)
from harnesscad.eval.pressure.briefs import brief_by_id
from harnesscad.eval.pressure.metrics import grade
from harnesscad.eval.pressure.model import ScriptedClient

V1_BRIEFS = (
    "plate_hole_four", "strip_hole_row", "l_bracket", "step_block",
    "flange_round", "flange_square", "shell_box_3mm", "trap_shell_too_thick",
    "trap_shell_too_thin", "trap_fillet_too_big", "trap_fillet_thin_plate",
    "trap_hole_oversize",
)


# --------------------------------------------------------------------------- #
# BUG 1: format_typed filtered on severity and never imported `soundness`, so
#        the experiment could not exercise the tiering fix at all.
# --------------------------------------------------------------------------- #
class TestTypedChannelIsSoundnessTiered(unittest.TestCase):

    PROVEN = {"severity": "warning", "code": "preflight-THICKNESS_TOO_LARGE",
              "message": "the wall consumes the whole solid", "where": "op[3]"}
    MEASURED = {"severity": "error", "code": "empty-solid",
                "message": "no solid after 4 features", "where": None}
    HEURISTIC_PRECHECK = {"severity": "error", "code": "infeasible-plan",
                          "message": "hole diameter 30 >= wall 8", "where": "op[3]"}
    HEURISTIC_FILLET = {"severity": "warning", "code": "preflight-RADIUS_TOO_LARGE",
                        "message": "radius 3.1 exceeds half the smallest extent",
                        "where": "op[3]"}

    def test_model_facing_keeps_proven_and_measured(self):
        kept = prompts.model_facing([self.PROVEN, self.MEASURED])
        self.assertEqual([d["code"] for d in kept],
                         ["preflight-THICKNESS_TOO_LARGE", "empty-solid"])

    def test_model_facing_drops_the_heuristics_that_caused_every_v1_regression(self):
        kept = prompts.model_facing([self.HEURISTIC_PRECHECK, self.HEURISTIC_FILLET])
        self.assertEqual(kept, [])

    def test_unknown_code_fails_closed(self):
        self.assertEqual(
            prompts.model_facing([{"severity": "error", "code": "brand-new-rule",
                                   "message": "?"}]), [])

    def test_format_typed_renders_only_the_sound_ones(self):
        text = prompts.format_typed({"ok": False, "diagnostics": [
            self.PROVEN, self.HEURISTIC_PRECHECK, self.HEURISTIC_FILLET]})
        self.assertIn("preflight-THICKNESS_TOO_LARGE", text)
        self.assertNotIn("infeasible-plan", text)
        self.assertNotIn("RADIUS_TOO_LARGE", text)

    def test_format_typed_is_silent_when_only_heuristics_fired(self):
        # v1 would have handed this straight to the model as an instruction.
        self.assertIsNone(prompts.format_typed(
            {"ok": True, "diagnostics": [self.HEURISTIC_PRECHECK]}))

    def test_style_lints_are_still_excluded_by_actionability(self):
        # The noise filter and the truth filter are different filters and both
        # must hold. missing-metadata is an ERROR and it is still dropped.
        self.assertEqual(prompts.model_facing(
            [{"severity": "error", "code": "missing-metadata", "message": "x"}]), [])


# --------------------------------------------------------------------------- #
# BUG 2: grade() constructed a raw CISPServer and BYPASSED io/gate.py -- the one
#        component that refuses a dilated shell.
# --------------------------------------------------------------------------- #
class TestGraderRoutesThroughTheOutputGate(unittest.TestCase):

    def test_grade_calls_the_gate(self):
        b = brief_by_id("plate_hole_four")
        from harnesscad.io import gate
        with mock.patch.object(gate, "check", wraps=gate.check) as spy:
            g = grade(b, [dict(o) for o in b.reference], shape=False)
        self.assertTrue(spy.called, "grade() must route through io/gate.py")
        self.assertTrue(g.gate_ok)
        self.assertEqual(g.gate_failures, [])
        self.assertTrue(g.solved)

    def test_a_gate_refusal_is_a_loss(self):
        b = brief_by_id("plate_hole_four")
        from harnesscad.io import gate
        bad = gate.GateReport(
            path=None, ok=False,
            failures=(gate.Failure("shell-grew-bbox", "declared",
                                   "shell(t=3) GREW the part along X, Y, Z"),),
            measurement={"triangle_count": 12})
        with mock.patch.object(gate, "check", return_value=bad):
            g = grade(b, [dict(o) for o in b.reference], shape=False)
        self.assertFalse(g.gate_ok)
        self.assertFalse(g.solved, "a gate refusal must not score as solved")
        self.assertTrue(any("output gate REFUSED" in r for r in g.reasons))


# --------------------------------------------------------------------------- #
# BUG 3: the grader was many-to-one -- bbox + volume + probes are all ENVELOPE
#        families. Every brief already carried a `reference` op stream.
# --------------------------------------------------------------------------- #
class TestShapeMetric(unittest.TestCase):

    def test_reference_matches_itself(self):
        b = brief_by_id("trap_hole_oversize")
        g = grade(b, [dict(o) for o in b.reference])
        self.assertEqual(g.shape["iou"], 1.0)
        self.assertTrue(g.solved_shape)

    def test_a_displaced_hole_is_caught_by_the_shape_metric(self):
        b = brief_by_id("trap_hole_oversize")
        ops = [dict(o) for o in b.reference]
        ops[-1] = dict(ops[-1], x=10, y=10)      # same bbox, same volume
        g = grade(b, ops)
        self.assertLess(g.shape["iou"], shape.IOU_SOLVED)
        self.assertFalse(g.solved_shape)

    def test_shape_is_reported_beside_the_envelope_verdict_not_instead_of_it(self):
        b = brief_by_id("l_bracket")
        g = grade(b, [dict(o) for o in b.reference])
        self.assertTrue(g.solved)            # the v1-comparable number survives
        self.assertTrue(g.solved_shape)      # and the new one sits beside it

    def test_iou_is_deterministic(self):
        b = brief_by_id("plate_hole_four")
        ops = [dict(o) for o in b.reference]
        self.assertEqual(grade(b, ops).shape["iou"], grade(b, ops).shape["iou"])


# --------------------------------------------------------------------------- #
# BUG 4: THE CORPUS WAS CONTAMINATED. The shell briefs probed their "inside"
#        point EXACTLY ON the outer face, so they could only pass because the
#        broken two-sided shell dilated the part outward.
# --------------------------------------------------------------------------- #
class TestShellCorpusContaminationIsFixed(unittest.TestCase):

    SHELLS = ("shell_box_3mm", "shell_tray_2mm", "shell_deep_4mm",
              "trap_shell_too_thick")

    def test_no_inside_probe_sits_on_the_outer_face(self):
        # x = 0 is the outer face of every one of these blocks. v1 probed there.
        for bid in self.SHELLS:
            b = brief_by_id(bid)
            for p in b.expect.inside:
                self.assertGreater(
                    p[0], 0.0,
                    f"{bid}: inside probe {p} sits ON the outer face x=0; it can "
                    f"only be satisfied by a part that grew outward")

    def test_every_shell_briefs_own_reference_solution_now_passes(self):
        # This is the test that would have caught the contamination in v1: with
        # the backend fixed, NO answer solved these briefs -- not even the
        # brief's own hand-written correct one.
        for bid in self.SHELLS:
            b = brief_by_id(bid)
            g = grade(b, [dict(o) for o in b.reference], shape=False)
            self.assertTrue(g.solved, f"{bid} reference: {g.reasons}")

    def test_the_whole_v1_brief_set_is_solvable_by_its_own_references(self):
        for bid in V1_BRIEFS:
            b = brief_by_id(bid)
            g = grade(b, [dict(o) for o in b.reference], shape=False)
            self.assertTrue(g.solved, f"{bid} is unsolvable: {g.reasons}")


# --------------------------------------------------------------------------- #
# BUG 5: NO STATISTICS AT ALL.
# --------------------------------------------------------------------------- #
class TestStatistics(unittest.TestCase):

    def test_wilson_interval_brackets_the_point_estimate(self):
        ci = stats.wilson(24, 72)
        self.assertLess(ci.lo, 24 / 72)
        self.assertGreater(ci.hi, 24 / 72)
        # v1's blind arm: 24/72. The audit quoted [23.6%, 44.6%] from a rounded
        # hand-calculation; the exact Wilson interval is [23.53%, 44.82%].
        self.assertAlmostEqual(ci.lo, 0.23535, places=4)
        self.assertAlmostEqual(ci.hi, 0.44820, places=4)

    def test_wilson_does_not_collapse_at_zero(self):
        ci = stats.wilson(0, 12)
        self.assertEqual(ci.lo, 0.0)
        self.assertGreater(ci.hi, 0.0)     # a Wald interval would return [0, 0]

    def test_mcnemar_on_v1s_own_eight_regressions(self):
        # v1: 8 cells blind solved and harness did not; 0 the other way.
        blind = [True] * 8 + [False] * 64
        harness = [False] * 72
        m = stats.mcnemar(blind, harness)
        self.assertEqual((m.b, m.c), (8, 0))
        self.assertAlmostEqual(m.p_value, 2.0 / 2 ** 8, places=6)
        self.assertLess(m.p_value, 0.01)   # v1 UNDER-claimed

    def test_mcnemar_ignores_concordant_cells(self):
        a = [True, True, False, False]
        b = [True, False, True, False]
        m = stats.mcnemar(a, b)
        self.assertEqual((m.b, m.c, m.discordant), (1, 1, 2))
        self.assertEqual(m.p_value, 1.0)

    def test_mcnemar_requires_aligned_arms(self):
        with self.assertRaises(ValueError):
            stats.mcnemar([True], [True, False])

    def test_pass_at_k_is_the_unbiased_estimator(self):
        self.assertEqual(stats.pass_at_k(3, 0, 1), 0.0)
        self.assertEqual(stats.pass_at_k(3, 3, 3), 1.0)
        self.assertAlmostEqual(stats.pass_at_k(3, 1, 2), 2 / 3)

    def test_pass_hat_k_is_the_conjunctive_metric(self):
        self.assertEqual(stats.pass_hat_k(3, 3, 3), 1.0)
        self.assertEqual(stats.pass_hat_k(3, 2, 3), 0.0)      # not ALL of them
        self.assertAlmostEqual(stats.pass_hat_k(3, 2, 2), 1 / 3)

    def test_pass_hat_k_is_never_above_pass_at_k(self):
        for c in range(4):
            for k in (1, 2, 3):
                self.assertLessEqual(stats.pass_hat_k(3, c, k),
                                     stats.pass_at_k(3, c, k))


# --------------------------------------------------------------------------- #
# BUG 6: THE MANDATORY BASELINE WAS NEVER RUN. Best-of-N with a reward model.
# --------------------------------------------------------------------------- #
GOOD = [{"op": "new_sketch", "plane": "XY"},
        {"op": "add_rectangle", "sketch": "sk1", "x": 0, "y": 0, "w": 40, "h": 40},
        {"op": "extrude", "sketch": "sk1", "distance": 10},
        {"op": "hole", "face_or_sketch": "solid", "x": 20, "y": 20,
         "diameter": 12, "through": True}]
#: Same brief, but the hole is bored outside the plate: it builds nothing useful.
BROKEN = [{"op": "new_sketch", "plane": "XY"},
          {"op": "add_rectangle", "sketch": "sk1", "x": 0, "y": 0, "w": 40, "h": 40},
          {"op": "extrude", "sketch": "sk1", "distance": 10},
          {"op": "shell", "faces": [], "thickness": 40}]


class TestOracleSelector(unittest.TestCase):

    def test_a_sound_candidate_outranks_a_broken_one(self):
        best, scores = oracle.rank([BROKEN, GOOD])
        self.assertEqual(best, 1)
        self.assertGreater(scores[1].key, scores[0].key)

    def test_the_oracle_reads_six_engines(self):
        s = oracle.score_ops(GOOD)
        self.assertTrue(s.built)
        self.assertTrue(s.gate_ok)
        self.assertGreaterEqual(s.engines_agreeing, 2)
        self.assertEqual(s.engines_crashed, 0)

    def test_ties_break_on_the_first_sample(self):
        best, _ = oracle.rank([GOOD, GOOD])
        self.assertEqual(best, 0)

    def test_an_empty_candidate_never_wins(self):
        best, _ = oracle.rank([[], GOOD])
        self.assertEqual(best, 1)


class TestSelectionArms(unittest.TestCase):

    def _client(self, responses):
        return ScriptedClient(responses, name="scripted")

    def test_oracle_bon_picks_the_candidate_the_oracle_prefers(self):
        b = brief_by_id("trap_hole_oversize")
        c = self._client([BROKEN, GOOD, BROKEN])
        arms = c and loops.run_sampling(c, b, seed=1, n=3)
        bon = arms[loops.ORACLE_BON]
        self.assertEqual(bon.selection["chosen"], 1)
        self.assertTrue(bon.solved)
        self.assertEqual(bon.model_calls, 3)

    def test_self_consistency_picks_the_majority_even_when_it_is_wrong(self):
        # THE control. Two votes for the broken stream, one for the correct one.
        b = brief_by_id("trap_hole_oversize")
        c = self._client([BROKEN, GOOD, BROKEN])
        arms = loops.run_sampling(c, b, seed=1, n=3)
        sc = arms[loops.SELF_CONSISTENCY]
        self.assertEqual(sc.selection["chosen"], 0)
        self.assertEqual(sc.selection["votes"], 2)
        self.assertFalse(sc.solved)

    def test_both_arms_share_the_same_draws_so_self_consistency_is_free(self):
        b = brief_by_id("trap_hole_oversize")
        c = self._client([GOOD, GOOD, GOOD])
        arms = loops.run_sampling(c, b, seed=1, n=3)
        self.assertEqual(len(c.calls), 3, "the two arms must share their N draws")
        for arm in arms.values():
            self.assertEqual(arm.model_calls, 3)

    def test_the_draws_are_seeded_and_hot(self):
        # Best-of-N does not exist at temperature 0: greedy decoding makes N
        # samples of one prompt N copies of one sample.
        b = brief_by_id("trap_hole_oversize")
        c = self._client([GOOD, GOOD, GOOD])
        loops.run_sampling(c, b, seed=100, n=3)
        self.assertEqual([s for s, _ in c.draws], [100, 101, 102])
        self.assertEqual({t for _, t in c.draws}, {loops.SAMPLING_TEMPERATURE})

    def test_no_feedback_is_ever_appended(self):
        b = brief_by_id("trap_hole_oversize")
        c = self._client([GOOD, BROKEN, GOOD])
        loops.run_sampling(c, b, seed=1, n=3)
        for _attempt, messages in c.calls:
            self.assertEqual(len(messages), 2,
                             "a selection arm has no feedback channel at all")

    def test_pass_at_k_inputs_are_recorded(self):
        b = brief_by_id("trap_hole_oversize")
        c = self._client([GOOD, BROKEN, GOOD])
        arms = loops.run_sampling(c, b, seed=1, n=3)
        sel = arms[loops.ORACLE_BON].selection
        self.assertEqual(sel["n"], 3)
        self.assertEqual(sel["n_correct"], 2)
        self.assertEqual(sel["draw_solved"], [True, False, True])


# --------------------------------------------------------------------------- #
# BUG 7: the ruler was not pinned. `DEFAULT_MESHER` is a module constant in code
#        this experiment does not own, and a mesher change silently confounds
#        every v1-vs-v2 delta -- volume, bbox and every probe are read off the
#        tessellation.
# --------------------------------------------------------------------------- #
class TestTheRulerIsPinned(unittest.TestCase):

    def test_the_experiment_pins_the_mesher_v1_ran(self):
        # The MESHER is pinned to v1's (marching cubes), against a repo default
        # that has since flipped to dual contouring. The RESOLUTION is NOT v1's
        # 48: the backend's new wall-resolution guard makes trap_shell_too_thick
        # unsolvable by every answer at 48 (feasible t<2.5 vs buildable t>=2.5),
        # so it is forced to 96 and that confound is disclosed in the report.
        self.assertEqual(session.MESHER, "marching_cubes")
        self.assertEqual(session.RESOLUTION, 96)

    def test_every_session_this_package_builds_is_pinned(self):
        s = session.frep_server("core")
        self.assertEqual(s.backend.mesher, session.MESHER)
        self.assertEqual(s.backend.resolution, session.RESOLUTION)

    def test_the_pin_survives_a_flipped_backend_default(self):
        # The property being bought: if somebody flips DEFAULT_MESHER to dual
        # contouring tomorrow (they should -- it is ~100x more accurate), this
        # experiment keeps measuring on v1's ruler and stays comparable.
        from harnesscad.io.backends import frep as frep_mod
        with mock.patch.object(frep_mod, "DEFAULT_MESHER", "dual_contouring"):
            s = session.frep_server("core")
        self.assertEqual(s.backend.mesher, "marching_cubes")

    def test_the_grader_measures_on_the_pinned_ruler(self):
        b = brief_by_id("plate_hole_four")
        from harnesscad.eval.pressure import session as sess
        seen = []
        real = sess.frep_server

        def spy(level):
            s = real(level)
            seen.append(s.backend.mesher)
            return s

        with mock.patch.object(sess, "frep_server", spy):
            grade(b, [dict(o) for o in b.reference], shape=False)
        self.assertTrue(seen)
        self.assertEqual(set(seen), {"marching_cubes"})


class TestArmsAreStillMatched(unittest.TestCase):

    def test_the_iterative_arms_still_send_one_prompt_and_one_budget(self):
        b = brief_by_id("plate_hole_four")
        seen = {}
        for arm in (loops.BLIND, loops.HARNESS):
            c = ScriptedClient(["not json"], name="m")
            loops.run_brief(c, b, arm, seed=7, max_attempts=1)
            seen[arm] = c.calls[0][1]
        self.assertEqual(seen[loops.BLIND], seen[loops.HARNESS])

    def test_the_iterative_arms_do_not_override_seed_or_temperature(self):
        b = brief_by_id("plate_hole_four")
        c = ScriptedClient(["[]"], name="m")
        loops.run_brief(c, b, loops.BLIND, seed=7, max_attempts=1)
        self.assertEqual(c.draws, [(None, None)])


if __name__ == "__main__":       # pragma: no cover
    unittest.main()
