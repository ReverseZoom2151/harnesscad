"""Offline tests for the pressure harness.

Not one of these touches ollama, litellm or the network: every model is a
``ScriptedClient`` replaying canned text. The geometry, the verifier fleet, the
grader, the loops, the cache and the report are all the real ones, so the suite
proves the machinery without proving anything about the models.
"""

from __future__ import annotations

import json
import os
import tempfile
import unittest

from harnesscad.eval.pressure import briefs as briefs_mod
from harnesscad.eval.pressure import prompts, report
from harnesscad.eval.pressure.briefs import BRIEFS, brief_by_id, briefs_for
from harnesscad.eval.pressure.cache import CompletionCache, cache_key
from harnesscad.eval.pressure.cli import add_arguments, run as cli_run
from harnesscad.eval.pressure.loops import BLIND, HARNESS, VERIFY_LEVEL, run_brief
from harnesscad.eval.pressure.metrics import grade
from harnesscad.eval.pressure.model import CachedClient, ScriptedClient, extract_ops
from harnesscad.eval.pressure.runner import run as run_grid

PLATE = [
    {"op": "new_sketch", "plane": "XY"},
    {"op": "add_rectangle", "sketch": "sk1", "x": 0, "y": 0, "w": 60, "h": 40},
    {"op": "extrude", "sketch": "sk1", "distance": 5},
]


# --------------------------------------------------------------------------- #
# the corpus is real and solvable
# --------------------------------------------------------------------------- #
class TestCorpus(unittest.TestCase):
    def test_corpus_is_nontrivial(self):
        self.assertGreaterEqual(len(BRIEFS), 20)
        self.assertLessEqual(len(BRIEFS), 30)
        self.assertEqual(len({b.id for b in BRIEFS}), len(BRIEFS), "duplicate ids")
        self.assertGreaterEqual(sum(1 for b in BRIEFS if b.trap), 4)

    def test_every_reference_solution_passes_its_own_grader(self):
        """If a brief's own known-good answer does not grade as solved, the brief
        is unsolvable and would silently depress BOTH arms."""
        failures = []
        for b in BRIEFS:
            g = grade(b, list(b.reference))
            if not g.solved:
                failures.append((b.id, g.reasons))
        self.assertEqual(failures, [], f"unsolvable briefs: {failures}")

    def test_selectors(self):
        self.assertEqual(len(briefs_for("all")), len(BRIEFS))
        self.assertTrue(all(b.trap for b in briefs_for("traps")))
        self.assertTrue(all(not b.trap for b in briefs_for("notraps")))
        self.assertEqual([b.id for b in briefs_for("plate_60x40x5")],
                         ["plate_60x40x5"])
        with self.assertRaises(KeyError):
            briefs_for("no_such_brief")


# --------------------------------------------------------------------------- #
# the grader is channel-blind and actually checks geometry
# --------------------------------------------------------------------------- #
class TestGrader(unittest.TestCase):
    def test_correct_plate_solves(self):
        g = grade(brief_by_id("plate_60x40x5"), PLATE)
        self.assertTrue(g.solved, g.reasons)
        self.assertTrue(g.built)

    def test_wrong_thickness_fails(self):
        ops = list(PLATE[:2]) + [
            {"op": "extrude", "sketch": "sk1", "distance": 50}]
        g = grade(brief_by_id("plate_60x40x5"), ops)
        self.assertFalse(g.solved)
        self.assertTrue(any("bounding box" in r for r in g.reasons), g.reasons)

    def test_hole_outside_the_plate_is_caught_by_the_grader(self):
        """The probe is the point: a hole at (500, 500) removes no material, and
        NOTHING in the verifier fleet notices. The grader must."""
        b = brief_by_id("plate_hole_centre")
        ops = [
            {"op": "new_sketch", "plane": "XY"},
            {"op": "add_rectangle", "sketch": "sk1", "x": 0, "y": 0, "w": 60, "h": 40},
            {"op": "extrude", "sketch": "sk1", "distance": 12},
            {"op": "hole", "face_or_sketch": "solid", "x": 500, "y": 500,
             "diameter": 8, "through": True},
        ]
        g = grade(b, ops)
        self.assertFalse(g.solved)
        self.assertTrue(any("should be empty" in r for r in g.reasons), g.reasons)
        # ... and this is a FLEET HOLE: wrong geometry, silent fleet.
        self.assertTrue(g.built)
        self.assertFalse(g.fleet_caught)
        self.assertTrue(g.fleet_missed)

    def test_no_ops_is_not_solved(self):
        g = grade(brief_by_id("plate_60x40x5"), [])
        self.assertFalse(g.solved)
        self.assertFalse(g.built)


# --------------------------------------------------------------------------- #
# the parser
# --------------------------------------------------------------------------- #
class TestExtractOps(unittest.TestCase):
    def test_bare_array(self):
        self.assertTrue(extract_ops(json.dumps(PLATE)).ok)

    def test_markdown_fenced(self):
        raw = "Here you go:\n```json\n" + json.dumps(PLATE) + "\n```\nEnjoy."
        parsed = extract_ops(raw)
        self.assertTrue(parsed.ok, parsed.error)
        self.assertEqual(len(parsed.ops), 3)

    def test_prose_then_array(self):
        parsed = extract_ops("Sure! " + json.dumps(PLATE))
        self.assertTrue(parsed.ok, parsed.error)

    def test_garbage_returns_a_feedable_error(self):
        parsed = extract_ops("I cannot help with that.")
        self.assertFalse(parsed.ok)
        self.assertIsInstance(parsed.error, str)
        self.assertTrue(parsed.error)

    def test_unknown_op_is_an_error(self):
        parsed = extract_ops('[{"op":"teleport","x":1}]')
        self.assertFalse(parsed.ok)
        self.assertIn("teleport", parsed.error)


# --------------------------------------------------------------------------- #
# the feedback channels -- the independent variable
# --------------------------------------------------------------------------- #
class TestFeedback(unittest.TestCase):
    def test_blind_says_nothing_when_the_kernel_is_happy(self):
        self.assertIsNone(prompts.format_blind({"ok": True, "diagnostics": []}))

    def test_blind_carries_no_code_and_no_location(self):
        result = {"ok": False, "rejected": {"op": "shell"}, "diagnostics": [
            {"severity": "error", "code": "bad-value",
             "message": "shell thickness must be > 0 (got -1)", "where": "op[3]"},
        ]}
        fb = prompts.format_blind(result)
        self.assertIn("shell thickness must be > 0", fb)
        self.assertNotIn("bad-value", fb)      # no stable code
        self.assertNotIn("op[3]", fb)          # no location
        self.assertIn("Traceback", fb)

    def test_typed_carries_the_code_and_the_location(self):
        result = {"ok": True, "diagnostics": [
            {"severity": "error", "code": "infeasible-plan",
             "message": "shell thickness 9 mm >= available stock 5 mm; the wall "
                        "consumes the whole solid.", "where": "op[3]"},
        ]}
        fb = prompts.format_typed(result)
        self.assertIn("infeasible-plan", fb)
        self.assertIn("op[3]", fb)
        self.assertIn("consumes the whole solid", fb)

    def test_typed_ignores_diagnostics_no_model_can_act_on(self):
        result = {"ok": True, "diagnostics": [
            {"severity": "error", "code": "missing-metadata",
             "message": "part carries no name", "where": "part.name"},
            {"severity": "warning", "code": "non-preferred-dimension",
             "message": "30 mm is not an ISO preferred number", "where": "h"},
            {"severity": "info", "code": "dfm-not-yet-measurable",
             "message": "thin-wall not measured", "where": None},
        ]}
        self.assertIsNone(prompts.format_typed(result))

    def test_typed_keeps_preflight_warnings_because_they_mean_infeasible(self):
        result = {"ok": True, "diagnostics": [
            {"severity": "warning", "code": "preflight-RADIUS_TOO_LARGE",
             "message": "Fillet radius 8 exceeds half the smallest extent (6).",
             "where": "op[3]:fillet"},
        ]}
        fb = prompts.format_typed(result)
        self.assertIsNotNone(fb)
        self.assertIn("RADIUS_TOO_LARGE", fb)

    def test_the_two_arms_differ_only_in_the_formatter(self):
        self.assertEqual(VERIFY_LEVEL[BLIND], "core")
        self.assertEqual(VERIFY_LEVEL[HARNESS], "full")
        self.assertIs(prompts.FEEDBACK[BLIND], prompts.format_blind)
        self.assertIs(prompts.FEEDBACK[HARNESS], prompts.format_typed)


# --------------------------------------------------------------------------- #
# the loops
# --------------------------------------------------------------------------- #
class TestLoops(unittest.TestCase):
    def test_first_shot_success_uses_one_attempt_in_both_arms(self):
        b = brief_by_id("plate_60x40x5")
        for loop in (BLIND, HARNESS):
            client = ScriptedClient([PLATE], name="m")
            res = run_brief(client, b, loop, seed=1, max_attempts=4)
            self.assertTrue(res.solved, (loop, res.final_reasons))
            self.assertEqual(res.attempts_used, 1, loop)
            self.assertEqual(res.attempts_to_solve, 1, loop)
            self.assertEqual(res.invalid_ops, 0, loop)

    def test_unparseable_then_correct_costs_a_retry(self):
        b = brief_by_id("plate_60x40x5")
        client = ScriptedClient(["I'm sorry, I can't do that.", PLATE], name="m")
        res = run_brief(client, b, HARNESS, seed=1, max_attempts=4)
        self.assertTrue(res.solved, res.final_reasons)
        self.assertEqual(res.attempts_used, 2)
        self.assertEqual(res.attempts_to_solve, 2)
        self.assertEqual(res.invalid_ops, 1)

    def test_budget_is_respected(self):
        b = brief_by_id("plate_60x40x5")
        client = ScriptedClient(["nope"] * 10, name="m")
        res = run_brief(client, b, BLIND, seed=1, max_attempts=3)
        self.assertFalse(res.solved)
        self.assertEqual(res.attempts_used, 3)
        self.assertEqual(res.invalid_ops, 3)

    def test_the_trap_is_the_whole_experiment(self):
        """On an infeasible shell, the blind arm's kernel is HAPPY -- it stops
        after one attempt with wrong geometry and is never told otherwise. The
        harness arm is handed the typed diagnostic and gets a second attempt."""
        b = brief_by_id("trap_shell_too_thick")
        bad = PLATE + [{"op": "shell", "faces": [], "thickness": 9}]
        good = PLATE + [{"op": "shell", "faces": [], "thickness": 1.5}]

        blind = run_brief(ScriptedClient([bad, good], name="m"), b, BLIND,
                          seed=1, max_attempts=4)
        self.assertEqual(blind.attempts_used, 1,
                         "core verify accepted an infeasible shell, so the blind "
                         "arm had no reason to retry -- that is the finding")
        self.assertFalse(blind.solved)

        harness = run_brief(ScriptedClient([bad, good], name="m"), b, HARNESS,
                            seed=1, max_attempts=4)
        self.assertEqual(harness.attempts_used, 2)
        self.assertTrue(harness.solved, harness.final_reasons)
        self.assertGreaterEqual(harness.fleet_caught, 1)
        # The diagnostic the model was shown named the mistake:
        fb = harness.records[0]["feedback"]
        self.assertIn("infeasible-plan", fb)
        self.assertIn("consumes the whole solid", fb)

    def test_a_regression_after_a_correct_attempt_is_scored_honestly(self):
        """If an arm has a correct plan and then breaks it, the FINAL plan is what
        it stood behind, so it loses the brief. Anything else would flatter the
        loop that churns."""
        b = brief_by_id("plate_60x40x5")
        wrong = list(PLATE[:2]) + [
            {"op": "extrude", "sketch": "sk1", "distance": 50}]
        client = ScriptedClient([wrong, wrong], name="m")
        res = run_brief(client, b, HARNESS, seed=1, max_attempts=2)
        self.assertFalse(res.solved)

    def test_unknown_loop_rejected(self):
        with self.assertRaises(ValueError):
            run_brief(ScriptedClient([PLATE]), brief_by_id("plate_60x40x5"),
                      "sideways", seed=1)


# --------------------------------------------------------------------------- #
# the cache
# --------------------------------------------------------------------------- #
class TestCache(unittest.TestCase):
    def test_key_is_stable_and_sensitive(self):
        msgs = [{"role": "user", "content": "hi"}]
        k1 = cache_key("m", 1, 0.0, 1, msgs)
        self.assertEqual(k1, cache_key("m", 1, 0.0, 1, msgs))
        self.assertNotEqual(k1, cache_key("m", 2, 0.0, 1, msgs))     # seed
        self.assertNotEqual(k1, cache_key("m", 1, 0.7, 1, msgs))     # temperature
        self.assertNotEqual(k1, cache_key("m", 1, 0.0, 2, msgs))     # attempt
        self.assertNotEqual(k1, cache_key("n", 1, 0.0, 1, msgs))     # model
        self.assertNotEqual(
            k1, cache_key("m", 1, 0.0, 1, [{"role": "user", "content": "ho"}]))

    def test_roundtrip_and_miss(self):
        with tempfile.TemporaryDirectory() as d:
            c = CompletionCache(d)
            self.assertIsNone(c.get("deadbeef"))
            c.put("deadbeef", {"text": "hello"})
            self.assertEqual(c.get("deadbeef")["text"], "hello")
            self.assertEqual(c.stats(), {"hits": 1, "misses": 1})

    def test_corrupt_entry_is_a_miss_not_a_crash(self):
        with tempfile.TemporaryDirectory() as d:
            c = CompletionCache(d)
            with open(c.path("abc"), "w", encoding="utf-8") as fh:
                fh.write("{ this is not json")
            self.assertIsNone(c.get("abc"))

    def test_cached_client_calls_the_model_once(self):
        with tempfile.TemporaryDirectory() as d:
            inner = ScriptedClient([json.dumps(PLATE)], name="m")
            client = CachedClient(inner, CompletionCache(d), seed=3,
                                  temperature=0.0)
            msgs = [{"role": "user", "content": "build a plate"}]
            a = client.complete(msgs, 1)
            b = client.complete(msgs, 1)
            self.assertEqual(a, b)
            self.assertEqual(len(inner.calls), 1, "the second call must be a cache hit")

    def test_a_rerun_is_free_and_identical(self):
        b = brief_by_id("plate_60x40x5")
        with tempfile.TemporaryDirectory() as d:
            cache = CompletionCache(d)
            inner = ScriptedClient([PLATE], name="m")
            run_brief(CachedClient(inner, cache, 5, 0.0), b, HARNESS, seed=5)
            self.assertEqual(len(inner.calls), 1)
            # Second run: the scripted client has NO responses left, so if the
            # cache were not working this would fail outright.
            res2 = run_brief(CachedClient(inner, cache, 5, 0.0), b, HARNESS, seed=5)
            self.assertTrue(res2.solved, res2.final_reasons)
            self.assertEqual(len(inner.calls), 1, "no new model calls on a re-run")


# --------------------------------------------------------------------------- #
# the runner, the report and the CLI
# --------------------------------------------------------------------------- #
class _RoutedClient:
    """A deterministic stand-in that answers by CONTENT, not by call order.

    A flat response queue cannot be used here, because the completion cache
    (correctly) makes both arms share attempt 1 -- their message lists are
    byte-identical, so the second arm gets a cache hit and never calls the model.
    That is a property worth having: it guarantees the two arms start from the
    SAME first plan, so any divergence is caused by the feedback and nothing
    else. This client therefore behaves like a real model: same input, same
    output; a shell brief gets an infeasible shell first, and only repairs it if
    it is actually told that the shell is infeasible.
    """

    def __init__(self, name="m"):
        self.name = name
        self.calls = 0

    def complete(self, messages, attempt):
        self.calls += 1
        last = messages[-1]["content"]
        first = messages[1]["content"]
        told_it_is_infeasible = "infeasible-plan" in last
        if "hollowed out" in first:                     # the trap brief
            t = 1.5 if told_it_is_infeasible else 9
            return json.dumps(PLATE + [{"op": "shell", "faces": [], "thickness": t}])
        return json.dumps(PLATE)                        # the plate brief


class TestRunnerAndReport(unittest.TestCase):
    def _grid(self, out, cache_dir, resume=True):
        return run_grid(
            models=["m"],
            briefs=[brief_by_id("plate_60x40x5"), brief_by_id("trap_shell_too_thick")],
            seed=11, out=out, cache_dir=cache_dir, resume=resume,
            client_factory=lambda name: _RoutedClient(name),
            log=lambda s: None,
        )

    def test_grid_runs_and_is_resumable(self):
        with tempfile.TemporaryDirectory() as d:
            out = os.path.join(d, "r.json")
            payload = self._grid(out, os.path.join(d, "cache"))
            self.assertEqual(len(payload["results"]), 4)   # 1 model x 2 briefs x 2 loops
            self.assertEqual(payload["meta"]["seed"], 11)
            self.assertTrue(os.path.exists(out))

            # Resume: every cell is already present, so nothing re-runs.
            again = self._grid(out, os.path.join(d, "cache"))
            self.assertEqual(len(again["results"]), 4)
            self.assertEqual(
                sorted(r["brief"] + "|" + r["loop"] for r in again["results"]),
                sorted(r["brief"] + "|" + r["loop"] for r in payload["results"]))

    def test_report_renders_and_reports_the_delta(self):
        with tempfile.TemporaryDirectory() as d:
            out = os.path.join(d, "r.json")
            payload = self._grid(out, os.path.join(d, "cache"))
            text = report.render_all(payload)
            self.assertIn("HEADLINE", text)
            self.assertIn("VERDICT", text)
            self.assertIn("blind", text)
            self.assertIn("harness", text)

            agg = report.aggregate(payload["results"])["cells"]
            self.assertEqual(agg[("m", BLIND)]["n"], 2)
            self.assertEqual(agg[("m", HARNESS)]["n"], 2)
            # The scripted model solves the plate in both arms, but only the
            # harness arm is ever TOLD about the infeasible shell.
            self.assertEqual(agg[("m", BLIND)]["solved"], 1)
            self.assertEqual(agg[("m", HARNESS)]["solved"], 2)

    def test_cli_list_briefs(self):
        import argparse
        p = add_arguments(argparse.ArgumentParser())
        self.assertEqual(cli_run(p.parse_args(["--list-briefs"])), 0)

    def test_cli_report_of_a_results_file(self):
        import argparse
        with tempfile.TemporaryDirectory() as d:
            out = os.path.join(d, "r.json")
            self._grid(out, os.path.join(d, "cache"))
            p = add_arguments(argparse.ArgumentParser())
            self.assertEqual(cli_run(p.parse_args(["--report", out])), 0)
            self.assertEqual(cli_run(p.parse_args(["--report", out, "--json"])), 0)

    def test_cli_report_of_a_missing_file_is_an_error(self):
        import argparse
        p = add_arguments(argparse.ArgumentParser())
        self.assertEqual(cli_run(p.parse_args(["--report", "nope.json"])), 2)


class TestMainCliStillWorks(unittest.TestCase):
    """The new subcommand must not disturb the existing ones."""

    def test_all_subcommands_still_parse(self):
        from harnesscad.core.cli import build_parser

        parser = build_parser()
        sub = [a for a in parser._actions if a.dest == "command"][0]
        for name in ("apply", "demo", "build", "formats", "export", "ingest",
                     "reconstruct", "program", "spec", "procedural", "bench",
                     "report", "dataset", "capabilities", "pressure"):
            self.assertIn(name, sub.choices, f"subcommand '{name}' went missing")

    def test_pressure_subcommand_parses(self):
        from harnesscad.core.cli import build_parser

        args = build_parser().parse_args(
            ["pressure", "--model", "qwen2.5-coder:3b", "--loop", "both",
             "--briefs", "all", "--out", "results.json"])
        self.assertEqual(args.model, ["qwen2.5-coder:3b"])
        self.assertEqual(args.loop, "both")


if __name__ == "__main__":
    unittest.main()
