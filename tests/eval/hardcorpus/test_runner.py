"""The hard-corpus runner and its CLI, proven offline.

The scorer is the one module allowed near the held-out briefs and grading it for
real needs the exact kernel, so the plumbing tests here swap ``score.score`` for
a stub that still CALLS THE SOLVER -- which is the part the runner owns. That
keeps these tests free of ollama, of a network and of a geometry backend, while
still exercising: the cell key, resume, persistence, the leaderboard row, and
the invalid accounting a model that emits prose must produce.

``test_a_scripted_model_is_scored_end_to_end`` is the one exception: it runs the
REAL scorer on a single brief and is skipped when cadquery/OCP is absent.
"""

from __future__ import annotations

import json
import os
import tempfile
import unittest

from harnesscad.eval.hardcorpus import cli as hc_cli
from harnesscad.eval.hardcorpus import runner as runner_mod
from harnesscad.eval.hardcorpus import score as hc_score
from harnesscad.eval.pressure.model import ScriptedClient

try:
    import cadquery  # noqa: F401
    HAVE_CQ = True
except Exception:  # noqa: BLE001
    HAVE_CQ = False

PLATE = [
    {"op": "new_sketch", "plane": "XY"},
    {"op": "add_rectangle", "sketch": "sk1", "x": 0, "y": 0, "w": 60, "h": 40},
    {"op": "extrude", "sketch": "sk1", "distance": 12},
]


class _StubScore:
    """Replace ``score.score`` with a scorer that runs the solver on N fake briefs.

    It grades nothing (no kernel): a brief "solves" when the solver returned any
    ops at all. That is enough to prove the runner's own contract -- the solver
    is called once per brief, an empty stream is a failure, and the counts reach
    the report.
    """

    def __init__(self, n=3):
        self.n = n
        self.previous = None

    def __enter__(self):
        self.previous = hc_score.score

        def stub(solver, limit=None):
            n = int(limit) if limit else self.n
            r = hc_score.HeldOutReport(n=n)
            for i in range(n):
                ops = list(solver("stub brief %d" % i))
                if ops:
                    r.built += 1
                    r.oracle_solved += 1
                    r.weak_passed += 1
                else:
                    r.failed["stub-%d" % i] = ["nothing was built"]
            return r

        hc_score.score = stub
        return self

    def __exit__(self, *exc):
        hc_score.score = self.previous
        return False


def _factory(responses):
    return lambda model: ScriptedClient(list(responses), name=model)


class TestCellId(unittest.TestCase):

    def test_every_knob_makes_a_new_cell(self):
        base = runner_mod.cell_id("m", None, 1, 0.0, 2)
        self.assertNotEqual(base, runner_mod.cell_id("m", 2, 1, 0.0, 2))
        self.assertNotEqual(base, runner_mod.cell_id("m", None, 2, 0.0, 2))
        self.assertNotEqual(base, runner_mod.cell_id("m", None, 1, 0.8, 2))
        self.assertNotEqual(base, runner_mod.cell_id("m", None, 1, 0.0, 4))
        self.assertNotEqual(base, runner_mod.cell_id("m", None, 1, 0.0, 2, 4096))
        self.assertEqual(base, runner_mod.cell_id("m", 0, 1, 0.0, 2))


class TestRunner(unittest.TestCase):

    def test_a_scripted_model_produces_a_persisted_report(self):
        with tempfile.TemporaryDirectory() as tmp, _StubScore(n=3):
            out = os.path.join(tmp, "results.json")
            payload = runner_mod.run(
                models=["scripted"], out=out, cache_dir=os.path.join(tmp, "c"),
                client_factory=_factory([PLATE, PLATE, PLATE]), log=lambda s: None)
            self.assertEqual(len(payload["submissions"]), 1)
            row = payload["submissions"][0]
            self.assertEqual(row["name"], "scripted")
            self.assertEqual(row["n"], 3)
            self.assertEqual(row["oracle_solved"], 3)
            self.assertEqual(row["solver"]["stats"]["invalid"], 0)
            with open(out, encoding="utf-8") as fh:
                self.assertEqual(json.load(fh)["submissions"][0]["name"], "scripted")

    def test_limit_scores_only_the_first_briefs(self):
        with tempfile.TemporaryDirectory() as tmp, _StubScore(n=9):
            payload = runner_mod.run(
                models=["scripted"], out=os.path.join(tmp, "r.json"), limit=2,
                cache_dir=os.path.join(tmp, "c"),
                client_factory=_factory([PLATE, PLATE]), log=lambda s: None)
            self.assertEqual(payload["submissions"][0]["n"], 2)
            self.assertEqual(payload["meta"]["limit"], 2)

    def test_unparseable_output_is_an_invalid_attempt_not_a_crash(self):
        with tempfile.TemporaryDirectory() as tmp, _StubScore(n=2):
            payload = runner_mod.run(
                models=["scripted"], out=os.path.join(tmp, "r.json"),
                cache_dir=os.path.join(tmp, "c"), max_attempts=1,
                client_factory=_factory(["I am a helpful assistant.", PLATE]),
                log=lambda s: None)
            row = payload["submissions"][0]
            self.assertEqual(row["n"], 2)
            self.assertEqual(row["solver"]["stats"]["invalid"], 1)
            self.assertEqual(row["oracle_solved"], 1)
            self.assertTrue(row["failed"])

    def test_an_unreachable_model_does_not_kill_the_run(self):
        class Dead:
            name = "dead"

            def complete(self, messages, attempt, seed=None, temperature=None):
                raise RuntimeError("connection refused")

        with tempfile.TemporaryDirectory() as tmp, _StubScore(n=2):
            payload = runner_mod.run(
                models=["dead"], out=os.path.join(tmp, "r.json"),
                cache_dir=os.path.join(tmp, "c"), max_attempts=1,
                client_factory=lambda m: Dead(), log=lambda s: None)
            stats = payload["submissions"][0]["solver"]["stats"]
            self.assertEqual(stats["errored"], 2)
            self.assertEqual(stats["invalid"], 2)

    def test_a_finished_model_is_skipped_on_a_re_run(self):
        with tempfile.TemporaryDirectory() as tmp, _StubScore(n=2):
            out = os.path.join(tmp, "r.json")
            cache = os.path.join(tmp, "c")
            runner_mod.run(models=["scripted"], out=out, cache_dir=cache,
                           client_factory=_factory([PLATE, PLATE]),
                           log=lambda s: None)
            calls = []

            def counting(model):
                calls.append(model)
                return ScriptedClient([PLATE, PLATE], name=model)

            payload = runner_mod.run(models=["scripted"], out=out,
                                     cache_dir=cache, client_factory=counting,
                                     log=lambda s: None)
            self.assertEqual(calls, [], "the finished cell was re-run")
            self.assertEqual(len(payload["submissions"]), 1)

    def test_no_resume_re_runs_the_cell(self):
        with tempfile.TemporaryDirectory() as tmp, _StubScore(n=2):
            out = os.path.join(tmp, "r.json")
            cache = os.path.join(tmp, "c")
            runner_mod.run(models=["scripted"], out=out, cache_dir=cache,
                           client_factory=_factory([PLATE, PLATE]),
                           log=lambda s: None)
            payload = runner_mod.run(models=["scripted"], out=out,
                                     cache_dir=cache, resume=False,
                                     client_factory=_factory([PLATE, PLATE]),
                                     log=lambda s: None)
            self.assertEqual(len(payload["submissions"]), 1)

    def test_the_completion_cache_serves_a_repeat_run(self):
        with tempfile.TemporaryDirectory() as tmp, _StubScore(n=2):
            cache = os.path.join(tmp, "c")
            runner_mod.run(models=["scripted"], out=os.path.join(tmp, "a.json"),
                           cache_dir=cache, client_factory=_factory([PLATE, PLATE]),
                           log=lambda s: None)
            # A different results file is a fresh cell, but the completions are
            # already on disk: the scripted client is never asked again.
            empty = ScriptedClient([], name="scripted")
            payload = runner_mod.run(models=["scripted"],
                                     out=os.path.join(tmp, "b.json"),
                                     cache_dir=cache,
                                     client_factory=lambda m: empty,
                                     log=lambda s: None)
            self.assertEqual(empty.calls, [])
            self.assertEqual(payload["submissions"][0]["oracle_solved"], 2)

    def test_standings_carry_both_columns(self):
        with tempfile.TemporaryDirectory() as tmp, _StubScore(n=2):
            payload = runner_mod.run(
                models=["scripted"], out=os.path.join(tmp, "r.json"),
                cache_dir=os.path.join(tmp, "c"),
                client_factory=_factory([PLATE, PLATE]), log=lambda s: None)
        rows = runner_mod.standings(payload)
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0].name, "scripted")
        self.assertEqual(rows[0].oracle_rate, 1.0)
        self.assertEqual(rows[0].weak_rate, 1.0)


class TestCli(unittest.TestCase):

    def _args(self, argv):
        import argparse

        parser = hc_cli.add_arguments(argparse.ArgumentParser())
        return parser.parse_args(argv)

    def test_the_flags_the_sweep_needs_all_parse(self):
        a = self._args(["--model", "ornith:9b", "--model", "qwen3.6:27b",
                        "--limit", "2", "--out", "r.json", "--cache", "cdir",
                        "--json"])
        self.assertEqual(a.model, ["ornith:9b", "qwen3.6:27b"])
        self.assertEqual(a.limit, 2)
        self.assertEqual(a.out, "r.json")
        self.assertEqual(a.cache, "cdir")
        self.assertTrue(a.json)

    def test_the_token_budget_is_exposed(self):
        a = self._args(["--max-tokens", "4096"])
        self.assertEqual(a.max_tokens, 4096)
        self.assertEqual(self._args([]).max_tokens,
                         runner_mod.DEFAULT_MAX_TOKENS)

    def test_the_verb_is_wired_into_the_product_cli(self):
        from harnesscad.core import cli as core_cli

        args = core_cli.build_parser().parse_args(
            ["hardcorpus", "--model", "ornith:9b", "--limit", "2"])
        self.assertEqual(args.func, core_cli.cmd_hardcorpus)
        self.assertEqual(args.limit, 2)

    def test_run_scores_and_prints_through_the_cli(self):
        with tempfile.TemporaryDirectory() as tmp, _StubScore(n=2):
            out = os.path.join(tmp, "r.json")
            args = self._args(["--model", "scripted", "--out", out,
                               "--cache", os.path.join(tmp, "c"), "--json"])
            # The CLI builds its own client, so point it at the stub by hand.
            payload = runner_mod.run(models=["scripted"], out=out,
                                     cache_dir=os.path.join(tmp, "c"),
                                     client_factory=_factory([PLATE, PLATE]),
                                     log=lambda s: None)
            args.report = out
            self.assertEqual(hc_cli.run(args), 0)
            self.assertEqual(payload["submissions"][0]["n"], 2)

    def test_report_on_a_missing_file_is_an_error_not_a_traceback(self):
        args = self._args(["--report", "no_such_results.json"])
        self.assertEqual(hc_cli.run(args), 2)

    def test_rows_render_both_columns_and_the_gap(self):
        from harnesscad.eval.leaderboard import hardcorpus_board as board

        s = board.Standing(name="m", n=10, built=8, oracle_solved=3,
                           weak_passed=7, field_fooled=4)
        text = hc_cli.render_rows([s])
        self.assertIn("weak", text)
        self.assertIn("oracle", text)
        self.assertIn("0.700", text)
        self.assertIn("0.300", text)

    def test_the_invalid_table_names_unmeasurable_output(self):
        payload = {"submissions": [{"name": "m", "n": 4,
                                    "solver": {"stats": {"invalid": 3,
                                                         "errored": 0,
                                                         "model_calls": 7}}}]}
        text = hc_cli._render_invalid(payload)
        self.assertIn("UNMEASURABLE", text)
        self.assertIn("3", text)


class TestRealScorer(unittest.TestCase):

    @unittest.skipUnless(HAVE_CQ, "cadquery/OCP not installed")
    def test_a_scripted_model_is_scored_end_to_end(self):
        """One brief, one scripted answer, the REAL oracle. No ollama."""
        from harnesscad.eval.hardcorpus.solver import ModelSolver

        solver = ModelSolver(ScriptedClient([PLATE], name="scripted"))
        report = hc_score.score(solver, limit=1)
        self.assertEqual(report.n, 1)
        self.assertEqual(len(solver.records), 1)
        d = report.to_dict()
        self.assertIn("oracle_rate", d)
        self.assertIn("weak_rate", d)

    @unittest.skipUnless(HAVE_CQ, "cadquery/OCP not installed")
    def test_the_reference_submission_is_the_ceiling(self):
        row = runner_mod.reference_submission(limit=2)
        self.assertEqual(row["name"], "reference")
        self.assertEqual(row["oracle_solved"], row["n"])


if __name__ == "__main__":
    unittest.main()
