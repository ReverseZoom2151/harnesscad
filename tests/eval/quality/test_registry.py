"""Tests for the quality analysis surface (harnesscad.eval.quality.registry).

Quality modules ANALYSE (a number, a graph, a matrix); verifiers GATE (a
diagnostic). What these tests pin down:

*   the registry DISCOVERS real quality modules from the tree (not a stub list);
*   it produces a report for a live model state (the demo op stream on the stub
    backend), and every number is stamped with the module that produced it;
*   an analyser whose input the state does not carry is SKIPPED, never guessed;
*   an analyser that raises becomes an error entry and the rest still run;
*   rival analysers (three anomaly scorers, two reward functions) are exposed by
    name, disagree on the SAME input, and are never averaged into one score;
*   the `report` CLI subcommand works and the existing subcommands still do.
"""

from __future__ import annotations

import io
import json
import os
import tempfile
import unittest
from contextlib import redirect_stdout

from harnesscad.core import cli
from harnesscad.core.cisp.ops import parse_op
from harnesscad.core.loop import HarnessSession
from harnesscad.eval.quality import registry as quality
from harnesscad.io.backends.stub import StubBackend

DEMO_OPS = [
    {"op": "new_sketch", "plane": "XY"},
    {"op": "add_rectangle", "sketch": "sk1", "x": 0.0, "y": 0.0, "w": 20.0, "h": 10.0},
    {"op": "extrude", "sketch": "sk1", "distance": 5.0},
]


def _state(extras=None):
    session = HarnessSession(StubBackend())
    result = session.apply_ops([parse_op(o) for o in DEMO_OPS])
    assert result.ok
    return quality.model_state(session.backend, session.opdag, extras)


class TestDiscovery(unittest.TestCase):
    def test_discovers_more_than_ten_real_modules(self):
        modules = quality.adapted_modules()
        self.assertGreater(len(modules), 10, modules)
        for dotted in modules:
            self.assertTrue(dotted.startswith("harnesscad.eval.quality."), dotted)
            self.assertIsNotNone(quality.capability_registry.get(dotted), dotted)

    def test_analysers_cover_several_kinds(self):
        got = {a.kind for a in quality.analysers()}
        for kind in ("geometry", "sequence", "physics", "graph", "report", "reward"):
            self.assertIn(kind, got)

    def test_unadapted_is_reported_not_hidden(self):
        left = quality.unadapted()
        self.assertTrue(left)  # honesty: not every quality module has a call site
        self.assertFalse(set(left) & set(quality.adapted_modules()))

    def test_order_is_deterministic(self):
        self.assertEqual([a.name for a in quality.analysers()],
                         [a.name for a in quality.analysers()])


class TestReport(unittest.TestCase):
    def test_report_for_a_model_state(self):
        rep = quality.report(_state({"brief": {"text": "a 20x10x5 plate"}}))
        self.assertEqual([], [(r.name, r.error) for r in rep.errors()])
        self.assertGreater(len(rep.ok()), 5)

        complexity = rep.value("sequence.complexity_taxonomy")
        self.assertIn("level", complexity)
        self.assertGreaterEqual(complexity["level"], 1)

        mass = rep.value("physics.mass_properties")
        self.assertAlmostEqual(20.0 * 10.0 * 5.0, mass["total_volume"])
        self.assertAlmostEqual(1000.0, mass["total_mass"])   # density 1.0

        pose = rep.value("geometry.canonical_pose")
        self.assertFalse(pose["centered"])       # the demo plate sits in the +XY quadrant
        self.assertEqual([20.0, 10.0, 5.0], list(pose["extents"]))

        params = rep.value("report.parameter_exposure")
        self.assertGreater(params["n_fields"], 0)

        graph = rep.value("graph.intent_graph")
        self.assertTrue(graph["causal_order"])

        trace = rep.value("report.traceability")
        self.assertGreater(trace["n_rows"], 0)

        # every number is stamped with the module that produced it
        for r in rep.ok():
            self.assertTrue(r.dotted.startswith("harnesscad.eval.quality."), r.dotted)

    def test_report_is_deterministic(self):
        a = quality.report(_state({"brief": {"text": "a 20x10x5 plate"}}))
        b = quality.report(_state({"brief": {"text": "a 20x10x5 plate"}}))
        self.assertEqual(a.to_json(), b.to_json())

    def test_absent_input_is_skipped_never_guessed(self):
        rep = quality.report(_state())
        skipped = {r.name for r in rep.skipped()}
        # the stub backend has no mesh, no reward components, no anomaly corpus
        for name in ("geometry.mesh_stability", "reward.composite",
                     "geometry.anomaly_zscore", "physics.beam_screening"):
            self.assertIn(name, skipped)
        for r in rep.skipped():
            self.assertEqual({}, r.value)   # a skip carries NO fabricated number

    def test_a_raising_analyser_is_survived(self):
        class Boom:
            name = "geometry.boom"
            kind = "geometry"
            dotted = "harnesscad.eval.quality.geometry.anomaly"

            def applies_to(self, state):
                return True

            def analyse(self, state):
                raise RuntimeError("detonate")

        fleet = list(quality.analysers()) + [Boom()]
        results = quality.analyse(_state(), fleet=fleet)
        errors = [r for r in results if r.status == "error"]
        self.assertEqual(1, len(errors))
        self.assertIn("RuntimeError: detonate", errors[0].error)
        # the rest of the fleet still produced its numbers
        self.assertTrue([r for r in results if r.status == "ok"])

    def test_report_audits_its_own_prose(self):
        rep = quality.report(_state())
        self.assertIn("findings", rep.claims)
        self.assertIsInstance(rep.claims["findings"], list)

    def test_analyse_can_be_filtered_by_kind(self):
        results = quality.analyse(_state(), kinds_=["physics"])
        self.assertTrue(results)
        self.assertEqual({"physics"}, {r.kind for r in results})


class TestRivalsAreNeverAveraged(unittest.TestCase):
    #: A reference corpus of known-good feature vectors, plus one obvious outlier.
    REFERENCE = [{"volume": 100.0 + i, "bbox_diagonal": 10.0 + i * 0.1}
                 for i in range(20)]

    def _anomaly_state(self):
        return _state({"anomaly": {"reference": self.REFERENCE}})

    def test_three_anomaly_scorers_are_exposed_by_name(self):
        names = {a.name for a in quality.analysers(kind="geometry")}
        for n in ("geometry.anomaly_zscore", "geometry.anomaly_iqr",
                  "geometry.anomaly_isolation"):
            self.assertIn(n, names)
        self.assertIn("anomaly_score", quality.rivals())
        self.assertEqual(3, len(quality.rivals()["anomaly_score"]))
        # all three adapt the SAME module under DIFFERENT protocols
        self.assertEqual({"harnesscad.eval.quality.geometry.anomaly"},
                         {quality.analyser(n).dotted
                          for n in quality.rivals()["anomaly_score"]})

    def test_rival_scores_stay_under_distinct_names(self):
        rep = quality.report(self._anomaly_state())
        keys = set(rep.scores)
        scored = [k for k in keys if k.startswith("geometry.anomaly_")]
        self.assertGreaterEqual(len(scored), 2, keys)
        # NO blended "anomaly" key exists anywhere in the report
        self.assertNotIn("anomaly", keys)
        self.assertNotIn("geometry.anomaly", keys)
        for key in scored:
            self.assertTrue(key.endswith(".score"), key)

    def test_rival_rewards_are_separate_analysers_on_separate_protocols(self):
        state = _state({"reward": {
            "components": {"code_valid": 1.0, "execution_valid": 1.0, "geometry": 0.4},
            "text": "solid = cq.Workplane('XY').rect(20, 10).extrude(5)",
            "cd": 0.25,
        }})
        rep = quality.report(state)
        composite = rep.value("reward.composite")
        execution = rep.value("reward.execution")
        self.assertIsNotNone(composite)
        self.assertIsNotNone(execution)
        # same question, different protocols -> different totals, both reported
        self.assertNotAlmostEqual(composite["total"], execution["total"])
        self.assertIn("reward.composite.total", rep.scores)
        self.assertIn("reward.execution.total", rep.scores)
        self.assertNotIn("reward.total", rep.scores)   # never pooled

    def test_every_rival_family_member_exists(self):
        for family, members in quality.RIVAL_FAMILIES:
            self.assertGreater(len(members), 1, family)
            for name in members:
                self.assertEqual(name, quality.analyser(name).name)


class TestCli(unittest.TestCase):
    def test_report_list_and_rivals(self):
        buf = io.StringIO()
        with redirect_stdout(buf):
            code = cli.main(["report", "--list"])
        self.assertEqual(0, code)
        self.assertIn("physics.mass_properties", buf.getvalue())

        buf = io.StringIO()
        with redirect_stdout(buf):
            code = cli.main(["report", "--rivals"])
        self.assertEqual(0, code)
        self.assertIn("never averaged", buf.getvalue())

    def test_report_analyses_the_demo_model(self):
        buf = io.StringIO()
        with redirect_stdout(buf):
            code = cli.main(["report", "--brief", "a 20x10x5 plate"])
        self.assertEqual(0, code, buf.getvalue())
        out = buf.getvalue()
        self.assertIn("sequence.complexity_taxonomy", out)
        self.assertIn("physics.mass_properties", out)
        self.assertIn("skipped (input absent", out)

    def test_report_json_over_an_ops_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "ops.json")
            with open(path, "w", encoding="utf-8") as fh:
                json.dump(DEMO_OPS, fh)
            buf = io.StringIO()
            with redirect_stdout(buf):
                code = cli.main(["report", "--ops", path, "--json"])
        self.assertEqual(0, code, buf.getvalue())
        payload = json.loads(buf.getvalue())
        self.assertIn("analyses", payload)
        self.assertIn("sequence.complexity_taxonomy", payload["analyses"])
        self.assertGreater(payload["counts"]["ok"], 0)

    def test_existing_subcommands_still_work(self):
        for argv in (["demo"], ["formats"], ["bench", "--suites"],
                     ["capabilities", "--stats"]):
            buf = io.StringIO()
            with redirect_stdout(buf):
                code = cli.main(list(argv))
            self.assertEqual(0, code, f"{argv} -> {buf.getvalue()}")


if __name__ == "__main__":
    unittest.main()
