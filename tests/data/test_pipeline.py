"""Tests for the data-engine pipeline (the seam over dataengine/ + datagen/).

What these pin down:

*   the stage registry DISCOVERS real data modules from the tree (not a stub list);
*   a pipeline runs end to end on synthetic samples and EMITS a dataset;
*   determinism: same records + same seed -> byte-identical dataset JSON;
*   a real :class:`HarnessSession` run emits usable training records through the
    trace stage (the flywheel);
*   a stage that raises is CAPTURED, not fatal;
*   rival strategies (scale-invariant vs exact-token dedup, the two reward
    functions) are exposed by name, disagree by design, and are never averaged --
    a pipeline that selects two members of one family cannot be constructed.
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
from harnesscad.core.trace import InMemoryTracer
from harnesscad.data import pipeline
from harnesscad.io.backends.stub import StubBackend


def _plate_ops(w: float, h: float, d: float) -> list:
    return [
        {"op": "new_sketch", "plane": "XY"},
        {"op": "add_rectangle", "sketch": "sk1", "x": 0.0, "y": 0.0, "w": w, "h": h},
        {"op": "extrude", "sketch": "sk1", "distance": d},
    ]


def _disc_ops(r: float, d: float) -> list:
    return [
        {"op": "new_sketch", "plane": "XY"},
        {"op": "add_circle", "sketch": "sk1", "cx": 0.0, "cy": 0.0, "r": r},
        {"op": "extrude", "sketch": "sk1", "distance": d},
    ]


def _records() -> list:
    """Three synthetic samples: a plate, the SAME plate scaled 2x, and a disc.

    The scaled plate is the whole point of the rival dedup pair: it is a duplicate
    under the scale-invariant strategy and a distinct design under exact tokens.
    """
    return [
        {"id": "plate-small", "prompt": "a plate", "ok": True, "source": "synthetic",
         "geometry_family": "plate", "ops": _plate_ops(20.0, 10.0, 5.0)},
        {"id": "plate-large", "prompt": "a plate", "ok": True, "source": "synthetic",
         "geometry_family": "plate", "ops": _plate_ops(40.0, 20.0, 10.0)},
        {"id": "disc", "prompt": "a disc", "ok": True, "source": "synthetic",
         "geometry_family": "disc", "ops": _disc_ops(8.0, 3.0)},
    ]


class TestDiscovery(unittest.TestCase):
    def test_discovers_more_than_ten_real_modules(self):
        modules = pipeline.adapted_modules()
        self.assertGreater(len(modules), 10, modules)
        for dotted in modules:
            self.assertTrue(dotted.startswith("harnesscad.data."), dotted)
            # every bound module is a REAL module in the static index
            self.assertIsNotNone(pipeline.capability_registry.get(dotted), dotted)

    def test_stages_span_the_whole_flow(self):
        got = {s.kind for s in pipeline.stages()}
        for kind in ("generate", "annotate", "curate", "augment", "emit"):
            self.assertIn(kind, got)

    def test_unadapted_is_reported_not_hidden(self):
        left = pipeline.unadapted()
        self.assertTrue(left)  # honesty: the seam does not claim to bind all ~120
        self.assertFalse(set(left) & set(pipeline.adapted_modules()))

    def test_stage_order_is_deterministic(self):
        self.assertEqual([s.name for s in pipeline.stages()],
                         [s.name for s in pipeline.stages()])


class TestEndToEnd(unittest.TestCase):
    def test_text2cad_pipeline_emits_a_dataset(self):
        ds = pipeline.run_preset("text2cad", _records(), pipeline.Context(seed=7))
        self.assertEqual([], [(e.name, e.error) for e in ds.errors()])
        self.assertEqual(3, len(ds.records))
        for rec in ds.records:
            self.assertIn("code", rec)                 # annotate.code ran
            self.assertIn("code_complexity", rec)      # annotate.code_complexity ran
            self.assertIn("tier", rec)                 # curate.tiers ran
            self.assertIn(rec["split"], ("train", "val", "test"))
        self.assertIn("command_balance", ds.artifacts)
        self.assertIn("bias", ds.artifacts)
        self.assertEqual(0, len(ds.artifacts["leakage"]["leaks"]))
        self.assertEqual(3, sum(ds.splits().values()))

    def test_same_seed_gives_byte_identical_output(self):
        a = pipeline.run_preset("text2cad", _records(), pipeline.Context(seed=7))
        b = pipeline.run_preset("text2cad", _records(), pipeline.Context(seed=7))
        self.assertEqual(a.to_json(), b.to_json())
        self.assertEqual(a.to_json().encode("utf-8"), b.to_json().encode("utf-8"))

    def test_emit_writes_jsonl(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "dataset.jsonl")
            ctx = pipeline.Context(seed=3, options={"emit.jsonl": {"path": path}})
            ds = pipeline.run_preset("text2cad", _records(), ctx)
            with open(path, "r", encoding="utf-8") as fh:
                rows = [json.loads(line) for line in fh if line.strip()]
        self.assertEqual(len(ds.records), len(rows))
        self.assertTrue(all("code" in r for r in rows))

    def test_augmentation_expands_the_corpus(self):
        ctx = pipeline.Context(seed=5)
        ds = pipeline.run_pipeline(
            ("annotate.features", "augment.parametric"), _records(), ctx, name="aug")
        self.assertGreater(len(ds.records), len(_records()))
        variants = [r for r in ds.records if r.get("augmented_from")]
        self.assertTrue(variants)
        # every variant keeps the op STRUCTURE: only numbers move
        for v in variants:
            origin = next(r for r in _records() if r["id"] == v["augmented_from"])
            self.assertEqual([o["op"] for o in origin["ops"]],
                             [o["op"] for o in v["ops"]])

    def test_a_raising_stage_is_captured_not_fatal(self):
        boom = pipeline.Stage(
            name="audit.boom", kind="audit", dotted="harnesscad.data.dataengine.audit.bias",
            fn=lambda records, ctx: (_ for _ in ()).throw(RuntimeError("detonate")),
            summary="a stage that always raises")
        pipeline._stage_map()["audit.boom"] = boom
        try:
            ds = pipeline.run_pipeline(
                ("annotate.features", "audit.boom", "curate.tiers"),
                _records(), pipeline.Context(seed=1), name="boom")
        finally:
            pipeline._stage_map().pop("audit.boom", None)
        errors = ds.errors()
        self.assertEqual(1, len(errors))
        self.assertIn("RuntimeError: detonate", errors[0].error)
        # the run carried on: the stage AFTER the explosion still did its work
        self.assertEqual(3, len(ds.records))
        self.assertTrue(all("tier" in r for r in ds.records))


class TestFlywheel(unittest.TestCase):
    """A HarnessSession run must emit usable training records."""

    def _session_events(self):
        tracer = InMemoryTracer()
        session = HarnessSession(StubBackend(), tracer=tracer)
        result = session.apply_ops([parse_op(o) for o in _plate_ops(20.0, 10.0, 5.0)])
        self.assertTrue(result.ok)
        return tracer.events

    def test_session_run_becomes_training_records(self):
        records = pipeline.records_from_session(
            self._session_events(), prompt="a 20x10x5 plate", session_id="s1")
        self.assertEqual(1, len(records))
        rec = records[0]
        self.assertEqual("a 20x10x5 plate", rec["prompt"])
        self.assertEqual(3, rec["n_steps"])          # one step per applied op
        self.assertTrue(rec["ok"])                    # the verifier said so
        self.assertEqual(3, len(rec["ops"]))
        self.assertEqual(0, rec["n_corrections"])

    def test_flywheel_pipeline_emits_star_dpo_grpo_rows(self):
        records = pipeline.records_from_session(
            self._session_events(), prompt="a 20x10x5 plate", session_id="s1")
        ctx = pipeline.Context(seed=1,
                               options={"ingest.session_capture": {"consent": True}})
        ds = pipeline.run_preset("flywheel", records, ctx)
        self.assertEqual([], [(e.name, e.error) for e in ds.errors()])

        rows = ds.artifacts["training_rows"]
        self.assertEqual(1, len(rows["star"]))        # the run verified -> SFT row
        star = rows["star"][0]
        self.assertEqual("a 20x10x5 plate", star["prompt"])
        self.assertEqual(3, star["n_ops"])
        self.assertEqual(3, len(star["completion"]))
        self.assertIn("grpo", rows)
        self.assertIn("dpo", rows)

        flywheel = ds.artifacts["flywheel"]
        self.assertEqual(1, flywheel["n_trajectories"])
        self.assertEqual(1, flywheel["n_success"])

        # the consented capture is attached to the record
        capture = ds.records[0]["capture"]
        self.assertTrue(capture["consent"]["granted"])
        self.assertEqual(3, len(capture["op_decisions"]))

    def test_capture_requires_consent(self):
        records = pipeline.records_from_session(self._session_events(), prompt="p")
        ds = pipeline.run_pipeline(("ingest.session_capture",), records,
                                   pipeline.Context(seed=1), name="no-consent")
        self.assertNotIn("capture", ds.records[0])
        self.assertIn("consent", ds.results[0].note)

    def test_flywheel_is_deterministic(self):
        records = pipeline.records_from_session(self._session_events(), prompt="p",
                                                session_id="s1")
        opts = {"ingest.session_capture": {"consent": True}}
        a = pipeline.run_preset("flywheel", records,
                                pipeline.Context(seed=2, options=dict(opts)))
        b = pipeline.run_preset("flywheel", records,
                                pipeline.Context(seed=2, options=dict(opts)))
        self.assertEqual(a.to_json(), b.to_json())


class TestRivalsAreNeverBlended(unittest.TestCase):
    def test_rival_dedup_strategies_disagree_by_design(self):
        recs = _records()
        scale = pipeline.run_pipeline(("annotate.features", "curate.dedup_scale"),
                                      recs, pipeline.Context(seed=0), name="scale")
        tokens = pipeline.run_pipeline(("annotate.features", "curate.dedup_tokens"),
                                       recs, pipeline.Context(seed=0), name="tokens")
        kept_scale = {r["id"] for r in scale.records}
        kept_tokens = {r["id"] for r in tokens.records}
        # the 2x plate is a duplicate under scale-invariance, a distinct design
        # under exact tokens. Two answers, both correct under their own protocol.
        self.assertNotIn("plate-large", kept_scale)
        self.assertIn("plate-large", kept_tokens)
        self.assertNotEqual(kept_scale, kept_tokens)

    def test_a_pipeline_cannot_select_two_rivals(self):
        with self.assertRaises(pipeline.RivalBlendError):
            pipeline.Pipeline("bad", "blends two dedup protocols",
                              ("curate.dedup_scale", "curate.dedup_tokens"))
        with self.assertRaises(pipeline.RivalBlendError):
            pipeline.run_pipeline(("reward.executability", "reward.geometry_semantics"),
                                  _records(), pipeline.Context(), name="bad")

    def test_rival_rewards_are_kept_under_their_own_names(self):
        recs = [dict(r, iou=0.8, similarity=0.9, judge_score=7.0, union=1.0,
                     intersection=0.8) for r in _records()]
        ctx_a = pipeline.Context(seed=0)
        a = pipeline.run_pipeline(("annotate.code", "reward.executability"), recs,
                                  ctx_a, name="exec")
        ctx_b = pipeline.Context(seed=0)
        b = pipeline.run_pipeline(("annotate.code", "reward.geometry_semantics"), recs,
                                  ctx_b, name="semantic")
        exec_r = a.records[0]["reward"]
        sem_r = b.records[0]["reward"]
        self.assertIn("executability", exec_r)
        self.assertNotIn("geometry_semantics", exec_r)
        self.assertIn("geometry_semantics", sem_r)
        self.assertNotIn("executability", sem_r)
        # different numbers for the same record -- and no blended mean anywhere
        self.assertNotAlmostEqual(exec_r["executability"], sem_r["geometry_semantics"])

    def test_every_preset_is_rival_free(self):
        for name in pipeline.presets():
            p = pipeline.preset(name)  # __post_init__ enforces it
            for _family, members in pipeline.RIVAL_FAMILIES:
                self.assertLessEqual(
                    len(set(p.stage_names) & set(members)), 1, f"{name} blends {members}")

    def test_dpo_pairs_use_one_named_reward(self):
        recs = [
            {"id": "cand-a", "prompt": "same prompt", "ok": True, "union": 1.0,
             "intersection": 0.9, "judge_score": 9.0, "ops": _plate_ops(20.0, 10.0, 5.0)},
            {"id": "cand-b", "prompt": "same prompt", "ok": False, "union": 1.0,
             "intersection": 0.1, "judge_score": 1.0, "ops": _plate_ops(21.0, 11.0, 5.0)},
        ]
        ds = pipeline.run_preset("preference", recs, pipeline.Context(seed=1))
        rows = ds.artifacts["dpo"]
        self.assertEqual(1, len(rows))
        self.assertGreater(rows[0]["chosen_reward"], rows[0]["rejected_reward"])


class TestCli(unittest.TestCase):
    def test_dataset_list_and_rivals(self):
        buf = io.StringIO()
        with redirect_stdout(buf):
            code = cli.main(["dataset", "--list"])
        self.assertEqual(0, code)
        self.assertIn("curate.dedup_scale", buf.getvalue())

        buf = io.StringIO()
        with redirect_stdout(buf):
            code = cli.main(["dataset", "--rivals"])
        self.assertEqual(0, code)
        self.assertIn("never run two of these", buf.getvalue())

    def test_dataset_runs_a_named_pipeline(self):
        with tempfile.TemporaryDirectory() as tmp:
            src = os.path.join(tmp, "records.json")
            out = os.path.join(tmp, "out.jsonl")
            with open(src, "w", encoding="utf-8") as fh:
                json.dump(_records(), fh)
            buf = io.StringIO()
            with redirect_stdout(buf):
                code = cli.main(["dataset", "--pipeline", "text2cad", "--input", src,
                                 "--out", out, "--seed", "3"])
            self.assertEqual(0, code, buf.getvalue())
            self.assertTrue(os.path.exists(out))
        self.assertIn("pipeline: text2cad", buf.getvalue())


if __name__ == "__main__":
    unittest.main()
