"""Tests for the observability/trace layer and its wiring into HarnessSession."""

import json
import os
import tempfile
import unittest

from harnesscad.io.backends.stub import StubBackend
from harnesscad.core.cisp.ops import NewSketch, AddRectangle, Constrain, Extrude
from harnesscad.core.loop import HarnessSession
from harnesscad.core.trace import InMemoryTracer, JsonlTracer, NullTracer, Tracer


def _rect_setup():
    return [NewSketch(), AddRectangle(sketch="sk1")]


def _four_constraints():
    return [Constrain(kind="distance", a="e1", value=10.0) for _ in range(4)]


class TestBackwardCompatibility(unittest.TestCase):
    def test_default_tracer_is_null(self):
        session = HarnessSession(StubBackend())
        self.assertIsInstance(session.tracer, NullTracer)

    def test_null_tracer_behaviour_identical(self):
        untraced = HarnessSession(StubBackend())
        traced = HarnessSession(StubBackend(), tracer=InMemoryTracer())
        ops = _rect_setup() + _four_constraints() + [Extrude(sketch="sk1", distance=5.0)]
        r1 = untraced.apply_ops(list(ops))
        r2 = traced.apply_ops(list(ops))
        self.assertEqual(r1.ok, r2.ok)
        self.assertEqual(r1.applied, r2.applied)
        self.assertEqual(r1.digest, r2.digest)


class TestValidBatchEvents(unittest.TestCase):
    def test_event_sequence_for_valid_batch(self):
        tracer = InMemoryTracer()
        session = HarnessSession(StubBackend(), tracer=tracer)
        ops = _rect_setup() + _four_constraints() + [Extrude(sketch="sk1", distance=5.0)]
        res = session.apply_ops(ops)
        self.assertTrue(res.ok)

        kinds = tracer.kinds()
        # First event is run_start, last is a successful run_end.
        self.assertEqual(kinds[0], "run_start")
        self.assertEqual(kinds[-1], "run_end")
        self.assertTrue(tracer.of_kind("run_end")[0]["data"]["ok"])

        # One op_applied per op, and no rejected events on the happy path.
        self.assertEqual(len(tracer.of_kind("op_applied")), len(ops))
        self.assertEqual(tracer.of_kind("rejected"), [])
        # A verify_result and checkpoint per applied op.
        self.assertEqual(len(tracer.of_kind("verify_result")), len(ops))
        self.assertEqual(len(tracer.of_kind("checkpoint")), len(ops))

        # run_start precedes every op_applied which precedes run_end (ordering).
        first_applied = kinds.index("op_applied")
        self.assertLess(kinds.index("run_start"), first_applied)
        self.assertLess(first_applied, len(kinds) - 1)

        # A single run_id threads the whole batch.
        run_ids = {e["run_id"] for e in tracer.events}
        self.assertEqual(len(run_ids), 1)

        # op_applied carries the resulting digest matching the final state.
        self.assertEqual(tracer.of_kind("op_applied")[-1]["data"]["digest"], res.digest)

    def test_run_id_deterministic(self):
        ops = _rect_setup() + _four_constraints() + [Extrude(sketch="sk1", distance=5.0)]
        t1, t2 = InMemoryTracer(), InMemoryTracer()
        HarnessSession(StubBackend(), tracer=t1).apply_ops(list(ops))
        HarnessSession(StubBackend(), tracer=t2).apply_ops(list(ops))
        self.assertEqual(t1.events[0]["run_id"], t2.events[0]["run_id"])


class TestRejectedEvents(unittest.TestCase):
    def test_backend_rejected_emits_rejected_event(self):
        tracer = InMemoryTracer()
        session = HarnessSession(StubBackend(), tracer=tracer)
        res = session.apply_ops([Extrude(sketch="nope", distance=5.0)])
        self.assertFalse(res.ok)

        rejected = tracer.of_kind("rejected")
        self.assertEqual(len(rejected), 1)
        self.assertEqual(rejected[0]["data"]["op"]["op"], "extrude")
        diags = rejected[0]["data"]["diagnostics"]
        self.assertTrue(diags)
        self.assertEqual(diags[0]["code"], "bad-ref")
        # run_end still fires, with ok False.
        self.assertFalse(tracer.of_kind("run_end")[0]["data"]["ok"])

    def test_verify_rejected_emits_rejected_with_diagnostics(self):
        tracer = InMemoryTracer()
        session = HarnessSession(StubBackend(), tracer=tracer)
        # 5th distance constraint over-constrains the rectangle -> verify error.
        ops = _rect_setup() + _four_constraints() + [
            Constrain(kind="distance", a="e1", value=10.0)]
        res = session.apply_ops(ops)
        self.assertFalse(res.ok)

        rejected = tracer.of_kind("rejected")
        self.assertEqual(len(rejected), 1)
        self.assertEqual(rejected[0]["data"]["reason"], "verify-failed")
        codes = [d["code"] for d in rejected[0]["data"]["diagnostics"]]
        self.assertIn("over-constrained", codes)


class TestJsonlTracer(unittest.TestCase):
    def test_writes_valid_json_lines(self):
        fd, path = tempfile.mkstemp(suffix=".jsonl")
        os.close(fd)
        try:
            tracer = JsonlTracer(path)
            self.assertIsInstance(tracer, Tracer)
            session = HarnessSession(StubBackend(), tracer=tracer)
            ops = _rect_setup() + _four_constraints() + [Extrude(sketch="sk1", distance=5.0)]
            session.apply_ops(ops)

            with open(path, "r", encoding="utf-8") as fh:
                lines = [ln for ln in fh.read().splitlines() if ln]
            self.assertTrue(lines)
            records = [json.loads(ln) for ln in lines]  # each line valid JSON
            for rec in records:
                self.assertEqual(set(rec.keys()), {"ts", "run_id", "kind", "data"})
                self.assertIsNone(rec["ts"])  # wall-clock-free default
            self.assertEqual(records[0]["kind"], "run_start")
            self.assertEqual(records[-1]["kind"], "run_end")
            self.assertTrue(records[-1]["data"]["ok"])
        finally:
            os.remove(path)

    def test_injected_clock_stamps_ts(self):
        fd, path = tempfile.mkstemp(suffix=".jsonl")
        os.close(fd)
        try:
            counter = iter(range(1000))
            tracer = JsonlTracer(path, clock=lambda: next(counter))
            session = HarnessSession(StubBackend(), tracer=tracer)
            session.apply_ops([NewSketch()])
            with open(path, "r", encoding="utf-8") as fh:
                records = [json.loads(ln) for ln in fh.read().splitlines() if ln]
            ts_values = [r["ts"] for r in records]
            self.assertEqual(ts_values, sorted(ts_values))
            self.assertTrue(all(isinstance(t, int) for t in ts_values))
        finally:
            os.remove(path)


if __name__ == "__main__":
    unittest.main()
