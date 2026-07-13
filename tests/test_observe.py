"""Tests for the observability layer (observe.py) — sec.15 of the blueprint.

Event streams are hand-built in the JsonlTracer shape (``{ts, run_id, kind,
data}``) so the tests are decoupled from the live loop yet exercise the exact
event vocabulary loop.py emits.
"""

import os
import tempfile
import unittest

from harnesscad.core.trace import JsonlTracer, monotonic_counter

from harnesscad.core.observe import (
    Classification,
    FailureTaxonomy,
    Mean,
    Metrics,
    Proportion,
    Replayer,
    SpanCollector,
    TARGETS,
    group_runs,
    load_jsonl,
    normal_interval,
    replay,
    render_trajectory,
    report,
    wilson_interval,
)


# --------------------------------------------------------------------------
# Event-stream builders (JsonlTracer shape).
# --------------------------------------------------------------------------

def _ev(kind, run_id, data, ts=None):
    return {"ts": ts, "run_id": run_id, "kind": kind, "data": data}


def _op(name, **kw):
    d = {"op": name}
    d.update(kw)
    return d


def successful_run(run_id="run-1", n_ops=3, ts_base=0):
    """A clean run: n ops each applied + verified ok + checkpointed."""
    evs = [_ev("run_start", run_id, {"op_count": n_ops}, ts=ts_base)]
    ts = ts_base + 1
    for i in range(n_ops):
        evs.append(_ev("op_applied", run_id,
                       {"op": _op(f"op{i}"), "index": i, "digest": f"d{i}"}, ts=ts))
        ts += 1
        evs.append(_ev("verify_result", run_id, {"ok": True, "diagnostics": []}, ts=ts))
        ts += 1
        evs.append(_ev("checkpoint", run_id, {"label": f"auto-{i + 1}", "index": i + 1}, ts=ts))
        ts += 1
    evs.append(_ev("run_end", run_id,
                   {"ok": True, "applied": n_ops, "digest": f"d{n_ops - 1}"}, ts=ts))
    return evs


def backend_rejected_run(run_id, code, message="bad", n_ok=1):
    """A run that applies n_ok ops then a backend-rejected op stops it."""
    evs = [_ev("run_start", run_id, {"op_count": n_ok + 1})]
    for i in range(n_ok):
        evs.append(_ev("op_applied", run_id,
                       {"op": _op(f"ok{i}"), "index": i, "digest": f"d{i}"}))
        evs.append(_ev("verify_result", run_id, {"ok": True, "diagnostics": []}))
        evs.append(_ev("checkpoint", run_id, {"label": f"auto-{i + 1}", "index": i + 1}))
    diags = [{"severity": "error", "code": code, "message": message, "where": None}]
    evs.append(_ev("rejected", run_id,
                   {"op": _op("badop"), "reason": "backend-rejected", "diagnostics": diags}))
    evs.append(_ev("run_end", run_id, {"ok": False, "applied": n_ok, "digest": f"d{n_ok - 1}"}))
    return evs


def verify_failed_run(run_id, code, message="predicate unmet"):
    """A run where an op applies but verification fails and it rolls back."""
    diags = [{"severity": "error", "code": code, "message": message, "where": None}]
    return [
        _ev("run_start", run_id, {"op_count": 1}),
        _ev("op_applied", run_id, {"op": _op("extrude"), "index": 0, "digest": "d0"}),
        _ev("verify_result", run_id, {"ok": False, "diagnostics": diags}),
        _ev("rejected", run_id,
            {"op": _op("extrude"), "reason": "verify-failed", "diagnostics": diags}),
        _ev("run_end", run_id, {"ok": False, "applied": 0, "digest": "start"}),
    ]


def loop_run(run_id="loop-1"):
    """A run that re-emits the SAME op repeatedly (a loop failure)."""
    same = _op("fillet", radius=99.0)
    diags = [{"severity": "error", "code": "radius-too-large",
              "message": "fillet radius exceeds edge", "where": None}]
    evs = [_ev("run_start", run_id, {"op_count": 3})]
    for _ in range(3):
        evs.append(_ev("rejected", run_id,
                       {"op": same, "reason": "backend-rejected", "diagnostics": diags}))
    evs.append(_ev("run_end", run_id, {"ok": False, "applied": 0, "digest": "start"}))
    return evs


# --------------------------------------------------------------------------
# Confidence intervals.
# --------------------------------------------------------------------------

class TestConfidenceIntervals(unittest.TestCase):
    def test_wilson_brackets_point_estimate(self):
        lo, hi = wilson_interval(8, 10)
        self.assertLess(lo, 0.8)
        self.assertGreater(hi, 0.8)
        self.assertGreaterEqual(lo, 0.0)
        self.assertLessEqual(hi, 1.0)

    def test_wilson_empty_is_zero(self):
        self.assertEqual(wilson_interval(0, 0), (0.0, 0.0))

    def test_wilson_extremes_stay_in_unit_interval(self):
        lo, hi = wilson_interval(10, 10)
        self.assertGreaterEqual(lo, 0.0)
        self.assertLessEqual(hi, 1.0)
        lo0, hi0 = wilson_interval(0, 10)
        self.assertGreaterEqual(lo0, 0.0)

    def test_normal_interval_of_constant_is_point(self):
        self.assertEqual(normal_interval([0.5, 0.5, 0.5]), (0.5, 0.5))

    def test_proportion_and_mean_helpers(self):
        p = Proportion(3, 4)
        self.assertEqual(p.value, 0.75)
        lo, hi = p.ci()
        self.assertLessEqual(lo, 0.75)
        self.assertGreaterEqual(hi, 0.75)
        m = Mean([0.2, 0.4, 0.6])
        self.assertAlmostEqual(m.value, 0.4)


# --------------------------------------------------------------------------
# Metrics.
# --------------------------------------------------------------------------

class TestMetrics(unittest.TestCase):
    def test_task_success_rate(self):
        events = successful_run("r1") + successful_run("r2") + \
            backend_rejected_run("r3", "bad-reference")
        m = Metrics(events)
        self.assertEqual(m.task_success_rate.k, 2)
        self.assertEqual(m.task_success_rate.n, 3)
        self.assertAlmostEqual(m.task_success_rate.value, 2 / 3)

    def test_tool_call_accuracy(self):
        # r1: 3 applied, 0 rejects. r2: 1 applied + 1 backend reject.
        events = successful_run("r1", n_ops=3) + backend_rejected_run("r2", "bad-reference", n_ok=1)
        m = Metrics(events)
        # applied=4, backend_rej=1, verify_rej=0 -> (4-0)/(4+1) = 0.8
        self.assertAlmostEqual(m.tool_call_accuracy.value, 0.8)
        self.assertEqual(m.tool_call_accuracy.n, 5)

    def test_tool_call_accuracy_counts_verify_failure_as_failed(self):
        events = successful_run("r1", n_ops=2) + verify_failed_run("r2", "under-constrained")
        m = Metrics(events)
        # applied = 2 (r1) + 1 (r2) = 3; backend_rej = 0; verify_rej = 1.
        # successful = 3-1 = 2; attempts = 3+0 = 3 -> 2/3.
        self.assertAlmostEqual(m.tool_call_accuracy.value, 2 / 3)

    def test_recovery_rate(self):
        # A failing run followed by a succeeding retry = recovered.
        events = backend_rejected_run("r1", "bad-reference") + successful_run("r2")
        m = Metrics(events)
        self.assertEqual(m.recovery_rate.k, 1)
        self.assertEqual(m.recovery_rate.n, 1)
        self.assertEqual(m.recovery_rate.value, 1.0)

    def test_escalation_rate_terminal_failure(self):
        # Two runs, both fail, no later success -> both escalate.
        events = backend_rejected_run("r1", "bad-reference") + verify_failed_run("r2", "dof")
        m = Metrics(events)
        self.assertEqual(m.escalation_rate.k, 2)
        self.assertEqual(m.escalation_rate.n, 2)

    def test_escalation_rate_recovered_failure_not_escalated(self):
        events = backend_rejected_run("r1", "bad-reference") + successful_run("r2")
        m = Metrics(events)
        self.assertEqual(m.escalation_rate.k, 0)

    def test_trajectory_efficiency_proxy(self):
        # r1 all applied -> eta 1.0; r2 1 applied of 2 attempts -> 0.5.
        events = successful_run("r1", n_ops=2) + backend_rejected_run("r2", "bad-reference", n_ok=1)
        m = Metrics(events)
        self.assertAlmostEqual(m.trajectory_efficiency.value, (1.0 + 0.5) / 2)

    def test_trajectory_efficiency_with_optimal(self):
        # r1 emits 4 ops but optimal is 2 -> eta = 2/4 = 0.5.
        events = successful_run("r1", n_ops=4)
        m = Metrics(events, optimal_lengths={"r1": 2})
        self.assertAlmostEqual(m.trajectory_efficiency.value, 0.5)

    def test_meets_targets_all_pass(self):
        events = sum((successful_run(f"r{i}") for i in range(10)), [])
        m = Metrics(events)
        met = m.meets_targets()
        self.assertTrue(met["task_success_rate"])
        self.assertTrue(met["tool_call_accuracy"])
        # escalation is an upper bound; 0 escalations passes.
        self.assertTrue(met["escalation_rate"])

    def test_escalation_target_is_upper_bound(self):
        # All fail -> escalation 100% -> fails the <15% target.
        events = sum((backend_rejected_run(f"r{i}", "bad-reference") for i in range(4)), [])
        m = Metrics(events)
        self.assertFalse(m.meets_targets()["escalation_rate"])


# --------------------------------------------------------------------------
# Failure taxonomy.
# --------------------------------------------------------------------------

class TestFailureTaxonomy(unittest.TestCase):
    def _classify(self, events):
        runs = [r for r in group_runs(events) if r.failed]
        self.assertEqual(len(runs), 1)
        return FailureTaxonomy.classify(runs[0])

    def test_classifies_loop(self):
        c = self._classify(loop_run("loop-1"))
        self.assertEqual(c.category, "loop")
        self.assertIn("loop", c.remediation.lower())

    def test_classifies_regen(self):
        c = self._classify(verify_failed_run("r", "non-manifold"))
        self.assertEqual(c.category, "regen")

    def test_classifies_hallucination(self):
        c = self._classify(backend_rejected_run("r", "unknown-ref", n_ok=0))
        self.assertEqual(c.category, "hallucination")

    def test_classifies_reasoning(self):
        c = self._classify(verify_failed_run("r", "under-constrained"))
        self.assertEqual(c.category, "reasoning")

    def test_classifies_context_overflow(self):
        c = self._classify(backend_rejected_run("r", "context-overflow", n_ok=0))
        self.assertEqual(c.category, "context-overflow")

    def test_classifies_refusal(self):
        c = self._classify(backend_rejected_run("r", "refusal", n_ok=0))
        self.assertEqual(c.category, "refusal")

    def test_loop_vs_regen_are_distinguished(self):
        loop_c = self._classify(loop_run("loop-x"))
        regen_c = self._classify(verify_failed_run("rg", "boolean-fail"))
        self.assertEqual(loop_c.category, "loop")
        self.assertEqual(regen_c.category, "regen")
        self.assertNotEqual(loop_c.remediation, regen_c.remediation)

    def test_every_category_has_remediation(self):
        from harnesscad.core.observe import CATEGORIES, REMEDIATION
        for cat in CATEGORIES:
            self.assertIn(cat, REMEDIATION)
            self.assertTrue(REMEDIATION[cat])

    def test_classify_events_only_returns_failures(self):
        events = successful_run("ok1") + loop_run("loop-1")
        cs = FailureTaxonomy.classify_events(events)
        self.assertEqual(len(cs), 1)
        self.assertEqual(cs[0].category, "loop")
        self.assertEqual(cs[0].run_id, "loop-1")


# --------------------------------------------------------------------------
# Spans.
# --------------------------------------------------------------------------

class TestSpans(unittest.TestCase):
    def test_span_aggregates_tokens_with_injected_clock(self):
        # Deterministic clock: ticks 0,1,2,... each span consumes two ticks
        # (enter -> start, exit -> end), so latency == 1 per span.
        sc = SpanCollector(clock=monotonic_counter())
        with sc.span("llm-call-1", "llm", tokens=100, cost_usd=0.01) as s:
            s.tokens += 20  # streamed completion tokens
        with sc.span("kernel-extrude", "tool", tokens=0) as s:
            s.attributes["op"] = "extrude"
        self.assertEqual(sc.total_tokens(), 120)
        self.assertAlmostEqual(sc.total_cost(), 0.01)
        self.assertEqual(sc.total_latency(), 2)  # 1 per span, 2 spans

    def test_injected_clock_makes_latency_deterministic(self):
        # A scripted clock returns exactly these values in order.
        ticks = iter([10, 25, 100, 130])
        sc = SpanCollector(clock=lambda: next(ticks))
        with sc.span("a", "llm"):
            pass
        with sc.span("b", "tool"):
            pass
        self.assertEqual(sc.spans[0].latency, 15)   # 25-10
        self.assertEqual(sc.spans[1].latency, 30)   # 130-100
        self.assertEqual(sc.total_latency(), 45)

    def test_record_completed_span(self):
        sc = SpanCollector(clock=monotonic_counter())
        sc.record("state-transition", "state", latency=5, run_id="r1")
        self.assertEqual(sc.spans[0].latency, 5)
        self.assertEqual(sc.spans[0].kind, "state")

    def test_by_kind_and_aggregate(self):
        sc = SpanCollector(clock=monotonic_counter())
        with sc.span("l", "llm", tokens=50):
            pass
        with sc.span("t", "tool", tokens=5):
            pass
        agg = sc.aggregate()
        self.assertEqual(agg["count"], 2)
        self.assertEqual(agg["tokens"], 55)
        self.assertIn("llm", agg["by_kind"])
        self.assertIn("tool", agg["by_kind"])

    def test_unknown_kind_rejected(self):
        sc = SpanCollector()
        with self.assertRaises(ValueError):
            with sc.span("x", "not-a-kind"):
                pass

    def test_from_events_builds_tool_and_state_spans(self):
        events = successful_run("r1", n_ops=2, ts_base=0)
        sc = SpanCollector.from_events(events)
        kinds = {s.kind for s in sc.spans}
        self.assertIn("tool", kinds)   # op_applied -> tool spans
        self.assertIn("state", kinds)  # run -> state span
        # 2 op_applied tool spans.
        self.assertEqual(sum(1 for s in sc.spans if s.kind == "tool"), 2)

    def test_from_events_latency_from_ts_deltas(self):
        events = successful_run("r1", n_ops=1, ts_base=0)
        sc = SpanCollector.from_events(events)
        run_span = next(s for s in sc.spans if s.kind == "state")
        # run_start ts=0 .. run_end ts>0, so latency is positive.
        self.assertGreater(run_span.latency, 0)


# --------------------------------------------------------------------------
# Replay.
# --------------------------------------------------------------------------

class TestReplay(unittest.TestCase):
    def test_replay_reconstructs_op_order(self):
        events = successful_run("r1", n_ops=3)
        runs = replay(events)
        self.assertEqual(len(runs), 1)
        order = [o["op"] for o in runs[0].op_order()]
        self.assertEqual(order, ["op0", "op1", "op2"])
        self.assertTrue(all(o.outcome == "applied" for o in runs[0].ops))

    def test_replay_marks_backend_rejected(self):
        events = backend_rejected_run("r1", "bad-reference", n_ok=2)
        run = replay(events)[0]
        self.assertEqual(len(run.ops), 3)  # 2 applied + 1 rejected
        self.assertEqual(run.ops[-1].outcome, "rejected-backend")
        self.assertEqual(run.ops[0].outcome, "applied")
        self.assertFalse(run.ok)

    def test_replay_marks_rollback_on_verify_failure(self):
        events = verify_failed_run("r1", "under-constrained")
        run = replay(events)[0]
        self.assertEqual(len(run.ops), 1)
        self.assertEqual(run.ops[0].outcome, "rolled-back")
        self.assertFalse(run.ops[0].verify_ok)

    def test_replay_checkpoint_flag(self):
        events = successful_run("r1", n_ops=2)
        run = replay(events)[0]
        self.assertTrue(all(o.checkpointed for o in run.ops))

    def test_replay_across_multiple_runs(self):
        events = successful_run("r1", n_ops=1) + backend_rejected_run("r2", "unknown-ref")
        runs = replay(events)
        self.assertEqual([r.run_id for r in runs], ["r1", "r2"])
        self.assertTrue(runs[0].ok)
        self.assertFalse(runs[1].ok)

    def test_render_trajectory_is_readable(self):
        events = successful_run("r1", n_ops=2) + backend_rejected_run("r2", "unknown-ref")
        text = render_trajectory(events)
        self.assertIn("r1", text)
        self.assertIn("FAILED", text)
        self.assertIn("op0", text)

    def test_replayer_class_api(self):
        events = successful_run("r1", n_ops=1)
        rp = Replayer()
        self.assertEqual(len(rp.replay(events)), 1)


# --------------------------------------------------------------------------
# JSONL round-trip via the real JsonlTracer.
# --------------------------------------------------------------------------

class TestJsonlRoundTrip(unittest.TestCase):
    def test_load_jsonl_reads_jsonltracer_output(self):
        tmpdir = tempfile.mkdtemp()
        path = os.path.join(tmpdir, "trace.jsonl")
        # Emit a couple of events through the actual JsonlTracer.
        tracer = JsonlTracer(path, clock=monotonic_counter())
        tracer.event("run_start", "run-1", {"op_count": 1})
        tracer.event("op_applied", "run-1", {"op": {"op": "extrude"}, "index": 0, "digest": "d0"})
        tracer.event("verify_result", "run-1", {"ok": True, "diagnostics": []})
        tracer.event("run_end", "run-1", {"ok": True, "applied": 1, "digest": "d0"})

        events = load_jsonl(path)
        self.assertEqual(len(events), 4)
        self.assertEqual(events[0]["kind"], "run_start")
        # And it replays.
        runs = replay(events)
        self.assertEqual(len(runs), 1)
        self.assertTrue(runs[0].ok)
        self.assertEqual(runs[0].applied, 1)

    def test_load_jsonl_skips_blank_lines(self):
        tmpdir = tempfile.mkdtemp()
        path = os.path.join(tmpdir, "trace.jsonl")
        with open(path, "w", encoding="utf-8") as fh:
            fh.write('{"ts":null,"run_id":"r","kind":"run_start","data":{"op_count":0}}\n')
            fh.write("\n")
            fh.write('{"ts":null,"run_id":"r","kind":"run_end","data":{"ok":true,"applied":0,"digest":"x"}}\n')
        events = load_jsonl(path)
        self.assertEqual(len(events), 2)


# --------------------------------------------------------------------------
# report() — the whole triad tied together.
# --------------------------------------------------------------------------

class TestReport(unittest.TestCase):
    def test_report_ties_everything(self):
        events = (successful_run("r1", n_ops=3) +
                  successful_run("r2", n_ops=2) +
                  loop_run("loop-1"))
        rep = report(events)
        self.assertEqual(rep["runs"]["total"], 3)
        self.assertEqual(rep["runs"]["ok"], 2)
        self.assertEqual(rep["runs"]["failed"], 1)
        # metrics present with CIs.
        self.assertIn("task_success_rate", rep["metrics"])
        self.assertIn("ci", rep["metrics"]["task_success_rate"])
        # taxonomy classified the loop failure.
        self.assertEqual(len(rep["failures"]), 1)
        self.assertEqual(rep["failures"][0]["category"], "loop")
        # spans synthesized.
        self.assertGreater(rep["spans"]["count"], 0)
        # targets echoed.
        self.assertEqual(rep["targets"], TARGETS)

    def test_report_targets_met_structure(self):
        events = sum((successful_run(f"r{i}") for i in range(10)), [])
        rep = report(events)
        self.assertTrue(rep["targets_met"]["task_success_rate"])
        self.assertTrue(rep["targets_met"]["tool_call_accuracy"])


if __name__ == "__main__":
    unittest.main()
