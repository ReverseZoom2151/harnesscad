"""Tests for the data-engine / training-trace layer (dataengine/) — sec.17/21.

Event streams are hand-built in the JsonlTracer shape (``{ts, run_id, kind,
data}``) so the tests exercise the exact event vocabulary loop.py emits without
booting the live loop. Two archetypes: a clean successful run, and a
failed-then-fixed session (a run that diverges, then a later run that completes).
"""

import os
import tempfile
import unittest

from harnesscad.data.dataengine import (
    Action,
    Step,
    SubGoal,
    Trajectory,
    from_events,
    to_grpo,
    to_dpo,
    to_star,
    flywheel_metrics,
    write_jsonl,
)
from harnesscad.data.dataengine.trajectory import (
    OUTCOME_APPLIED,
    OUTCOME_ROLLED_BACK,
    OUTCOME_REJECTED,
    REWARD_PASS,
    REWARD_FAIL,
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


def successful_run(run_id="run-1", n_ops=3, prompt="a bracket", plan=None):
    """A clean run: n ops each applied + verified ok + checkpointed."""
    plan = plan or ["sketch", "extrude", "fillet"]
    evs = [_ev("run_start", run_id, {"op_count": n_ops, "prompt": prompt, "plan": plan})]
    for i in range(n_ops):
        op = _op(f"op{i}", reasoning=f"do step {i}")
        evs.append(_ev("op_applied", run_id, {"op": op, "index": i, "digest": f"d{i}"}))
        evs.append(_ev("verify_result", run_id, {"ok": True, "diagnostics": []}))
        evs.append(_ev("checkpoint", run_id, {"label": f"auto-{i + 1}", "index": i + 1}))
    evs.append(_ev("run_end", run_id, {"ok": True, "applied": n_ops, "digest": f"d{n_ops - 1}"}))
    return evs


def verify_failed_run(run_id, code="over-constrained", n_ok=1, prompt="a bracket"):
    """A run that applies n_ok ops, then one op applies but fails verify -> rollback."""
    evs = [_ev("run_start", run_id, {"op_count": n_ok + 1, "prompt": prompt})]
    for i in range(n_ok):
        op = _op(f"ok{i}", reasoning=f"good step {i}")
        evs.append(_ev("op_applied", run_id, {"op": op, "index": i, "digest": f"d{i}"}))
        evs.append(_ev("verify_result", run_id, {"ok": True, "diagnostics": []}))
        evs.append(_ev("checkpoint", run_id, {"label": f"auto-{i + 1}", "index": i + 1}))
    diags = [{"severity": "error", "code": code, "message": "bad", "where": None}]
    bad = _op("extrude", reasoning="over-extrude")
    evs.append(_ev("op_applied", run_id, {"op": bad, "index": n_ok, "digest": f"d{n_ok}"}))
    evs.append(_ev("verify_result", run_id, {"ok": False, "diagnostics": diags}))
    evs.append(_ev("rejected", run_id, {"op": bad, "reason": "verify-failed", "diagnostics": diags}))
    evs.append(_ev("run_end", run_id, {"ok": False, "applied": n_ok, "digest": f"d{n_ok - 1}"}))
    return evs


def failed_then_fixed(prompt="a bracket"):
    """A session: run-a diverges (verify-fail), run-b then completes cleanly."""
    return verify_failed_run("run-a", n_ok=1, prompt=prompt) + successful_run(
        "run-b", n_ops=2, prompt=prompt)


# --------------------------------------------------------------------------
# from_events: step count + rewards
# --------------------------------------------------------------------------

class TestFromEvents(unittest.TestCase):

    def test_successful_run_step_count_and_rewards(self):
        traj = from_events(successful_run(n_ops=3))
        self.assertEqual(traj.length, 3)
        self.assertTrue(traj.success)
        self.assertEqual(traj.final_reward, REWARD_PASS)
        self.assertEqual(traj.dense_rewards(), [REWARD_PASS, REWARD_PASS, REWARD_PASS])
        self.assertEqual(traj.total_reward(), 3 * REWARD_PASS)
        # Every op reached a sub-goal.
        self.assertTrue(all(sg.reached for sg in traj.sub_goal_labels))
        self.assertEqual(len(traj.sub_goal_labels), 3)

    def test_prompt_and_plan_harvested_from_run_start(self):
        traj = from_events(successful_run(prompt="a flange", plan=["s", "e"]))
        self.assertEqual(traj.prompt, "a flange")
        self.assertEqual(traj.plan, ["s", "e"])

    def test_action_is_reasoning_and_tool_call(self):
        traj = from_events(successful_run(n_ops=1))
        step = traj.steps[0]
        self.assertIsInstance(step.action, Action)
        self.assertEqual(step.action.reasoning, "do step 0")
        self.assertEqual(step.action.tool_call["op"], "op0")
        pair = step.action.as_pair()
        self.assertEqual(pair[0], "do step 0")
        self.assertEqual(pair[1]["op"], "op0")

    def test_state_transition_digests(self):
        traj = from_events(successful_run(n_ops=2))
        s0, s1 = traj.steps
        # S_0 starts from the empty (None) digest; S_{t+1} chains forward.
        self.assertIsNone(s0.state_before["digest"])
        self.assertEqual(s0.state_after["digest"], "d0")
        self.assertEqual(s1.state_before["digest"], "d0")
        self.assertEqual(s1.state_after["digest"], "d1")

    def test_verify_failed_gives_negative_reward_and_divergence(self):
        traj = from_events(verify_failed_run("run-a", n_ok=2))
        # 2 good ops + 1 rolled-back op = 3 steps.
        self.assertEqual(traj.length, 3)
        self.assertEqual(traj.dense_rewards(), [REWARD_PASS, REWARD_PASS, REWARD_FAIL])
        self.assertEqual(traj.steps[-1].outcome, OUTCOME_ROLLED_BACK)
        self.assertTrue(traj.steps[-1].divergent)
        self.assertFalse(traj.success)
        self.assertEqual(traj.final_reward, 0.0)
        self.assertEqual(traj.first_divergence(), 2)

    def test_backend_rejected_is_negative_and_never_applied(self):
        diags = [{"severity": "error", "code": "unknown-ref", "message": "x", "where": None}]
        evs = [
            _ev("run_start", "run-x", {"op_count": 1, "prompt": "p"}),
            _ev("rejected", "run-x", {"op": _op("bad"), "reason": "backend-rejected",
                                      "diagnostics": diags}),
            _ev("run_end", "run-x", {"ok": False, "applied": 0, "digest": "start"}),
        ]
        traj = from_events(evs)
        self.assertEqual(traj.length, 1)
        self.assertEqual(traj.steps[0].outcome, OUTCOME_REJECTED)
        self.assertEqual(traj.steps[0].reward, REWARD_FAIL)
        self.assertFalse(traj.success)

    def test_failed_then_fixed_spans_two_runs(self):
        traj = from_events(failed_then_fixed())
        # run-a: 1 ok + 1 rolled-back = 2 steps; run-b: 2 ok = 2 steps.
        self.assertEqual(traj.length, 4)
        self.assertEqual(traj.run_ids, ["run-a", "run-b"])
        # Terminal verdict is the LAST run (which succeeded).
        self.assertTrue(traj.success)
        self.assertEqual(traj.dense_rewards(),
                         [REWARD_PASS, REWARD_FAIL, REWARD_PASS, REWARD_PASS])


# --------------------------------------------------------------------------
# trajectory_slice: cut at first divergence
# --------------------------------------------------------------------------

class TestTrajectorySlice(unittest.TestCase):

    def test_slice_cuts_at_first_divergence_inclusive(self):
        traj = from_events(failed_then_fixed())
        sliced = traj.trajectory_slice(to_first_divergence=True)
        # First divergence is step index 1 (the rolled-back op in run-a).
        self.assertEqual(sliced.length, 2)
        self.assertTrue(sliced.steps[-1].divergent)
        self.assertEqual(sliced.dense_rewards(), [REWARD_PASS, REWARD_FAIL])
        self.assertFalse(sliced.success)
        self.assertTrue(sliced.metadata.get("sliced_to_first_divergence"))
        # Denser signal: fewer steps than the full trace.
        self.assertLess(sliced.length, traj.length)

    def test_slice_of_clean_success_is_unchanged(self):
        traj = from_events(successful_run(n_ops=3))
        sliced = traj.trajectory_slice(to_first_divergence=True)
        self.assertEqual(sliced.length, traj.length)
        self.assertTrue(sliced.success)

    def test_slice_disabled_returns_full(self):
        traj = from_events(failed_then_fixed())
        full = traj.trajectory_slice(to_first_divergence=False)
        self.assertEqual(full.length, traj.length)


# --------------------------------------------------------------------------
# Exporters: GRPO / DPO / STaR
# --------------------------------------------------------------------------

class TestExporters(unittest.TestCase):

    def _group(self):
        # Same prompt, three traces of differing quality -> one group.
        good = from_events(successful_run("run-g", n_ops=3, prompt="P"))
        mid = from_events(successful_run("run-m", n_ops=2, prompt="P"))
        bad = from_events(verify_failed_run("run-b", n_ok=1, prompt="P"))
        return [good, mid, bad]

    def test_grpo_group_normalised_rows(self):
        rows = to_grpo(self._group())
        self.assertEqual(len(rows), 3)
        for r in rows:
            self.assertEqual(r["group_size"], 3)
            self.assertIn("advantage", r)
            self.assertIn("response", r)
            self.assertIn("dense_rewards", r)
        # Advantages within a group are centred at ~0.
        self.assertAlmostEqual(sum(r["advantage"] for r in rows), 0.0, places=6)
        # The best trace has the highest advantage.
        best = max(rows, key=lambda r: r["reward"])
        self.assertEqual(best["advantage"], max(r["advantage"] for r in rows))

    def test_grpo_singleton_group_zero_advantage(self):
        rows = to_grpo([from_events(successful_run(prompt="solo"))])
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["advantage"], 0.0)

    def test_dpo_pair_chosen_beats_rejected(self):
        rows = to_dpo(self._group())
        self.assertEqual(len(rows), 1)
        pair = rows[0]
        self.assertGreater(pair["chosen"]["reward"], pair["rejected"]["reward"])
        self.assertIn("response", pair["chosen"])
        self.assertIn("response", pair["rejected"])

    def test_dpo_no_pair_when_group_has_no_separation(self):
        # Two identical successful traces -> equal reward -> no preference.
        a = from_events(successful_run("r1", n_ops=2, prompt="Q"))
        b = from_events(successful_run("r2", n_ops=2, prompt="Q"))
        self.assertEqual(to_dpo([a, b]), [])

    def test_dpo_skips_singleton_group(self):
        self.assertEqual(to_dpo([from_events(successful_run(prompt="lone"))]), [])

    def test_star_only_successes_applied_ops(self):
        rows = to_star(self._group())
        # Only the two successful traces qualify (the verify-failed one is dropped).
        self.assertEqual(len(rows), 2)
        for r in rows:
            self.assertGreater(r["reward"], 0.0)
            self.assertEqual(r["n_ops"], len(r["completion"]))
            self.assertTrue(all("tool_call" in c for c in r["completion"]))

    def test_star_excludes_diverged_ops_from_completion(self):
        # A success cannot contain a divergent op, so completion == applied ops.
        traj = from_events(successful_run(n_ops=3, prompt="P"))
        rows = to_star([traj])
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["n_ops"], 3)


# --------------------------------------------------------------------------
# Flywheel metrics: corrections-per-plan
# --------------------------------------------------------------------------

class TestFlywheelMetrics(unittest.TestCase):

    def test_corrections_per_plan_from_divergences(self):
        clean = from_events(successful_run("r1", n_ops=3))       # 0 corrections
        one = from_events(verify_failed_run("r2", n_ok=2))       # 1 correction
        m = flywheel_metrics([clean, one])
        self.assertEqual(m["n_trajectories"], 2)
        self.assertEqual(m["corrections_total"], 1)
        self.assertEqual(m["corrections_per_plan"], 0.5)
        self.assertEqual(m["per_plan_corrections"], [0, 1])

    def test_explicit_human_corrections_metadata_overrides_proxy(self):
        traj = from_events(successful_run("r1", n_ops=3),
                           metadata={"human_corrections": 4})
        self.assertEqual(traj.corrections(), 4)
        m = flywheel_metrics([traj])
        self.assertEqual(m["corrections_per_plan"], 4.0)

    def test_corrections_trend_falls_over_time(self):
        # Early sessions need corrections; later ones are clean -> trend falls.
        early = [from_events(verify_failed_run(f"e{i}", n_ok=1)) for i in range(2)]
        late = [from_events(successful_run(f"l{i}", n_ops=2)) for i in range(2)]
        m = flywheel_metrics(early + late)
        trend = m["corrections_trend"]
        self.assertTrue(trend["falling"])
        self.assertLess(trend["second_half_mean"], trend["first_half_mean"])

    def test_success_and_efficiency_aggregates(self):
        clean = from_events(successful_run("r1", n_ops=2))
        failed = from_events(verify_failed_run("r2", n_ok=1))
        m = flywheel_metrics([clean, failed])
        self.assertEqual(m["n_success"], 1)
        self.assertEqual(m["success_rate"], 0.5)
        # clean: 2/2 applied = 1.0; failed: 1 applied of 2 steps = 0.5 -> mean 0.75.
        self.assertAlmostEqual(m["mean_efficiency"], 0.75, places=6)

    def test_empty_input(self):
        m = flywheel_metrics([])
        self.assertEqual(m["n_trajectories"], 0)
        self.assertEqual(m["corrections_per_plan"], 0.0)


# --------------------------------------------------------------------------
# write_jsonl round-trip
# --------------------------------------------------------------------------

class TestWriteJsonl(unittest.TestCase):

    def test_write_jsonl_round_trips(self):
        rows = to_grpo([from_events(successful_run("r1", n_ops=2, prompt="P")),
                        from_events(successful_run("r2", n_ops=3, prompt="P"))])
        fd, path = tempfile.mkstemp(suffix=".jsonl")
        os.close(fd)
        try:
            n = write_jsonl(path, rows)
            self.assertEqual(n, len(rows))
            import json
            with open(path, encoding="utf-8") as fh:
                back = [json.loads(line) for line in fh if line.strip()]
            self.assertEqual(len(back), len(rows))
            self.assertIn("advantage", back[0])
        finally:
            os.remove(path)


if __name__ == "__main__":
    unittest.main()
