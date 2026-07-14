"""The trajectory format: versioned, stable, and it refuses to read a stranger."""

from __future__ import annotations

import json
import os
import tempfile
import unittest

from harnesscad.agents.selftrain import SCHEMA_VERSION
from harnesscad.agents.selftrain import trajectory as tj


def _traj(i: int = 0) -> tj.Trajectory:
    return tj.Trajectory(
        trajectory_id=tj.trajectory_id("m", "blind", "b%d" % i, 1, 7),
        brief_id="b%d" % i,
        brief_text="a plate",
        model="m", loop="blind", seed=7, attempt=1,
        prompt="a plate", raw="[]",
        ops=[{"op": "new_sketch", "plane": "XY"}],
        parse_ok=True,
        verdict={"accepted": True},
        step_rewards=[{"index": 0, "op": "new_sketch", "applied": True,
                       "reward": 1.0, "divergent": False, "detail": ""}],
        reward_total=1.0,
    )


class TestRoundTrip(unittest.TestCase):

    def test_write_read(self):
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "t.jsonl")
            n = tj.write_jsonl(path, [_traj(0), _traj(1)])
            self.assertEqual(n, 2)
            back = tj.read_jsonl(path)
            self.assertEqual(len(back), 2)
            self.assertEqual(back[0].to_dict(), _traj(0).to_dict())
            self.assertTrue(back[0].accepted)

    def test_a_foreign_schema_is_refused_not_guessed(self):
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "t.jsonl")
            with open(path, "w", encoding="utf-8") as fh:
                fh.write(json.dumps({"schema": "selftrain/0", "ops": []}) + "\n")
            with self.assertRaises(ValueError):
                tj.read_jsonl(path)

    def test_schema_is_stamped(self):
        self.assertEqual(_traj().schema, SCHEMA_VERSION)

    def test_id_is_stable_and_has_no_clock_in_it(self):
        a = tj.trajectory_id("qwen", "harness", "l_bracket", 2, 20260713)
        b = tj.trajectory_id("qwen", "harness", "l_bracket", 2, 20260713)
        self.assertEqual(a, b)
        self.assertEqual(a, "qwen|harness|l_bracket|a2|s20260713")


class TestSink(unittest.TestCase):

    def test_the_hook_buffers_and_does_not_grade(self):
        sink = tj.TrajectorySink()
        sink.on_attempt(tj.AttemptCapture(
            brief_id="b", prompt="p", raw="r", ops=[], parse_ok=False,
            parse_error="nope", attempt=1, diagnostics=[], feedback=None))
        self.assertEqual(len(sink), 1)
        self.assertEqual(sink.captures[0].brief_id, "b")


class TestPressureCapture(unittest.TestCase):

    def test_lifts_an_attempt_out_of_a_results_cell(self):
        cell = {"brief": "l_bracket", "model": "qwen", "loop": "harness",
                "seed": 20260713, "records": []}
        record = {"attempt": 2, "raw": "[]", "parse_ok": True,
                  "parse_error": None, "ops": [{"op": "new_sketch"}],
                  "feedback": "the harness said something false",
                  "grade": {"diagnostics": [{"code": "infeasible-plan"}]}}
        cap = tj.capture_from_pressure_record(cell, record)
        self.assertEqual(cap.brief_id, "l_bracket")
        self.assertEqual(cap.attempt, 2)
        self.assertTrue(cap.parse_ok)
        self.assertEqual(cap.diagnostics[0]["code"], "infeasible-plan")
        # The fleet's verdict rides along as DATA. It is never a label.
        self.assertEqual(cap.feedback, "the harness said something false")


if __name__ == "__main__":                                # pragma: no cover
    unittest.main()
