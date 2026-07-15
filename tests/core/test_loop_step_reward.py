"""The loop emits a PER-OP reward, and it attributes the break to ONE op.

Before this wiring the loop emitted no per-step signal at all: a six-op plan that
failed produced one scalar about the finished solid and condemned ops 1-5 with it.
`agents/agent/tool_reward.py` had implemented the process reward and nothing in
`core/loop.py` or `core/trace.py` called it.
"""

from __future__ import annotations

import unittest

from harnesscad.core.cisp.ops import parse_op
from harnesscad.core.loop import HarnessSession
from harnesscad.core.trace import EVENT_KINDS, InMemoryTracer
from harnesscad.io.backends.frep import FRepBackend


GOOD = [
    {"op": "new_sketch", "plane": "XY"},
    {"op": "add_rectangle", "sketch": "sk1", "x": 0, "y": 0, "w": 50, "h": 30},
    {"op": "extrude", "sketch": "sk1", "distance": 6},
]

# op 2 (index 1) references a sketch that does not exist: the batch must break
# THERE, and ops before it must keep their credit.
BREAKS_AT_1 = [
    {"op": "new_sketch", "plane": "XY"},
    {"op": "extrude", "sketch": "sk9", "distance": 6},
    {"op": "new_sketch", "plane": "XY"},
]


def _session(tracer=None):
    return HarnessSession(FRepBackend(), tracer=tracer)


def _ops(raw):
    return [parse_op(dict(o)) for o in raw]


class TestStepReward(unittest.TestCase):
    def test_step_reward_is_a_trace_event_kind(self):
        self.assertIn("step_reward", EVENT_KINDS)

    def test_every_accepted_op_earns_one(self):
        s = _session()
        res = s.apply_ops(_ops(GOOD))
        self.assertTrue(res.ok)
        self.assertEqual([r["reward"] for r in s.step_rewards], [1.0, 1.0, 1.0])
        self.assertEqual([r["op"] for r in s.step_rewards],
                         ["new_sketch", "add_rectangle", "extrude"])
        self.assertEqual(s.mean_step_reward(), 1.0)

    def test_the_broken_op_is_the_only_one_punished(self):
        s = _session()
        res = s.apply_ops(_ops(BREAKS_AT_1))
        self.assertFalse(res.ok)
        rewards = [r["reward"] for r in s.step_rewards]
        # op 0 applied and verified; op 1 broke it; op 2 was never reached and is
        # never scored. THAT is trajectory slicing.
        self.assertEqual(rewards, [1.0, 0.0])
        self.assertEqual(s.step_rewards[1]["index"], 1)
        self.assertEqual(s.mean_step_reward(), 0.5)

    def test_the_tracer_sees_one_event_per_scored_op(self):
        tracer = InMemoryTracer()
        s = _session(tracer)
        s.apply_ops(_ops(BREAKS_AT_1))
        events = tracer.of_kind("step_reward")
        self.assertEqual(len(events), 2)
        self.assertEqual([e["data"]["index"] for e in events], [0, 1])
        self.assertEqual([e["data"]["reward"] for e in events], [1.0, 0.0])

    def test_run_end_carries_the_whole_vector(self):
        tracer = InMemoryTracer()
        s = _session(tracer)
        s.apply_ops(_ops(GOOD))
        end = tracer.of_kind("run_end")[-1]["data"]
        self.assertEqual(end["mean_step_reward"], 1.0)
        self.assertEqual(len(end["step_rewards"]), 3)

    def test_the_vector_is_per_batch_not_cumulative(self):
        s = _session()
        s.apply_ops(_ops(GOOD))
        s.apply_ops(_ops([{"op": "new_sketch", "plane": "XY"}]))
        self.assertEqual(len(s.step_rewards), 1)


if __name__ == "__main__":
    unittest.main()
