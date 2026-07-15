"""The memory A/B rig, proven offline with a ScriptedClient.

These tests prove the RIG, not the hypothesis: that the two arms differ in
exactly one thing (the memory block), that the answer key never reaches memory,
and that a negative result is reported as a negative result. The hypothesis is
settled by running it against a real model, which is what `ab.main` does.
"""

import unittest

from harnesscad.eval.memory.ab import (
    ARM_OFF,
    ARM_ON,
    ABReport,
    format_text,
    run_arm,
)
from harnesscad.eval.pressure.briefs import briefs_for
from harnesscad.eval.pressure.model import ScriptedClient


class RecordingClient(ScriptedClient):
    """A ScriptedClient that always answers with the brief's OWN reference ops.

    A perfect model. Used to prove the plumbing: every brief solves, so every
    trajectory is oracle-verified, so the ON arm's memory fills with real
    exemplars and the OFF arm's stays empty.
    """

    def __init__(self, briefs):
        super().__init__([], name="perfect")
        self._by_index = [list(b.reference) for b in briefs]
        self._i = 0

    def complete(self, messages, attempt):
        self.calls.append((attempt, [dict(m) for m in messages]))
        import json
        ops = self._by_index[min(self._i, len(self._by_index) - 1)]
        if attempt == 0:
            self._i += 1
        return json.dumps(ops)


class TestArmsDifferByExactlyOneThing(unittest.TestCase):
    def setUp(self):
        self.briefs = briefs_for("all")[:3]

    def test_off_arm_prompt_never_mentions_memory(self):
        client = RecordingClient(self.briefs)
        run_arm(client, self.briefs, memory_on=False, max_attempts=1)
        for _, messages in client.calls:
            self.assertNotIn("MEMORY", messages[-1]["content"])

    def test_on_arm_accumulates_verified_exemplars_across_briefs(self):
        client = RecordingClient(self.briefs)
        arm = run_arm(client, self.briefs, memory_on=True, max_attempts=1)
        self.assertEqual(arm.arm, ARM_ON)
        # Every reference stream builds, so the gate admits them.
        self.assertGreater(arm.memory_stats["commits_admitted"], 0)

    def test_off_arm_writes_nothing_to_memory(self):
        client = RecordingClient(self.briefs)
        arm = run_arm(client, self.briefs, memory_on=False, max_attempts=1)
        self.assertEqual(arm.arm, ARM_OFF)
        self.assertEqual(arm.memory_stats, {})

    def test_the_answer_key_never_reaches_memory(self):
        """`grade.solved` is the hidden ground truth. If memory could see it,
        memory would be a channel for the answer key and the whole experiment
        would be worthless. It is not passed, and this pins that."""
        import inspect
        from harnesscad.eval.memory import ab
        src = inspect.getsource(ab.run_brief)
        commit = src[src.index("memory.commit"):src.index("if g.apply_ok")]
        self.assertNotIn("g.solved", commit)
        self.assertNotIn("solved_shape", commit)


class TestReportingIsHonest(unittest.TestCase):
    def _report(self, off_rate, on_rate):
        from harnesscad.eval.memory.ab import ArmResult, BriefRun
        rep = ABReport(seed=1, max_attempts=1, brief_order=["a"] * 10)
        for arm_name, rate in ((ARM_OFF, off_rate), (ARM_ON, on_rate)):
            a = ArmResult(model="m", arm=arm_name)
            for i in range(10):
                a.runs.append(BriefRun(brief_id=f"b{i}", arm=arm_name,
                                       solved=i < rate * 10))
            rep.arms.append(a)
        return rep

    def test_a_negative_result_is_reported_as_a_negative_result(self):
        text = format_text(self._report(off_rate=0.8, on_rate=0.5))
        self.assertIn("MEMORY HURT", text)
        self.assertIn("Do not ship memory on", text)
        self.assertIn("-30.0 pp", text.replace("−", "-"))

    def test_a_positive_result_says_so(self):
        text = format_text(self._report(off_rate=0.5, on_rate=0.8))
        self.assertIn("Memory helped", text)

    def test_a_null_result_says_delete_it(self):
        text = format_text(self._report(off_rate=0.5, on_rate=0.5))
        self.assertIn("nothing measurable", text)


if __name__ == "__main__":
    unittest.main()
