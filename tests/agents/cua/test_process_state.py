"""Tests for the ghost-os-derived agent/process state model."""

import unittest

from harnesscad.agents.cua.process_state import (
    AgentRuntime, AgentState, LongRunningProcess, ProcessStatus, StateError, Tool,
)


class ProcessTest(unittest.TestCase):
    def test_lifecycle_pending_running_succeeded(self):
        p = LongRunningProcess(id="p1", name="export")
        self.assertEqual(p.status, ProcessStatus.PENDING)
        p.start()
        self.assertEqual(p.status, ProcessStatus.RUNNING)
        p.observe(ProcessStatus.SUCCEEDED, result={"volume": 1000.0})
        self.assertTrue(p.succeeded)
        self.assertEqual(p.result, {"volume": 1000.0})
        self.assertEqual(p.history, [ProcessStatus.PENDING, ProcessStatus.RUNNING,
                                     ProcessStatus.SUCCEEDED])

    def test_failed_carries_error(self):
        p = LongRunningProcess(id="p", name="recompute").start()
        p.observe(ProcessStatus.FAILED, error="topology broke")
        self.assertEqual(p.status, ProcessStatus.FAILED)
        self.assertEqual(p.error, "topology broke")

    def test_illegal_transition_refused(self):
        p = LongRunningProcess(id="p", name="x")
        # PENDING -> SUCCEEDED is not allowed (must RUN first).
        with self.assertRaises(StateError):
            p.observe(ProcessStatus.SUCCEEDED)

    def test_terminal_process_is_frozen(self):
        p = LongRunningProcess(id="p", name="x").start()
        p.observe(ProcessStatus.SUCCEEDED)
        with self.assertRaises(StateError):
            p.observe(ProcessStatus.FAILED)

    def test_cancel_from_running(self):
        p = LongRunningProcess(id="p", name="x").start()
        p.observe(ProcessStatus.CANCELLED)
        self.assertTrue(p.terminal)
        self.assertFalse(p.succeeded)


class AgentInlineTest(unittest.TestCase):
    def test_inline_tool_runs_and_returns_to_thinking(self):
        calls = []
        tool = Tool("measure", handler=lambda **kw: calls.append(kw) or 42)
        rt = AgentRuntime([tool])
        rt.think()
        result = rt.dispatch("measure", target="body")
        self.assertEqual(result, 42)
        self.assertEqual(rt.state, AgentState.THINKING)
        self.assertEqual(calls, [{"target": "body"}])

    def test_dispatch_unknown_tool_raises(self):
        rt = AgentRuntime()
        rt.think()
        with self.assertRaises(StateError):
            rt.dispatch("nope")

    def test_cannot_dispatch_from_idle(self):
        rt = AgentRuntime([Tool("t")])
        with self.assertRaises(StateError):
            rt.dispatch("t")


class AgentLongRunningTest(unittest.TestCase):
    def setUp(self):
        self.rt = AgentRuntime([Tool("export", long_running=True)])
        self.rt.think()

    def test_long_running_moves_to_waiting(self):
        proc = self.rt.dispatch("export")
        self.assertEqual(self.rt.state, AgentState.WAITING)
        self.assertEqual(proc.status, ProcessStatus.RUNNING)
        self.assertIn(proc.id, self.rt.processes)
        self.assertEqual(len(self.rt.pending_processes), 1)

    def test_observe_completion_returns_to_thinking(self):
        proc = self.rt.dispatch("export")
        self.rt.observe_process(proc.id, ProcessStatus.SUCCEEDED,
                                result={"ok": True})
        self.assertEqual(self.rt.state, AgentState.THINKING)
        self.assertEqual(self.rt.processes[proc.id].result, {"ok": True})

    def test_stays_waiting_while_another_process_runs(self):
        p1 = self.rt.dispatch("export")
        p2 = self.rt.dispatch("export")   # a second launched while waiting
        self.assertEqual(self.rt.state, AgentState.WAITING)
        self.rt.observe_process(p1.id, ProcessStatus.SUCCEEDED)
        # still one running -> still waiting
        self.assertEqual(self.rt.state, AgentState.WAITING)
        self.rt.observe_process(p2.id, ProcessStatus.SUCCEEDED)
        self.assertEqual(self.rt.state, AgentState.THINKING)

    def test_observe_unknown_process_raises(self):
        with self.assertRaises(StateError):
            self.rt.observe_process("ghost#9", ProcessStatus.SUCCEEDED)

    def test_finish_refused_while_process_running(self):
        self.rt.dispatch("export")
        with self.assertRaises(StateError):
            self.rt.finish()

    def test_finish_after_all_resolved(self):
        proc = self.rt.dispatch("export")
        self.rt.observe_process(proc.id, ProcessStatus.SUCCEEDED)
        self.rt.finish()
        self.assertEqual(self.rt.state, AgentState.DONE)


class AgentTransitionTest(unittest.TestCase):
    def test_illegal_agent_transition_refused(self):
        rt = AgentRuntime()
        rt.think()
        rt.fail("boom")
        self.assertEqual(rt.state, AgentState.FAILED)
        # terminal: no further transitions
        with self.assertRaises(StateError):
            rt.think()

    def test_done_is_terminal(self):
        rt = AgentRuntime()
        rt.think()
        rt.finish()
        with self.assertRaises(StateError):
            rt.fail()


if __name__ == "__main__":
    unittest.main()
