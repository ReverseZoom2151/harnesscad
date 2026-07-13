"""Behaviour tests for the ToolExecutor orchestration layer.

Every test injects a logical clock and a no-op sleeper (or a recording sleeper) so
retry/backoff/timeout are exercised WITHOUT real time. Backends/sessions are real
where possible (StubBackend + HarnessSession) and doubles only where the behaviour
under test needs one (transient failures, oversized payloads, Tier-3 ops).
"""

import unittest
from dataclasses import dataclass
from typing import ClassVar, List, Optional

from harnesscad.core.cisp.ops import Op, NewSketch, Extrude
from harnesscad.core.cisp.protocol import ApplyOpsResult
from harnesscad.io.backends.base import ApplyResult
from harnesscad.io.backends.stub import StubBackend
from harnesscad.core.loop import HarnessSession
from harnesscad.eval.verifiers.verify import Diagnostic, Severity

from harnesscad.eval.reliability.executor import ExecResult, LogicalClock, ToolExecutor


# --- test doubles ----------------------------------------------------------
class StepClock:
    """A callable clock returning successive values from a list, then holding.

    Lets a test make one ``execute`` step *appear* to take a controlled amount of
    logical time (t1 - t0) with no wall-clock involved.
    """

    def __init__(self, values: List[float]) -> None:
        self._values = list(values)
        self._i = 0

    def __call__(self) -> float:
        v = self._values[min(self._i, len(self._values) - 1)]
        self._i += 1
        return v


class RecordingSleeper:
    """Records each backoff delay it is asked to sleep (never sleeps for real)."""

    def __init__(self) -> None:
        self.delays: List[float] = []

    def __call__(self, delay: float) -> None:
        self.delays.append(delay)


class FlakyBackend:
    """Wraps a real StubBackend but RAISES on the first ``fail_times`` applies.

    Models a transient backend fault (kernel hiccup) the executor should retry.
    """

    def __init__(self, fail_times: int) -> None:
        self._inner = StubBackend()
        self._fail_times = fail_times
        self._applies = 0

    def reset(self) -> None:
        self._inner.reset()

    def apply(self, op: Op) -> ApplyResult:
        self._applies += 1
        if self._applies <= self._fail_times:
            raise RuntimeError(f"transient kernel fault #{self._applies}")
        return self._inner.apply(op)

    def regenerate(self):
        return self._inner.regenerate()

    def query(self, q: str) -> dict:
        return self._inner.query(q)

    def export(self, fmt: str):
        return self._inner.export(fmt)

    def state_digest(self) -> str:
        return self._inner.state_digest()


class FakeSession:
    """A minimal session double returning a preconfigured ApplyOpsResult.

    Records every ``apply_ops`` call so a test can assert the session was (or was
    not) touched.
    """

    def __init__(self, result: ApplyOpsResult, backend=None) -> None:
        self._result = result
        self.backend = backend
        self.calls: List[list] = []

    def apply_ops(self, ops) -> ApplyOpsResult:
        self.calls.append(list(ops))
        return self._result


@dataclass(frozen=True)
class DeleteOp(Op):
    """A Tier-3 (destructive) op — 'delete' maps to REQUIRE in ui.approval."""

    OP: ClassVar[str] = "delete"
    target: str = ""


def _ok_result(diags: Optional[List[Diagnostic]] = None) -> ApplyOpsResult:
    return ApplyOpsResult(True, 1, "digest-x", diags or [])


# --- (1) guardrail hard gate ----------------------------------------------
class TestGuardrailBlock(unittest.TestCase):
    def test_invalid_op_blocked_without_mutating_session(self):
        session = HarnessSession(StubBackend())
        before = session.digest()
        ex = ToolExecutor()
        # Non-positive extrude depth -> guardrail 'extrude-nonpositive' hard block.
        res = ex.execute(Extrude(sketch="sk1", distance=0.0), session)
        self.assertFalse(res.ok)
        self.assertTrue(res.blocked)
        self.assertFalse(res.approved)
        self.assertEqual(res.attempts, 0)
        self.assertIsNone(res.result)
        self.assertTrue(any(d.code == "extrude-nonpositive" for d in res.diagnostics))
        # The session was never touched: block-and-correct.
        self.assertEqual(session.digest(), before)


# --- (2) valid op, one attempt --------------------------------------------
class TestValidOneAttempt(unittest.TestCase):
    def test_valid_op_executes_in_one_attempt(self):
        session = HarnessSession(StubBackend())
        ex = ToolExecutor()
        res = ex.execute(NewSketch(), session)
        self.assertTrue(res.ok)
        self.assertFalse(res.blocked)
        self.assertTrue(res.approved)
        self.assertFalse(res.timed_out)
        self.assertEqual(res.attempts, 1)
        self.assertIsNotNone(res.result)
        self.assertTrue(res.result.ok)


# --- (3) transient retry + backoff ----------------------------------------
class TestTransientRetry(unittest.TestCase):
    def test_transient_failure_retried_with_backoff_then_succeeds(self):
        session = HarnessSession(FlakyBackend(fail_times=1))
        sleeper = RecordingSleeper()
        ex = ToolExecutor(max_retries=2, backoff_base=0.5,
                          clock=LogicalClock(), sleeper=sleeper)
        res = ex.execute(NewSketch(), session)
        self.assertTrue(res.ok)
        self.assertEqual(res.attempts, 2)  # failed once, succeeded on retry
        # Exactly one backoff delay was taken, and it was the exponential base.
        self.assertEqual(sleeper.delays, [0.5])

    def test_exceeding_max_retries_returns_not_ok(self):
        # Always-failing transient backend; 1 retry -> 2 attempts, both raise.
        session = HarnessSession(FlakyBackend(fail_times=99))
        sleeper = RecordingSleeper()
        ex = ToolExecutor(max_retries=1, backoff_base=0.5,
                          clock=LogicalClock(), sleeper=sleeper)
        res = ex.execute(NewSketch(), session)
        self.assertFalse(res.ok)
        self.assertFalse(res.blocked)
        self.assertEqual(res.attempts, 2)
        self.assertEqual(sleeper.delays, [0.5])  # one backoff between the 2 attempts
        self.assertTrue(any(d.code == "transient-error" for d in res.diagnostics))

    def test_deterministic_invalid_op_is_not_retried(self):
        # An op the backend deterministically rejects (ok=False, not an exception)
        # must NOT be retried unchanged. Extrude on a missing sketch: passes the
        # guardrail (positive distance) but the backend rejects the bad reference.
        session = HarnessSession(StubBackend())
        sleeper = RecordingSleeper()
        ex = ToolExecutor(max_retries=3, clock=LogicalClock(), sleeper=sleeper)
        res = ex.execute(Extrude(sketch="missing", distance=5.0), session)
        self.assertFalse(res.ok)
        self.assertFalse(res.blocked)
        self.assertEqual(res.attempts, 1)       # tried exactly once
        self.assertEqual(sleeper.delays, [])    # no backoff/retry


# --- (4) timeout ----------------------------------------------------------
class TestTimeout(unittest.TestCase):
    def test_slow_op_past_timeout_returns_timed_out(self):
        session = HarnessSession(StubBackend())
        # Clock jumps 0 -> 100 across the single step; 100 > timeout(10).
        clock = StepClock([0, 100])
        ex = ToolExecutor(timeout=10, clock=clock)
        res = ex.execute(NewSketch(), session)
        self.assertTrue(res.timed_out)
        self.assertFalse(res.ok)
        self.assertEqual(res.attempts, 1)
        self.assertTrue(any(d.code == "timeout" for d in res.diagnostics))


# --- (5) output truncation -------------------------------------------------
class TestOutputTruncation(unittest.TestCase):
    def test_oversized_diagnostic_is_truncated(self):
        big = "M" * 10_000  # a mesh/log dump that must not blow the context
        payload = ApplyOpsResult(True, 1, "d", [Diagnostic(Severity.WARNING, "mesh", big)])
        session = FakeSession(payload, backend=StubBackend())
        ex = ToolExecutor(max_output=100)
        res = ex.execute(NewSketch(), session)
        self.assertTrue(res.ok)
        self.assertTrue(res.truncated)
        # The clipped message is bounded by max_output plus a short marker.
        clipped = res.diagnostics[0].message
        self.assertTrue(clipped.startswith("M" * 100))
        self.assertLess(len(clipped), 100 + 40)
        self.assertIn("truncated", clipped)


# --- (2b) Tier-3 human-approval gate --------------------------------------
class TestApproval(unittest.TestCase):
    def test_tier3_auto_approves_by_default(self):
        session = FakeSession(_ok_result(), backend=StubBackend())
        ex = ToolExecutor()  # default: auto-approve with a note
        res = ex.execute(DeleteOp(target="body1"), session)
        self.assertTrue(res.approved)
        self.assertTrue(res.ok)
        self.assertIn("auto-approved", res.note)
        self.assertEqual(len(session.calls), 1)  # op reached the session

    def test_tier3_denied_blocks_without_touching_session(self):
        session = FakeSession(_ok_result(), backend=StubBackend())
        ex = ToolExecutor(approval=lambda op: False)  # human says no
        res = ex.execute(DeleteOp(target="body1"), session)
        self.assertFalse(res.ok)
        self.assertFalse(res.approved)
        self.assertFalse(res.blocked)  # denied by approval, not the guardrail
        self.assertEqual(len(session.calls), 0)  # session never touched
        self.assertTrue(any(d.code == "approval-denied" for d in res.diagnostics))

    def test_tier2_op_does_not_require_approval(self):
        session = FakeSession(_ok_result(), backend=StubBackend())
        # A Tier-2 op with a denying approver: approver must NOT be consulted.
        consulted = []
        ex = ToolExecutor(approval=lambda op: consulted.append(op) or False)
        res = ex.execute(NewSketch(), session)
        self.assertTrue(res.ok)
        self.assertTrue(res.approved)
        self.assertEqual(consulted, [])  # Tier-2 auto-proceeds, approver skipped


# --- error-recovery ladder hook -------------------------------------------
class TestHandleFailure(unittest.TestCase):
    def test_handle_failure_maps_to_recovery_ladder(self):
        ex = ToolExecutor()
        timed_out = ExecResult(ok=False, timed_out=True, attempts=1,
                               diagnostics=[Diagnostic(Severity.ERROR, "timeout", "slow")])
        plan = ex.handle_failure(timed_out)
        self.assertEqual(plan["detect"], "timeout")
        self.assertIn(plan["handle"], plan["ladder"]["handle"])
        self.assertIn(plan["recover"], plan["ladder"]["recover"])
        self.assertEqual(list(plan["ladder"].keys()), ["detect", "handle", "recover"])

    def test_handle_failure_denied_escalates(self):
        ex = ToolExecutor()
        denied = ExecResult(ok=False, approved=False, attempts=0,
                            diagnostics=[Diagnostic(Severity.ERROR, "approval-denied", "no")])
        plan = ex.handle_failure(denied)
        self.assertEqual(plan["recover"], "escalate")


if __name__ == "__main__":
    unittest.main()
