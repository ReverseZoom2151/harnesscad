"""ToolExecutor — the sandboxed, gated, retrying tool-call orchestration layer.

Per HARNESS_BLUEPRINT.md sec.3 (``ToolExecutor``) and sec.10 (guardrails &
error-recovery ladder). This is a THIN orchestration layer over pieces that
already exist — it contains no geometry logic of its own:

  1. ``before_tool_callback`` HARD GATE — every kernel op runs through
     ``guardrails.GuardrailGate`` BEFORE it can touch the model. A guardrail
     violation is *blocked and corrected*: the executor returns the correction
     diagnostics and NEVER applies the op to the session (block-and-correct,
     sec.10). The gate is composed, not reimplemented.
  2. HUMAN-APPROVAL GATE — Tier-3 ops (export/delete/irreversible, per
     ``ui.approval``) require approval. ``ui.approval`` is composed via lazy
     import to classify the tier and surface a risk indicator + dry-run preview;
     the actual yes/no comes from a pluggable ``approve(op) -> bool``. The default
     auto-approves with a note, so tests need no human.
  3. RETRY + EXPONENTIAL BACKOFF — a *transient* failure (a backend/session that
     raises) is retried with exponential backoff up to ``max_retries``. A
     *deterministic* rejection (``ApplyOpsResult.ok is False`` — an invalid op the
     backend/verifier blocked) is NEVER retried unchanged (sec.10: "retry with
     adjusted params — never the same invalid op unchanged").
  4. TIMEOUT — a step whose elapsed time (measured on the injected clock) exceeds
     ``timeout`` returns a timed-out result.
  5. OUTPUT TRUNCATION — any oversized diagnostic/log payload is truncated to
     ``max_output`` so a mesh/log dump can't blow the context window.

Determinism: there is NO wall-clock on the default path. ``clock`` (a monotonic
logical counter by default) and ``sleeper`` (a no-op by default) are injected, so
retry/backoff/timeout are fully testable without real time.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, List, Optional

from cisp.ops import Op
from cisp.protocol import ApplyOpsResult
from guardrails import ErrorRecovery, GuardrailGate
from verify import Diagnostic, Severity


# --- injected time primitives ---------------------------------------------
class LogicalClock:
    """A deterministic, wall-clock-free monotonic clock.

    Each call returns a strictly increasing integer. This is the executor's
    DEFAULT clock so the standard path never reads wall time; tests that need to
    force a timeout inject their own clock that jumps past ``timeout``.
    """

    def __init__(self, start: int = 0) -> None:
        self._t = start

    def __call__(self) -> float:
        self._t += 1
        return self._t


def _noop_sleeper(_delay: float) -> None:
    """Default sleeper: consumes a backoff delay without touching real time."""
    return None


# --- result type -----------------------------------------------------------
@dataclass
class ExecResult:
    """The verdict of one ``ToolExecutor.execute`` call.

    ``blocked`` is reserved for the guardrail hard gate (an op that never reached
    the session). ``approved`` is False when a Tier-3 op was denied. ``result`` is
    the underlying ``ApplyOpsResult`` when the op reached the session, else None.
    ``diagnostics`` are already output-truncated; ``truncated`` records whether any
    payload was clipped. ``note`` carries an advisory (e.g. the auto-approval note).
    """

    ok: bool
    blocked: bool = False
    approved: bool = True
    timed_out: bool = False
    attempts: int = 0
    result: Optional[ApplyOpsResult] = None
    diagnostics: List[Diagnostic] = field(default_factory=list)
    truncated: bool = False
    note: str = ""

    def to_dict(self) -> dict:
        return {
            "ok": self.ok,
            "blocked": self.blocked,
            "approved": self.approved,
            "timed_out": self.timed_out,
            "attempts": self.attempts,
            "result": self.result.to_dict() if self.result is not None else None,
            "diagnostics": [d.to_dict() for d in self.diagnostics],
            "truncated": self.truncated,
            "note": self.note,
        }


class ToolExecutor:
    """Sandboxed, gated, retrying orchestration over ``HarnessSession.apply_ops``.

    Composes ``guardrails.GuardrailGate`` (hard pre-apply gate) and, when present,
    ``ui.approval`` (Tier-3 classification + approval events). It owns no geometry
    logic — it only sequences: gate -> approve -> apply-with-retry/backoff/timeout
    -> truncate.
    """

    def __init__(
        self,
        gate: Optional[GuardrailGate] = None,
        approval: Optional[Callable[[Op], bool]] = None,
        max_retries: int = 2,
        backoff_base: float = 0.5,
        timeout: Optional[float] = None,
        max_output: int = 2000,
        clock: Optional[Callable[[], float]] = None,
        sleeper: Optional[Callable[[float], None]] = None,
    ) -> None:
        self.gate = gate if gate is not None else GuardrailGate()
        # ``approval`` is the pluggable yes/no decider for Tier-3 ops. None => the
        # default auto-approve-with-a-note behaviour (so tests need no human).
        self.approval = approval
        self.max_retries = max_retries
        self.backoff_base = backoff_base
        self.timeout = timeout
        self.max_output = max_output
        self.clock = clock if clock is not None else LogicalClock()
        self.sleeper = sleeper if sleeper is not None else _noop_sleeper
        # Composed ui.approval gate (lazy) — records approval_required events with
        # a risk indicator + dry-run preview for inspection. None if ui absent.
        self._approval_gate = self._make_approval_gate()

    # --- ui.approval composition (lazy import; degrades gracefully) --------
    @staticmethod
    def _make_approval_gate():
        try:
            from ui.approval import ApprovalGate
        except Exception:
            return None
        return ApprovalGate()

    def _requires_approval(self, op: Op) -> bool:
        """True iff ``op`` is Tier-3 (REQUIRE) per ui.approval. False if ui absent."""
        try:
            from ui.approval import ApprovalTier, tier_for
        except Exception:
            return False
        try:
            return tier_for(op) is ApprovalTier.REQUIRE
        except Exception:
            return False

    def _approve(self, op: Op) -> tuple:
        """Return (approved, note) for ``op`` running the human-approval gate.

        Tier-1/Tier-2 auto-proceed. Tier-3 REQUIRE: surface the approval_required
        event (composed ui.approval, for a risk indicator + preview) then consult
        the pluggable approver; the default auto-approves with a note.
        """
        if not self._requires_approval(op):
            return True, ""
        # Surface the risk indicator + dry-run preview (best-effort; never fatal).
        if self._approval_gate is not None:
            try:
                self._approval_gate.evaluate(op)
            except Exception:
                pass
        if self.approval is None:
            return True, "tier-3 auto-approved (default): no human approver configured"
        if callable(self.approval):
            return bool(self.approval(op)), "tier-3: pluggable approver consulted"
        # Fallback: an object exposing may_proceed(op) -> bool.
        if hasattr(self.approval, "may_proceed"):
            return bool(self.approval.may_proceed(op)), "tier-3: approval gate consulted"
        return True, "tier-3 auto-approved (default): unrecognised approver"

    # --- output truncation ------------------------------------------------
    def _truncate(self, diags: List[Diagnostic]) -> tuple:
        """Clip each diagnostic message to ``max_output`` chars.

        Returns (new_diagnostics, truncated_flag). New ``Diagnostic`` instances are
        built so the session's own diagnostics are never mutated. A single oversized
        payload (a mesh/log dump) is what this defends the context window against.
        """
        if self.max_output is None:
            return list(diags), False
        out: List[Diagnostic] = []
        truncated = False
        for d in diags:
            msg = d.message
            if isinstance(msg, str) and len(msg) > self.max_output:
                clipped = len(msg) - self.max_output
                msg = msg[: self.max_output] + f"...[truncated {clipped} chars]"
                out.append(Diagnostic(d.severity, d.code, msg, d.where))
                truncated = True
            else:
                out.append(d)
        return out, truncated

    # --- main pipeline ----------------------------------------------------
    def execute(self, op: Op, session) -> ExecResult:
        """Run one op through the full gate -> approve -> apply pipeline."""
        backend = getattr(session, "backend", None)

        # (1) before_tool_callback HARD GATE. A guardrail violation is blocked and
        # corrected — the op NEVER touches the session.
        gate_diags = self.gate.check(op, backend=backend)
        if gate_diags:
            diags, truncated = self._truncate(gate_diags)
            return ExecResult(
                ok=False, blocked=True, approved=False, timed_out=False,
                attempts=0, result=None, diagnostics=diags, truncated=truncated,
                note="blocked by before_tool_callback hard gate (block-and-correct)",
            )

        # (2) HUMAN-APPROVAL GATE (Tier-3 only).
        approved, note = self._approve(op)
        if not approved:
            deny = Diagnostic(
                Severity.ERROR, "approval-denied",
                f"Tier-3 op '{getattr(op, 'OP', type(op).__name__)}' was denied "
                f"by the human-approval gate; op not applied")
            diags, truncated = self._truncate([deny])
            return ExecResult(
                ok=False, blocked=False, approved=False, timed_out=False,
                attempts=0, result=None, diagnostics=diags, truncated=truncated,
                note="denied by human-approval gate",
            )

        # (3)+(4)+(5) apply through the session with retry + backoff, timeout, and
        # output truncation.
        attempts = 0
        last_diags: List[Diagnostic] = []
        last_result: Optional[ApplyOpsResult] = None

        for attempt in range(self.max_retries + 1):
            if attempt > 0:
                # Exponential backoff before a transient retry (no-op sleeper by
                # default keeps this wall-clock-free and deterministic).
                self.sleeper(self.backoff_base * (2 ** (attempt - 1)))
            attempts += 1

            t0 = self.clock()
            try:
                result = session.apply_ops([op])
            except Exception as exc:
                # Transient/backend error — retry with backoff (never a deterministic
                # invalid op; those come back as ok=False below and are NOT retried).
                last_result = None
                last_diags = [Diagnostic(
                    Severity.ERROR, "transient-error",
                    f"transient backend/session failure on attempt {attempts}: {exc}")]
                continue
            t1 = self.clock()

            # (4) TIMEOUT — a step exceeding the budget on the injected clock.
            if self.timeout is not None and (t1 - t0) > self.timeout:
                diags, truncated = self._truncate(
                    list(result.diagnostics) if result is not None else [])
                diags.append(Diagnostic(
                    Severity.ERROR, "timeout",
                    f"op exceeded timeout ({t1 - t0} > {self.timeout})"))
                return ExecResult(
                    ok=False, blocked=False, approved=True, timed_out=True,
                    attempts=attempts, result=result, diagnostics=diags,
                    truncated=truncated, note=note,
                )

            if result.ok:
                diags, truncated = self._truncate(list(result.diagnostics))
                return ExecResult(
                    ok=True, blocked=False, approved=True, timed_out=False,
                    attempts=attempts, result=result, diagnostics=diags,
                    truncated=truncated, note=note,
                )

            # Deterministic rejection (block-and-correct at the session/verifier
            # layer). Retrying the SAME op unchanged would only fail identically, so
            # we return immediately (sec.10).
            diags, truncated = self._truncate(list(result.diagnostics))
            return ExecResult(
                ok=False, blocked=False, approved=True, timed_out=False,
                attempts=attempts, result=result, diagnostics=diags,
                truncated=truncated, note=note,
            )

        # Retries exhausted — every attempt raised a transient failure.
        diags, truncated = self._truncate(last_diags)
        return ExecResult(
            ok=False, blocked=False, approved=True, timed_out=False,
            attempts=attempts, result=last_result, diagnostics=diags,
            truncated=truncated,
            note=f"exhausted {self.max_retries} retries on transient failure",
        )

    # --- error-recovery ladder hook ---------------------------------------
    def handle_failure(self, result: ExecResult) -> dict:
        """Map an ``ExecResult`` failure onto the ErrorRecovery detect->handle->recover
        ladder (guardrails.ErrorRecovery — composed, not reimplemented).

        Returns an advisory plan: the inferred *detect* signal, a recommended
        *handle* and *recover* strategy, and the full ladder for the caller to walk.
        This is metadata the loop consults to pick its next move; it does not act.
        """
        detect = self._detect(result)
        handle, recover = self._recommend(result, detect)
        return {
            "detect": detect,
            "handle": handle,
            "recover": recover,
            "ladder": {
                stage: ErrorRecovery.strategies(stage)
                for stage in ErrorRecovery.stages()
            },
        }

    # --- ladder inference helpers -----------------------------------------
    # Map a diagnostic code to a DETECT-stage signal from the blueprint ladder.
    _CODE_TO_DETECT = {
        "over-constrained": "over-constrained",
        "under-constrained": "under-constrained",
        "empty-solid": "empty",
        "boolean-nulls-body": "boolean-fail",
        "boolean-bad-kind": "boolean-fail",
        "timeout": "timeout",
        "transient-error": "regen-fail",
    }

    def _detect(self, result: ExecResult) -> str:
        if result.timed_out:
            return "timeout"
        for d in result.diagnostics:
            if d.code in self._CODE_TO_DETECT:
                return self._CODE_TO_DETECT[d.code]
        # A guardrail block is, by construction, an invalid-parameter detection;
        # an empty result is the generic "empty" signal.
        if result.blocked:
            return "over-constrained" if any(
                d.code == "over-constrained" for d in result.diagnostics) else "empty"
        return "empty"

    def _recommend(self, result: ExecResult, detect: str) -> tuple:
        """Pick a (handle, recover) pair for the detected failure."""
        if not result.approved:
            # A denied Tier-3 op is a human decision — escalate, don't retry.
            return "graceful-degradation", "escalate"
        if result.timed_out:
            return "retry-adjusted-params", "rollback-feature-tree"
        if result.blocked:
            # Never re-emit the same invalid op unchanged.
            return "retry-adjusted-params", "reflect-diagnose"
        if detect == "regen-fail":
            # Exhausted transient retries — fall back to a simpler strategy.
            return "fallback-simpler-strategy", "rollback-feature-tree"
        return "retry-adjusted-params", "rollback-feature-tree"
