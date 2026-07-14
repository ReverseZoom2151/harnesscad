"""ToolExecutor — the sandboxed, gated, retrying tool-call orchestration layer.

Per docs/blueprint.md sec.3 (``ToolExecutor``) and sec.10 (guardrails &
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

from harnesscad.core.cisp.ops import Op
from harnesscad.core.cisp.protocol import ApplyOpsResult
from harnesscad.eval.reliability.guardrails import ErrorRecovery, GuardrailGate
from harnesscad.eval.verifiers.verify import Diagnostic, Severity
from harnesscad.io.surfaces.ui.approval import ApprovalPolicy


def _make_policy(approval: Optional[Callable[[Op], bool]],
                 *, surface: str) -> ApprovalPolicy:
    """Adapt a legacy ``approval`` callable / gate object into an ApprovalPolicy.

    ``None`` -> the refuse-by-default headless policy (never a silent approve).
    A callable -> that callable is the human channel.
    An object with ``may_proceed`` -> its ``may_proceed`` is the human channel.
    """
    if approval is None:
        return ApprovalPolicy(None, surface=surface)
    if callable(approval):
        return ApprovalPolicy(approval, principal="approver", surface=surface)
    may_proceed = getattr(approval, "may_proceed", None)
    if callable(may_proceed):
        return ApprovalPolicy(may_proceed, principal="approval-gate", surface=surface)
    raise TypeError(
        "approval must be a callable(op) -> bool, an object exposing "
        "may_proceed(op) -> bool, or None (headless: refuse by default). "
        "An unrecognised approver used to be treated as an auto-approve; it is "
        "now an error, because that is how a destructive op ships unguarded.")


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
        policy: Optional["ApprovalPolicy"] = None,
    ) -> None:
        self.gate = gate if gate is not None else GuardrailGate()
        # ``approval`` is the pluggable human yes/no decider for Tier-3 ops; it is
        # kept for callers that pass a bare callable (the ACP surface passes its
        # blocking session/request_permission round-trip). Either way it becomes
        # an ``ApprovalPolicy``: the gate is MECHANICAL (``require`` raises) and
        # every decision it makes is recorded on ``policy.audit``.
        #
        # THE DEFAULT USED TO BE A SILENT AUTO-APPROVE ("tier-3 auto-approved
        # (default): no human approver configured"), which is exactly the bypass
        # the audit found: the gate existed, classified correctly, emitted its
        # risk indicator, and then let every destructive op through anyway. The
        # default is now REFUSE, and an unattended surface that legitimately needs
        # to proceed must say so by name via
        # ``ApprovalPolicy.headless_auto_approve(reason=...)``.
        self.approval = approval
        self.policy = policy if policy is not None else _make_policy(
            approval, surface="harness")
        self.max_retries = max_retries
        self.backoff_base = backoff_base
        self.timeout = timeout
        self.max_output = max_output
        self.clock = clock if clock is not None else LogicalClock()
        self.sleeper = sleeper if sleeper is not None else _noop_sleeper

    @property
    def approval_audit(self) -> List[dict]:
        """The auditable record of every approval decision this executor made."""
        return self.policy.audit_dicts() if self.policy is not None else []

    def _approve(self, op: Op) -> tuple:
        """Return (approved, note) for ``op`` running the human-approval gate.

        Tier-1/Tier-2 auto-proceed (and are still recorded). Tier-3 REQUIRE blocks
        on the policy: a human approver if one is attached, else the explicit
        headless policy (REFUSE by default).
        """
        if self.policy is None:  # ui.approval unavailable: fail closed on nothing
            return True, ""
        record = self.policy.decide(op)
        return record.approved, f"tier-{record.tier.value}: {record.reason}"

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


# ---------------------------------------------------------------------------
# The batch adapter — how the harness actually dispatches.
# ---------------------------------------------------------------------------
class SessionToolExecutor:
    """Drive a whole op batch through a :class:`ToolExecutor`, one op at a time.

    ``AgentHarness`` dispatches a BATCH (`apply_ops(ops) -> ApplyOpsResult`);
    ``ToolExecutor`` gates a single OP (`execute(op, session) -> ExecResult`).
    This is the adapter between them, and it is what makes the guardrail gate,
    the human-approval gate, retry/backoff and output truncation apply on the
    DEFAULT harness path instead of only on the ACP editor surface.

    Semantics preserved from ``HarnessSession.apply_ops``: ops are applied in
    order; the first op that is blocked, denied or rejected stops the batch and
    is reported as ``rejected``; every diagnostic seen along the way is returned.
    """

    def __init__(self, session, executor: Optional[ToolExecutor] = None) -> None:
        self.session = session
        self.executor = executor if executor is not None else ToolExecutor()

    def apply_ops(self, ops: List[Op]) -> ApplyOpsResult:
        applied = 0
        diags: List[Diagnostic] = []
        rejected: Optional[dict] = None

        for op in ops:
            res = self.executor.execute(op, self.session)
            diags.extend(res.diagnostics)
            if res.ok:
                applied += res.result.applied if res.result is not None else 1
                continue
            rejected = {
                "op": op.to_dict() if hasattr(op, "to_dict") else str(op),
                "reason": ("guardrail-blocked" if res.blocked
                           else "approval-denied" if not res.approved
                           else "timeout" if res.timed_out
                           else "verifier-rejected"),
                "note": res.note,
            }
            break

        ok = rejected is None and not any(
            d.severity is Severity.ERROR for d in diags)
        return ApplyOpsResult(ok, applied, self.session.digest(),
                              diagnostics=diags, rejected=rejected)
