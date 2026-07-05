"""AgentHarness — the top-level ReAct orchestrator (HARNESS_BLUEPRINT.md sec.3+4).

This is the single from-scratch module that ties the already-built pieces
together. It COMPOSES; it reimplements nothing:

  * ``HarnessSession`` (loop.py) owns the applyOps -> regen -> verify -> checkpoint
    spine (block-and-correct + transactional rollback live there).
  * ``Planner`` (agent/planner.py) turns brief + state + diagnostics into ops.
  * ``ContextManager`` (context/manager.py) does the pre-flight / assemble.
  * ``LoopDetector`` (loopdetect.py) flags oscillation over emitted op signatures.
  * a ``ToolExecutor`` (executor.py, if present) may run the ops; otherwise the
    harness dispatches straight through ``session.apply_ops``.
  * ``Tracer`` (trace.py) receives structured events under one stable ``run_id``.
  * ``verify.Verifier`` / ``ContractCheck`` (verify.py, contract.py) provide the
    plural geometry verifier and the machine-verifiable acceptance contract.

The relation to ``agent/runner.py``: runner is the minimal single-shot
plan -> apply -> replan loop. ``AgentHarness`` is the fuller ReAct orchestrator
that adds context pre-flight, loop detection, an optional tool-executor seam,
optional harness-level verifiers, contract satisfaction, structured tracing, and
a per-run trajectory audit trail around that same spine.

Determinism: no wall clock. The ``run_id`` is derived from the brief content
(matching the loop.py / trace.py conventions), so replaying the same brief yields
the same id and the same trajectory.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from cisp.ops import Op, canonical_json
from cisp.protocol import ApplyOpsResult
from contract import Contract, ContractCheck
from llm.structured import ParsedOps
from loop import HarnessSession
from trace import NullTracer, Tracer
from verify import Diagnostic, Severity


# The harness-level event kinds (in addition to the loop's own EVENT_KINDS in
# trace.py). Exposed so downstream tooling can route without literals.
HARNESS_EVENT_KINDS = (
    "harness_start",
    "iteration_start",
    "context",
    "plan",
    "loop_detected",
    "dispatch",
    "verify",
    "contract",
    "checkpoint",
    "harness_end",
)


@dataclass
class HarnessRun:
    """The result + audit trail of one ``AgentHarness.run``.

    ``trajectory`` is the ordered list of per-iteration records (the replayable
    audit trail); each entry is a plain dict so the whole run is JSON-serialisable.
    """

    ok: bool
    iterations: int
    applied: int
    digest: str
    diagnostics: List[dict] = field(default_factory=list)
    contract_ok: bool = True
    stop_reason: str = ""
    trajectory: List[dict] = field(default_factory=list)
    run_id: str = ""

    def to_dict(self) -> dict:
        return {
            "ok": self.ok,
            "iterations": self.iterations,
            "applied": self.applied,
            "digest": self.digest,
            "diagnostics": self.diagnostics,
            "contract_ok": self.contract_ok,
            "stop_reason": self.stop_reason,
            "trajectory": self.trajectory,
            "run_id": self.run_id,
        }


class AgentHarness:
    """The ReAct loop: pre-flight -> plan -> loop-detect -> dispatch -> verify ->
    observe -> repair, up to ``max_iterations``; checkpoint on success.

    Every collaborator except ``session`` and ``planner`` is optional and the
    harness degrades gracefully when one is absent (no context manager -> no
    pre-flight; no loop detector -> no oscillation guard; no executor ->
    ``session.apply_ops``; no verifiers -> the session's own plural verifier is
    the only gate; no contract -> ``contract_ok`` is trivially True).
    """

    def __init__(
        self,
        session: HarnessSession,
        planner: Any,
        *,
        context: Any = None,
        loop_detector: Any = None,
        executor: Any = None,
        tracer: Optional[Tracer] = None,
        verifiers: Optional[List[Any]] = None,
        max_iterations: int = 8,
    ) -> None:
        if max_iterations < 1:
            raise ValueError("max_iterations must be >= 1")
        self.session = session
        self.planner = planner
        self.context = context
        self.loop_detector = loop_detector
        self.executor = executor
        self.tracer = tracer if tracer is not None else NullTracer()
        self.verifiers = list(verifiers) if verifiers else []
        self.max_iterations = max_iterations

    # --- run id -----------------------------------------------------------
    def _make_run_id(self, brief: str) -> str:
        """Deterministic, wall-clock-free run id derived from the brief."""
        h = hashlib.sha256(brief.encode("utf-8")).hexdigest()
        return f"harness-{h[:12]}"

    # --- the ReAct loop ---------------------------------------------------
    def run(self, brief: str, contract: Optional[Contract] = None) -> HarnessRun:
        run_id = self._make_run_id(brief)
        self.tracer.event("harness_start", run_id, {
            "brief_chars": len(brief),
            "max_iterations": self.max_iterations,
            "has_contract": contract is not None,
        })

        trajectory: List[dict] = []
        diagnostics: Optional[List[Any]] = None
        last_diag_dicts: List[dict] = []
        applied_total = 0
        contract_ok = contract is None
        ok = False
        stop_reason = "max_iterations"

        for i in range(self.max_iterations):
            self.tracer.event("iteration_start", run_id, {"iteration": i})
            state_summary = self.session.summary()

            # 1. (optional) context pre-flight / assemble.
            self._preflight(run_id, i, brief, state_summary, diagnostics)

            # 2. plan: brief + latest state + fed-back diagnostics -> ops.
            parsed = self._plan(brief, state_summary, diagnostics)
            self.tracer.event("plan", run_id, {
                "iteration": i,
                "ok": parsed.ok,
                "op_count": len(parsed.ops),
                "error": parsed.error,
            })
            if not parsed.ok:
                # Malformed plan: surface as a diagnostic and re-prompt (repair).
                pe = {"severity": "error", "code": "plan-parse-error",
                      "message": parsed.error or "planner produced no valid ops"}
                diagnostics = [pe]
                last_diag_dicts = [pe]
                trajectory.append(self._entry(
                    i, [], None, verified=False, contract_ok=contract_ok,
                    converged=False, diagnostics=[pe], looped=False))
                continue

            ops = parsed.ops

            # 3. loop-detect over the emitted op signatures (pre-dispatch, so a
            #    detected oscillation never re-mutates state).
            if self._detect_loop(ops):
                looped_sig = [canonical_json(op) for op in ops]
                self.tracer.event("loop_detected", run_id, {
                    "iteration": i, "signatures": looped_sig})
                trajectory.append(self._entry(
                    i, ops, None, verified=False, contract_ok=contract_ok,
                    converged=False, diagnostics=last_diag_dicts, looped=True))
                stop_reason = "loop"
                ok = False
                break

            # 4. dispatch (ToolExecutor if provided, else session.apply_ops).
            result = self._dispatch(ops)
            applied_total += result.applied
            self.tracer.event("dispatch", run_id, {
                "iteration": i,
                "ok": result.ok,
                "applied": result.applied,
                "digest": result.digest,
                "rejected": result.rejected,
            })

            # 5. observe + verify: session already ran its plural verifier inside
            #    apply_ops; run any harness-level verifiers on top.
            hverify_ok, hverify_diags = self._harness_verify()
            self.tracer.event("verify", run_id, {
                "iteration": i,
                "dispatch_ok": result.ok,
                "harness_verify_ok": hverify_ok,
                "diagnostics": _diag_dicts(hverify_diags),
            })

            # 6. contract satisfaction (+ ContractCheck if a contract is given).
            contract_ok, contract_diags = self._check_contract(contract)
            if contract is not None:
                self.tracer.event("contract", run_id, {
                    "iteration": i,
                    "contract_ok": contract_ok,
                    "diagnostics": _diag_dicts(contract_diags),
                })

            verified = result.ok and hverify_ok
            converged = verified and contract_ok

            step_diags = (list(result.diagnostics) + list(hverify_diags)
                          + list(contract_diags))
            last_diag_dicts = _diag_dicts(step_diags)
            trajectory.append(self._entry(
                i, ops, result, verified=verified, contract_ok=contract_ok,
                converged=converged, diagnostics=last_diag_dicts, looped=False))

            # 7. terminal check (verified & contract-satisfied -> stop + checkpoint)
            #    OR repair (feed diagnostics back and iterate).
            if converged:
                label = f"{run_id}-i{i}-converged"
                self.session.checkpoint(label)
                self.tracer.event("checkpoint", run_id, {
                    "iteration": i, "label": label})
                stop_reason = "converged"
                ok = True
                break

            # Repair: feed the ERROR-severity diagnostics back to the planner.
            diagnostics = [d for d in step_diags if _is_error(d)] or step_diags

        run = HarnessRun(
            ok=ok,
            iterations=len(trajectory),
            applied=applied_total,
            digest=self.session.digest(),
            diagnostics=last_diag_dicts,
            contract_ok=contract_ok,
            stop_reason=stop_reason,
            trajectory=trajectory,
            run_id=run_id,
        )
        self.tracer.event("harness_end", run_id, {
            "ok": run.ok,
            "iterations": run.iterations,
            "applied": run.applied,
            "digest": run.digest,
            "contract_ok": run.contract_ok,
            "stop_reason": run.stop_reason,
        })
        return run

    # --- collaborators (each defensive / optional) ------------------------
    def _plan(
        self,
        brief: str,
        state_summary: Dict[str, Any],
        diagnostics: Optional[List[Any]],
    ) -> ParsedOps:
        """Ask the planner for ops. Prefer the non-raising ``plan_parsed`` seam;
        fall back to ``plan`` (which raises ``PlanError``) if that's all there is."""
        pp = getattr(self.planner, "plan_parsed", None)
        if callable(pp):
            return pp(brief, state_summary=state_summary, diagnostics=diagnostics)
        # Fallback for a minimal planner exposing only ``plan``.
        try:
            ops = self.planner.plan(
                brief, state_summary=state_summary, diagnostics=diagnostics)
        except Exception as exc:  # noqa: BLE001 - surface as re-promptable error
            return ParsedOps([], error=str(exc))
        return ParsedOps(list(ops))

    def _preflight(
        self,
        run_id: str,
        iteration: int,
        brief: str,
        state_summary: Dict[str, Any],
        diagnostics: Optional[List[Any]],
    ) -> None:
        """Run the ContextManager pre-flight if a context manager + a planner that
        can build messages are both present. Never fatal: a budget overflow is
        recorded as a trace event, not raised into the loop."""
        if self.context is None:
            return
        build = getattr(self.planner, "build_messages", None)
        preflight = getattr(self.context, "preflight", None)
        if not (callable(build) and callable(preflight)):
            return
        try:
            messages = build(brief, state_summary, diagnostics)
            report = preflight(messages, tools=None, strict=False)
            data = report.to_dict() if hasattr(report, "to_dict") else {}
            data["iteration"] = iteration
            self.tracer.event("context", run_id, data)
        except Exception as exc:  # noqa: BLE001 - pre-flight must not break the run
            self.tracer.event("context", run_id, {
                "iteration": iteration, "error": str(exc)})

    def _detect_loop(self, ops: List[Op]) -> bool:
        """Feed emitted op signatures to the loop detector; True on oscillation."""
        if self.loop_detector is None:
            return False
        looped = False
        for op in ops:
            # Observe every op (keeps the sliding window correct) but remember if
            # any observation completes a loop.
            if self.loop_detector.observe(op):
                looped = True
        return looped

    def _dispatch(self, ops: List[Op]) -> ApplyOpsResult:
        """Run the ops through the ToolExecutor if one is provided, else straight
        through the session. Degrades to ``session.apply_ops`` whenever no usable
        executor entry point is found (executor.py may land concurrently)."""
        ex = self.executor
        if ex is not None:
            for meth in ("apply_ops", "execute", "run", "dispatch"):
                fn = getattr(ex, meth, None)
                if callable(fn):
                    try:
                        return fn(ops)
                    except TypeError:
                        # Executor may want the session too: execute(session, ops).
                        return fn(self.session, ops)
        return self.session.apply_ops(ops)

    def _harness_verify(self):
        """Run any harness-level verifiers against the current backend state.

        These compose ON TOP of the session's own plural verifier (which already
        ran inside apply_ops and rolled back any ERROR). Returns (ok, diagnostics).
        """
        diags: List[Diagnostic] = []
        for v in self.verifiers:
            diags += v.check(self.session.backend, self.session.opdag).diagnostics
        ok = not any(_is_error_diag(d) for d in diags)
        return ok, diags

    def _check_contract(self, contract: Optional[Contract]):
        """Verify the current state against the acceptance contract, if given."""
        if contract is None:
            return True, []
        report = ContractCheck(contract).check(
            self.session.backend, self.session.opdag)
        return report.ok, list(report.diagnostics)

    # --- trajectory -------------------------------------------------------
    def _entry(
        self,
        iteration: int,
        ops: List[Op],
        result: Optional[ApplyOpsResult],
        *,
        verified: bool,
        contract_ok: bool,
        converged: bool,
        diagnostics: List[dict],
        looped: bool,
    ) -> dict:
        return {
            "iteration": iteration,
            "op_count": len(ops),
            "op_signatures": [canonical_json(op) for op in ops],
            "dispatch_ok": (result.ok if result is not None else None),
            "applied": (result.applied if result is not None else 0),
            "rejected": (result.rejected if result is not None else None),
            "verified": verified,
            "contract_ok": contract_ok,
            "converged": converged,
            "looped": looped,
            "digest": self.session.digest(),
            "diagnostics": diagnostics,
        }


# --- diagnostic helpers ----------------------------------------------------
def _diag_dicts(diags: List[Any]) -> List[dict]:
    out: List[dict] = []
    for d in diags:
        if hasattr(d, "to_dict"):
            out.append(d.to_dict())
        elif isinstance(d, dict):
            out.append(d)
        else:
            out.append({"severity": "error", "code": "diagnostic",
                        "message": str(d)})
    return out


def _is_error_diag(d: Diagnostic) -> bool:
    return getattr(d, "severity", None) is Severity.ERROR


def _is_error(d: Any) -> bool:
    """True if a diagnostic (Diagnostic object or dict) is ERROR severity."""
    if hasattr(d, "severity"):
        sev = d.severity
        return sev is Severity.ERROR or getattr(sev, "value", sev) == "error"
    if isinstance(d, dict):
        return d.get("severity", "error") == "error"
    return True
