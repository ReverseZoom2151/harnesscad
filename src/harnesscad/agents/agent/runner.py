"""`run` — the plan -> apply -> observe -> replan loop, as a THIN VIEW of the one loop.

There is exactly one agent loop in this product and it is
:class:`core.harness.AgentHarness`. This module used to be a second, weaker
implementation of the same pattern -- no loop detection, no context pre-flight,
no write gate, no contract, no trajectory -- and `core/pipeline.build`, the
SHIPPING PATH, used it. Improvements to the harness did not reach the product.

`run` is now an adapter, not a loop: it drives `AgentHarness` and projects the
`HarnessRun` back onto the `ApplyOpsResult` its existing callers expect. Every
surface therefore gets loop detection, the guardrail + approval write gate, the
soundness feedback gate, and a replayable trajectory, whether it asked for them
or not.

BEHAVIOUR CHANGE, deliberately: a caller of `run` now gets a gated write path
and a gated feedback channel. Both were previously absent here. See
`tests/agents/agent/test_planner.py` and `tests/core/test_harness.py`.
"""

from __future__ import annotations

from typing import Any, List, Optional

from harnesscad.core.cisp.protocol import ApplyOpsResult
from harnesscad.core.loop import HarnessSession
from harnesscad.eval.verifiers.verify import Diagnostic, Severity


def run(
    session: HarnessSession,
    planner: Any,
    brief: str,
    max_iters: int = 5,
    **harness_kwargs: Any,
) -> ApplyOpsResult:
    """Drive THE loop; return the final ApplyOpsResult (ok or not).

    `harness_kwargs` are passed straight to `AgentHarness` (`context=`,
    `loop_detector=`, `executor=`, `tracer=`, `verifiers=`, `feedback_tiers=`,
    `gated=`), so a surface that needs different behaviour configures the one
    loop rather than writing a second one.
    """
    # Imported here, not at module scope: `agents.agent.__init__` imports this
    # module, and `core.harness` imports `agents.agent.feedback`. A top-level
    # import would close that cycle.
    from harnesscad.core.harness import AgentHarness

    harness = AgentHarness(session, planner, max_iterations=max_iters,
                           **harness_kwargs)
    hrun = harness.run(brief)

    # `ApplyOpsResult` is the verdict on ONE batch, so this projects the FINAL
    # iteration -- not `HarnessRun.applied`, which is the cumulative count across
    # every iteration including ops that were later rolled back.
    return ApplyOpsResult(
        ok=hrun.ok,
        applied=_final_applied(hrun),
        digest=hrun.digest,
        diagnostics=_as_diagnostics(hrun.diagnostics),
        rejected=_last_rejected(hrun),
    )


def _final_applied(hrun: Any) -> int:
    for entry in reversed(hrun.trajectory or []):
        if entry.get("dispatch_ok") is not None:
            return int(entry.get("applied") or 0)
    return 0


def _as_diagnostics(dicts: List[dict]) -> List[Diagnostic]:
    """Rehydrate the harness's serialised diagnostics, tier included."""
    out: List[Diagnostic] = []
    for d in dicts or []:
        try:
            sev = Severity(d.get("severity", "error"))
        except ValueError:
            sev = Severity.ERROR
        out.append(Diagnostic(
            severity=sev,
            code=str(d.get("code", "diagnostic")),
            message=str(d.get("message", "")),
            where=d.get("where"),
            soundness=d.get("soundness"),
        ))
    return out


def _last_rejected(hrun: Any) -> Optional[dict]:
    for entry in reversed(hrun.trajectory or []):
        rejected = entry.get("rejected")
        if rejected:
            return rejected
    return None
