"""The plan -> apply -> observe -> replan loop.

`run` closes the correction loop the whole harness is built around: ask the
planner for ops, apply them through the session (which itself does block-and-
correct + transactional verify), and if the batch isn't `ok`, feed the returned
diagnostics back into the next `plan()` call. Repeat until the model reaches a
verified state or `max_iters` is exhausted. The final `ApplyOpsResult` is
returned either way.
"""

from __future__ import annotations

from typing import List, Optional

from harnesscad.core.cisp.protocol import ApplyOpsResult
from harnesscad.core.loop import HarnessSession
from harnesscad.agents.agent.planner import Planner, PlanError


def run(
    session: HarnessSession,
    planner: Planner,
    brief: str,
    max_iters: int = 5,
) -> ApplyOpsResult:
    """Drive the agent loop; return the final ApplyOpsResult (ok or not)."""
    diagnostics: Optional[List] = None
    result: Optional[ApplyOpsResult] = None

    for _ in range(max_iters):
        state = session.summary()
        try:
            ops = planner.plan(brief, state_summary=state, diagnostics=diagnostics)
        except PlanError as e:
            # Malformed model output: surface it as a diagnostic and re-prompt.
            diagnostics = [{
                "severity": "error",
                "code": "plan-parse-error",
                "message": str(e),
            }]
            result = ApplyOpsResult(False, 0, session.digest(),
                                    diagnostics=[], rejected=None)
            continue

        result = session.apply_ops(ops)
        if result.ok:
            return result
        # Not ok: feed the diagnostics from this batch back into the next plan.
        diagnostics = list(result.diagnostics)

    # Exhausted iterations without a verified state.
    return result if result is not None else ApplyOpsResult(
        False, 0, session.digest(), diagnostics=[], rejected=None
    )
