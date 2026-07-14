"""Wire `tool_reward` to the loop — the process reward the harness never used.

`agents/agent/tool_reward.py` implements ``R = alpha*R_ORM + beta*mean(R_step) +
gamma*R_format`` with a per-step binary execution reward. Its only importers were
`agents/registry.py` (a dispatch table) and its own unit test. It was not in
`core/loop.py`, not in `core/trace.py`, and not in `eval/pressure/`. The harness
therefore ran **outcome-only supervision on a 3-8-op trajectory** while carrying a
finished process-reward implementation — the configuration the literature marks
"sparse" and "easier to hack" — and it took 6 models x 12 briefs x 2 loops to
discover a poisoning that a per-step delta shows on the first regressed brief.

This module is the seam. `core/loop.py` now emits a `step_reward` event per op
and accumulates `session.step_rewards`; here we lift that vector into the
`ToolTrajectory` `tool_reward` already speaks, and score a whole attempt:

    reward_for_session(session, orm_verdict=grader_says_solved)
        -> ToolUseReward(outcome, step_mean, fmt, total)

`first_divergence` is the credit-assignment primitive: the index of the op that
broke the trajectory. A six-op plan that fails does not condemn ops 1-5.

Deterministic, stdlib-only, no model call.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

from harnesscad.agents.agent.tool_reward import ToolUseReward, aggregate_reward
from harnesscad.agents.agent.tool_schema import InterfaceResult, ToolCall
from harnesscad.agents.agent.tool_trajectory import ToolTrajectory, TrajectoryStep

__all__ = [
    "STEP_REWARD_EVENT",
    "trajectory_from_steps",
    "trajectory_from_trace",
    "first_divergence",
    "step_accuracy",
    "reward_for_steps",
    "reward_for_session",
]

STEP_REWARD_EVENT = "step_reward"


def trajectory_from_steps(step_rewards: Sequence[Dict[str, Any]],
                          completed: Optional[bool] = None) -> ToolTrajectory:
    """Lift the loop's per-op reward vector into a `ToolTrajectory`.

    Each record is ``{"index", "op", "reward", "reason"}`` as emitted by
    `HarnessSession._step_reward`. A trajectory is ``completed`` when every op it
    decided about succeeded (no op broke it).
    """
    steps: List[TrajectoryStep] = []
    for rec in step_rewards:
        ok = float(rec.get("reward", 0.0)) > 0.0
        steps.append(TrajectoryStep(
            think="",
            call=ToolCall(name=str(rec.get("op") or "unknown"), arguments={}),
            result=InterfaceResult(success=ok,
                                   description=str(rec.get("reason") or "")),
        ))
    done = all(s.succeeded for s in steps) if completed is None else bool(completed)
    return ToolTrajectory(steps=steps, completed=bool(steps) and done)


def trajectory_from_trace(events: Sequence[Dict[str, Any]],
                          run_id: Optional[str] = None) -> ToolTrajectory:
    """Rebuild a trajectory from a tracer's event stream (e.g. `InMemoryTracer`).

    Reads only `step_reward` events, in emission order, optionally filtered to one
    ``run_id`` — so a JSONL trace of a whole session can be sliced per batch.
    """
    recs = [e["data"] for e in events
            if e.get("kind") == STEP_REWARD_EVENT
            and (run_id is None or e.get("run_id") == run_id)]
    return trajectory_from_steps(recs)


def first_divergence(step_rewards: Sequence[Dict[str, Any]]) -> Optional[int]:
    """Index of the FIRST op that broke the trajectory, or None if none did.

    This is trajectory slicing: assign the negative reward to that op alone and
    leave the correct prefix intact. Grading the finished solid attributes
    nothing; this attributes exactly one thing.
    """
    for rec in step_rewards:
        if float(rec.get("reward", 0.0)) <= 0.0:
            return int(rec.get("index", 0))
    return None


def step_accuracy(step_rewards: Sequence[Dict[str, Any]]) -> float:
    """Fraction of ops that applied AND verified. The step-level eval metric.

    A loop whose step accuracy DROPS after typed feedback is a loop being
    poisoned, and that is visible without waiting for the final solid.
    """
    if not step_rewards:
        return 0.0
    good = sum(1 for r in step_rewards if float(r.get("reward", 0.0)) > 0.0)
    return good / len(step_rewards)


def reward_for_steps(step_rewards: Sequence[Dict[str, Any]],
                     *,
                     orm_verdict: bool,
                     format_text: str = "",
                     alpha: float = 1.0,
                     beta: float = 1.0,
                     gamma: float = 1.0) -> ToolUseReward:
    """Score a per-op vector with the full TOOLCAD aggregate reward."""
    traj = trajectory_from_steps(step_rewards)
    return aggregate_reward(traj, orm_verdict=orm_verdict, format_text=format_text,
                            alpha=alpha, beta=beta, gamma=gamma)


def reward_for_session(session: Any,
                       *,
                       orm_verdict: bool,
                       format_text: str = "",
                       alpha: float = 1.0,
                       beta: float = 1.0,
                       gamma: float = 1.0) -> ToolUseReward:
    """Score the LAST batch a `HarnessSession` applied.

    ``orm_verdict`` is the OUTCOME signal and must come from an oracle
    (`selftest.golden` / `selftest.differential`) or from a grader — never from
    the verifier fleet, whose false-positive rate is measured and non-zero.
    """
    return reward_for_steps(list(getattr(session, "step_rewards", []) or []),
                            orm_verdict=orm_verdict, format_text=format_text,
                            alpha=alpha, beta=beta, gamma=gamma)
