"""TOOLCAD tool-use efficiency metrics — tools-per-task, redundancy, latency.

Beyond geometric success, TOOLCAD reports *tool-use* behaviour of the agent:
tool-calling accuracy (Fig. 4), and average token usage / latency per tool call
(Table 7, App. D.3), where efficient agents keep interaction cheap while still
completing the task. This module provides deterministic, stdlib-only metrics
over a :class:`agent.toolcad_trajectory.ToolTrajectory`:

  * **tools-per-task** — trajectory length (fewer tools for the same completed
    task == more efficient).

  * **success rate / failure rate** — fraction of tool calls the engine
    accepted, the step-level signal underpinning the paper's step-wise reward.

  * **redundant-call detection** — repeated identical tool calls, and no-op
    boolean self-operations (operating an object with itself), which waste turns
    and inflate the trajectory without advancing geometry. The paper motivates
    "Acting less is reasoning more" efficiency; this quantifies wasted calls.

  * **effective progress** — successful *object-producing* calls per total call
    (how much of the interaction actually advanced the geometric-object list).

  * **token/latency summary** — a length-weighted aggregate matching Table 7's
    per-tool-call token and latency comparison, so competing rollouts can be
    ranked on interaction cost.

These are tool-use metrics, orthogonal to the repository's geometry benchmarks
(``bench/metrics.py`` etc.) which score the *produced solid*.

Stdlib only, deterministic.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence, Tuple

from agent.toolcad_trajectory import ToolTrajectory


def _call_key(call) -> Tuple:
    """Hashable identity of a tool call (name + sorted argument items)."""
    items = tuple(
        (k, tuple(v) if isinstance(v, (list, tuple)) else v)
        for k, v in sorted(call.arguments.items())
    )
    return (call.name, items)


def tools_per_task(traj: ToolTrajectory) -> int:
    """Number of tool calls issued in the trajectory."""
    return len(traj)


def success_rate(traj: ToolTrajectory) -> float:
    """Fraction of tool calls the engine accepted; 0 for an empty trajectory."""
    if len(traj) == 0:
        return 0.0
    return traj.num_success / len(traj)


def count_redundant_calls(traj: ToolTrajectory) -> int:
    """Count wasted calls: exact-duplicate calls and no-op boolean self-ops.

    A duplicate is any tool call whose (name, arguments) identity has already
    appeared earlier in the trajectory. A no-op boolean self-op is a
    boolean_operation whose base and tool operand names are identical.
    """
    seen: set = set()
    redundant = 0
    for step in traj.steps:
        call = step.call
        key = _call_key(call)
        if key in seen:
            redundant += 1
        else:
            seen.add(key)
        if call.name == "boolean_operation":
            base = call.arguments.get("base_object_name")
            other = call.arguments.get("tool_object_name")
            if base is not None and base == other:
                redundant += 1
    return redundant


def effective_progress(traj: ToolTrajectory) -> float:
    """Fraction of calls that succeeded AND produced a new geometry object."""
    if len(traj) == 0:
        return 0.0
    produced = sum(
        1 for s in traj.steps if s.succeeded and s.result.produced_object
    )
    return produced / len(traj)


@dataclass(frozen=True)
class ToolUseMetrics:
    """Bundle of tool-use efficiency metrics for one trajectory."""

    tools_per_task: int
    success_rate: float
    redundant_calls: int
    effective_progress: float
    completed: bool

    @property
    def efficiency(self) -> float:
        """Completed tasks reward fewer, non-redundant, productive calls.

        0 for an incomplete task; otherwise success_rate scaled down by the
        share of redundant calls. In [0, 1].
        """
        if not self.completed or self.tools_per_task == 0:
            return 0.0
        redundancy_share = self.redundant_calls / self.tools_per_task
        return self.success_rate * (1.0 - min(1.0, redundancy_share))


def summarize(traj: ToolTrajectory) -> ToolUseMetrics:
    """Compute the tool-use metric bundle for a trajectory."""
    return ToolUseMetrics(
        tools_per_task=tools_per_task(traj),
        success_rate=success_rate(traj),
        redundant_calls=count_redundant_calls(traj),
        effective_progress=effective_progress(traj),
        completed=traj.completed,
    )


@dataclass(frozen=True)
class InteractionCost:
    """Table-7-style per-tool-call token/latency summary of a trajectory."""

    total_tokens: int
    total_latency_ms: float
    n_calls: int

    @property
    def avg_tokens_per_call(self) -> float:
        return self.total_tokens / self.n_calls if self.n_calls else 0.0

    @property
    def avg_latency_ms(self) -> float:
        return self.total_latency_ms / self.n_calls if self.n_calls else 0.0


def interaction_cost(
    traj: ToolTrajectory,
    tokens_per_call: Sequence[int],
    latency_per_call_ms: Sequence[float],
) -> InteractionCost:
    """Aggregate per-call token and latency measurements (Table 7)."""
    n = len(traj)
    if len(tokens_per_call) != n or len(latency_per_call_ms) != n:
        raise ValueError("per-call measurement length must match trajectory length")
    if any(t < 0 for t in tokens_per_call) or any(l < 0 for l in latency_per_call_ms):
        raise ValueError("token and latency measurements must be non-negative")
    return InteractionCost(
        total_tokens=int(sum(tokens_per_call)),
        total_latency_ms=float(sum(latency_per_call_ms)),
        n_calls=n,
    )
