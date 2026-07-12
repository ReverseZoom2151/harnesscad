"""TOOLCAD tool-use reward — format + step-wise execution + tool-selection.

TOOLCAD's post-training reward (App. B.3, Eq. 8) is a weighted combination of
signals defined *over the tool-using trajectory*, distinct from the repository's
many geometry rewards (cadrille/cmecad/intent2exec/recad, which score IoU /
chamfer / executability of a produced solid). The pieces this module implements:

  * **Format reward** (App. B.3): 0.5 if the trajectory's tags are all present
    and follow ``<think> -> <tool_call> -> <tool_response>`` in the correct
    order, else 0.

  * **Step-wise execution reward** (Eq. 8, "Step-wise Execution Reward"): a
    binary 0/1 per step, 1 only when the CAD engine returns "Success" for the
    primitive execution, averaged over the ``T`` steps to keep the scale
    trajectory-length-independent.

  * **Outcome reward** (Eq. 8, "Outcome Reward"): the ORM's terminal 0/1 verdict
    (injected — the ORM is an external model), gated to require a completed
    trajectory.

  * **Aggregate reward** ``R = alpha*R_ORM + beta*mean(R_step) + gamma*R_format``.

Additionally, because TOOLCAD is fundamentally about *correct tool use*, this
module scores **tool selection** and **argument correctness** against a
reference (gold) tool-call sequence — the component that makes "tool use"
learnable and is absent from every existing geometry reward here. This is a
deterministic, reference-based signal (no model call): it measures whether the
agent picked the right tools with the right typed arguments, which the paper's
"tool-calling accuracy" (Fig. 4) captures qualitatively.

Stdlib only, deterministic.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence, Tuple

from agent.toolcad_tool_schema import ToolCall
from agent.toolcad_trajectory import ToolTrajectory, check_format_order


FORMAT_REWARD_VALUE = 0.5


def format_reward(text: str) -> float:
    """0.5 if the tagged transcript has correct tag order/completeness, else 0."""
    return FORMAT_REWARD_VALUE if check_format_order(text) else 0.0


def step_execution_rewards(traj: ToolTrajectory) -> Tuple[float, ...]:
    """Per-step binary execution rewards (1.0 for engine 'Success', else 0)."""
    return tuple(1.0 if s.succeeded else 0.0 for s in traj.steps)


def mean_step_reward(traj: ToolTrajectory) -> float:
    """Mean step-wise execution reward; 0 for an empty trajectory."""
    rewards = step_execution_rewards(traj)
    if not rewards:
        return 0.0
    return sum(rewards) / len(rewards)


def outcome_reward(orm_verdict: bool, traj: ToolTrajectory) -> float:
    """ORM terminal reward: 1 only if the ORM says success AND traj completed."""
    return 1.0 if (orm_verdict and traj.completed) else 0.0


@dataclass(frozen=True)
class ToolUseReward:
    """The decomposed reward terms plus the aggregate scalar."""

    outcome: float
    step_mean: float
    fmt: float
    total: float


def aggregate_reward(
    traj: ToolTrajectory,
    *,
    orm_verdict: bool,
    format_text: str,
    alpha: float = 1.0,
    beta: float = 1.0,
    gamma: float = 1.0,
) -> ToolUseReward:
    """R = alpha*R_ORM + beta*mean(R_step) + gamma*R_format (Eq. 8)."""
    if min(alpha, beta, gamma) < 0:
        raise ValueError("reward weights must be non-negative")
    r_out = outcome_reward(orm_verdict, traj)
    r_step = mean_step_reward(traj)
    r_fmt = format_reward(format_text)
    total = alpha * r_out + beta * r_step + gamma * r_fmt
    return ToolUseReward(outcome=r_out, step_mean=r_step, fmt=r_fmt, total=total)


# --- Reference-based tool-selection / argument-correctness -----------------

def _args_match(pred: Mapping[str, object], gold: Mapping[str, object]) -> bool:
    """Exact match on the gold argument keys/values (extra pred keys ignored)."""
    for key, value in gold.items():
        if key not in pred:
            return False
        pv, gv = pred[key], value
        # Normalise list/tuple so [0,0,0] == (0,0,0).
        if isinstance(gv, (list, tuple)) and isinstance(pv, (list, tuple)):
            if list(pv) != list(gv):
                return False
        elif pv != gv:
            return False
    return True


@dataclass(frozen=True)
class ToolSelectionScore:
    """Position-wise scoring of a predicted vs. gold tool-call sequence."""

    name_matches: int
    arg_matches: int  # among name matches, how many also have correct args
    n_pred: int
    n_gold: int

    @property
    def selection_accuracy(self) -> float:
        """Fraction of gold positions whose tool name is correct."""
        return self.name_matches / self.n_gold if self.n_gold else 0.0

    @property
    def argument_accuracy(self) -> float:
        """Fraction of gold positions with correct name AND arguments."""
        return self.arg_matches / self.n_gold if self.n_gold else 0.0

    @property
    def length_penalty(self) -> float:
        """1.0 if lengths match, decaying with over/under-generation."""
        if self.n_gold == 0:
            return 1.0
        extra = abs(self.n_pred - self.n_gold)
        return 1.0 / (1.0 + extra)

    @property
    def reward(self) -> float:
        """Combined tool-use reward in [0, 1]: argument accuracy * length_penalty."""
        return self.argument_accuracy * self.length_penalty


def score_tool_selection(
    predicted: Sequence[ToolCall],
    gold: Sequence[ToolCall],
) -> ToolSelectionScore:
    """Score predicted tool calls against a reference sequence, position-wise."""
    name_matches = 0
    arg_matches = 0
    for pred, ref in zip(predicted, gold):
        if pred.name == ref.name:
            name_matches += 1
            if _args_match(pred.arguments, ref.arguments):
                arg_matches += 1
    return ToolSelectionScore(
        name_matches=name_matches,
        arg_matches=arg_matches,
        n_pred=len(predicted),
        n_gold=len(gold),
    )
