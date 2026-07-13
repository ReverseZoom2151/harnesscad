"""Export layer — one trace record, three training formats (sec.17) + the
flywheel metrics (sec.21).

The blueprint is explicit that the single ``(S_t, A_t, R_t, S_{t+1})`` record
"supports GRPO (group-normalize N verified trajectories -- the agentic default,
no critic), DPO (chosen=best trace / rejected=worst), and STaR/RFT (SFT on
verified successes)." This module turns a list of :class:`Trajectory` objects
into JSON-serialisable rows for each:

  * ``to_grpo``  — group-normalised advantages over the traces sharing a prompt.
  * ``to_dpo``   — a chosen (best) / rejected (worst) preference pair per prompt.
  * ``to_star``  — SFT rows from verified successes only.

Plus ``flywheel_metrics`` — the sec.21 data-flywheel dashboard, headlined by
**human-corrections-per-plan** (should fall over time), with success and
efficiency aggregates. And ``write_jsonl`` to persist any row list.

Absolute imports, stdlib only.
"""

from __future__ import annotations

import json
import math
from typing import Any, Dict, Iterable, List, Optional

from harnesscad.data.dataengine.trace.trajectory import Trajectory


# =====================================================================
# Serialisation helpers
# =====================================================================

def write_jsonl(path: str, rows: Iterable[dict], encoding: str = "utf-8") -> int:
    """Write ``rows`` as JSON Lines to ``path``. Returns the row count.

    One compact, sorted-key JSON object per line — the same convention as
    trace.JsonlTracer, so downstream tooling reads every artefact identically.
    """
    n = 0
    with open(path, "w", encoding=encoding) as fh:
        for row in rows:
            fh.write(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n")
            n += 1
    return n


def _messages(traj: Trajectory) -> List[dict]:
    """The action sequence as chat-style turns: an assistant reasoning/tool_call
    per step. This is the model-facing 'response' the trainers consume.
    """
    turns: List[dict] = []
    for s in traj.steps:
        turns.append({
            "role": "assistant",
            "reasoning": s.action.reasoning,
            "tool_call": s.action.tool_call,
            "reward": s.reward,
            "outcome": s.outcome,
        })
    return turns


def _completion(traj: Trajectory) -> List[dict]:
    """The verified action sequence (applied ops only) — the SFT target for STaR."""
    return [
        {"reasoning": s.action.reasoning, "tool_call": s.action.tool_call}
        for s in traj.steps if not s.divergent
    ]


def _prompt_key(traj: Trajectory) -> str:
    """Group key: the prompt text, or a stable placeholder when unlabelled."""
    return traj.prompt if traj.prompt is not None else "<unlabelled>"


def _group_by_prompt(trajectories: Iterable[Trajectory]) -> "Dict[str, List[Trajectory]]":
    groups: Dict[str, List[Trajectory]] = {}
    for t in trajectories:
        groups.setdefault(_prompt_key(t), []).append(t)
    return groups


# =====================================================================
# GRPO — group-relative advantages (no critic)
# =====================================================================

def to_grpo(trajectories: Iterable[Trajectory],
            reward: str = "total") -> List[dict]:
    """Group-normalise the traces sharing a prompt into GRPO rows.

    GRPO needs no value network: within each prompt group it standardises the
    scalar reward to an advantage ``A = (r - mean) / std`` (std falls back to 1.0
    for a degenerate group so a single/identical group yields zero advantage
    rather than a divide-by-zero). ``reward`` selects the scalar: ``"total"`` (the
    dense per-step return) or ``"final"`` (the terminal verifier verdict).

    Each row carries the prompt, the group id/size, the raw reward, the
    normalised advantage, the dense per-step reward vector, and the action
    sequence — everything a GRPO step consumes.
    """
    rows: List[dict] = []
    groups = _group_by_prompt(trajectories)
    for gi, (prompt, group) in enumerate(sorted(groups.items(), key=lambda kv: kv[0])):
        rewards = [_scalar(t, reward) for t in group]
        mean = sum(rewards) / len(rewards)
        if len(rewards) > 1:
            var = sum((r - mean) ** 2 for r in rewards) / len(rewards)
            std = math.sqrt(var)
        else:
            std = 0.0
        denom = std if std > 1e-12 else 1.0
        for t, r in zip(group, rewards):
            rows.append({
                "prompt": t.prompt,
                "plan": t.plan,
                "group_id": gi,
                "group_size": len(group),
                "reward": r,
                "reward_kind": reward,
                "group_mean": mean,
                "group_std": std,
                "advantage": (r - mean) / denom,
                "dense_rewards": t.dense_rewards(),
                "sub_goal_rewards": t.sub_goal_rewards(),
                "success": t.success,
                "run_ids": t.run_ids,
                "response": _messages(t),
            })
    return rows


def _scalar(traj: Trajectory, reward: str) -> float:
    if reward == "final":
        return traj.final_reward
    if reward == "total":
        return traj.total_reward()
    raise ValueError(f"unknown reward kind {reward!r}; expected 'total' or 'final'")


# =====================================================================
# DPO — best/worst preference pair per prompt
# =====================================================================

def to_dpo(trajectories: Iterable[Trajectory],
           reward: str = "total") -> List[dict]:
    """One preference pair per prompt group: chosen=best trace, rejected=worst.

    A pair is emitted only when the group holds at least two traces whose scalar
    rewards differ (a pair with chosen==rejected carries no preference signal and
    is dropped). ``chosen``'s reward is always strictly greater than
    ``rejected``'s.
    """
    rows: List[dict] = []
    groups = _group_by_prompt(trajectories)
    for prompt, group in sorted(groups.items(), key=lambda kv: kv[0]):
        if len(group) < 2:
            continue
        scored = sorted(group, key=lambda t: _scalar(t, reward))
        worst, best = scored[0], scored[-1]
        r_best, r_worst = _scalar(best, reward), _scalar(worst, reward)
        if r_best <= r_worst:
            continue  # no separable preference in this group
        rows.append({
            "prompt": best.prompt,
            "plan": best.plan,
            "reward_kind": reward,
            "chosen": {
                "reward": r_best,
                "success": best.success,
                "run_ids": best.run_ids,
                "response": _messages(best),
            },
            "rejected": {
                "reward": r_worst,
                "success": worst.success,
                "run_ids": worst.run_ids,
                "response": _messages(worst),
            },
        })
    return rows


# =====================================================================
# STaR / RFT — SFT on verified successes only
# =====================================================================

def to_star(trajectories: Iterable[Trajectory]) -> List[dict]:
    """SFT rows from verified successes only (STaR / rejection-sampling FT).

    Only trajectories whose terminal verifier verdict was a success are kept, and
    within each only the applied (verified) ops form the completion — never a
    rolled-back or rejected op. This is the highest-precision training signal the
    verifier can hand a model.
    """
    rows: List[dict] = []
    for t in trajectories:
        if not t.success:
            continue
        completion = _completion(t)
        if not completion:
            continue
        rows.append({
            "prompt": t.prompt,
            "plan": t.plan,
            "reward": t.final_reward,
            "n_ops": len(completion),
            "run_ids": t.run_ids,
            "completion": completion,
        })
    return rows


# =====================================================================
# Flywheel metrics — sec.21 dashboard (human-corrections-per-plan et al.)
# =====================================================================

def flywheel_metrics(trajectories: Iterable[Trajectory]) -> dict:
    """The data-flywheel dashboard (sec.21).

    Headline: **human-corrections-per-plan** — the mean number of corrections the
    agent/human had to make against a plan, plus a trend (first-half vs
    second-half mean, and ``falling`` = is it going down as the flywheel spins).
    Trajectories are treated as chronological in the order given. Also reports
    success rate and mean trajectory efficiency (applied ops / attempted ops).
    """
    trajs = list(trajectories)
    n = len(trajs)
    if n == 0:
        return {
            "n_trajectories": 0,
            "n_success": 0,
            "success_rate": 0.0,
            "corrections_per_plan": 0.0,
            "corrections_total": 0,
            "corrections_trend": {
                "first_half_mean": 0.0,
                "second_half_mean": 0.0,
                "delta": 0.0,
                "falling": False,
            },
            "mean_efficiency": 0.0,
            "per_plan_corrections": [],
        }

    corrections = [t.corrections() for t in trajs]
    total_corr = sum(corrections)
    n_success = sum(1 for t in trajs if t.success)

    # Trajectory efficiency: fraction of attempted ops that stuck (applied /
    # total steps). A clean trace = 1.0; every correction drags it down.
    effs: List[float] = []
    for t in trajs:
        if t.length == 0:
            continue
        applied = sum(1 for s in t.steps if not s.divergent)
        effs.append(applied / t.length)
    mean_eff = sum(effs) / len(effs) if effs else 0.0

    # Corrections trend: does the flywheel reduce corrections over time?
    half = n // 2
    first = corrections[:half] if half else corrections[: max(1, n)]
    second = corrections[half:] if half else corrections[: max(1, n)]
    first_mean = sum(first) / len(first) if first else 0.0
    second_mean = sum(second) / len(second) if second else 0.0

    return {
        "n_trajectories": n,
        "n_success": n_success,
        "success_rate": n_success / n,
        "corrections_per_plan": total_corr / n,
        "corrections_total": total_corr,
        "corrections_trend": {
            "first_half_mean": first_mean,
            "second_half_mean": second_mean,
            "delta": second_mean - first_mean,
            "falling": second_mean < first_mean,
        },
        "mean_efficiency": mean_eff,
        "per_plan_corrections": corrections,
    }
