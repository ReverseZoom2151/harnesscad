"""CME-CAD Expert-Internal Advantage Estimation (CME-CAD, 2025).

The MERL stage samples ``G`` responses *per heterogeneous expert* (each expert
is a fixed system prompt ``P_n``), giving ``N x G`` responses total. Rather than
pooling all ``N x G`` rewards into a single GRPO group, CME-CAD estimates the
group-relative advantage **within each expert's own group** (Eq. 9)::

    A_ng = R_ng - (1/G) * sum_{g'} R_ng'

and applies a **non-negative truncation** (Eq. 10)::

    A~_ng = max(A_ng, 0)

so that small negative advantages -- caused by minor process errors on a hard
code-generation task -- do not immediately discourage exploration. The GRPO
surrogate then weights ``log pi_theta`` by ``A~_ng``.

Contrast with ``dataengine.cadrille_drcppo`` (a single std-free group + CPPO
top-|A| selection + PPO clip) and ``dataengine.export`` GRPO (std-normalised,
single group): here the baseline is *per-expert* and the advantage is *floored
at zero*, never clipped by importance ratio. Pure-stdlib, deterministic.
"""

from __future__ import annotations


def group_baseline(rewards) -> float:
    """Mean reward (1/G * sum) over one expert's group of G responses."""
    values = [float(r) for r in rewards]
    if not values:
        raise ValueError("rewards must be non-empty")
    return sum(values) / len(values)


def expert_advantages(rewards):
    """Within-expert relative advantages ``A_ng = R_ng - mean_g R`` (Eq. 9)."""
    values = [float(r) for r in rewards]
    if not values:
        raise ValueError("rewards must be non-empty")
    base = sum(values) / len(values)
    return [v - base for v in values]


def truncate_nonneg(advs):
    """Non-negative truncation ``max(A, 0)`` applied element-wise (Eq. 10)."""
    return [a if a > 0.0 else 0.0 for a in advs]


def expert_advantages_truncated(rewards):
    """Convenience: per-expert advantages already floored at zero."""
    return truncate_nonneg(expert_advantages(rewards))


def estimate(expert_rewards):
    """Estimate per-expert advantages for all experts.

    ``expert_rewards`` maps ``expert_id -> [R_n1, ..., R_nG]`` (any mapping or an
    iterable of ``(expert_id, rewards)`` pairs). Returns a dict per expert with
    the raw advantages, the non-negatively truncated advantages, and the group
    baseline.
    """
    if hasattr(expert_rewards, "items"):
        items = list(expert_rewards.items())
    else:
        items = list(expert_rewards)
    out = {}
    for expert_id, rewards in items:
        raw = expert_advantages(rewards)
        out[expert_id] = {
            "baseline": group_baseline(rewards),
            "advantages": raw,
            "truncated": truncate_nonneg(raw),
        }
    return out


def grpo_surrogate_term(log_prob: float, advantage: float) -> float:
    """Single GRPO term ``log pi_theta * max(A, 0)`` (Eq. 10, per response).

    The truncation is applied here so callers can pass a raw advantage.
    """
    return float(log_prob) * max(float(advantage), 0.0)


def expert_grpo_loss(log_probs, rewards):
    """Mean GRPO loss ``L = -E[ log pi * max(A, 0) ]`` for one expert's group.

    ``log_probs`` and ``rewards`` are aligned per-response sequences. Returns the
    negated mean surrogate (a loss to minimise).
    """
    log_probs = [float(x) for x in log_probs]
    rewards = list(rewards)
    if len(log_probs) != len(rewards):
        raise ValueError("log_probs and rewards must have equal length")
    advs = expert_advantages(rewards)
    terms = [grpo_surrogate_term(lp, a) for lp, a in zip(log_probs, advs)]
    if not terms:
        return 0.0
    return -sum(terms) / len(terms)
