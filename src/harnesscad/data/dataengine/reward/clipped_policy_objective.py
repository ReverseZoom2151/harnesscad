"""Dr. CPPO advantage estimation and clipped policy objective (cadrille, 2026).

cadrille combines two GRPO variants for its online RL stage:

* **Dr. GRPO** (Liu et al., 2025b): drops the reference model and the standard
  deviation normalisation of GRPO. For a group of ``G`` sampled sequences with
  rewards ``r_g``, the advantage is simply ``A_g = r_g - mean({r_i})`` (no
  division by std, so the reward scale is preserved).
* **CPPO** (Lin et al., 2025): keeps only the ``N`` samples with the strongest
  learning signal, i.e. the largest ``|A_g|``, and forms the update batch from
  them.

The retained samples update the policy by maximising the clipped PPO surrogate
``min(rho * A, clip(rho, 1-eps, 1+eps) * A)`` where ``rho`` is the importance
ratio ``pi_theta / pi_old``.

This differs from the repository's generic ``dataengine.export.to_grpo``, which
standardises advantages by the group std (GRPO) rather than using the raw,
std-free Dr. GRPO advantage plus CPPO top-|A| selection and a PPO clip. Numeric
only — no autograd, no model; deterministic.
"""

from __future__ import annotations

DEFAULT_EPSILON = 0.1
DEFAULT_GROUP_SIZE = 16
DEFAULT_CPPO_SAMPLES = 4


def advantages(rewards):
    """Dr. GRPO advantages: ``A_g = r_g - mean(r)`` (no std normalisation)."""
    values = [float(r) for r in rewards]
    if not values:
        raise ValueError("rewards must be non-empty")
    mean = sum(values) / len(values)
    return [v - mean for v in values]


def select_strongest(advs, n: int):
    """CPPO selection: indices of the ``n`` samples with the largest ``|A|``.

    Ties break by ascending index so the choice is deterministic. Returns the
    indices sorted ascending (batch order preserved).
    """
    if n <= 0:
        raise ValueError("n must be positive")
    count = min(n, len(advs))
    order = sorted(range(len(advs)), key=lambda i: (-abs(advs[i]), i))
    return sorted(order[:count])


def clip(value: float, low: float, high: float) -> float:
    return low if value < low else high if value > high else value


def ppo_clip_objective(ratio: float, advantage: float,
                       epsilon: float = DEFAULT_EPSILON) -> float:
    """Per-sample clipped PPO surrogate ``min(rho*A, clip(rho)*A)``."""
    rho = float(ratio)
    adv = float(advantage)
    unclipped = rho * adv
    clipped = clip(rho, 1.0 - epsilon, 1.0 + epsilon) * adv
    return min(unclipped, clipped)


def drcppo_step(rewards, ratios, n: int = DEFAULT_CPPO_SAMPLES,
                epsilon: float = DEFAULT_EPSILON) -> dict:
    """Run one Dr. CPPO batch computation over a group of samples.

    ``rewards`` and ``ratios`` are aligned per-sample sequences (``ratios`` are
    the importance weights ``pi_theta/pi_old``). Computes std-free advantages,
    selects the top-``n`` by ``|A|``, and returns the mean clipped surrogate over
    the selected batch together with the selection metadata.
    """
    rewards = list(rewards)
    ratios = list(ratios)
    if len(rewards) != len(ratios):
        raise ValueError("rewards and ratios must have equal length")
    advs = advantages(rewards)
    selected = select_strongest(advs, n)
    terms = [ppo_clip_objective(ratios[i], advs[i], epsilon) for i in selected]
    surrogate = sum(terms) / len(terms) if terms else 0.0
    return {
        "advantages": advs,
        "selected": selected,
        "objective_terms": terms,
        "surrogate": surrogate,
        "epsilon": float(epsilon),
    }
