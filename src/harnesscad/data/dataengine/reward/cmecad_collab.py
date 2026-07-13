"""CME-CAD Multi-Expert Collaborative Learning: best/worst expert credit
assignment and best-expert routing (CME-CAD, 2025).

To make heterogeneous experts *learn from one another* rather than merely
aggregate, CME-CAD (Sec. 3.2, "Multi-Expert Collaborative Learning") computes,
for each input ``I_i``, the average absolute reward ``r_bar_n`` of every expert
across its ``G`` responses, then identifies:

* the **best expert** ``E+`` = argmax_n r_bar_n, and
* the **worst expert** ``E-`` = argmin_n r_bar_n.

A directional KL penalty (Eq. 11) then forces the worse expert to imitate the
better one::

    L_KL = KL( pi_theta(A+ | P_{E-}, I) || pi_theta(A_correct | P_{E+}, I) )

This module builds the *deterministic* half of that mechanism -- the reward
aggregation, the E+/E- credit assignment, and the KL direction (which
distribution imitates which). It also implements the inference-time **best-expert
routing/gating**: at test time only the highest-reward expert is run, cutting the
cost of a full multi-expert ensemble.

The learned policy log-probs / the actual KL optimisation are external; here the
KL is computed on caller-supplied discrete distributions so the transfer
direction and magnitude can be unit-tested. Pure-stdlib, deterministic; ties
break by ascending expert id.
"""

from __future__ import annotations

import math


def average_absolute_reward(rewards) -> float:
    """r_bar_n = mean absolute reward of one expert over its G responses."""
    values = [abs(float(r)) for r in rewards]
    if not values:
        raise ValueError("rewards must be non-empty")
    return sum(values) / len(values)


def _expert_means(expert_rewards):
    if hasattr(expert_rewards, "items"):
        items = list(expert_rewards.items())
    else:
        items = list(expert_rewards)
    if not items:
        raise ValueError("expert_rewards must be non-empty")
    return [(eid, average_absolute_reward(rw)) for eid, rw in items]


def best_expert(expert_rewards):
    """E+ = expert with the highest average absolute reward. Ties -> lowest id."""
    means = _expert_means(expert_rewards)
    return min(means, key=lambda kv: (-kv[1], _sort_key(kv[0])))[0]


def worst_expert(expert_rewards):
    """E- = expert with the lowest average absolute reward. Ties -> lowest id."""
    means = _expert_means(expert_rewards)
    return min(means, key=lambda kv: (kv[1], _sort_key(kv[0])))[0]


def _sort_key(expert_id):
    # Stable, type-tolerant tie-break: (is_not_number, numeric_or_zero, str)
    try:
        return (0, float(expert_id), "")
    except (TypeError, ValueError):
        return (1, 0.0, str(expert_id))


def credit_assignment(expert_rewards) -> dict:
    """Assign collaborative credit for one input.

    Returns the best/worst expert ids, their mean rewards, the full ranking, and
    the KL transfer direction (``teacher -> student`` = ``E+ -> E-``).
    """
    means = dict(_expert_means(expert_rewards))
    ranking = sorted(means.items(), key=lambda kv: (-kv[1], _sort_key(kv[0])))
    e_plus = ranking[0][0]
    e_minus = ranking[-1][0]
    return {
        "best_expert": e_plus,
        "worst_expert": e_minus,
        "mean_rewards": means,
        "ranking": [eid for eid, _ in ranking],
        "kl_teacher": e_plus,
        "kl_student": e_minus,
    }


def route_best_expert(expert_rewards):
    """Inference-time gating: pick the single best-performing expert to run."""
    return best_expert(expert_rewards)


def kl_divergence(p, q) -> float:
    """Discrete KL( p || q ) = sum_x p(x) log(p(x)/q(x)) in nats.

    ``p`` and ``q`` are aligned probability sequences over the same support.
    Terms with ``p(x) == 0`` contribute 0; ``q(x) == 0`` where ``p(x) > 0`` is an
    infinite divergence and raises.
    """
    p = [float(x) for x in p]
    q = [float(x) for x in q]
    if len(p) != len(q):
        raise ValueError("p and q must have equal length")
    if any(x < 0.0 for x in p) or any(x < 0.0 for x in q):
        raise ValueError("probabilities must be non-negative")
    total = 0.0
    for pi, qi in zip(p, q):
        if pi == 0.0:
            continue
        if qi == 0.0:
            raise ValueError("q assigns zero probability where p is positive")
        total += pi * math.log(pi / qi)
    return total


def collaborative_kl_penalty(student_dist, teacher_dist) -> float:
    """L_KL forcing the worst expert (``student``) toward the best (``teacher``).

    Equation 11: ``KL( pi(A+|P_{E-}) || pi(A_correct|P_{E+}) )``. The student is
    the first argument (the distribution being pushed), the teacher the second.
    """
    return kl_divergence(student_dist, teacher_dist)
