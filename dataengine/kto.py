"""Numerically stable prospect-theory utilities for binary preference rows."""

from __future__ import annotations

import math


def _sigmoid(value):
    if value >= 0:
        z = math.exp(-value)
        return 1/(1+z)
    z = math.exp(value)
    return z/(1+z)


def implied_reward(policy_logprob, reference_logprob):
    return float(policy_logprob) - float(reference_logprob)


def kto_utility(reward, reference_point, desirable, *, beta=1.0,
                desirable_weight=1.0, undesirable_weight=1.0):
    if beta <= 0 or desirable_weight < 0 or undesirable_weight < 0:
        raise ValueError("invalid KTO weights")
    if desirable:
        return desirable_weight * _sigmoid(beta*(reward-reference_point))
    return undesirable_weight * _sigmoid(beta*(reference_point-reward))


def kto_row(preference, *, policy_logprob, reference_logprob, reference_point,
            beta=1.0, desirable_weight=1.0, undesirable_weight=1.0):
    reward = implied_reward(policy_logprob, reference_logprob)
    return {
        "prompt": preference.prompt, "candidate": preference.candidate,
        "desirable": preference.desirable, "implied_reward": reward,
        "reference_point": reference_point,
        "utility": kto_utility(reward, reference_point, preference.desirable,
                               beta=beta, desirable_weight=desirable_weight,
                               undesirable_weight=undesirable_weight),
        "candidate_digest": preference.candidate_digest,
    }
