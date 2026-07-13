"""Trust Region Stretch (TRS) -- asymmetric PPO clip surrogate (CAD-RL, 2026).

Standard PPO / GRPO constrains the policy update with a *symmetric* clip
``clip(r_t, 1 - eps, 1 + eps)`` to keep the new policy close to the reference.
In CAD code generation, where many valid reasoning trajectories exist for a
single design, this symmetric constraint prematurely collapses exploration and
the model converges to narrow, near-deterministic policies (mode collapse).

CAD-RL's **Trust Region Stretch** (Eq. 7) *relaxes* and *asymmetrically widens*
the clip bounds to allow larger updates and encourage trajectory diversity::

    L_TRS = E_t[ min( r_t * A_t, clip(r_t, eps_low, eps_high) * A_t ) ]

with, e.g., ``eps_low = 0.6`` and ``eps_high = 1.8`` (versus PPO's symmetric
``0.8 / 1.2`` for eps=0.2). Note the bounds are absolute ratio bounds, not the
``1 +/- eps`` form, and ``eps_low < 1 < eps_high``.

This differs from the repository's existing PPO-style surrogates:

  * ``dataengine.cadrille_drcppo`` -- standard *symmetric* PPO clip plus CPPO
    top-|A| token selection.
  * ``dataengine.export`` GRPO -- symmetric clip with std-normalised advantage.

Here the objective is the *asymmetric relaxed-bound* surrogate itself. The
probability ratio ``r_t = pi_theta / pi_ref`` and advantages ``A_t`` are
injected (they come from the policy/optimiser); this module only computes the
deterministic surrogate. Pure stdlib, deterministic.
"""

from __future__ import annotations

from typing import Sequence

DEFAULT_EPS_LOW = 0.6
DEFAULT_EPS_HIGH = 1.8


def clip_ratio(ratio: float, eps_low: float = DEFAULT_EPS_LOW,
               eps_high: float = DEFAULT_EPS_HIGH) -> float:
    """Clamp ``ratio`` to the relaxed absolute bounds ``[eps_low, eps_high]``."""
    if not (0.0 < eps_low < 1.0 < eps_high):
        raise ValueError("require 0 < eps_low < 1 < eps_high")
    r = float(ratio)
    if r < eps_low:
        return eps_low
    if r > eps_high:
        return eps_high
    return r


def trs_token_objective(ratio: float, advantage: float,
                        eps_low: float = DEFAULT_EPS_LOW,
                        eps_high: float = DEFAULT_EPS_HIGH) -> float:
    """Per-token TRS surrogate ``min(r*A, clip(r)*A)`` (Eq. 7)."""
    r = float(ratio)
    a = float(advantage)
    unclipped = r * a
    clipped = clip_ratio(r, eps_low, eps_high) * a
    return min(unclipped, clipped)


def trs_objective(ratios: Sequence[float], advantages: Sequence[float],
                  eps_low: float = DEFAULT_EPS_LOW,
                  eps_high: float = DEFAULT_EPS_HIGH) -> float:
    """Mean TRS surrogate over a batch of ``(ratio, advantage)`` tokens.

    This is the quantity a maximiser ascends; it equals the average of the
    per-token surrogates. Raises if the sequences differ in length or are empty.
    """
    ratios = list(ratios)
    advantages = list(advantages)
    if len(ratios) != len(advantages):
        raise ValueError("ratios and advantages must have equal length")
    if not ratios:
        raise ValueError("cannot compute objective over an empty batch")
    total = sum(
        trs_token_objective(r, a, eps_low, eps_high)
        for r, a in zip(ratios, advantages)
    )
    return total / len(ratios)


def is_clipped(ratio: float, eps_low: float = DEFAULT_EPS_LOW,
               eps_high: float = DEFAULT_EPS_HIGH) -> bool:
    """True if ``ratio`` lies outside the relaxed trust region (would be clipped)."""
    r = float(ratio)
    return r < eps_low or r > eps_high
