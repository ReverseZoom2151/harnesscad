"""Overlong Filtering -- exclude truncated sequences from RL loss (CAD-RL, 2026).

Autoregressive RL sets a maximum generation length ``T_max``. The standard
treatment of sequences that hit this limit is to truncate and *penalise* them
(zero reward or a harsh penalty). CAD-RL argues this injects noise into the
reward signal, because in CAD modeling output verbosity does not always imply a
functional error -- a long-but-correct program should not be punished merely for
length.

**Overlong Filtering** (Eq. 9) instead *removes* truncated sequences from the
reward computation entirely::

    L_RL = - E_{s ~ S \\ S_trunc}[ R(s) * log pi_theta(s) ]

where ``S_trunc`` is the set of truncated sequences in the batch. This module
implements the truncation test (length >= ``T_max``) and the batch filter, plus
the resulting masked objective magnitude.

This is DISTINCT from ``dataengine.cadrille_reward``'s hard-example *mining*
(which keeps only low-reward samples to speed convergence): overlong filtering
*discards* over-length samples regardless of reward, to reduce gradient noise.
Pure stdlib, deterministic.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence


@dataclass(frozen=True)
class Sequence_(object):
    """A sampled sequence with its token length, reward, and log-prob.

    ``length`` is the number of generated tokens; ``reward`` is ``R(s)``;
    ``log_prob`` is ``log pi_theta(s)`` (summed over the sequence).
    """

    length: int
    reward: float
    log_prob: float


def is_truncated(length: int, t_max: int) -> bool:
    """True if a sequence of ``length`` tokens reached / exceeded ``T_max``."""
    if t_max <= 0:
        raise ValueError("t_max must be positive")
    return int(length) >= int(t_max)


def filter_overlong(sequences: Sequence[Sequence_], t_max: int) -> list:
    """Return the subset ``S \\ S_trunc`` -- sequences strictly under ``T_max``."""
    return [s for s in sequences if not is_truncated(s.length, t_max)]


def truncated_indices(sequences: Sequence[Sequence_], t_max: int) -> list:
    """Indices of the truncated sequences ``S_trunc`` (excluded from the loss)."""
    return [i for i, s in enumerate(sequences)
            if is_truncated(s.length, t_max)]


def rl_objective(sequences: Sequence[Sequence_], t_max: int) -> float:
    """Masked RL objective magnitude ``E_{s in S\\S_trunc}[R(s)*log pi(s)]`` (Eq. 9).

    Returns the mean of ``reward * log_prob`` over the *retained* sequences
    (the quantity whose negation is minimised). An all-truncated batch
    contributes nothing and returns 0.0.
    """
    kept = filter_overlong(sequences, t_max)
    if not kept:
        return 0.0
    return sum(s.reward * s.log_prob for s in kept) / len(kept)
