"""Parametric Mean Squared Error (P-MSE) loss for parametric primitive analysis.

Standard VLMs optimise a cross-entropy (CE) loss over token likelihoods, which
does not penalise *numerical* deviation between a predicted geometric parameter
and its ground truth -- fatal for fine-grained parametric primitive analysis.
A P-MSE loss is added over the numeric outputs of the four dedicated
regression heads and combined with CE by a weighted sum:

    L_CE    = - sum_i t_i * log(t_hat_i)
    L_P-MSE = (1/N) sum_i | f_theta_i(h_i) - p_i |^2
    L       = lambda_CE * L_CE  +  lambda_P-MSE * L_P-MSE

This module computes all three deterministically in pure Python. ``cross_entropy``
takes a ground-truth distribution and a predicted distribution; ``p_mse`` takes
regressed parameters and their targets; ``total_loss`` forms the weighted sum.
Ablation shows P-MSE lowers ParamMSE and ImgMSE; this is the
train-time objective behind that gain.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

_EPS = 1e-12


def cross_entropy(targets: list[float], predicted: list[float]) -> float:
    """CE loss ``-sum t_i log(p_hat_i)`` for one token's distributions (Eq. 5).

    ``targets`` is the ground-truth distribution (need not be one-hot but must be
    non-negative); ``predicted`` are model probabilities. Probabilities are
    floored at ``_EPS`` to avoid ``log(0)``.
    """
    if len(targets) != len(predicted):
        raise ValueError("targets and predicted must have equal length")
    if not targets:
        raise ValueError("empty distribution")
    total = 0.0
    for t, p in zip(targets, predicted):
        if t < 0:
            raise ValueError("target probabilities must be non-negative")
        total -= t * math.log(max(p, _EPS))
    return total


def p_mse(predicted: list[float], targets: list[float]) -> float:
    """Parametric MSE ``(1/N) sum |f(h_i) - p_i|^2`` (Eq. 6).

    Averaged over the N predicted numeric parameters.
    """
    if len(predicted) != len(targets):
        raise ValueError("predicted and targets must have equal length")
    if not predicted:
        raise ValueError("no parameters to score")
    return sum((f - p) ** 2 for f, p in zip(predicted, targets)) / len(predicted)


@dataclass(frozen=True)
class LossBreakdown:
    """The three loss terms and their weighted total (Eq. 7)."""

    ce: float
    p_mse: float
    lambda_ce: float
    lambda_p_mse: float
    total: float


def total_loss(ce: float, pmse: float,
               lambda_ce: float = 1.0, lambda_p_mse: float = 1.0) -> LossBreakdown:
    """Weighted sum ``lambda_CE * L_CE + lambda_P-MSE * L_P-MSE`` (Eq. 7)."""
    if lambda_ce < 0 or lambda_p_mse < 0:
        raise ValueError("loss weights must be non-negative")
    total = lambda_ce * ce + lambda_p_mse * pmse
    return LossBreakdown(ce, pmse, lambda_ce, lambda_p_mse, total)


def combined_loss(targets: list[float], predicted: list[float],
                  param_targets: list[float], param_pred: list[float],
                  lambda_ce: float = 1.0,
                  lambda_p_mse: float = 1.0) -> LossBreakdown:
    """Convenience: compute CE + P-MSE from raw inputs and combine them."""
    ce = cross_entropy(targets, predicted)
    pm = p_mse(param_pred, param_targets)
    return total_loss(ce, pm, lambda_ce, lambda_p_mse)
