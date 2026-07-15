"""Soft target distribution for CAD parameter prediction (Drawing2CAD, Qin et
al., MM '25, "Sequence-to-Sequence Learning for CAD Generation from Vector
Drawings").

Drawing2CAD replaces the hard one-hot Parameter Loss with a *soft target
distribution* (Section 4.5, Eqs. 6-7) that tolerates minor parameter deviations
while preserving design intent. For a ground-truth quantised category ``t`` over
``num_classes`` discrete levels, the smoothed weight on category ``k`` is::

    w~_k = (beta - |k - t|) / Z      for k in [t - delta, t + delta]
    w~_k = 0                          otherwise

normalised so the weights sum to 1 (``Z`` is the normaliser). The paper sets
``beta = 2.0`` and ``delta = 3``. The Parameter Loss is then the cross-entropy of
the predicted distribution against this soft target rather than the one-hot
target, relaxing over-strict penalties for near-miss predictions.

This module builds the soft target and the soft cross-entropy deterministically
(stdlib only). It complements ``sequence/parameter_accuracy.py`` (which scores a
prediction with a tolerance threshold): here we produce the *training target
distribution* that motivates that same tolerance.
"""

from __future__ import annotations

from math import log

DEFAULT_BETA = 2.0
DEFAULT_DELTA = 3


def soft_target(target: int, num_classes: int, delta: int = DEFAULT_DELTA,
                beta: float = DEFAULT_BETA):
    """Triangular soft target distribution over ``num_classes`` categories.

    Weights fall off linearly as ``beta - |k - target|`` within the window
    ``[target - delta, target + delta]`` (clipped to valid categories), are
    clamped at zero, and are normalised to sum to 1.

    Requires ``0 <= target < num_classes``, ``num_classes >= 1``, ``delta >= 0``
    and ``beta > 0``. Raises otherwise. The window is clipped at the array
    bounds, so targets near an edge still yield a valid (re-normalised)
    distribution.
    """
    if num_classes < 1:
        raise ValueError("num_classes must be >= 1")
    if not 0 <= target < num_classes:
        raise ValueError("target out of range")
    if delta < 0:
        raise ValueError("delta must be >= 0")
    if beta <= 0:
        raise ValueError("beta must be > 0")
    weights = [0.0] * num_classes
    lo = max(0, target - delta)
    hi = min(num_classes - 1, target + delta)
    for k in range(lo, hi + 1):
        w = beta - abs(k - target)
        if w > 0:
            weights[k] = w
    total = sum(weights)
    if total <= 0:  # degenerate: beta <= 0 handled above, but guard anyway
        weights[target] = 1.0
        return weights
    return [w / total for w in weights]


def soft_cross_entropy(pred_probs, target: int, delta: int = DEFAULT_DELTA,
                       beta: float = DEFAULT_BETA, eps: float = 1e-12):
    """Cross-entropy of a predicted distribution against the soft target.

    ``pred_probs`` is a predicted probability distribution over the categories
    (need not be exactly normalised; it is renormalised defensively). Returns
    ``-sum_k w~_k log p_k`` with the soft target ``w~`` from :func:`soft_target`.
    Only categories with non-zero soft weight contribute, matching the paper's
    windowed penalisation.
    """
    n = len(pred_probs)
    tgt = soft_target(target, n, delta, beta)
    s = sum(pred_probs)
    if s <= 0:
        raise ValueError("pred_probs must have positive mass")
    loss = 0.0
    for k in range(n):
        if tgt[k] > 0:
            p = max(pred_probs[k] / s, eps)
            loss -= tgt[k] * log(p)
    return loss
