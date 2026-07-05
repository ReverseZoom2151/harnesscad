"""Scaled gradient-variance consistency metric (CFD, Appx. F, Eq. 15).

CFD reports that a more *consistent* flow yields lower gradient variance during
optimization, and measures it with a scaled statistic built from the Adam-style
exponential-moving-average moments:

    m_hat ~ E[g]              (first moment)
    v_hat ~ E[g^2]           (second moment)
    sigma = sqrt(sum(v_hat - m_hat^2)) / sqrt(sum(v_hat))
          = sqrt(sum(Var(g))) / sqrt(sum(E[g^2]))

This is a purely deterministic diagnostic: given a stream of per-step gradient
vectors it returns a scalar in [0, 1] where 0 means a perfectly steady gradient
direction/magnitude and values near 1 mean the gradient is dominated by
zero-mean noise. It is directly transferable to any iterative CAD-optimization
harness (shape fitting, constraint solving, distillation-style refinement) as a
convergence / update-consistency signal, independent of the generative model.

Stdlib-only, deterministic.
"""

from __future__ import annotations

import math
from typing import List, Sequence


def ema_moments(
    grads: Sequence[Sequence[float]],
    beta1: float = 0.9,
    beta2: float = 0.999,
    bias_correction: bool = True,
) -> tuple:
    """Return (m_hat, v_hat): Adam-style EMA of g and g^2 over the stream.

    ``grads`` is a sequence of equal-length gradient vectors, one per step.
    """
    if not grads:
        raise ValueError("need at least one gradient")
    dim = len(grads[0])
    if dim == 0:
        raise ValueError("gradients must be non-empty vectors")
    m = [0.0] * dim
    v = [0.0] * dim
    for step, g in enumerate(grads, start=1):
        if len(g) != dim:
            raise ValueError("all gradients must share the same dimension")
        for i in range(dim):
            m[i] = beta1 * m[i] + (1.0 - beta1) * g[i]
            v[i] = beta2 * v[i] + (1.0 - beta2) * g[i] * g[i]
    if bias_correction:
        n = len(grads)
        c1 = 1.0 - beta1 ** n
        c2 = 1.0 - beta2 ** n
        m_hat = [mi / c1 for mi in m]
        v_hat = [vi / c2 for vi in v]
    else:
        m_hat = list(m)
        v_hat = list(v)
    return m_hat, v_hat


def scaled_gradient_variance(
    grads: Sequence[Sequence[float]],
    beta1: float = 0.9,
    beta2: float = 0.999,
    bias_correction: bool = True,
    eps: float = 1e-12,
) -> float:
    """Compute sigma of Eq. 15 from a stream of gradient vectors.

    Returns a scalar in [0, 1]: sqrt(sum(v_hat - m_hat^2)) / sqrt(sum(v_hat)).
    """
    m_hat, v_hat = ema_moments(grads, beta1, beta2, bias_correction)
    num = 0.0
    den = 0.0
    for mi, vi in zip(m_hat, v_hat):
        var_i = vi - mi * mi
        if var_i < 0.0:  # guard tiny negative from float round-off
            var_i = 0.0
        num += var_i
        den += vi
    if den <= eps:
        return 0.0
    return math.sqrt(num) / math.sqrt(den)


def scaled_gradient_variance_direct(grads: Sequence[Sequence[float]], eps: float = 1e-12) -> float:
    """Same statistic computed from exact (unweighted) sample moments.

    Useful as a reference/ground-truth for the EMA estimator: uses the arithmetic
    mean of g and g^2 across all steps instead of an exponential average.
    """
    if not grads:
        raise ValueError("need at least one gradient")
    dim = len(grads[0])
    n = len(grads)
    means: List[float] = [0.0] * dim
    sq: List[float] = [0.0] * dim
    for g in grads:
        if len(g) != dim:
            raise ValueError("all gradients must share the same dimension")
        for i in range(dim):
            means[i] += g[i]
            sq[i] += g[i] * g[i]
    means = [mi / n for mi in means]
    sq = [si / n for si in sq]
    num = 0.0
    den = 0.0
    for mi, si in zip(means, sq):
        var_i = si - mi * mi
        if var_i < 0.0:
            var_i = 0.0
        num += var_i
        den += si
    if den <= eps:
        return 0.0
    return math.sqrt(num) / math.sqrt(den)
