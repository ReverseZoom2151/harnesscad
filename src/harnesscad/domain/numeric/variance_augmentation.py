"""Variance-schedule augmentation for Gaussian-Softmax diffusion (the model 3.4).

the model observes that a continuous variance schedule (e.g. the cosine schedule)
cannot be reused verbatim inside Gaussian-Softmax diffusion: the softmax
projection distorts the injected noise so that the *discrete* label switches far
too abruptly. This approach therefore augments the
schedule so that the argmax of ``y_t`` follows a controlled categorical marginal

    argmax(y_t) ~ C( b_t y_0 + (1 - b_t) / D )

for a chosen *discrete* schedule ``b_t``. Appendix A.4 derives the mapping via a
Gumbel-max approximation of the argmax of a Gaussian vector. With

    f(x) = log( (1 - x) / ((D - 1) x + 1) )

the augmented continuous retention coefficient is

    alpha_t = f(b_t)^2 / ( f(b_t)^2 + f(k)^2 )

where ``k`` is the label-smoothing constant (default 0.99). The inverse -- the
discrete keep-probability ``b_t`` implied by a given continuous ``alpha_t`` -- is
obtained by solving for ``f(b_t)`` and inverting ``f``:

    f(b_t) = -sqrt( alpha_t / (1 - alpha_t) ) * |f(k)|,
    b_t    = (1 - e^y) / ( (D - 1) e^y + 1 ),   y = f(b_t).

This module also provides the cosine schedule (Nichol & Dhariwal) used by the
paper and the categorical argmax-marginal target, all stdlib-only and
deterministic.
"""

from __future__ import annotations

import math
from typing import Sequence

Vector = list[float]


def cosine_alpha_bar(t: int, steps: int, s: float = 0.008) -> float:
    """Nichol & Dhariwal cosine cumulative retention ``abar_t`` (normalised)."""
    if steps < 1:
        raise ValueError("steps must be >= 1")
    if not 0 <= t <= steps:
        raise IndexError("t out of range")

    def f(u: int) -> float:
        return math.cos(((u / steps + s) / (1.0 + s)) * (math.pi / 2.0)) ** 2

    return f(t) / f(0)


def gumbel_f(x: float, num_classes: int) -> float:
    """``f(x) = log( (1-x) / ((D-1) x + 1) )`` from Appendix A.4.

    Defined for ``x`` in ``[0, 1)``; ``f(0) = 0`` and ``f -> -inf`` as ``x->1``.
    """
    if num_classes < 2:
        raise ValueError("num_classes must be >= 2")
    if not 0.0 <= x < 1.0:
        raise ValueError("x must be in [0, 1)")
    d = num_classes
    return math.log((1.0 - x) / ((d - 1) * x + 1.0))


def augmented_alpha(
    b_t: float, num_classes: int, k: float = 0.99
) -> float:
    """Continuous ``alpha_t`` realising a discrete keep-schedule ``b_t``.

    ``alpha_t = f(b_t)^2 / (f(b_t)^2 + f(k)^2)``. At ``b_t = 0`` (fully noised
    label) this returns 0; as ``b_t -> 1`` it approaches 1.
    """
    fb = gumbel_f(b_t, num_classes)
    fk = gumbel_f(k, num_classes)
    fb2 = fb * fb
    denom = fb2 + fk * fk
    if denom == 0.0:
        return 0.0
    return fb2 / denom


def inverse_gumbel_f(y: float, num_classes: int) -> float:
    """Invert ``f``: given ``y = f(x)`` recover ``x = (1 - e^y)/((D-1) e^y + 1)``."""
    if num_classes < 2:
        raise ValueError("num_classes must be >= 2")
    d = num_classes
    ey = math.exp(y)
    return (1.0 - ey) / ((d - 1) * ey + 1.0)


def implied_discrete_keep(
    alpha_t: float, num_classes: int, k: float = 0.99
) -> float:
    """Discrete keep-probability ``b_t`` implied by continuous ``alpha_t``.

    True inverse of :func:`augmented_alpha`: solves for ``f(b_t)`` (taking
    the ``f < 0`` branch, since ``f`` is negative on ``(0, 1)``) and inverts
    ``f``. ``alpha_t = 0`` maps to ``b_t = 0``; ``alpha_t -> 1`` maps to
    ``b_t -> 1``.
    """
    if not 0.0 <= alpha_t < 1.0:
        raise ValueError("alpha_t must be in [0, 1)")
    if alpha_t == 0.0:
        return 0.0
    fk = gumbel_f(k, num_classes)
    fb = -math.sqrt(alpha_t / (1.0 - alpha_t)) * abs(fk)
    return inverse_gumbel_f(fb, num_classes)


def argmax_marginal(y0: Sequence[float], b_t: float) -> Vector:
    """Target categorical marginal ``C(b_t y_0 + (1 - b_t)/D)`` for argmax(y_t).

    Interpolates the clean one-hot ``y_0`` with the uniform distribution by the
    discrete schedule ``b_t``.
    """
    if not 0.0 <= b_t <= 1.0:
        raise ValueError("b_t must be in [0, 1]")
    d = len(y0)
    if d == 0:
        raise ValueError("y0 must be non-empty")
    uni = (1.0 - b_t) / d
    return [b_t * v + uni for v in y0]


def augmented_schedule(
    discrete_keep: Sequence[float], num_classes: int, k: float = 0.99
) -> Vector:
    """Map a whole discrete keep-schedule ``b_1..b_T`` to continuous alphas."""
    return [augmented_alpha(b, num_classes, k) for b in discrete_keep]
