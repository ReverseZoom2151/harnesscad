"""VQ-Diffusion "mask-and-replace" hybrid transition for VQ-CAD (Sec. 3.2.1, Eq. 7).

VQ-CAD (Wang et al., CAGD 2024) runs a discrete diffusion over the VQ code tree
using the *mask-and-replace* strategy of VQ-Diffusion (Gu et al., 2022). This is a
genuinely distinct transition from the two standard discrete-diffusion families
already in the repo (``numeric.sketchdnn_categorical_diffusion`` supplies the pure
*uniform*/Multinomial matrix and the pure *absorbing*/[MASK] matrix). Mask-and-replace
is a **hybrid** of the two, applied over ``K`` real code categories plus one extra
``[MASK]`` category (index ``K``). For a real token ``i`` the forward transition is:

* keep the token with probability ``alpha_t + beta_t``;
* *replace* it by any of the other ``K-1`` real categories, each with probability
  ``beta_t`` (the uniform-diffusion part);
* *mask* it (jump to ``[MASK]``) with probability ``gamma_t`` (the absorbing part).

The ``[MASK]`` state is absorbing (row = identity). This is exactly Eq. 7 of the
paper written row-stochastically (row ``i`` = ``q(x_t | x_{t-1} = i)``), matching the
row-vector convention of ``sketchdnn_categorical_diffusion``. The three rates obey
the row-sum constraint

    alpha_t + K * beta_t + gamma_t = 1,

so given a per-step ``(alpha_t, gamma_t)`` schedule, ``beta_t = (1 - alpha_t -
gamma_t) / K`` is fixed.

The key deterministic content of VQ-Diffusion is its **closed-form cumulative**
schedule: the product ``Qbar_t = Q_1 ... Q_t`` keeps the same mask-and-replace form
with cumulative rates

    alpha_bar_t = prod_{s<=t} alpha_s,
    gamma_bar_t = 1 - prod_{s<=t} (1 - gamma_s),
    beta_bar_t  = (1 - alpha_bar_t - gamma_bar_t) / K,

so the one-shot marginal ``q(x_t | x_0)`` never needs an explicit ``(K+1)x(K+1)``
matrix product. As ``t -> T`` the schedule is designed so ``gamma_bar_t -> 1`` and
every token converges to ``[MASK]`` -- the state sampling starts from at inference.

Pure stdlib, deterministic (randomness routed through ``random.Random``).
"""

from __future__ import annotations

import random
from typing import Sequence

Matrix = list[list[float]]
Vector = list[float]

_TOL = 1e-9


def mask_index(num_classes: int) -> int:
    """Index of the ``[MASK]`` category (appended after the ``K`` real ones)."""
    if num_classes < 1:
        raise ValueError("num_classes must be >= 1")
    return num_classes


def beta_from(num_classes: int, alpha: float, gamma: float) -> float:
    """Derive the uniform-replace rate ``beta`` from the row-sum constraint.

    ``beta = (1 - alpha - gamma) / K``. Raises if ``(alpha, gamma)`` are infeasible
    (negative or ``alpha + gamma > 1``).
    """
    if num_classes < 1:
        raise ValueError("num_classes must be >= 1")
    if alpha < -_TOL or gamma < -_TOL:
        raise ValueError("alpha and gamma must be non-negative")
    beta = (1.0 - alpha - gamma) / num_classes
    if beta < -_TOL:
        raise ValueError("alpha + gamma must not exceed 1")
    return max(0.0, beta)


def mask_and_replace_matrix(num_classes: int, alpha: float, gamma: float) -> Matrix:
    """Row-stochastic mask-and-replace transition (Eq. 7) over ``K + 1`` states.

    States ``0 .. K-1`` are real code categories; state ``K`` is ``[MASK]``. Row ``i``
    is ``q(x_t | x_{t-1} = i)``. The mask row is the identity row (absorbing).
    """
    k = num_classes
    beta = beta_from(k, alpha, gamma)
    diag = alpha + beta
    q: Matrix = []
    for i in range(k):
        row = [beta] * k + [gamma]
        row[i] = diag
        q.append(row)
    # absorbing [MASK] row
    mask_row = [0.0] * k + [1.0]
    q.append(mask_row)
    return q


def cumulative_parameters(
    alphas: Sequence[float], gammas: Sequence[float]
) -> list[tuple[float, float, float]]:
    """Closed-form cumulative ``(alpha_bar_t, beta_bar_t, gamma_bar_t)`` per step.

    ``alphas[i]`` and ``gammas[i]`` are the step-``(i+1)`` rates. Returns a list whose
    entry ``i`` corresponds to timestep ``t = i + 1``. ``beta_bar_t`` still needs the
    class count, so it is returned *per class* factored out: the caller multiplies by
    nothing -- ``beta_bar_t`` here is the per-off-diagonal mass ``(1 - alpha_bar -
    gamma_bar) / K`` and requires ``num_classes``. Use :func:`cumulative_for` when the
    class count is known; this lower-level helper omits ``K`` and returns
    ``beta_bar_t`` as the *residual* ``1 - alpha_bar_t - gamma_bar_t`` (total off-mask
    replace mass, not yet divided by ``K``).
    """
    if len(alphas) != len(gammas):
        raise ValueError("alphas and gammas must have equal length")
    out: list[tuple[float, float, float]] = []
    alpha_prod = 1.0
    keep_prod = 1.0  # prod (1 - gamma_s)
    for a, g in zip(alphas, gammas):
        if not -_TOL <= a <= 1.0 + _TOL:
            raise ValueError("alpha out of [0, 1]")
        if not -_TOL <= g <= 1.0 + _TOL:
            raise ValueError("gamma out of [0, 1]")
        alpha_prod *= a
        keep_prod *= (1.0 - g)
        alpha_bar = alpha_prod
        gamma_bar = 1.0 - keep_prod
        residual = 1.0 - alpha_bar - gamma_bar
        out.append((alpha_bar, residual, gamma_bar))
    return out


def cumulative_for(
    num_classes: int, alphas: Sequence[float], gammas: Sequence[float]
) -> list[tuple[float, float, float]]:
    """Cumulative ``(alpha_bar_t, beta_bar_t, gamma_bar_t)`` with ``beta_bar`` per class.

    ``beta_bar_t = (1 - alpha_bar_t - gamma_bar_t) / K`` -- the probability an original
    real token maps to any one *specific* other real token after ``t`` steps.
    """
    if num_classes < 1:
        raise ValueError("num_classes must be >= 1")
    raw = cumulative_parameters(alphas, gammas)
    return [(a, residual / num_classes, g) for (a, residual, g) in raw]


def forward_marginal_index(
    x0_index: int,
    num_classes: int,
    cum_param: tuple[float, float, float],
) -> Vector:
    """Closed-form marginal ``q(x_t | x_0)`` as a ``(K + 1)``-vector.

    ``cum_param`` is the ``(alpha_bar, beta_bar, gamma_bar)`` triple from
    :func:`cumulative_for` (``beta_bar`` per class). For a real ``x_0 = i`` the mass is
    ``alpha_bar + beta_bar`` on ``i``, ``beta_bar`` on every other real category, and
    ``gamma_bar`` on ``[MASK]``. A masked ``x_0`` stays masked.
    """
    k = num_classes
    if not 0 <= x0_index <= k:
        raise IndexError("x0_index out of range for K + 1 states")
    alpha_bar, beta_bar, gamma_bar = cum_param
    if x0_index == k:  # already [MASK]: absorbing
        return [0.0] * k + [1.0]
    marg = [beta_bar] * k + [gamma_bar]
    marg[x0_index] = alpha_bar + beta_bar
    return marg


def converges_to_mask(
    gammas: Sequence[float], tol: float = 1e-6
) -> bool:
    """Whether the cumulative ``gamma_bar_T`` reaches ~1 (all mass on ``[MASK]``)."""
    keep_prod = 1.0
    for g in gammas:
        keep_prod *= (1.0 - g)
    return (1.0 - keep_prod) >= 1.0 - tol


def linear_gamma_schedule(num_steps: int, gamma_final: float = 1.0) -> list[float]:
    """A simple increasing per-step ``gamma`` schedule with ``gamma_bar_T = gamma_final``.

    Uses a uniform per-step keep factor so ``1 - prod(1 - gamma_s) = gamma_final``;
    i.e. every ``gamma_s = 1 - (1 - gamma_final)**(1 / T)``. Deterministic.
    """
    if num_steps < 1:
        raise ValueError("num_steps must be >= 1")
    if not 0.0 <= gamma_final <= 1.0:
        raise ValueError("gamma_final must lie in [0, 1]")
    per = 1.0 - (1.0 - gamma_final) ** (1.0 / num_steps)
    return [per] * num_steps


def diffuse_index(
    x0_index: int,
    num_classes: int,
    cum_params: Sequence[tuple[float, float, float]],
    t: int,
    rng: random.Random,
) -> int:
    """Sample ``x_t`` from the closed-form marginal ``q(x_t | x_0)`` (deterministic rng).

    ``cum_params`` comes from :func:`cumulative_for`; ``t`` runs ``1 .. len``.
    """
    if not 1 <= t <= len(cum_params):
        raise IndexError("t out of range for the cumulative schedule")
    marg = forward_marginal_index(x0_index, num_classes, cum_params[t - 1])
    total = sum(marg)
    r = rng.random() * total
    acc = 0.0
    for i, p in enumerate(marg):
        acc += p
        if r < acc:
            return i
    return len(marg) - 1
