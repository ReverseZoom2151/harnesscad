"""Gaussian-Softmax (GS) discrete diffusion from SketchDNN (ICML 2025), Sec. 3.3.

SketchDNN's core contribution is a *simplex-constrained* discrete-diffusion
process that -- unlike conventional categorical/Multinomial diffusion -- allows
**superposition**: a class label may be a blended probability vector rather than
a hard one-hot. The trick is to run ordinary Gaussian diffusion in log-space and
project back onto the probability simplex with a softmax:

    if  x ~ N(mu, sigma^2 I)  then  softmax(x) ~ GS(mu, sigma^2 I).

The forward process noises a (label-smoothed) one-hot ``y_0`` toward the uniform
distribution ``GS(0, I)``:

  * forward step (Eq. 5):
        y_{t+1} = softmax( sqrt(a_{t+1}) log(y_t) + sqrt(1 - a_{t+1}) eps )
  * cumulative one-shot (Eq. 6):
        y_t = softmax( sqrt(abar_t) log(y'_0) + sqrt(1 - abar_t) eps )
    with label smoothing ``y'_0 = k y_0 + (1-k)/D`` (``k = 0.99``) to avoid
    ``log 0`` singularities.

The reverse process samples from the GS posterior (Eq. 7), which is analogous to
the DDPM posterior but computed on the *logits*:

    y_{t-1} = softmax( mu_{t-1}(y_t, y_0) + sigma_{t-1} eps )

    mu_{t-1} = ( sqrt(a_t)(1 - abar_{t-1}) log y_t
                 + sqrt(abar_{t-1})(1 - a_t) log y_0 ) / (1 - abar_t)
    sigma_{t-1} = sqrt( (1 - a_t)(1 - abar_{t-1}) / (1 - abar_t) )

Here ``a_t`` are the per-step retention coefficients and ``abar_t`` their
cumulative product (matching the paper's ``alpha`` / ``alpha_bar`` overline).
This module is stdlib-only, deterministic (all randomness via ``random.Random``)
and operates on plain Python probability vectors.
"""

from __future__ import annotations

import math
import random
from typing import Sequence

Vector = list[float]

DEFAULT_LABEL_SMOOTH_K = 0.99


def softmax(logits: Sequence[float]) -> Vector:
    """Numerically stable softmax mapping ``R^D`` onto the probability simplex."""
    m = max(logits)
    exps = [math.exp(v - m) for v in logits]
    s = sum(exps)
    return [e / s for e in exps]


def label_smooth(y0: Sequence[float], k: float = DEFAULT_LABEL_SMOOTH_K) -> Vector:
    """``y'_0 = k y_0 + (1-k)/D`` -- smoothing so ``log y'_0`` is finite (Eq. 6)."""
    if not 0.0 < k <= 1.0:
        raise ValueError("k must be in (0, 1]")
    d = len(y0)
    if d == 0:
        raise ValueError("y0 must be non-empty")
    base = (1.0 - k) / d
    return [k * v + base for v in y0]


def safe_log(v: Sequence[float]) -> Vector:
    """Elementwise log with a tiny floor (inputs should be label-smoothed)."""
    return [math.log(max(x, 1e-12)) for x in v]


def gs_forward_step(
    y_t: Sequence[float],
    alpha_step: float,
    rng: random.Random,
) -> Vector:
    """Single GS forward transition (Eq. 5).

    ``y_{t+1} = softmax( sqrt(a) log y_t + sqrt(1-a) eps )``.
    """
    if not 0.0 <= alpha_step <= 1.0:
        raise ValueError("alpha_step must be in [0, 1]")
    sa = math.sqrt(alpha_step)
    sm = math.sqrt(1.0 - alpha_step)
    logits = [sa * lv + sm * rng.gauss(0.0, 1.0) for lv in safe_log(y_t)]
    return softmax(logits)


def gs_cumulative_sample(
    y0: Sequence[float],
    alpha_bar_t: float,
    rng: random.Random,
    k: float = DEFAULT_LABEL_SMOOTH_K,
) -> Vector:
    """One-shot GS marginal ``y_t ~ q(y_t | y_0)`` (Eq. 6).

    Label-smooths ``y_0``, diffuses its log with cumulative ``abar_t``, and
    projects back to the simplex. At ``abar_t = 0`` this is ``softmax(eps)``,
    a uniform-in-expectation sample whose argmax is uniform over classes.
    """
    if not 0.0 <= alpha_bar_t <= 1.0:
        raise ValueError("alpha_bar_t must be in [0, 1]")
    y0s = label_smooth(y0, k)
    sa = math.sqrt(alpha_bar_t)
    sm = math.sqrt(1.0 - alpha_bar_t)
    logits = [sa * lv + sm * rng.gauss(0.0, 1.0) for lv in safe_log(y0s)]
    return softmax(logits)


def gs_posterior_logit_coeffs(
    alpha_t: float, alpha_bar_t: float, alpha_bar_prev: float
) -> tuple[float, float]:
    """Coefficients ``(c_t, c_0)`` of the GS posterior mean on the logits (Eq. 7).

    ``mu_{t-1} = c_t log y_t + c_0 log y_0`` with
    ``c_t = sqrt(a_t)(1 - abar_{t-1}) / (1 - abar_t)`` and
    ``c_0 = sqrt(abar_{t-1})(1 - a_t) / (1 - abar_t)``.
    """
    denom = 1.0 - alpha_bar_t
    if denom <= 0.0:
        raise ValueError("1 - alpha_bar_t must be positive")
    c_t = (math.sqrt(alpha_t) * (1.0 - alpha_bar_prev)) / denom
    c_0 = (math.sqrt(alpha_bar_prev) * (1.0 - alpha_t)) / denom
    return c_t, c_0


def gs_posterior_sigma(
    alpha_t: float, alpha_bar_t: float, alpha_bar_prev: float
) -> float:
    """Posterior std ``sigma_{t-1} = sqrt((1-a_t)(1-abar_{t-1})/(1-abar_t))``."""
    denom = 1.0 - alpha_bar_t
    if denom <= 0.0:
        raise ValueError("1 - alpha_bar_t must be positive")
    var = (1.0 - alpha_t) * (1.0 - alpha_bar_prev) / denom
    return math.sqrt(max(var, 0.0))


def gs_posterior_mean(
    y_t: Sequence[float],
    y0: Sequence[float],
    alpha_t: float,
    alpha_bar_t: float,
    alpha_bar_prev: float,
) -> Vector:
    """Posterior mean logits ``mu_{t-1}(y_t, y_0)`` (Eq. 7, log-space interp)."""
    if len(y_t) != len(y0):
        raise ValueError("y_t and y0 must have equal length")
    c_t, c_0 = gs_posterior_logit_coeffs(alpha_t, alpha_bar_t, alpha_bar_prev)
    lt = safe_log(y_t)
    l0 = safe_log(y0)
    return [c_t * a + c_0 * b for a, b in zip(lt, l0)]


def gs_reverse_step(
    y_t: Sequence[float],
    y0_pred: Sequence[float],
    alpha_t: float,
    alpha_bar_t: float,
    alpha_bar_prev: float,
    rng: random.Random,
) -> Vector:
    """Sample ``y_{t-1}`` from the GS reverse transition (Eq. 7).

    ``y_{t-1} = softmax( mu_{t-1}(y_t, y0_pred) + sigma_{t-1} eps )``.
    """
    mean = gs_posterior_mean(y_t, y0_pred, alpha_t, alpha_bar_t, alpha_bar_prev)
    sigma = gs_posterior_sigma(alpha_t, alpha_bar_t, alpha_bar_prev)
    logits = [mv + sigma * rng.gauss(0.0, 1.0) for mv in mean]
    return softmax(logits)


def gs_reverse_step_mean(
    y_t: Sequence[float],
    y0_pred: Sequence[float],
    alpha_t: float,
    alpha_bar_t: float,
    alpha_bar_prev: float,
) -> Vector:
    """Deterministic (noise-free) reverse step: ``softmax(mu_{t-1})``."""
    mean = gs_posterior_mean(y_t, y0_pred, alpha_t, alpha_bar_t, alpha_bar_prev)
    return softmax(mean)


def argmax_class(y: Sequence[float]) -> int:
    """Recover the class label ``argmax(y)`` from a simplex vector."""
    best_i, best_v = 0, y[0]
    for i, v in enumerate(y):
        if v > best_v:
            best_i, best_v = i, v
    return best_i
