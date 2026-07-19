"""Clean-flow ODE / SDE integrator on an analytic Gaussian target.

The diffusion probability-flow ODE is reparametrized into the *clean variable*
``x_hat_c = (x_t - sigma_t * eps) / alpha_t``, which is a non-noisy
image for all t, is initialized at zero (``x_hat_c(T) = 0``), and whose ODE
endpoint ``x_hat_c(0) = x_0`` is a sample from the target distribution fully
determined by the fixed noise ``eps``. On top of that sits a second-order Heun
stochastic sampler in the log-signal-to-noise time variable.

The full method distills a learned diffusion model and is out of scope, but the
*integrator itself* is a deterministic numerical primitive. This module
implements it against a closed-form Gaussian score, where everything is exact:

For a data distribution ``x_0 ~ N(mu, s^2 I)`` under a VP schedule
``x_t = alpha_t x_0 + sigma_t eps`` the marginal is
``N(alpha_t mu, (alpha_t^2 s^2 + sigma_t^2) I)`` so the network prediction is

    eps_phi(x_t, t) = sigma_t (x_t - alpha_t mu) / (alpha_t^2 s^2 + sigma_t^2)

and the sample prediction is ``D(x_t, t) = (x_t - sigma_t eps_phi) / alpha_t``.

Known exact fact used as a test oracle: the probability-flow ODE for a Gaussian
maps the standardized initial noise linearly to ``x_0 = mu + s * eps_tilde``,
i.e. it preserves the z-score. The integrators here are validated against that.

Stdlib-only, deterministic (a seeded PRNG is used only for the stochastic
sampler's noise injection).
"""

from __future__ import annotations

import math
import random
from typing import Callable, List, Sequence, Tuple

Schedule = Callable[[float], Tuple[float, float]]


def cosine_schedule(t: float) -> Tuple[float, float]:
    """A simple VP schedule on t in [0, 1]: alpha_t = cos, sigma_t = sin."""
    ang = 0.5 * math.pi * t
    return math.cos(ang), math.sin(ang)


def eps_gaussian(x: float, alpha: float, sigma: float, mu: float, s: float) -> float:
    """Analytic diffusion-model prediction eps_phi for an N(mu, s^2) target."""
    denom = alpha * alpha * s * s + sigma * sigma
    return sigma * (x - alpha * mu) / denom


def sample_prediction(x: float, alpha: float, sigma: float, mu: float, s: float) -> float:
    """Sample prediction D = E[x0|xt] = (x_t - sigma eps_phi) / alpha."""
    eps = eps_gaussian(x, alpha, sigma, mu, s)
    return (x - sigma * eps) / alpha


def clean_flow_ode_endpoint(
    eps_tilde: float,
    mu: float,
    s: float,
    steps: int = 200,
    schedule: Schedule = cosine_schedule,
    t_start: float = 0.999,
    t_end: float = 1e-3,
) -> float:
    """Integrate the clean-flow ODE from t_start to t_end.

    Marches the *clean variable* ``x_hat_c`` directly, which stays
    bounded (no division by the vanishing ``alpha_t`` near t=T):

        d x_hat_c = d(sigma/alpha) * (eps_phi(alpha x_hat_c + sigma eps_tilde, t)
                                      - eps_tilde)

    with initial condition ``x_hat_c(T) = 0``. First-order (explicit Euler in the
    ``lambda = sigma/alpha`` time) discretization, first-order accurate.
    Returns ``x_hat_c(t_end) ~= x_0``.
    """
    if steps <= 0:
        raise ValueError("steps must be positive")
    xhat = 0.0  # x_hat_c(T) = 0
    ts = [t_start + (t_end - t_start) * i / steps for i in range(steps + 1)]
    for i in range(steps):
        a0, sig0 = schedule(ts[i])
        a1, sig1 = schedule(ts[i + 1])
        xt = a0 * xhat + sig0 * eps_tilde
        eps = eps_gaussian(xt, a0, sig0, mu, s)
        dlam = sig1 / a1 - sig0 / a0
        xhat += dlam * (eps - eps_tilde)
    return xhat


def edm_heun_sampler(
    mu: float,
    s: float,
    steps: int = 50,
    schedule: Schedule = cosine_schedule,
    t_start: float = 0.999,
    t_end: float = 1e-3,
    gamma: float = 0.0,
    seed: int = 0,
    eps_tilde: float | None = None,
) -> float:
    """Second-order Heun sampler on the clean variable.

    ``gamma`` is the per-step noise-injection rate (0 => deterministic Heun /
    clean-flow ODE). ``eps_tilde`` is the fixed trajectory-identity noise; when
    ``gamma > 0`` it is refreshed each step by the variance-preserving update
    ``eps <- sqrt(1-gamma) eps + sqrt(gamma) z``. Returns x_0. When
    ``gamma == 0`` and ``eps_tilde`` is given the result is deterministic and
    second-order accurate.
    """
    if steps <= 0:
        raise ValueError("steps must be positive")
    if gamma < 0.0 or gamma >= 1.0:
        raise ValueError("gamma must lie in [0, 1)")
    rng = random.Random(seed)
    eps_cur = rng.gauss(0.0, 1.0) if eps_tilde is None else eps_tilde
    xhat = 0.0  # x_hat_c(T) = 0
    ts = [t_start + (t_end - t_start) * i / steps for i in range(steps + 1)]
    for i in range(steps):
        a0, sig0 = schedule(ts[i])
        a1, sig1 = schedule(ts[i + 1])
        if gamma > 0.0:
            eps_cur = math.sqrt(1.0 - gamma) * eps_cur + math.sqrt(gamma) * rng.gauss(0.0, 1.0)
        dlam = sig1 / a1 - sig0 / a0
        xt0 = a0 * xhat + sig0 * eps_cur
        eps0 = eps_gaussian(xt0, a0, sig0, mu, s)
        slope0 = dlam * (eps0 - eps_cur)
        xhat_e = xhat + slope0
        xt1 = a1 * xhat_e + sig1 * eps_cur
        eps1 = eps_gaussian(xt1, a1, sig1, mu, s)
        slope1 = dlam * (eps1 - eps_cur)
        xhat += 0.5 * (slope0 + slope1)
    return xhat


def target_sample(eps_tilde: float, mu: float, s: float) -> float:
    """Exact PF-ODE oracle for a Gaussian target: x_0 = mu + s * eps_tilde."""
    return mu + s * eps_tilde
