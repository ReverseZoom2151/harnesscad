"""Deterministic DDIM sampling + diffuse-denoise from LION (Zeng et al., 2022).

LION's main experiments use 1000-step DDPM synthesis, but Section 5.5 highlights
that switching to **DDIM** [Song et al., 2021] produces high-quality shapes in a
handful of steps ("25-step DDIM samples, 0.89s per shape") for interactive use.
The DDIM reverse update is *deterministic* (the eta=0 case): given a schedule and
a noise-prediction ``eps(x_t, t)``, every step is a closed-form combination with
no injected noise. LION also uses a **diffuse-denoise** procedure (SDEdit-style):
encode a shape, run the forward diffusion only ``tau < T`` steps so only local
details are destroyed, then denoise back from ``tau`` -- yielding controlled
variations of the original.

This module implements those deterministic schedulers. The learned denoiser is
out of scope, so the noise-prediction function is a caller-supplied callable; the
existing ``numeric.diffusioncad_sqrt_schedule`` provides a compatible schedule
(it exposes ``alpha_bar(t)``), but this DDIM stepper is new -- diffusioncad only
implements the DDPM ancestral posterior, not the DDIM ODE-style update.

Pure stdlib, deterministic. Vectors are plain Python float lists.
"""

from __future__ import annotations

from math import sqrt
from typing import Callable, List, Protocol, Sequence

Vector = Sequence[float]
# eps_model(x_t, t) -> predicted noise vector of the same length as x_t.
EpsModel = Callable[[Sequence[float], int], Sequence[float]]


class HasAlphaBar(Protocol):
    def alpha_bar(self, t: int) -> float: ...


def predict_x0(x_t: Vector, eps: Vector, alpha_bar_t: float) -> List[float]:
    """Recover the predicted clean sample ``x0`` from ``x_t`` and noise ``eps``.

    ``x0 = (x_t - sqrt(1 - alpha_bar_t) * eps) / sqrt(alpha_bar_t)``.
    """
    if not 0.0 < alpha_bar_t <= 1.0:
        raise ValueError("alpha_bar_t must be in (0, 1]")
    sab = sqrt(alpha_bar_t)
    somab = sqrt(max(0.0, 1.0 - alpha_bar_t))
    return [(x - somab * e) / sab for x, e in zip(x_t, eps)]


def ddim_step(
    x_t: Vector,
    eps: Vector,
    alpha_bar_t: float,
    alpha_bar_prev: float,
) -> List[float]:
    """One deterministic (eta=0) DDIM reverse step ``x_t -> x_{prev}``.

    ``x_prev = sqrt(alpha_bar_prev) * x0_pred + sqrt(1 - alpha_bar_prev) * eps``
    where ``x0_pred`` is the current clean-sample estimate. With eta=0 no noise
    is injected, so the whole trajectory is a deterministic function of ``eps``.
    """
    if not 0.0 <= alpha_bar_prev <= 1.0:
        raise ValueError("alpha_bar_prev must be in [0, 1]")
    x0 = predict_x0(x_t, eps, alpha_bar_t)
    sab_prev = sqrt(alpha_bar_prev)
    dir_coeff = sqrt(max(0.0, 1.0 - alpha_bar_prev))
    return [sab_prev * x0_i + dir_coeff * e for x0_i, e in zip(x0, eps)]


def make_timesteps(total_steps: int, sample_steps: int) -> List[int]:
    """Evenly spaced descending DDIM sub-sequence of ``1 .. total_steps``.

    Returns ``sample_steps`` distinct timesteps in *decreasing* order, always
    including the terminal step ``total_steps``. DDIM subsamples the training
    schedule; with ``sample_steps == total_steps`` every step is visited.
    """
    if total_steps < 1:
        raise ValueError("total_steps must be >= 1")
    if not 1 <= sample_steps <= total_steps:
        raise ValueError("sample_steps must be in [1, total_steps]")
    if sample_steps == 1:
        return [total_steps]
    steps = []
    for i in range(sample_steps):
        # map i in [0, sample_steps-1] -> t in [1, total_steps]
        t = 1 + round(i * (total_steps - 1) / (sample_steps - 1))
        steps.append(int(t))
    # dedupe while preserving, then sort descending
    uniq = sorted(set(steps), reverse=True)
    return uniq


def ddim_sample(
    x_T: Vector,
    schedule: HasAlphaBar,
    eps_model: EpsModel,
    total_steps: int,
    sample_steps: int | None = None,
) -> List[float]:
    """Full deterministic DDIM sampling loop from ``x_T`` down to ``x_0``.

    Iterates the sub-sequence returned by :func:`make_timesteps`, calling
    ``eps_model(x_t, t)`` at each visited step. ``alpha_bar`` for the step below
    the smallest timestep is taken as ``alpha_bar(0) `` (clean signal). Returns
    the final ``x_0`` estimate. Deterministic given a deterministic eps model.
    """
    if sample_steps is None:
        sample_steps = total_steps
    timesteps = make_timesteps(total_steps, sample_steps)
    x = [float(v) for v in x_T]
    for idx, t in enumerate(timesteps):
        eps = list(eps_model(x, t))
        if len(eps) != len(x):
            raise ValueError("eps_model must return a vector matching x_t")
        ab_t = schedule.alpha_bar(t)
        prev_t = timesteps[idx + 1] if idx + 1 < len(timesteps) else 0
        ab_prev = schedule.alpha_bar(prev_t)
        x = ddim_step(x, eps, ab_t, ab_prev)
    return x


def diffuse_denoise_steps(total_steps: int, tau: int) -> List[int]:
    """Timestep sequence for LION's diffuse-denoise reverse pass from ``tau``.

    The forward pass noises a clean encoding up to intermediate step ``tau``; the
    reverse pass then denoises from ``tau`` down to ``0``. Returns the descending
    reverse timesteps ``tau, tau-1, ..., 1`` (empty when ``tau == 0``). Smaller
    ``tau`` preserves more of the original shape; larger ``tau`` yields more
    novel variations.
    """
    if not 0 <= tau <= total_steps:
        raise ValueError("tau must be in [0, total_steps]")
    return list(range(tau, 0, -1))


def diffuse_denoise_sample(
    x0: Vector,
    schedule: HasAlphaBar,
    eps_model: EpsModel,
    total_steps: int,
    tau: int,
    forward_noise: Sequence[float] | None = None,
) -> List[float]:
    """Deterministic diffuse-denoise: noise ``x0`` to ``tau`` then DDIM-denoise.

    ``forward_noise`` is the (caller-supplied, deterministic) ``eps`` used to
    construct ``x_tau = sqrt(alpha_bar_tau) x0 + sqrt(1 - alpha_bar_tau) eps``.
    When omitted (or ``tau == 0``) the input is returned unchanged after a no-op
    round trip. The reverse pass reuses the DDIM stepper over ``tau..1``.
    """
    if tau == 0:
        return [float(v) for v in x0]
    ab_tau = schedule.alpha_bar(tau)
    sab = sqrt(ab_tau)
    somab = sqrt(max(0.0, 1.0 - ab_tau))
    noise = list(forward_noise) if forward_noise is not None else [0.0] * len(x0)
    if len(noise) != len(x0):
        raise ValueError("forward_noise must match x0 length")
    x = [sab * v + somab * n for v, n in zip(x0, noise)]
    steps = diffuse_denoise_steps(total_steps, tau)
    for idx, t in enumerate(steps):
        eps = list(eps_model(x, t))
        if len(eps) != len(x):
            raise ValueError("eps_model must return a vector matching x_t")
        ab_t = schedule.alpha_bar(t)
        prev_t = steps[idx + 1] if idx + 1 < len(steps) else 0
        ab_prev = schedule.alpha_bar(prev_t)
        x = ddim_step(x, eps, ab_t, ab_prev)
    return x
