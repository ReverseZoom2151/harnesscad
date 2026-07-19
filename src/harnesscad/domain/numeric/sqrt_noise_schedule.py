"""Deterministic diffusion primitives from Diffusion-CAD (TVCG 2025).

Diffusion-CAD trains a DDPM over *CAD vectors* (continuous embeddings of a
discretised CAD command sequence). The learned denoiser is out of scope, but
several pieces of this approach are fully deterministic given a schedule and a seed:

1.  The **sqrt noise schedule** (Diffusion-LM style, adopted verbatim by the
    paper: "1000 diffusion steps with a sqrt noise schedule"). The schedule is
    defined through the cumulative signal-retention coefficient

        alpha_bar(t) = 1 - sqrt(t / T + s)

    with a small offset ``s`` so ``alpha_bar(0)`` is not exactly 1. Per-step
    ``beta_t`` and ``alpha_t = 1 - beta_t`` are recovered from the ratio of
    consecutive ``alpha_bar`` values, matching the Markov chain in Eq. (2).

2.  The **reparameterised forward diffusion** (Eq. (3)):

        q(x_t | x_0) = N( sqrt(alpha_bar_t) * x_0 , (1 - alpha_bar_t) I )

    i.e. ``x_t = sqrt(alpha_bar_t) x_0 + sqrt(1 - alpha_bar_t) * eps`` with
    ``eps ~ N(0, I)``. This is the training-pair construction (noising a
    ground-truth CAD vector to step ``t``).

3.  The **classifier-free conditional noise seeding** used at generation time
    for command-type / dimension / partial-sketch control. The
    user-specified coordinates of the CAD vector are seeded from the *shifted*
    distribution ``N( sqrt(alpha_bar_T) e_c , (1 - alpha_bar_T) I )`` while the
    remaining coordinates are drawn from the standard Gaussian ``N(0, I)``. This
    is the deterministic (given seed) recipe that biases the reverse process
    toward the user's constraints without retraining.

Everything here is stdlib-only, deterministic (all randomness flows through an
explicit ``random.Random`` seed), and works on plain Python float vectors so it
never depends on the learned network.
"""

from __future__ import annotations

import math
import random
from typing import Iterable, Sequence


class SqrtNoiseSchedule:
    """The sqrt DDPM schedule used by Diffusion-CAD (T=1000 by default).

    ``alpha_bar[t]`` is the signal-retention coefficient after ``t`` noising
    steps; ``alpha_bar[0]`` corresponds to the clean vector. Indices run
    ``0 .. T`` inclusive (``T + 1`` entries) so ``t`` is a genuine step count.
    """

    def __init__(self, steps: int = 1000, offset: float = 1e-4) -> None:
        if steps < 1:
            raise ValueError("steps must be >= 1")
        if not 0.0 <= offset < 1.0:
            raise ValueError("offset must be in [0, 1)")
        self.steps = int(steps)
        self.offset = float(offset)
        # alpha_bar(t) = 1 - sqrt(t/T + s), clamped to (0, 1].
        ab = []
        for t in range(self.steps + 1):
            v = 1.0 - math.sqrt(t / self.steps + offset)
            # Numerical guard: keep strictly positive and monotone non-increasing.
            v = max(v, 1e-12)
            ab.append(v)
        # Enforce monotone non-increasing (sqrt already is, guard rounding).
        for t in range(1, len(ab)):
            if ab[t] > ab[t - 1]:
                ab[t] = ab[t - 1]
        self._alpha_bar = ab

    def alpha_bar(self, t: int) -> float:
        """Cumulative signal coefficient ``\\bar{alpha}_t``."""
        if not 0 <= t <= self.steps:
            raise IndexError(f"t must be in [0, {self.steps}]")
        return self._alpha_bar[t]

    def beta(self, t: int) -> float:
        """Per-step diffusion rate ``beta_t`` for ``t`` in ``1 .. T``.

        Recovered from ``alpha_t = alpha_bar_t / alpha_bar_{t-1}`` and
        ``beta_t = 1 - alpha_t``.
        """
        if not 1 <= t <= self.steps:
            raise IndexError(f"t must be in [1, {self.steps}]")
        prev = self._alpha_bar[t - 1]
        cur = self._alpha_bar[t]
        alpha_t = cur / prev if prev > 0 else 0.0
        return 1.0 - alpha_t

    def alpha(self, t: int) -> float:
        """Per-step retention ``alpha_t = 1 - beta_t``."""
        return 1.0 - self.beta(t)

    def sqrt_alpha_bar(self, t: int) -> float:
        return math.sqrt(self.alpha_bar(t))

    def sqrt_one_minus_alpha_bar(self, t: int) -> float:
        return math.sqrt(1.0 - self.alpha_bar(t))

    def snr(self, t: int) -> float:
        """Signal-to-noise ratio ``alpha_bar_t / (1 - alpha_bar_t)``."""
        ab = self.alpha_bar(t)
        denom = 1.0 - ab
        return float("inf") if denom <= 0 else ab / denom


def forward_diffuse(
    x0: Sequence[float],
    t: int,
    schedule: SqrtNoiseSchedule,
    rng: random.Random,
) -> list[float]:
    """Sample ``x_t`` from ``q(x_t | x_0)`` (Eq. (3)).

    ``x_t = sqrt(alpha_bar_t) * x_0 + sqrt(1 - alpha_bar_t) * eps``.
    """
    sab = schedule.sqrt_alpha_bar(t)
    somab = schedule.sqrt_one_minus_alpha_bar(t)
    return [sab * v + somab * rng.gauss(0.0, 1.0) for v in x0]


def posterior_mean_coeffs(t: int, schedule: SqrtNoiseSchedule) -> tuple[float, float]:
    """Coefficients of the DDPM posterior mean for the ``x0`` parameterisation.

    Diffusion-CAD predicts ``x0`` directly (not the noise). The reverse-step
    posterior ``q(x_{t-1} | x_t, x_0)`` has mean

        mu = c0 * x0 + ct * x_t

    with the standard closed form. Returns ``(c0, ct)``. For ``t == 1`` the
    chain terminates at ``x_0`` so ``(1.0, 0.0)`` is returned.
    """
    if t == 1:
        return (1.0, 0.0)
    ab_t = schedule.alpha_bar(t)
    ab_prev = schedule.alpha_bar(t - 1)
    beta_t = schedule.beta(t)
    alpha_t = 1.0 - beta_t
    one_minus_ab_t = 1.0 - ab_t
    c0 = (math.sqrt(ab_prev) * beta_t) / one_minus_ab_t
    ct = (math.sqrt(alpha_t) * (1.0 - ab_prev)) / one_minus_ab_t
    return (c0, ct)


def conditional_noise_seed(
    dim: int,
    schedule: SqrtNoiseSchedule,
    conditioned: dict[int, float],
    rng: random.Random,
    t: int | None = None,
) -> list[float]:
    """Classifier-free conditional noise seeding.

    Coordinates listed in ``conditioned`` (index -> embedding value ``e_c``) are
    drawn from ``N( sqrt(alpha_bar_t) * e_c , (1 - alpha_bar_t) I )``; all other
    coordinates are drawn from the standard Gaussian ``N(0, I)``. ``t`` defaults
    to the terminal step ``T`` (this approach seeds the input noise at time ``T``).

    This is the deterministic construction of the biased input noise used for
    command-type control, dimension control, and partial-sketch completion.
    """
    if dim < 0:
        raise ValueError("dim must be non-negative")
    step = schedule.steps if t is None else t
    sab = schedule.sqrt_alpha_bar(step)
    somab = schedule.sqrt_one_minus_alpha_bar(step)
    out = []
    for i in range(dim):
        if i in conditioned:
            mean = sab * conditioned[i]
            out.append(rng.gauss(mean, somab))
        else:
            out.append(rng.gauss(0.0, 1.0))
    return out


def classifier_free_mix(
    uncond: Sequence[float],
    cond: Sequence[float],
    guidance: float,
) -> list[float]:
    """Classifier-free guidance mixing rule.

    ``out = uncond + guidance * (cond - uncond)``. ``guidance == 0`` returns the
    unconditional prediction; ``guidance == 1`` returns the conditional one;
    larger values extrapolate toward the condition (the standard CFG rule).
    """
    if len(uncond) != len(cond):
        raise ValueError("uncond and cond must have equal length")
    return [u + guidance * (c - u) for u, c in zip(uncond, cond)]


def quantize_levels(value: float, low: float, high: float, levels: int = 256) -> int:
    """Quantise a continuous parameter into ``levels`` discrete bins.

    Diffusion-CAD (following conventional CAD sequence models) unifies mixed discrete/continuous CAD
    parameters by quantising continuous values into 256 levels. Deterministic
    nearest-bin rounding with clamping to ``[0, levels-1]``.
    """
    if levels < 2:
        raise ValueError("levels must be >= 2")
    if high <= low:
        raise ValueError("high must exceed low")
    frac = (value - low) / (high - low)
    idx = int(round(frac * (levels - 1)))
    return max(0, min(levels - 1, idx))


def dequantize_levels(index: int, low: float, high: float, levels: int = 256) -> float:
    """Inverse of :func:`quantize_levels` (bin centre value)."""
    if levels < 2:
        raise ValueError("levels must be >= 2")
    idx = max(0, min(levels - 1, int(index)))
    return low + (idx / (levels - 1)) * (high - low)
