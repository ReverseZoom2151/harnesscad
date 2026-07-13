"""DDIM inversion + Gaussian perturbation from CADiffusion (Regularized Diffusion
Modeling for CAD Representation Generation, ICLR 2025 submission, Sec. 3.3).

CADiffusion's decoder-regularization strategy "navigates through the noise space"
of the diffusion model in three deterministic stages:

1. **Inverse mapping to noise space** (paper Eq. 6). A clean training latent
   ``z0`` is mapped *back up* to a noised latent ``z_hat_T`` with DDIM inversion.
   The inversion is the exact reverse direction of the DDIM sampler and is
   governed by::

       z_hat_t = sqrt(alpha_t) * x0_pred(z_hat_{t-1}, eps)
                 + sqrt(1 - alpha_t) * eps

   where ``x0_pred = (z_hat_{t-1} - sqrt(1 - alpha_{t-1}) eps) / sqrt(alpha_{t-1})``
   and ``eps = eps_theta(z_hat_{t-1}, t-1)``. This is the standard deterministic
   DDIM inversion recurrence (Song et al., 2021, used by the paper). It steps the
   timestep *upwards* (adding structure-preserving noise), the mirror image of the
   downward DDIM reverse step in :mod:`numeric.lion_ddim_sampler` -- which
   implements only the denoising direction, never inversion.

2. **Gaussian perturbation** (paper Eq. 7). The inverted latent is nudged toward
   isotropic Gaussian noise by a convex blend with a scaling factor ``sigma``
   (the paper uses ``sigma = 0.1``)::

       z_hat_T' = (1 - sigma) * z_hat_T + sigma * N(0, I)

3. (Re-generation + decoder distance penalty live in
   :mod:`numeric.regdiff_decoder_regularizer`.)

This module is the fully deterministic core of stages 1-2. The learned score model
``eps_theta`` is caller-supplied (as in the existing DDIM sampler). All randomness
is routed through an explicit ``random.Random`` instance so runs are reproducible
with no wall-clock dependence.

Pure stdlib. Vectors are plain Python float lists.
"""

from __future__ import annotations

import random
from math import sqrt
from typing import Callable, List, Protocol, Sequence

from harnesscad.domain.numeric.ddim_sampler import predict_x0

Vector = Sequence[float]
EpsModel = Callable[[Sequence[float], int], Sequence[float]]


class HasAlphaBar(Protocol):
    def alpha_bar(self, t: int) -> float: ...


def ddim_inversion_step(
    z_prev: Vector,
    eps: Vector,
    alpha_bar_prev: float,
    alpha_bar_t: float,
) -> List[float]:
    """One deterministic DDIM inversion step ``z_{t-1} -> z_t`` (paper Eq. 6).

    ``z_t = sqrt(alpha_bar_t) * x0_pred + sqrt(1 - alpha_bar_t) * eps`` where
    ``x0_pred`` is the clean-sample estimate recovered from ``z_{t-1}``. The
    timestep moves *up* (``prev < t``) so this re-injects noise. Exactly inverts
    :func:`numeric.lion_ddim_sampler.ddim_step` when ``eps`` is held fixed.
    """
    if not 0.0 < alpha_bar_prev <= 1.0:
        raise ValueError("alpha_bar_prev must be in (0, 1]")
    if not 0.0 < alpha_bar_t <= 1.0:
        raise ValueError("alpha_bar_t must be in (0, 1]")
    x0 = predict_x0(z_prev, eps, alpha_bar_prev)
    sab_t = sqrt(alpha_bar_t)
    dir_coeff = sqrt(max(0.0, 1.0 - alpha_bar_t))
    return [sab_t * x0_i + dir_coeff * e for x0_i, e in zip(x0, eps)]


def make_inversion_timesteps(total_steps: int, sample_steps: int) -> List[int]:
    """Ascending DDIM sub-sequence of ``1 .. total_steps`` for inversion.

    Mirror of :func:`numeric.lion_ddim_sampler.make_timesteps` but in *increasing*
    order (inversion climbs from the clean signal up to the fully-noised latent).
    Always includes the terminal step ``total_steps``.
    """
    if total_steps < 1:
        raise ValueError("total_steps must be >= 1")
    if not 1 <= sample_steps <= total_steps:
        raise ValueError("sample_steps must be in [1, total_steps]")
    if sample_steps == 1:
        return [total_steps]
    steps = []
    for i in range(sample_steps):
        t = 1 + round(i * (total_steps - 1) / (sample_steps - 1))
        steps.append(int(t))
    return sorted(set(steps))


def ddim_invert(
    z0: Vector,
    schedule: HasAlphaBar,
    eps_model: EpsModel,
    total_steps: int,
    sample_steps: int | None = None,
) -> List[float]:
    """Full deterministic DDIM inversion loop ``z0 -> z_hat_T`` (paper Eq. 6).

    Climbs the ascending sub-sequence from :func:`make_inversion_timesteps`,
    evaluating ``eps_model(z_prev, prev_t)`` at each source step (the previous,
    lower-noise latent -- the standard DDIM-inversion approximation). The step
    below the smallest timestep is ``alpha_bar(0)`` (the clean signal). Returns
    the noised latent ``z_hat_T``. Deterministic given a deterministic eps model.
    """
    if sample_steps is None:
        sample_steps = total_steps
    timesteps = make_inversion_timesteps(total_steps, sample_steps)
    z = [float(v) for v in z0]
    prev_t = 0
    for t in timesteps:
        eps = list(eps_model(z, prev_t))
        if len(eps) != len(z):
            raise ValueError("eps_model must return a vector matching z")
        ab_prev = schedule.alpha_bar(prev_t)
        ab_t = schedule.alpha_bar(t)
        z = ddim_inversion_step(z, eps, ab_prev, ab_t)
        prev_t = t
    return z


def gaussian_noise(n: int, rng: random.Random) -> List[float]:
    """``n`` samples of isotropic standard Gaussian noise from ``rng``.

    Randomness is entirely determined by the supplied ``random.Random`` instance,
    so results are reproducible and free of any wall-clock dependence.
    """
    if n < 0:
        raise ValueError("n must be >= 0")
    return [rng.gauss(0.0, 1.0) for _ in range(n)]


def gaussian_perturb(
    z_T: Vector,
    sigma: float,
    noise: Vector,
) -> List[float]:
    """Convex blend toward isotropic Gaussian noise (paper Eq. 7).

    ``z_hat_T' = (1 - sigma) * z_T + sigma * noise``. ``sigma`` is the perturbation
    scaling factor (paper default ``0.1``); ``noise`` is a caller-supplied
    (deterministic) isotropic Gaussian sample of the same length as ``z_T``. With
    ``sigma == 0`` the input is returned unchanged; with ``sigma == 1`` the output
    is pure noise.
    """
    if not 0.0 <= sigma <= 1.0:
        raise ValueError("sigma must be in [0, 1]")
    z = list(z_T)
    nz = list(noise)
    if len(nz) != len(z):
        raise ValueError("noise must match z_T length")
    return [(1.0 - sigma) * v + sigma * n for v, n in zip(z, nz)]


def perturb_with_seed(z_T: Vector, sigma: float, seed: int) -> List[float]:
    """Convenience: :func:`gaussian_perturb` with noise drawn from ``Random(seed)``."""
    rng = random.Random(seed)
    noise = gaussian_noise(len(list(z_T)), rng)
    return gaussian_perturb(z_T, sigma, noise)
