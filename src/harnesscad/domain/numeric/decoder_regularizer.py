"""Decoder distance-minimization regularizer from the decoder-regularized model (Regularized
Diffusion Modeling for CAD Representation Generation, 

The central *new* contribution is a regularization term that trains the
CAD decoder ``D`` to be robust to the kind of latent variations the diffusion
model produces at sampling time. Rather than only decoding clean training latents
``z0`` (which "leads to some unrealistic decoded results"), it augments the
decoder objective with a **perturb-and-match** energy:

1. invert a clean latent ``z0`` up to noise space ``z_hat_T``
2. perturb it toward isotropic Gaussian: ``z_hat_T' = (1-sigma) z_hat_T + sigma n``
3. regenerate a nearby latent by DDIM denoising: ``z_hat_0' = DDIM(z_hat_T')``
4. penalize the decoded distance to the original CAD::

       min_D || D(z_hat_0') - CAD ||

Steps 1-2 are provided by :mod:`numeric.regdiff_ddim_inversion`; the DDIM
regeneration in step 3 reuses :func:`numeric.lion_ddim_sampler.ddim_sample`. This
module supplies the deterministic *energy* of step 4 and the full pipeline that
composes all four steps, plus a batch-averaged regularizer suitable as an
additive training loss term.

The learned decoder ``D`` and score model ``eps_theta`` are caller-supplied
callables (their weights are out of scope). Given deterministic callables and an
explicit ``random.Random`` seed, every number here is reproducible with no
wall-clock dependence.

Pure stdlib. Vectors / CAD targets are plain Python float lists.
"""

from __future__ import annotations

import random
from math import sqrt
from typing import Callable, List, Protocol, Sequence

from harnesscad.domain.numeric.ddim_sampler import ddim_sample
from harnesscad.domain.numeric.ddim_inversion import (
    ddim_invert,
    gaussian_noise,
    gaussian_perturb,
)

Vector = Sequence[float]
EpsModel = Callable[[Sequence[float], int], Sequence[float]]
# Decoder(latent) -> decoded CAD parameter vector.
Decoder = Callable[[Sequence[float]], Sequence[float]]


class HasAlphaBar(Protocol):
    def alpha_bar(self, t: int) -> float: ...


def decoder_distance(decoded: Vector, target: Vector) -> float:
    """Euclidean (L2) distance ``|| decoded - target ||``.

    The regularization objective ``min_D || D(z_hat_0') - CAD ||`` is measured with
    the L2 norm between the decoded CAD parameter vector and the ground-truth CAD.
    """
    d = list(decoded)
    t = list(target)
    if len(d) != len(t):
        raise ValueError("decoded and target must have equal length")
    return sqrt(sum((a - b) * (a - b) for a, b in zip(d, t)))


def regularization_energy(
    z0: Vector,
    cad_target: Vector,
    schedule: HasAlphaBar,
    eps_model: EpsModel,
    decoder: Decoder,
    total_steps: int,
    sigma: float,
    seed: int,
    sample_steps: int | None = None,
) -> float:
    """Full deterministic the decoder-regularized model decoder-regularization energy.

    Composes: DDIM-invert ``z0`` to ``z_hat_T``; Gaussian-perturb by ``sigma``
    using noise from ``random.Random(seed)``; DDIM-regenerate to ``z_hat_0'``;
    decode and return the L2 distance to ``cad_target``. This scalar is the term
    added to the decoder training loss so the decoder maps a *neighbourhood* of
    each latent to the same CAD, smoothing the noise space.
    """
    z_T = ddim_invert(z0, schedule, eps_model, total_steps, sample_steps)
    rng = random.Random(seed)
    noise = gaussian_noise(len(z_T), rng)
    z_T_pert = gaussian_perturb(z_T, sigma, noise)
    z0_pert = ddim_sample(z_T_pert, schedule, eps_model, total_steps, sample_steps)
    decoded = decoder(z0_pert)
    return decoder_distance(decoded, cad_target)


def batch_regularization_loss(
    latents: Sequence[Vector],
    cad_targets: Sequence[Vector],
    schedule: HasAlphaBar,
    eps_model: EpsModel,
    decoder: Decoder,
    total_steps: int,
    sigma: float,
    seed: int,
    sample_steps: int | None = None,
) -> float:
    """Mean :func:`regularization_energy` over a batch of ``(latent, cad)`` pairs.

    Each pair uses a distinct derived sub-seed (``seed + index``) so perturbations
    differ across items yet stay fully reproducible. Returns the average energy;
    an empty batch gives ``0.0``.
    """
    if len(latents) != len(cad_targets):
        raise ValueError("latents and cad_targets must have equal length")
    if not latents:
        return 0.0
    total = 0.0
    for i, (z0, cad) in enumerate(zip(latents, cad_targets)):
        total += regularization_energy(
            z0, cad, schedule, eps_model, decoder,
            total_steps, sigma, seed + i, sample_steps,
        )
    return total / len(latents)


def combined_decoder_loss(
    z0: Vector,
    cad_target: Vector,
    schedule: HasAlphaBar,
    eps_model: EpsModel,
    decoder: Decoder,
    total_steps: int,
    sigma: float,
    seed: int,
    reg_weight: float = 1.0,
    sample_steps: int | None = None,
) -> float:
    """Reconstruction + weighted regularization decoder loss.

    This approach trains the decoder on both the clean latent ``z0`` (reconstruction
    ``|| D(z0) - CAD ||``) and the perturbed regularization energy: "The latent
    representations z0 of the original data are also used to train this decoder."
    Returns ``recon + reg_weight * reg`` where ``reg`` is
    :func:`regularization_energy`. With ``reg_weight == 0`` this reduces to the
    plain reconstruction term (the "w/o reg" ablation).
    """
    if reg_weight < 0.0:
        raise ValueError("reg_weight must be >= 0")
    recon = decoder_distance(decoder(z0), cad_target)
    if reg_weight == 0.0:
        return recon
    reg = regularization_energy(
        z0, cad_target, schedule, eps_model, decoder,
        total_steps, sigma, seed, sample_steps,
    )
    return recon + reg_weight * reg
