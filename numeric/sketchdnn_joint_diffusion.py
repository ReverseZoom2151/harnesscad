"""Joint continuous-discrete diffusion coordination (SketchDNN Sec. 4 & 4.1).

SketchDNN diffuses a whole sketch ``X = {x_i}`` where each primitive
``x = (b, c, p)`` mixes DISCRETE variables (construction flag ``b``, class label
``c``) with CONTINUOUS parameters ``p``. The joint process treats each component
independently:

* discrete ``b`` and ``c`` follow **Gaussian-Softmax** diffusion (Eqs. 6/7);
* continuous ``p`` follows **standard Gaussian** diffusion (Eqs. 2/3);

driven by a single shared retention schedule so the two modalities are noised in
lock-step. Because every primitive is corrupted *independently* and identically
(Eqs. 9/10), the forward and reverse processes are **permutation-equivariant**:
permuting the primitives before noising yields exactly the permuted noised
sketch, ``p(pi(X_t) | pi(X_0)) = p(X_t | X_0)``.

This module operates on the composite feature vectors of
``reconstruction.sketchdnn_primitive_representation`` -- the first two blocks
(construction one-hot, class one-hot) are the discrete part, everything after is
the continuous parameter part. It reuses the Gaussian-Softmax primitives from
``numeric.sketchdnn_gaussian_softmax``. Everything is stdlib-only and
deterministic (randomness via explicit ``random.Random`` streams).
"""

from __future__ import annotations

import math
import random
from typing import Sequence

from numeric.sketchdnn_gaussian_softmax import (
    gs_cumulative_sample,
    gs_posterior_mean,
    softmax,
)
from reconstruction.sketchdnn_primitive_representation import (
    CLASS_DIM,
    CONSTRUCTION_DIM,
    FEATURE_DIM,
)

Vector = list[float]

# Discrete blocks (start, end) and the continuous tail, derived from the layout.
_DISCRETE_BLOCKS = [
    (0, CONSTRUCTION_DIM),
    (CONSTRUCTION_DIM, CONSTRUCTION_DIM + CLASS_DIM),
]
_CONT_START = CONSTRUCTION_DIM + CLASS_DIM


def discrete_slices() -> list[tuple[int, int]]:
    """Return the ``(start, end)`` spans of the discrete (simplex) blocks."""
    return [s for s in _DISCRETE_BLOCKS]


def continuous_slice() -> tuple[int, int]:
    """Return the ``(start, end)`` span of the continuous parameter block."""
    return (_CONT_START, FEATURE_DIM)


def _check(vec: Sequence[float]) -> None:
    if len(vec) != FEATURE_DIM:
        raise ValueError(f"expected feature dim {FEATURE_DIM}, got {len(vec)}")


def corrupt_primitive(
    vec: Sequence[float],
    alpha_bar_t: float,
    rng: random.Random,
    k: float = 0.99,
) -> Vector:
    """Jointly corrupt one primitive at cumulative retention ``alpha_bar_t``.

    Discrete blocks are noised with Gaussian-Softmax diffusion (staying on the
    simplex); continuous parameters with Gaussian diffusion
    ``sqrt(abar) p + sqrt(1-abar) eps``.
    """
    _check(vec)
    if not 0.0 <= alpha_bar_t <= 1.0:
        raise ValueError("alpha_bar_t must be in [0, 1]")
    out = list(vec)
    # discrete blocks -> Gaussian-Softmax
    for start, end in _DISCRETE_BLOCKS:
        block = vec[start:end]
        out[start:end] = gs_cumulative_sample(block, alpha_bar_t, rng, k)
    # continuous block -> Gaussian
    sab = math.sqrt(alpha_bar_t)
    som = math.sqrt(1.0 - alpha_bar_t)
    for i in range(_CONT_START, FEATURE_DIM):
        out[i] = sab * vec[i] + som * rng.gauss(0.0, 1.0)
    return out


def corrupt_sketch(
    primitives: Sequence[Sequence[float]],
    alpha_bar_t: float,
    seeds: Sequence[int],
    k: float = 0.99,
) -> list[Vector]:
    """Corrupt a sketch primitive-by-primitive (factorised forward, Eq. 10).

    Each primitive is noised with its *own* ``random.Random(seed)`` stream so the
    noise is bound to the primitive, not to its position -- this is exactly what
    makes the process permutation-equivariant.
    """
    if len(primitives) != len(seeds):
        raise ValueError("primitives and seeds must have equal length")
    return [
        corrupt_primitive(p, alpha_bar_t, random.Random(s), k)
        for p, s in zip(primitives, seeds)
    ]


def is_permutation_equivariant(
    primitives: Sequence[Sequence[float]],
    seeds: Sequence[int],
    perm: Sequence[int],
    alpha_bar_t: float,
    k: float = 0.99,
) -> bool:
    """Check ``corrupt(pi(X)) == pi(corrupt(X))`` for the given permutation.

    Permuting ``(primitive, seed)`` pairs and corrupting must equal corrupting
    then permuting -- the deterministic witness of Eqs. 9/10.
    """
    n = len(primitives)
    if sorted(perm) != list(range(n)):
        raise ValueError("perm must be a permutation of range(n)")
    base = corrupt_sketch(primitives, alpha_bar_t, seeds, k)
    permuted_inputs = [primitives[i] for i in perm]
    permuted_seeds = [seeds[i] for i in perm]
    corrupted_perm = corrupt_sketch(permuted_inputs, alpha_bar_t, permuted_seeds, k)
    expected = [base[i] for i in perm]
    return all(
        all(abs(a - b) < 1e-12 for a, b in zip(row_c, row_e))
        for row_c, row_e in zip(corrupted_perm, expected)
    )


def _ddpm_posterior_coeffs(
    alpha_t: float, alpha_bar_t: float, alpha_bar_prev: float
) -> tuple[float, float]:
    """DDPM posterior-mean coeffs ``(c0, ct)`` for the ``x0`` parameterisation."""
    denom = 1.0 - alpha_bar_t
    if denom <= 0.0:
        raise ValueError("1 - alpha_bar_t must be positive")
    beta_t = 1.0 - alpha_t
    c0 = (math.sqrt(alpha_bar_prev) * beta_t) / denom
    ct = (math.sqrt(alpha_t) * (1.0 - alpha_bar_prev)) / denom
    return c0, ct


def joint_posterior_mean(
    vec_t: Sequence[float],
    vec_0: Sequence[float],
    alpha_t: float,
    alpha_bar_t: float,
    alpha_bar_prev: float,
) -> Vector:
    """Joint reverse posterior mean for one primitive.

    Discrete blocks use the Gaussian-Softmax posterior (projected back to the
    simplex with softmax); the continuous block uses the DDPM posterior mean
    ``c0 x_0 + ct x_t``.
    """
    _check(vec_t)
    _check(vec_0)
    out = list(vec_t)
    for start, end in _DISCRETE_BLOCKS:
        mean_logits = gs_posterior_mean(
            vec_t[start:end], vec_0[start:end], alpha_t, alpha_bar_t, alpha_bar_prev
        )
        out[start:end] = softmax(mean_logits)
    c0, ct = _ddpm_posterior_coeffs(alpha_t, alpha_bar_t, alpha_bar_prev)
    for i in range(_CONT_START, FEATURE_DIM):
        out[i] = c0 * vec_0[i] + ct * vec_t[i]
    return out
