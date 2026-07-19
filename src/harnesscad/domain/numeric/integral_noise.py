"""Variance-preserving integral-noise transport and SDE noise schedule.

Consistent multi-view flow distillation needs a noise function that (i) stays
per-pixel unit-variance Gaussian in every view, yet (ii) has correct
correspondence on the object surface between views. This is achieved with a
*noise transport* rule: each query pixel is warped onto a
high-resolution reference noise map, the reference cells it covers are summed,
and the sum is normalized by ``1 / sqrt(|area|)`` (NOT ``1 / |area|`` as for an
ordinary image), which preserves unit variance while inducing correlation
between pixels that cover overlapping reference regions.

This module isolates the deterministic, model-free pieces:

* :func:`aggregate_cell` / :func:`transport` -- area-normalized noise
  aggregation with the ``1/sqrt(n)`` rule that keeps output variance at
  1 regardless of coverage size, and gives correlated outputs for overlapping
  coverage sets (multi-view correspondence).
* :func:`inject_noise` -- the variance-preserving SDE noise-injection update
  ``eps' = sqrt(1-gamma) eps + sqrt(gamma) z``, the
  Ornstein-Uhlenbeck step that keeps ``eps`` at unit variance across
  optimization steps.
* :func:`gamma_from_beta_integral`, :func:`ddpm_equivalent_gamma` -- the
  closed-form gamma from the beta integral and its discrete-step equivalent.

Stdlib-only and deterministic (all randomness is via a seeded ``random.Random``).
"""

from __future__ import annotations

import math
import random
from typing import List, Sequence


class ReferenceNoise:
    """A deterministic high-resolution reference noise map W on E_ref.

    Values are i.i.d. unit-variance Gaussians addressed by a flat integer cell
    index, generated lazily and memoized so a given seed always yields the same
    field.
    """

    def __init__(self, seed: int = 0):
        self._seed = seed
        self._cache: dict = {}

    def value(self, cell: int) -> float:
        v = self._cache.get(cell)
        if v is None:
            # Deterministic per-cell draw independent of access order.
            v = random.Random("%d:%d" % (self._seed, cell)).gauss(0.0, 1.0)
            self._cache[cell] = v
        return v

    def values(self, cells: Sequence[int]) -> List[float]:
        return [self.value(c) for c in cells]


def aggregate_cell(values: Sequence[float]) -> float:
    """Area-normalized aggregation: sum(W) / sqrt(|coverage|).

    If the covered reference cells are independent unit-variance samples, the
    returned value is also unit variance (Var = n * 1 / n = 1). An empty
    coverage returns 0.
    """
    n = len(values)
    if n == 0:
        return 0.0
    return sum(values) / math.sqrt(n)


def transport(ref: ReferenceNoise, coverage: Sequence[Sequence[int]]) -> List[float]:
    """Warp reference noise to query pixels given per-pixel coverage cell lists.

    ``coverage[p]`` is the list of reference-cell indices covered by query pixel
    ``p`` (i.e. Omega_p). Returns the per-pixel consistent noise values.
    """
    return [aggregate_cell(ref.values(cells)) for cells in coverage]


def inject_noise(eps: Sequence[float], gamma: float, rng: random.Random) -> List[float]:
    """One SDE noise-injection step.

    ``eps' = sqrt(1 - gamma) * eps + sqrt(gamma) * z`` with fresh
    ``z ~ N(0, I)``. For ``gamma in [0, 1)`` this preserves unit variance:
    Var(eps') = (1-gamma) Var(eps) + gamma. gamma = 0 leaves eps unchanged
    (ODE / deterministic limit).
    """
    if gamma < 0.0 or gamma >= 1.0:
        raise ValueError("gamma must lie in [0, 1)")
    a = math.sqrt(1.0 - gamma)
    b = math.sqrt(gamma)
    return [a * e + b * rng.gauss(0.0, 1.0) for e in eps]


def gamma_from_beta_integral(beta_integral: float) -> float:
    """Closed form: gamma = 1 - exp(-2 * integral(beta_s ds))."""
    if beta_integral < 0.0:
        raise ValueError("beta_integral must be non-negative")
    return 1.0 - math.exp(-2.0 * beta_integral)


def ddpm_equivalent_gamma(sig_alpha_t: float, sig_alpha_T: float, k: int) -> float:
    """Exact gamma equivalent to ``k`` discrete denoising-diffusion steps.

    gamma = 1 - (sig_alpha_t / sig_alpha_T) ** (2 / k)

    where sig_alpha = sigma_t / alpha_t and ``k`` is the number of steps.
    """
    if k <= 0:
        raise ValueError("k must be positive")
    if sig_alpha_t <= 0.0 or sig_alpha_T <= 0.0:
        raise ValueError("sigma/alpha ratios must be positive")
    return 1.0 - (sig_alpha_t / sig_alpha_T) ** (2.0 / k)


def ddpm_equivalent_gamma_approx(sig_alpha_t: float, sig_alpha_T: float, k: int) -> float:
    """First-order approximation: 2 * log(sig_alpha_T / sig_alpha_t) / k."""
    if k <= 0:
        raise ValueError("k must be positive")
    return 2.0 * math.log(sig_alpha_T / sig_alpha_t) / k


def sample_variance(values: Sequence[float]) -> float:
    """Population variance helper (deterministic; used by tests)."""
    n = len(values)
    if n == 0:
        return 0.0
    mean = sum(values) / n
    return sum((v - mean) ** 2 for v in values) / n
