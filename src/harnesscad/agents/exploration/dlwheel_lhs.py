"""Normal-distribution Latin hypercube sampling (LHSnorm) in a latent space.

Deterministic design-of-experiment sampler of the DOE-in-latent-space stage
(Section 4.3) of:

    Yoo et al., "Integrating deep learning into CAD/CAE system: generative design
    and evaluation of 3D conceptual wheel", Struct. Multidisc. Optim. 64 (2021)
    2725-2747.

The paper samples 2D wheel designs from a learned latent space whose encoded
training data are *normally* distributed (not uniform).  It therefore uses
``LHSnorm`` -- Latin hypercube sampling transformed to a normal distribution --
rather than the plain uniform LHS, "because a good wheel-shaped image can be
sampled when we used the LHSnorm and not the LHS".

This differs from :mod:`exploration.datacon_designspace_sampler`, whose
stratified sampler spreads uniformly over ``[low, high]`` ranges.  Here each
dimension is stratified in *probability* space and then mapped through the
inverse normal CDF so the marginal is Gaussian with a per-dimension mean and
standard deviation.

Algorithm (per dimension, ``n`` samples):

    1. Partition ``[0, 1]`` into ``n`` equal probability strata.
    2. Draw one uniform value inside each stratum (jitter), giving one
       probability per stratum.
    3. Map each probability through the inverse standard-normal CDF (probit)
       and scale/shift by the dimension's ``(mean, std)``.
    4. Permute the stratum order independently per dimension so the columns are
       decorrelated.

All randomness flows through ``random.Random(seed)`` so results are
deterministic given a seed.  Stdlib-only (``math``, ``random``).
"""

from __future__ import annotations

import math
import random
from typing import List, Sequence, Tuple


# ---------------------------------------------------------------------------
# Inverse standard-normal CDF (probit) -- Acklam's rational approximation.
# Relative error < 1.15e-9 over the whole domain.
# ---------------------------------------------------------------------------
_A = (
    -3.969683028665376e+01, 2.209460984245205e+02, -2.759285104469687e+02,
    1.383577518672690e+02, -3.066479806614716e+01, 2.506628277459239e+00,
)
_B = (
    -5.447609879822406e+01, 1.615858368580409e+02, -1.556989798598866e+02,
    6.680131188771972e+01, -1.328068155288572e+01,
)
_C = (
    -7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00,
    -2.549732539343734e+00, 4.374664141464968e+00, 2.938163982698783e+00,
)
_D = (
    7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e+00,
    3.754408661907416e+00,
)
_P_LOW = 0.02425
_P_HIGH = 1.0 - _P_LOW


def inverse_normal_cdf(p: float) -> float:
    """Inverse standard-normal CDF (probit) for ``0 < p < 1``.

    Raises ``ValueError`` for ``p`` outside the open interval ``(0, 1)``.
    """
    if not (0.0 < p < 1.0):
        raise ValueError("p must be in the open interval (0, 1)")
    if p < _P_LOW:
        q = math.sqrt(-2.0 * math.log(p))
        return (((((_C[0] * q + _C[1]) * q + _C[2]) * q + _C[3]) * q + _C[4]) * q + _C[5]) / \
               ((((_D[0] * q + _D[1]) * q + _D[2]) * q + _D[3]) * q + 1.0)
    if p <= _P_HIGH:
        q = p - 0.5
        r = q * q
        return (((((_A[0] * r + _A[1]) * r + _A[2]) * r + _A[3]) * r + _A[4]) * r + _A[5]) * q / \
               (((((_B[0] * r + _B[1]) * r + _B[2]) * r + _B[3]) * r + _B[4]) * r + 1.0)
    q = math.sqrt(-2.0 * math.log(1.0 - p))
    return -(((((_C[0] * q + _C[1]) * q + _C[2]) * q + _C[3]) * q + _C[4]) * q + _C[5]) / \
            ((((_D[0] * q + _D[1]) * q + _D[2]) * q + _D[3]) * q + 1.0)


def _stratified_probabilities(n: int, rng: random.Random) -> List[float]:
    """Return one jittered probability per equal ``[0, 1]`` stratum."""
    out = []
    for i in range(n):
        out.append((i + rng.random()) / n)
    return out


def lhs_normal(
    n: int,
    means: Sequence[float],
    stds: Sequence[float],
    seed: int,
) -> List[List[float]]:
    """Latin-hypercube sample ``n`` points from independent normals (LHSnorm).

    ``means`` and ``stds`` give the per-dimension normal parameters (equal
    length = dimensionality ``d``).  Returns an ``n x d`` list of rows.  Each
    marginal is Latin-hypercube stratified in probability space and mapped
    through the inverse normal CDF, then permuted per dimension.  Deterministic
    given ``seed``.
    """
    if n < 0:
        raise ValueError("n must be >= 0")
    if len(means) != len(stds):
        raise ValueError("means and stds must have equal length")
    d = len(means)
    if d == 0:
        raise ValueError("need at least one dimension")
    for s in stds:
        if s <= 0.0:
            raise ValueError("all standard deviations must be positive")
    if n == 0:
        return []

    rng = random.Random(seed)
    columns: List[List[float]] = []
    for j in range(d):
        probs = _stratified_probabilities(n, rng)
        rng.shuffle(probs)
        col = [means[j] + stds[j] * inverse_normal_cdf(p) for p in probs]
        columns.append(col)

    return [[columns[j][i] for j in range(d)] for i in range(n)]


def lhs_standard_normal(n: int, d: int, seed: int) -> List[List[float]]:
    """LHSnorm with zero mean and unit variance in ``d`` dimensions."""
    return lhs_normal(n, [0.0] * d, [1.0] * d, seed)


def column_stats(samples: Sequence[Sequence[float]]) -> List[Tuple[float, float]]:
    """Return per-column ``(mean, std)`` (population std) of a sample matrix.

    Useful for verifying that an LHSnorm sample reproduces the requested normal
    marginals.  Raises ``ValueError`` for empty input.
    """
    if not samples:
        raise ValueError("no samples")
    n = len(samples)
    d = len(samples[0])
    stats: List[Tuple[float, float]] = []
    for j in range(d):
        col = [row[j] for row in samples]
        mean = math.fsum(col) / n
        var = math.fsum((v - mean) ** 2 for v in col) / n
        stats.append((mean, math.sqrt(var)))
    return stats
