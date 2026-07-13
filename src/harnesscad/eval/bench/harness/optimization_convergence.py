"""Convergence and diversity metrics for evolutionary design optimisation.

Paper: T. Rios, S. Menzel, B. Sendhoff, "Large Language and Text-to-3D Models
for Engineering Design Optimization" (Honda Research Institute Europe).

The paper's results section characterises the optimisation runs with a handful
of deterministic statistics computed from per-generation populations:

  * Mean population performance (cd) per generation with a 95% confidence band
    (Figs. 5, 10) -- "the translucent areas represent a confidence interval of
    95% based on the population data for each generation".
  * The minimum (best) cd per generation and its monotonic improvement under the
    elitist (mu + lambda) strategy (Fig. 10).
  * The "highly-oscillating population performance" the paper repeatedly notes
    for the (mu, lambda) runs -- quantified here as an oscillation index.
  * Convergence of the design parameters: "the variance of the WUP values
    decreases and stabilizes over the generations" (Fig. 11) -- a decreasing
    per-generation parameter-variance trajectory.

All metrics are pure functions over lists of numbers.  Deterministic, no
randomness, no wall clock.  Distinct from ``bench`` shape metrics (Chamfer,
etc.): this module scores the *optimisation trajectory*, not geometry.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Sequence

# z value for a two-sided 95% normal confidence interval.
Z_95 = 1.959963984540054


def _mean(xs: Sequence[float]) -> float:
    if not xs:
        raise ValueError("empty sequence")
    return sum(xs) / len(xs)


def _variance(xs: Sequence[float]) -> float:
    """Population variance (divisor n)."""
    n = len(xs)
    if n == 0:
        raise ValueError("empty sequence")
    m = sum(xs) / n
    return sum((x - m) ** 2 for x in xs) / n


def std(xs: Sequence[float]) -> float:
    return math.sqrt(_variance(xs))


@dataclass(frozen=True)
class GenerationStats:
    generation: int
    mean: float
    minimum: float
    maximum: float
    std: float
    ci95_low: float
    ci95_high: float


def generation_stats(generation: int,
                     population: Sequence[float]) -> GenerationStats:
    """Summary of one generation's population performance values."""
    if not population:
        raise ValueError("population is empty")
    n = len(population)
    m = _mean(population)
    s = std(population)
    # 95% CI of the mean: mean +/- z * std / sqrt(n).
    half = Z_95 * s / math.sqrt(n) if n > 0 else 0.0
    return GenerationStats(
        generation=generation,
        mean=m,
        minimum=min(population),
        maximum=max(population),
        std=s,
        ci95_low=m - half,
        ci95_high=m + half,
    )


def trajectory_stats(populations: Sequence[Sequence[float]]) -> List[GenerationStats]:
    """Per-generation stats for a whole run (index = generation)."""
    return [generation_stats(i, pop) for i, pop in enumerate(populations)]


def running_minimum(populations: Sequence[Sequence[float]]) -> List[float]:
    """Best-so-far minimum after each generation (elitist convergence curve)."""
    best = math.inf
    out: List[float] = []
    for pop in populations:
        if not pop:
            raise ValueError("empty generation")
        best = min(best, min(pop))
        out.append(best)
    return out


def oscillation_index(means: Sequence[float]) -> float:
    """Total-variation of the mean curve normalised by its net displacement.

    Sum of absolute successive changes divided by the span (max - min).  A
    smooth monotone curve scores ~1.0; a highly oscillating curve scores much
    higher (the paper's "highly-oscillating population performance").  Returns
    0.0 for a flat curve.
    """
    if len(means) < 2:
        return 0.0
    total_var = sum(abs(b - a) for a, b in zip(means, means[1:]))
    span = max(means) - min(means)
    if span == 0.0:
        return 0.0
    return total_var / span


def is_monotonic_non_increasing(values: Sequence[float],
                                tol: float = 1e-12) -> bool:
    """True if the curve never increases (elitist best-so-far property)."""
    return all(b <= a + tol for a, b in zip(values, values[1:]))


def parameter_variance_trajectory(
        parameter_populations: Sequence[Sequence[float]]) -> List[float]:
    """Per-generation variance of a design parameter across the population.

    A decreasing trajectory indicates the optimisation is converging (design
    parameters stabilising), as the paper observes for the WUP values under the
    elitist strategy.
    """
    return [_variance(pop) for pop in parameter_populations]


def has_converged(variances: Sequence[float], tol: float,
                  window: int = 3) -> bool:
    """True if the last ``window`` variances are all below ``tol``.

    A deterministic stand-in for the paper's "convergence of step size
    adaption" / parameter-variance stabilisation stopping criterion.
    """
    if window <= 0:
        raise ValueError("window must be positive")
    if len(variances) < window:
        return False
    return all(v <= tol for v in variances[-window:])
