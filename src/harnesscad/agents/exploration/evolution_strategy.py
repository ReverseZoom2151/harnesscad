"""Self-adaptive (mu, lambda) / (mu + lambda) evolution strategy optimiser.

Paper: T. Rios, S. Menzel, B. Sendhoff, "Large Language and Text-to-3D Models
for Engineering Design Optimization" (Honda Research Institute Europe).

Section III-B uses CMA-ES to optimise a real-valued design vector that a
text-to-3D model turns into a car mesh, then minimises the CFD drag coefficient.
The paper reports two selection schemes and compares them (Fig. 10):

  * (mu, lambda): non-elitist -- parents chosen only from the lambda offspring
    of the current generation.  Oscillates but explores.
  * (mu + lambda): elitist -- parents chosen from the union of parents and
    offspring, so the best-so-far is never lost.  Converges to lower cd.

Settings from the paper (Sec. III-B / IV-B): lambda = 10, mu = 3, up to 100
iterations, continuous variables, and for tokenisation the sampled values are
rounded to the nearest integer and clamped to a token range.

CMA-ES itself (full covariance adaptation) is a large algorithm; here we
implement the surrounding, deterministic *evolution-strategy skeleton* with
self-adaptive isotropic step-size control (the derandomised sigma path idea of
Hansen & Ostermeier that CMA-ES generalises).  This is deliberately distinct
from ``exploration/evocad_evolution.py`` (rank-based *genetic* evolution of CAD
*programs* with LLM crossover/mutation): here individuals are real vectors, the
operators are Gaussian mutation with a self-adapted global step size, and
selection is truncation over either (mu, lambda) or (mu + lambda).

The objective is an injected callable (the paper's Shap-E + OpenFOAM pipeline is
external); tests use a plain quadratic and the drag proxy.  Determinism: every
draw flows through a single ``random.Random(seed)``.
"""
from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import Callable, List, Optional, Sequence, Tuple

Objective = Callable[[Sequence[float]], float]


@dataclass(frozen=True)
class Individual:
    genome: Tuple[float, ...]
    fitness: float


@dataclass
class ESResult:
    best: Individual
    history_best: List[float]          # best fitness after each generation
    history_mean: List[float]          # mean fitness of the mu parents
    history_sigma: List[float]         # global step size per generation
    generations: int
    converged: bool                    # True if step-size convergence hit


def _clamp_round(value: float, bounds: Optional[Tuple[float, float]],
                 integer: bool) -> float:
    if integer:
        value = float(round(value))
    if bounds is not None:
        lo, hi = bounds
        if value < lo:
            value = lo
        elif value > hi:
            value = hi
        if integer:
            value = float(round(value))
    return value


def _apply_bounds(genome: Sequence[float],
                  bounds: Optional[Tuple[float, float]],
                  integer: bool) -> Tuple[float, ...]:
    return tuple(_clamp_round(g, bounds, integer) for g in genome)


def optimise(
    objective: Objective,
    x0: Sequence[float],
    *,
    seed: int,
    mu: int = 3,
    lam: int = 10,
    max_generations: int = 100,
    sigma0: float = 1.0,
    plus_selection: bool = False,
    bounds: Optional[Tuple[float, float]] = None,
    integer: bool = False,
    sigma_tol: float = 1e-8,
    minimise: bool = True,
) -> ESResult:
    """Run a self-adaptive ES.

    ``plus_selection=False`` -> (mu, lambda); ``True`` -> (mu + lambda).
    ``integer`` + ``bounds`` reproduce the tokenisation encoding (round to
    nearest integer, clamp to token range).  Stops on ``max_generations`` or
    when the global step size shrinks below ``sigma_tol`` (the paper's
    "convergence of step size adaption" criterion).
    """
    if not (1 <= mu <= lam):
        raise ValueError("require 1 <= mu <= lam")
    rng = random.Random(seed)
    dim = len(x0)
    if dim == 0:
        raise ValueError("x0 must be non-empty")

    sign = 1.0 if minimise else -1.0

    def evaluate(genome: Tuple[float, ...]) -> Individual:
        return Individual(genome=genome, fitness=objective(genome))

    def better(a: Individual, b: Individual) -> bool:
        return sign * a.fitness < sign * b.fitness

    # Learning rate for log-normal self-adaptation of the global step size.
    tau = 1.0 / math.sqrt(2.0 * dim)

    start = _apply_bounds(x0, bounds, integer)
    parents: List[Individual] = [evaluate(start) for _ in range(mu)]
    sigma = float(sigma0)

    best = min(parents, key=lambda ind: sign * ind.fitness)
    history_best: List[float] = []
    history_mean: List[float] = []
    history_sigma: List[float] = []
    converged = False
    gens_run = 0

    for _ in range(max_generations):
        gens_run += 1
        # Each parent's mean is the current best-parent centroid? The paper
        # samples offspring around parents; we recombine the mu parents into a
        # centroid mean (intermediate recombination) as CMA-ES does.
        centroid = tuple(
            sum(p.genome[i] for p in parents) / mu for i in range(dim)
        )
        offspring: List[Individual] = []
        for _ in range(lam):
            child_sigma = sigma * math.exp(tau * rng.gauss(0.0, 1.0))
            genome = tuple(
                centroid[i] + child_sigma * rng.gauss(0.0, 1.0)
                for i in range(dim)
            )
            genome = _apply_bounds(genome, bounds, integer)
            offspring.append(evaluate(genome))
            # remember the step that produced each child for sigma update
            offspring[-1] = Individual(genome=genome, fitness=offspring[-1].fitness)

        pool = offspring + parents if plus_selection else offspring
        pool.sort(key=lambda ind: sign * ind.fitness)
        parents = pool[:mu]

        # Adapt the global step size toward the mean magnitude of the surviving
        # deviation from the centroid (a simple, deterministic proxy for the
        # CMA-ES cumulative step-size path).
        mean_dev = 0.0
        for p in parents:
            mean_dev += math.sqrt(
                sum((p.genome[i] - centroid[i]) ** 2 for i in range(dim))
            )
        mean_dev /= mu
        sigma = max(mean_dev / math.sqrt(dim), sigma * 0.5)

        gen_best = parents[0]
        if better(gen_best, best):
            best = gen_best
        history_best.append(best.fitness)
        history_mean.append(sum(p.fitness for p in parents) / mu)
        history_sigma.append(sigma)

        if sigma < sigma_tol:
            converged = True
            break

    return ESResult(
        best=best,
        history_best=history_best,
        history_mean=history_mean,
        history_sigma=history_sigma,
        generations=gens_run,
        converged=converged,
    )
