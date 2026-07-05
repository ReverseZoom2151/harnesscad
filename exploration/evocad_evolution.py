"""EvoCAD evolutionary CAD-code generation loop (Preintner et al., 2025).

This is the deterministic *evolutionary algorithm* skeleton of EvoCAD
(Algorithm 1): population initialisation, rank-based fitness, exponential
rank-to-probability selection (Eq. 1), weighted parent mating, probabilistic
mutation, single-elite generational replacement, and multi-repeat ranking
averaging.

It is deliberately **distinct** from ``exploration/tournament.py`` (Co-Scientist
Elo tournament): that layer *ranks* a fixed population by pairwise Elo debates;
this layer *evolves* the population across ``N`` generations, producing new CAD
programs by crossover + mutation and replacing the population each generation.
It is also distinct from ``reliability.strategies.best_of_n`` (single-shot draw).

Everything model-touching is an **injected seam** (research-heavy / external in
the paper): the VLM that describes rendered images and the RLM that ranks them
are folded into one deterministic ``ranker`` callable; the LM crossover and
mutation operators are ``crossover_op`` / ``mutation_op`` callables. The paper's
actual operators are prompt-driven LLM calls; ``exploration.evocad_variation``
provides a concrete deterministic program-level implementation for testing.

Paper parameters (Sec. IV-A): M=6, N=4, p_m=0.5, lambda=0.5, elites=1, and the
RLM ranking is repeated 3 times and averaged.

Determinism: no wall clock; all stochastic choices flow through a single
``random.Random(seed)`` threaded from a fixed seed.
"""
from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import Any, Callable, List, Sequence, Tuple

# ----- injected seams -------------------------------------------------------
# ranker(population, generation, repeat_index) -> list[int] ranks, one per
#   individual; 0 == best alignment, larger == worse (RLM "best to worst").
Ranker = Callable[[Sequence[Any], int, int], Sequence[float]]
# crossover_op(parent_a, parent_b, rng) -> child program.
CrossoverOp = Callable[[Any, Any, random.Random], Any]
# mutation_op(child, rng) -> mutated child program.
MutationOp = Callable[[Any, random.Random], Any]


# ----- fitness / selection primitives --------------------------------------

def average_rankings(rankings: Sequence[Sequence[float]], size: int) -> List[float]:
    """Average ``R_1..R_r`` rank vectors elementwise (Algorithm 1, line 11).

    Each ranking is a per-individual rank (0 == best). Repeating the RLM ranking
    and averaging reduces the variance of a single stochastic judgement.
    """
    if not rankings:
        raise ValueError("at least one ranking required")
    for r in rankings:
        if len(r) != size:
            raise ValueError("ranking length does not match population size")
    return [sum(r[i] for r in rankings) / len(rankings) for i in range(size)]


def ordering_to_ranks(order: Sequence[int], size: int) -> List[float]:
    """Convert an ordered id list (best-first) into a per-index rank vector.

    ``order`` lists individual indices from best alignment to worst; the result
    maps ``index -> rank`` where the best-aligned individual gets rank 0.
    """
    if sorted(order) != list(range(size)):
        raise ValueError("order must be a permutation of range(size)")
    ranks = [0.0] * size
    for rank, idx in enumerate(order):
        ranks[idx] = float(rank)
    return ranks


def rank_probabilities(avg_ranks: Sequence[float], lam: float) -> List[float]:
    """Exponential rank-to-probability distribution, EvoCAD Eq. 1.

    ``p(r_i) = exp(-lambda * r_i) / sum_j exp(-lambda * r_j)``. Lower ranks
    (better alignment) receive exponentially higher mating probability.
    """
    if lam < 0:
        raise ValueError("lambda must be non-negative")
    weights = [math.exp(-lam * r) for r in avg_ranks]
    total = sum(weights)
    if total <= 0.0:
        n = len(avg_ranks)
        return [1.0 / n] * n
    return [w / total for w in weights]


def select_parents(
    probabilities: Sequence[float],
    num_pairs: int,
    rng: random.Random,
    *,
    distinct: bool = True,
) -> List[Tuple[int, int]]:
    """Weighted-random selection of ``num_pairs`` mating pairs (Eq. 1 dist.).

    Draws parent indices proportional to ``probabilities``. With ``distinct``
    the two parents of a pair differ when the population allows it.
    """
    n = len(probabilities)
    if n == 0:
        raise ValueError("empty probability vector")
    indices = list(range(n))
    pairs: List[Tuple[int, int]] = []
    for _ in range(num_pairs):
        a = rng.choices(indices, weights=probabilities, k=1)[0]
        b = rng.choices(indices, weights=probabilities, k=1)[0]
        if distinct and n > 1:
            guard = 0
            while b == a and guard < 8:
                b = rng.choices(indices, weights=probabilities, k=1)[0]
                guard += 1
        pairs.append((a, b))
    return pairs


def elite_indices(avg_ranks: Sequence[float], num_elites: int) -> List[int]:
    """Indices of the ``num_elites`` best individuals (lowest average rank).

    Ties break by index for determinism. These move unmodified to the next
    generation (Sec. IV-A: elites=1) so the best object never deteriorates.
    """
    order = sorted(range(len(avg_ranks)), key=lambda i: (avg_ranks[i], i))
    return order[: max(0, num_elites)]


# ----- generation record ----------------------------------------------------

@dataclass(frozen=True)
class GenerationRecord:
    """Per-generation snapshot of the evolutionary run."""

    generation: int
    avg_ranks: Tuple[float, ...]
    probabilities: Tuple[float, ...]
    elite_indices: Tuple[int, ...]
    parent_pairs: Tuple[Tuple[int, int], ...]
    mutated_children: Tuple[int, ...]
    best_index: int


@dataclass
class EvoResult:
    """Outcome of :func:`evolve`."""

    best_program: Any
    best_avg_rank: float
    final_population: List[Any]
    history: List[GenerationRecord] = field(default_factory=list)


# ----- main loop ------------------------------------------------------------

def evolve(
    initial_population: Sequence[Any],
    ranker: Ranker,
    crossover_op: CrossoverOp,
    mutation_op: MutationOp,
    *,
    generations: int = 4,
    mutation_prob: float = 0.5,
    lam: float = 0.5,
    num_elites: int = 1,
    seed: int = 0,
    rank_repeats: int = 3,
) -> EvoResult:
    """Run EvoCAD Algorithm 1 over an already-initialised population.

    The population is fixed size ``M``. Each generation: rank ``rank_repeats``
    times and average; convert ranks to selection probabilities (Eq. 1); carry
    ``num_elites`` best unchanged; fill the rest by crossing over weighted
    parent pairs then mutating each child with probability ``mutation_prob``.
    """
    if generations < 0:
        raise ValueError("generations must be non-negative")
    if not (0.0 <= mutation_prob <= 1.0):
        raise ValueError("mutation_prob must be in [0, 1]")
    if not initial_population:
        raise ValueError("initial population is empty")

    rng = random.Random(seed)
    population: List[Any] = list(initial_population)
    m = len(population)
    num_elites = min(num_elites, m)
    history: List[GenerationRecord] = []

    def evaluate(pop: Sequence[Any], gen: int) -> List[float]:
        rankings = [list(ranker(pop, gen, rep)) for rep in range(rank_repeats)]
        return average_rankings(rankings, len(pop))

    avg = evaluate(population, 0)
    for gen in range(generations):
        probs = rank_probabilities(avg, lam)
        elites = elite_indices(avg, num_elites)
        elite_programs = [population[i] for i in elites]

        num_children = m - num_elites
        pairs = select_parents(probs, num_children, rng)
        children: List[Any] = []
        mutated: List[int] = []
        for slot, (pa, pb) in enumerate(pairs):
            child = crossover_op(population[pa], population[pb], rng)
            if rng.random() < mutation_prob:
                child = mutation_op(child, rng)
                mutated.append(slot)
            children.append(child)

        best = min(range(m), key=lambda i: (avg[i], i))
        history.append(GenerationRecord(
            generation=gen,
            avg_ranks=tuple(avg),
            probabilities=tuple(probs),
            elite_indices=tuple(elites),
            parent_pairs=tuple(pairs),
            mutated_children=tuple(mutated),
            best_index=best,
        ))

        population = elite_programs + children
        avg = evaluate(population, gen + 1)

    best = min(range(len(population)), key=lambda i: (avg[i], i))
    return EvoResult(
        best_program=population[best],
        best_avg_rank=avg[best],
        final_population=population,
        history=history,
    )
