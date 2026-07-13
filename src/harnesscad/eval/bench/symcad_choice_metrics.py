"""Evaluation metrics for 'choose one of N options to minimise a cost' heuristics.

Domain-agnostic protocol extracted from del Rio & England, *Lessons on Datasets
and Paradigms in Machine Learning for Symbolic Computation: A Case Study on CAD*
(Section 3.7). Their case study chooses a CAD *variable ordering* to minimise
build time, but the metrics judge any strategy that, per instance, must pick one
of several options given the (possibly ground-truth) cost of each. That covers
plenty of mechanical-CAD choices: picking a build order, a meshing parameter set,
a solver, or a fixturing plan to minimise time / cell-count / cost.

Each *instance* provides a vector ``costs`` -- the true cost of every option
(``None`` marks a timed-out/failed option whose cost is unknown). A *strategy*
picks an index per instance. The metrics:

* :func:`n_solved`      -- how many picks avoided a timeout (Section 3.7.1).
* :func:`choice_accuracy` -- fraction picking a truly-optimal option (3.7.2).
* :func:`total_cost`    -- summed realised cost, timeouts penalised (3.7.3/3.7.6).
* :func:`time_markup`   -- mean of ``(chosen - optimal) / (optimal + 1)`` (3.7.4):
  a scale-forgiving regret that neither over-penalises noise on cheap instances
  nor ignores it, with the paper's ``+1`` denominator guard.

Timeout penalisation (Section 3.7.6): an unknown (``None``) cost is scored as
``timeout_penalty_factor * time_limit`` (the paper uses twice the limit).

:func:`rank_select` is the paper's regression-to-rank decision rule (Sections 5
and 6): given *estimated* costs per option, pick the cheapest option that also
satisfies an optional feasibility predicate -- so an estimator can honour
constraints a plain classifier cannot (e.g. quantifier-block orderings).

Stdlib-only, deterministic (no wall clock, no randomness).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, List, Optional, Sequence

Cost = Optional[float]


def _resolve_cost(cost: Cost, *, time_limit: float, factor: float) -> float:
    if cost is None:
        return factor * time_limit
    if cost < 0:
        raise ValueError("costs must be non-negative (or None for timeout)")
    return float(cost)


def optimal_cost(costs: Sequence[Cost]) -> Cost:
    """The best (lowest) known cost, or ``None`` if every option timed out."""
    known = [c for c in costs if c is not None]
    return min(known) if known else None


def is_optimal_choice(costs: Sequence[Cost], choice: int, *, tol: float = 0.0) -> bool:
    best = optimal_cost(costs)
    picked = costs[choice]
    if best is None:
        return False
    if picked is None:
        return False
    return picked <= best + tol


def n_solved(instances: Sequence[Sequence[Cost]], choices: Sequence[int]) -> int:
    """Number of instances whose chosen option did not time out."""
    _check(instances, choices)
    return sum(1 for costs, c in zip(instances, choices) if costs[c] is not None)


def choice_accuracy(
    instances: Sequence[Sequence[Cost]],
    choices: Sequence[int],
    *,
    tol: float = 0.0,
) -> float:
    """Fraction of instances for which an optimal option was chosen."""
    _check(instances, choices)
    if not instances:
        return 0.0
    hits = sum(
        1
        for costs, c in zip(instances, choices)
        if is_optimal_choice(costs, c, tol=tol)
    )
    return hits / len(instances)


def total_cost(
    instances: Sequence[Sequence[Cost]],
    choices: Sequence[int],
    *,
    time_limit: float,
    timeout_penalty_factor: float = 2.0,
) -> float:
    """Summed realised cost, timed-out picks scored as factor * time_limit."""
    _check(instances, choices)
    if time_limit <= 0:
        raise ValueError("time_limit must be positive")
    return sum(
        _resolve_cost(costs[c], time_limit=time_limit, factor=timeout_penalty_factor)
        for costs, c in zip(instances, choices)
    )


def time_markup(
    instances: Sequence[Sequence[Cost]],
    choices: Sequence[int],
    *,
    time_limit: float,
    timeout_penalty_factor: float = 2.0,
) -> float:
    """Mean per-instance markup ``(chosen - optimal) / (optimal + 1)``.

    Instances where every option timed out (no known optimum) are skipped, since
    no meaningful baseline exists. Returns 0.0 if no instance is scorable.
    """
    _check(instances, choices)
    if time_limit <= 0:
        raise ValueError("time_limit must be positive")
    total = 0.0
    scored = 0
    for costs, c in zip(instances, choices):
        best = optimal_cost(costs)
        if best is None:
            continue
        chosen = _resolve_cost(
            costs[c], time_limit=time_limit, factor=timeout_penalty_factor
        )
        total += (chosen - best) / (best + 1.0)
        scored += 1
    return total / scored if scored else 0.0


@dataclass(frozen=True)
class StrategyReport:
    n_instances: int
    solved: int
    accuracy: float
    total_cost: float
    markup: float


def evaluate_strategy(
    instances: Sequence[Sequence[Cost]],
    choices: Sequence[int],
    *,
    time_limit: float,
    timeout_penalty_factor: float = 2.0,
    tol: float = 0.0,
) -> StrategyReport:
    """Bundle all Section 3.7 metrics into one report."""
    return StrategyReport(
        n_instances=len(instances),
        solved=n_solved(instances, choices),
        accuracy=choice_accuracy(instances, choices, tol=tol),
        total_cost=total_cost(
            instances,
            choices,
            time_limit=time_limit,
            timeout_penalty_factor=timeout_penalty_factor,
        ),
        markup=time_markup(
            instances,
            choices,
            time_limit=time_limit,
            timeout_penalty_factor=timeout_penalty_factor,
        ),
    )


def rank_select(
    estimates: Sequence[float],
    *,
    feasible: Optional[Callable[[int], bool]] = None,
) -> int:
    """Regression-to-rank decision rule: cheapest estimated *feasible* option.

    Implements the paper's Section 5/6 recasting: rather than classify the best
    option directly, estimate each option's cost and pick the minimum. Ties break
    to the lowest index (deterministic). When ``feasible`` is given, options for
    which it returns False are excluded first, so constraints (e.g. an ordering
    that must respect quantifier blocks) can be honoured -- something a plain
    classifier's single output cannot do.
    """
    if not estimates:
        raise ValueError("estimates must be non-empty")
    candidates = [
        i for i in range(len(estimates)) if feasible is None or feasible(i)
    ]
    if not candidates:
        raise ValueError("no feasible option")
    return min(candidates, key=lambda i: (estimates[i], i))


def strategy_choices(
    estimates_per_instance: Sequence[Sequence[float]],
    *,
    feasible: Optional[Callable[[int, int], bool]] = None,
) -> List[int]:
    """Apply :func:`rank_select` across a dataset of per-option estimate vectors.

    ``feasible(instance_index, option_index)`` optionally filters options.
    """
    out: List[int] = []
    for idx, estimates in enumerate(estimates_per_instance):
        pred = None if feasible is None else (lambda o, _i=idx: feasible(_i, o))
        out.append(rank_select(estimates, feasible=pred))
    return out


def _check(instances: Sequence[Sequence[Cost]], choices: Sequence[int]) -> None:
    if len(instances) != len(choices):
        raise ValueError("instances and choices must have equal length")
    for costs, c in zip(instances, choices):
        if not 0 <= c < len(costs):
            raise ValueError("choice index out of range for instance")
