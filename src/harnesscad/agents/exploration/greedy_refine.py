"""Narrow-focus greedy refinement of sparsely-sampled seeds.

From Séquin, *Interactive Procedural Computer-Aided Design*, Sections 2 and 3.1.
The paper's recurring pattern is to combine a broad (stochastic) search with a
narrowly-focused greedy optimisation: "it is preferable to introduce an inner
loop with a continuous optimization procedure, such as gradient descent, to
fine-tune the ... parameters so as to obtain the desired ... behavior", and
(OPASYN) "it was possible to sparsely sample that solution space and then refine
these solutions with a gradient descent optimization to find all the local minima
that might be compatible with the specifications. If there is more than one such
'optimal' solution, they all get presented to the designer".

This module implements the deterministic greedy pieces:

* :func:`gradient_descent` -- finite-difference steepest descent inside an
  optional axis-aligned box (the "narrowly focused" domain);
* :func:`match_target` -- greedily drive a scalar response ``g(x)`` to a
  non-negotiable ``target`` (minimise ``(g(x) - target)**2``);
* :func:`multistart_local_optima` -- refine each of several seeds and return the
  *distinct* local optima found, so more than one can be presented to a designer.

Determinism: no wall clock; the only randomness (seed generation) uses an
explicit ``random.Random(seed)``.
"""

from __future__ import annotations

import random
from typing import Callable, List, Optional, Sequence, Tuple

Vector = Tuple[float, ...]
Objective = Callable[[Sequence[float]], float]
Bounds = Sequence[Tuple[float, float]]


def _clamp(x: Sequence[float], bounds: Optional[Bounds]) -> Vector:
    if bounds is None:
        return tuple(x)
    return tuple(min(hi, max(lo, xi)) for xi, (lo, hi) in zip(x, bounds))


def _gradient(f: Objective, x: Sequence[float], eps: float) -> Vector:
    """Central finite-difference gradient."""
    g: List[float] = []
    xl = list(x)
    for i in range(len(x)):
        orig = xl[i]
        xl[i] = orig + eps
        fp = f(xl)
        xl[i] = orig - eps
        fm = f(xl)
        xl[i] = orig
        g.append((fp - fm) / (2 * eps))
    return tuple(g)


def gradient_descent(
    f: Objective,
    x0: Sequence[float],
    *,
    step: float = 0.1,
    eps: float = 1e-6,
    iters: int = 500,
    bounds: Optional[Bounds] = None,
    tol: float = 1e-12,
) -> Tuple[Vector, float, int]:
    """Steepest descent with a fixed step, confined to ``bounds``.

    Returns ``(x_best, f_best, iterations_used)``. Stops early when the update
    is smaller than ``tol``. Deterministic.
    """
    x = _clamp(x0, bounds)
    fx = f(x)
    used = 0
    for used in range(1, iters + 1):
        grad = _gradient(f, x, eps)
        nxt = _clamp([xi - step * gi for xi, gi in zip(x, grad)], bounds)
        fn = f(nxt)
        if fn < fx:
            move = max(abs(a - b) for a, b in zip(x, nxt))
            x, fx = nxt, fn
            if move < tol:
                break
        else:
            # overshoot: shrink the step and retry
            step *= 0.5
            if step < tol:
                break
    return x, fx, used


def match_target(
    g: Objective,
    x0: Sequence[float],
    target: float,
    **kwargs,
) -> Tuple[Vector, float, int]:
    """Greedily drive scalar response ``g(x)`` to ``target``.

    Minimises ``(g(x) - target)**2``; returns ``(x, achieved_g, iters)``.
    """
    x, _, used = gradient_descent(lambda x: (g(x) - target) ** 2, x0, **kwargs)
    return x, g(x), used


def multistart_local_optima(
    f: Objective,
    bounds: Bounds,
    *,
    n_starts: int = 8,
    seed: int = 0,
    dedup_tol: float = 1e-3,
    **descent_kwargs,
) -> Tuple[Tuple[Vector, float], ...]:
    """Sparsely sample the box, refine each seed, return distinct local optima.

    Seeds are drawn deterministically from ``random.Random(seed)``. Results are
    sorted by objective value (best first); optima closer than ``dedup_tol`` in
    every coordinate are treated as the same basin and merged.
    """
    rng = random.Random(seed)
    found: List[Tuple[Vector, float]] = []
    for _ in range(n_starts):
        x0 = tuple(rng.uniform(lo, hi) for (lo, hi) in bounds)
        x, fx, _ = gradient_descent(f, x0, bounds=bounds, **descent_kwargs)
        if not _is_duplicate(x, found, dedup_tol):
            found.append((x, fx))
    found.sort(key=lambda pair: pair[1])
    return tuple(found)


def _is_duplicate(x: Vector, found: Sequence[Tuple[Vector, float]], tol: float) -> bool:
    for existing, _ in found:
        if all(abs(a - b) <= tol for a, b in zip(x, existing)):
            return True
    return False
