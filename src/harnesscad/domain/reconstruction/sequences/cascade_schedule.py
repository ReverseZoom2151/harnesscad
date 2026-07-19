"""Cascade stage schema and masked-autoregressive reveal schedule.

A B-Rep is generated with a *cascade* of two masked autoregressive networks that
embed the "edges contour surfaces" prior:

  1. the **edge** stage generates the edge tokens from the condition embedding;
  2. the **surface** stage generates the surface tokens conditioned on the
     condition embedding *and* the generated edges.

Each stage follows the standard masked-autoregressive recipe: at inference a
subset of tokens is revealed per step following a cosine schedule, generating
one token at a time when the number of sampling steps equals the sequence
length. Typical step budgets are 64/32, 32/16, 16/8, and so on.

The neural transformer + diffusion decoder are external. What is deterministic --
and implemented here -- is (a) the cascade stage graph and its dependency
ordering, and (b) the cosine reveal schedule plus the seeded reveal order.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
import random


@dataclass(frozen=True)
class Stage:
    name: str
    depends_on: tuple[str, ...] = ()


def cascade_stages() -> tuple[Stage, ...]:
    """The cascade: edge first, then surface conditioned on edges."""
    return (Stage("edge", ()), Stage("surface", ("edge",)))


def validate_stage_order(order: tuple[Stage, ...]) -> bool:
    """True iff every stage appears after all of its dependencies."""
    seen: set[str] = set()
    names = [s.name for s in order]
    if len(names) != len(set(names)):
        return False
    for stage in order:
        if any(dep not in seen for dep in stage.depends_on):
            return False
        seen.add(stage.name)
    return True


def cosine_reveal_counts(n: int, steps: int) -> tuple[int, ...]:
    """How many tokens to reveal at each of ``steps`` steps for ``n`` tokens.

    Follows the cosine reveal schedule: the masked fraction after step
    ``i`` (1-based) is ``cos(pi/2 * i/steps)``, so the cumulative number of
    revealed tokens is ``round(n * (1 - cos(pi/2 * i/steps)))``. As ``steps <= n``
    always holds, every step reveals at least one token and enough tokens are
    held back for the remaining steps; per-step counts sum to ``n``. With
    ``steps == n`` every clamp collapses to exactly one token per step, the
    default of generating one token at a time.
    """
    if n < 0:
        raise ValueError("n must be non-negative")
    if steps <= 0:
        raise ValueError("steps must be positive")
    if steps > n and n != 0:
        raise ValueError("steps must not exceed the number of tokens")
    if n == 0:
        return tuple(0 for _ in range(steps))
    counts: list[int] = []
    revealed = 0
    for i in range(1, steps + 1):
        cumulative = round(n * (1.0 - math.cos(math.pi / 2 * i / steps)))
        # at least one token per step so far, and leave one per remaining step
        cumulative = max(i, min(n - (steps - i), cumulative))
        counts.append(cumulative - revealed)
        revealed = cumulative
    return tuple(counts)


def reveal_order(n: int, seed: int = 0) -> tuple[int, ...]:
    """A seeded permutation giving the order in which token slots are revealed."""
    order = list(range(n))
    random.Random(seed).shuffle(order)
    return tuple(order)


def mar_schedule(n: int, steps: int, seed: int = 0) -> tuple[tuple[int, ...], ...]:
    """Group the seeded reveal order into per-step batches by the cosine counts.

    Returns a tuple with one entry per step, each a tuple of the token indices
    revealed at that step. With ``steps == n`` every step reveals exactly one
    token (the default of one token at a time).
    """
    counts = cosine_reveal_counts(n, steps)
    order = reveal_order(n, seed)
    schedule: list[tuple[int, ...]] = []
    cursor = 0
    for count in counts:
        schedule.append(tuple(order[cursor:cursor + count]))
        cursor += count
    return tuple(schedule)
