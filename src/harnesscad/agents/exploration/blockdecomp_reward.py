"""Deterministic quality-based reward for RL block decomposition.

From *Reinforcement Learning for Block Decomposition of CAD Models* (DiPrete
et al., AAAI-2022), Sec. "Reward Function" and Eq. (1). The reward is a critical,
fully deterministic component (the *learned* policy consumes it but does not
define it). It is designed to (Sec. "Reward Function"):

  * encourage creating quadrilateral parts,
  * discourage cuts that do not affect the model (e.g. cutting along a side),
  * discourage high variance in the areas of the decomposed parts,
  * discourage high aspect ratios in the decomposed parts.

Given a splitting action producing ``N`` parts, ``Nq`` of them quads, with areas
``A_i`` (mean ``Abar``) and aspect ratios ``R_i``, the paper's reward combines:

  * an **aspect term** ``(1/N) * sum(1/R_i)`` -- since ``R_i >= 1`` this is at
    most 1, attained when every part is a square;
  * an **area-variance term** ``sqrt(sum (A_i - Abar)^2) / sum A_i`` -- at
    minimum 0 when all areas are equal (subtracted);
  * a **quad term** ``Nq / N`` -- maximal when the action yields all quads;
  * a **no-effect penalty** ``p(N)`` = 1 when ``N == 1`` (no new shapes), else 0.

The OCR of Eq. (1) is partially garbled in the source; we reconstruct it from the
stated monotonic properties ("maximum reward when the action cuts the shape into
all squares of equal area") as a weighted combination with the paper's visible
constants, exposing every component and the weights so the exact form is
transparent and adjustable.

Pure stdlib; deterministic (no randomness, no wall clock).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

from harnesscad.domain.geometry.blockdecomp_domain import Shape
from harnesscad.domain.geometry.blockdecomp_quality import area_variance_ratio, quad_fraction

_EPS = 1e-9


def aspect_term(parts: Sequence[Shape]) -> float:
    """(1/N) * sum(1/R_i): at most 1, equal to 1 iff every part is a square."""
    if not parts:
        return 0.0
    total = 0.0
    for p in parts:
        r = p.aspect_ratio()
        total += 0.0 if math.isinf(r) else 1.0 / r
    return total / len(parts)


def variance_term(parts: Sequence[Shape]) -> float:
    """sqrt(sum (A_i - Abar)^2) / sum A_i: >= 0, zero iff all areas equal."""
    return area_variance_ratio(parts)


def quad_term(parts: Sequence[Shape]) -> float:
    """Nq / N: fraction of parts that are quadrilateral blocks."""
    return quad_fraction(parts)


def no_effect_penalty(parts: Sequence[Shape]) -> float:
    """p(N): 1 when the action created no new shapes (N == 1), else 0."""
    return 1.0 if len(parts) <= 1 else 0.0


@dataclass(frozen=True)
class RewardComponents:
    """The individual reward terms plus the combined scalar reward."""

    aspect: float
    variance: float
    quad: float
    penalty: float
    total: float


def reward(
    parts: Sequence[Shape],
    quad_weight: float = 10.0,
    penalty_weight: float = 5.0,
    scale: float = 1.0 / 3.0,
    offset: float = 1.0,
) -> RewardComponents:
    """Deterministic block-decomposition reward for a cut's resulting parts.

    Combines the four terms following Eq. (1)'s reconstructed form::

        total = scale * (aspect - variance + quad_weight * quad)
                - penalty_weight * penalty - offset

    Maximised when a cut yields all squares of equal area (aspect = quad = 1,
    variance = penalty = 0); an ineffective cut (single part) incurs the
    ``penalty_weight`` penalty.
    """
    a = aspect_term(parts)
    v = variance_term(parts)
    q = quad_term(parts)
    p = no_effect_penalty(parts)
    total = scale * (a - v + quad_weight * q) - penalty_weight * p - offset
    return RewardComponents(aspect=a, variance=v, quad=q, penalty=p, total=total)


def terminal_bonus(all_quads: bool, bonus: float = 10.0) -> float:
    """Episode-completion bonus once the model is fully decomposed into blocks.

    "Once the geometric model is fully decomposed into blocks, the agent gets a
    bonus reward and the episode concludes" (Sec. "Reward Function").
    """
    return bonus if all_quads else 0.0
