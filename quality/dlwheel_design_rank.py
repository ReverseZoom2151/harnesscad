"""Wheel-concept diversity filtering and stiffness-based candidate ranking.

Deterministic design-evaluation pieces of the generative-design and
visualization/analysis stages (Sections 4.1 and 5.2.3) of:

    Yoo et al., "Integrating deep learning into CAD/CAE system: generative design
    and evaluation of 3D conceptual wheel", Struct. Multidisc. Optim. 64 (2021)
    2725-2747.

The learned generator produces many near-duplicate 2D wheel designs, and the
learned surrogate predicts each concept's frequency and mass.  Two deterministic
screening operations sit around those learned components:

    * **Pixelwise L1 de-duplication** (Section 4.1).  The paper computes the
      pixelwise L1 distance between all generated designs and removes designs
      within a distance threshold, keeping only sufficiently distinct shapes
      ("we calculated the pixelwise L1 distance for all designs and removed the
      designs that are within the threshold").  This yields a diverse subset.

    * **Stiffness ranking and elimination** (Section 5.2.3).  Concepts are
      ranked "in the order of highest stiffness", and "an automaker has its own
      stiffness standard, and unsatisfactory designs can be eliminated on the
      basis of this value".

The stiffness is recovered from the predicted frequency and mass via the modal
relation (see :mod:`verifiers.dlwheel_modal`); this module keeps that dependency
optional by accepting a precomputed stiffness or computing it inline.

All functions are deterministic and stdlib-only (``math``).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, List, Sequence, Tuple


# ---------------------------------------------------------------------------
# Pixelwise L1 de-duplication (Section 4.1).
# ---------------------------------------------------------------------------
def l1_distance(a: Sequence[float], b: Sequence[float]) -> float:
    """Sum of absolute differences between two equal-length flat images."""
    if len(a) != len(b):
        raise ValueError("images must have equal length")
    return math.fsum(abs(float(x) - float(y)) for x, y in zip(a, b))


def flatten(image: Sequence[Sequence[float]]) -> List[float]:
    """Flatten a 2D image (row-major) into a 1D list of pixels."""
    return [float(v) for row in image for v in row]


def deduplicate_l1(
    designs: Sequence[Sequence[float]],
    threshold: float,
) -> List[int]:
    """Greedily keep designs that are at least ``threshold`` apart in L1.

    Iterates in input order; a design is kept only if its L1 distance to every
    already-kept design is ``> threshold``.  Returns the list of kept indices
    (input order).  Deterministic.  ``threshold`` must be non-negative.
    """
    if threshold < 0.0:
        raise ValueError("threshold must be non-negative")
    kept: List[int] = []
    kept_vecs: List[Sequence[float]] = []
    for i, design in enumerate(designs):
        distinct = True
        for kv in kept_vecs:
            if l1_distance(design, kv) <= threshold:
                distinct = False
                break
        if distinct:
            kept.append(i)
            kept_vecs.append(design)
    return kept


def mean_pairwise_l1(designs: Sequence[Sequence[float]]) -> float:
    """Average pairwise L1 distance -- a diversity score (Section 4.3).

    Higher means the set is more spread out.  Returns 0.0 for fewer than two
    designs.
    """
    n = len(designs)
    if n < 2:
        return 0.0
    total = 0.0
    count = 0
    for i in range(n):
        for j in range(i + 1, n):
            total += l1_distance(designs[i], designs[j])
            count += 1
    return total / count


# ---------------------------------------------------------------------------
# Stiffness ranking and elimination (Section 5.2.3).
# ---------------------------------------------------------------------------
@dataclass
class RankedConcept:
    """A wheel concept scored by recovered modal stiffness."""

    index: int
    mass: float
    frequency: float
    stiffness: float
    meets_standard: bool


def _stiffness(frequency: float, mass: float) -> float:
    if frequency <= 0.0 or mass <= 0.0:
        raise ValueError("frequency and mass must be positive")
    omega = 2.0 * math.pi * frequency
    return mass * omega * omega


def rank_by_stiffness(
    concepts: Sequence[Tuple[float, float]],
    stiffness_standard: float,
) -> List[RankedConcept]:
    """Rank ``(mass, frequency)`` concepts by descending recovered stiffness.

    Each concept's stiffness is recovered from ``k = m * (2*pi*f)**2`` and
    compared against ``stiffness_standard``.  Returns a list of
    :class:`RankedConcept` sorted stiffest-first; ties break by original index
    (stable, deterministic).
    """
    ranked: List[RankedConcept] = []
    for i, (mass, freq) in enumerate(concepts):
        k = _stiffness(freq, mass)
        ranked.append(
            RankedConcept(
                index=i,
                mass=mass,
                frequency=freq,
                stiffness=k,
                meets_standard=k >= stiffness_standard,
            )
        )
    ranked.sort(key=lambda c: (-c.stiffness, c.index))
    return ranked


def eliminate_below_standard(ranked: Sequence[RankedConcept]) -> List[RankedConcept]:
    """Keep only concepts meeting the stiffness standard, order preserved."""
    return [c for c in ranked if c.meets_standard]


def top_k(ranked: Sequence[RankedConcept], k: int) -> List[RankedConcept]:
    """Return the ``k`` stiffest concepts (``ranked`` assumed already sorted)."""
    if k < 0:
        raise ValueError("k must be >= 0")
    return list(ranked[:k])
