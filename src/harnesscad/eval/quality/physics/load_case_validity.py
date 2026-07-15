"""Functional-validity scoring for load-bearing CAD designs (Physics-in-the-Loop).

Mined from *Physics-in-the-Loop: A Hybrid Agentic Architecture for Validated CAD
Engineering Design*. The paper's agent loop calls a trained VLM, but its
**objective** is a deterministic engineering criterion: a part is functionally
valid when its FEA safety factor lands in the engineering band ``[2.0, 5.0]``
(under-designed below 2.0, wasteful above 5.0), and among valid parts the one that
minimises volume is preferred.

This module ports that criterion:

*   :class:`LoadCase` -- fixed supports, applied forces, and the design-space
    bounding box (the structured input the paper conditions on).
*   :func:`in_safety_band` / :func:`functional_validity` -- the band membership and
    a graded validity score that peaks inside the band and decays outside it.
*   :func:`design_objective` -- a single score rewarding in-band validity while
    minimising volume relative to the design-space envelope.

Everything is deterministic and stdlib-only (no FEA is run; the safety factor and
volume are supplied by the caller).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Sequence, Tuple

__all__ = [
    "LoadCase",
    "SAFETY_BAND",
    "in_safety_band",
    "functional_validity",
    "design_objective",
]

#: The engineering-practice safety-factor band (Budynas & Nisbett).
SAFETY_BAND: Tuple[float, float] = (2.0, 5.0)

Vec3 = Tuple[float, float, float]


@dataclass(frozen=True)
class LoadCase:
    """A structured load case: supports, forces, and the design-space box."""

    fixed_supports: Tuple[Vec3, ...]
    forces: Tuple[Tuple[Vec3, Vec3], ...]  # (application point, force vector)
    design_space: Tuple[Vec3, Vec3]        # (min corner, max corner)

    def __post_init__(self) -> None:
        lo, hi = self.design_space
        if any(h <= l for l, h in zip(lo, hi)):
            raise ValueError("design_space max corner must exceed min corner")
        if not self.fixed_supports:
            raise ValueError("a load case needs at least one fixed support")
        if not self.forces:
            raise ValueError("a load case needs at least one applied force")

    def envelope_volume(self) -> float:
        """Volume of the design-space bounding box."""
        lo, hi = self.design_space
        return (hi[0] - lo[0]) * (hi[1] - lo[1]) * (hi[2] - lo[2])


def in_safety_band(
    safety_factor: float, band: Tuple[float, float] = SAFETY_BAND
) -> bool:
    """True iff ``safety_factor`` lies within the engineering band (inclusive)."""
    low, high = band
    return low <= safety_factor <= high


def functional_validity(
    safety_factor: float, band: Tuple[float, float] = SAFETY_BAND
) -> float:
    """Graded validity in ``[0, 1]``: 1.0 inside the band, linear decay outside.

    Below the band the score decays to 0 at ``sf = 0`` (structural failure). Above
    the band it decays to 0 at ``sf = 2*high`` (grossly over-engineered).
    """
    if safety_factor < 0:
        raise ValueError("safety factor must be non-negative")
    low, high = band
    if low <= safety_factor <= high:
        return 1.0
    if safety_factor < low:
        return max(0.0, safety_factor / low)
    # above the band
    return max(0.0, (2 * high - safety_factor) / high)


def design_objective(
    safety_factor: float,
    volume: float,
    load_case: LoadCase,
    band: Tuple[float, float] = SAFETY_BAND,
    volume_weight: float = 0.5,
) -> float:
    """Combined score: in-band validity, minus a volume penalty (both in ``[0,1]``).

    ``volume`` is the part's volume; it is normalised by the design-space envelope,
    so a lighter valid part scores higher. An out-of-band part gets no volume
    credit -- validity gates the reward, matching the paper's "structurally sound
    first, then minimal" objective.
    """
    if not 0.0 <= volume_weight <= 1.0:
        raise ValueError("volume_weight must be in [0, 1]")
    if volume < 0:
        raise ValueError("volume must be non-negative")
    validity = functional_validity(safety_factor, band)
    if validity == 0.0:
        return 0.0
    envelope = load_case.envelope_volume()
    fill = min(1.0, volume / envelope) if envelope > 0 else 1.0
    volume_reward = 1.0 - fill
    return (1.0 - volume_weight) * validity + volume_weight * validity * volume_reward
