"""Analytic CAE modal-frequency surrogate for conceptual road-wheel evaluation.

This module implements the *deterministic* structural-evaluation relations used
in the CAE-automation and design-evaluation stages of:

    Yoo, Lee, Kim, Hwang, Park & Kang, "Integrating deep learning into CAD/CAE
    system: generative design and evaluation of 3D conceptual wheel",
    Structural and Multidisciplinary Optimization 64 (2021) 2725-2747.

The paper trains a learned CNN surrogate (external) to predict the modal natural
frequency and mass of a wheel from a 2D disk-view image.  It does so, though, on
top of a small set of closed-form CAE relations that are fully deterministic and
locally reproducible.  This module implements those relations so a wheel concept
can be evaluated (and screened against a manufacturer's stiffness standard)
without either the learned surrogate or a real FEA solver.

Relations implemented (paper Section 5.1):

    * Equation (4) -- the single-degree-of-freedom natural-frequency relation::

          f = (1 / (2*pi)) * sqrt(k / m)

      the natural frequency ``f`` (Hz) is proportional to the square root of the
      structural stiffness ``k`` and inversely proportional to the square root of
      the mass ``m``.  Given any two of ``{f, k, m}`` the third follows.

    * Free-free modal taxonomy (Section 5.1) -- an unconstrained 3D wheel model
      has six rigid-body modes with zero frequency (three translations, three
      rotations).  From the seventh mode onward a non-zero elastic frequency
      appears.  The paper labels the low elastic modes::

          modes 7, 8   -> rim mode 1
          modes 9, 10  -> rim mode 2
          mode  11     -> spoke lateral mode   (the mode evaluated in the paper)
          modes 12, 13 -> rim mode 3
          modes 14, 15 -> spoke bending mode

    * Stiffness design constraint (Section 5.1) -- "manufacturers consider a
      lower bound of stiffness for each mode as a design constraint, based on the
      correlation between the stiffness for each mode and road noise".  A concept
      is screened by comparing its per-mode stiffness against that lower bound.

All functions are deterministic and depend only on the Python standard library
(``math``).  Units mirror the paper: frequency in Hz, mass in consistent mass
units, stiffness in ``force/length`` consistent units.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# Free-free modal taxonomy (Section 5.1).
# ---------------------------------------------------------------------------
RIGID_BODY_MODE_COUNT = 6

# Mode index (1-based) -> descriptive label for the low elastic modes.
MODE_LABELS: Dict[int, str] = {
    7: "rim mode 1",
    8: "rim mode 1",
    9: "rim mode 2",
    10: "rim mode 2",
    11: "spoke lateral mode",
    12: "rim mode 3",
    13: "rim mode 3",
    14: "spoke bending mode",
    15: "spoke bending mode",
}

# The mode the paper selects to evaluate spoke shape sensitivity.
LATERAL_MODE_INDEX = 11


def is_rigid_body_mode(mode_index: int) -> bool:
    """Return ``True`` if ``mode_index`` (1-based) is a zero-frequency rigid mode.

    An unconstrained (free-free) 3D model has ``RIGID_BODY_MODE_COUNT`` rigid
    body modes; these are modes 1..6.  Raises ``ValueError`` for a non-positive
    index.
    """
    if mode_index < 1:
        raise ValueError("mode_index must be a positive 1-based index")
    return mode_index <= RIGID_BODY_MODE_COUNT


def mode_label(mode_index: int) -> str:
    """Return the descriptive label for a 1-based ``mode_index``.

    Rigid-body modes (1..6) return ``"rigid body mode"``.  Modes with a paper
    label return that label; higher unlabeled elastic modes return
    ``"elastic mode <n>"``.  Raises ``ValueError`` for a non-positive index.
    """
    if mode_index < 1:
        raise ValueError("mode_index must be a positive 1-based index")
    if mode_index <= RIGID_BODY_MODE_COUNT:
        return "rigid body mode"
    return MODE_LABELS.get(mode_index, "elastic mode {0}".format(mode_index))


# ---------------------------------------------------------------------------
# Equation (4): natural-frequency / stiffness / mass relations.
# ---------------------------------------------------------------------------
def natural_frequency(stiffness: float, mass: float) -> float:
    """Natural frequency ``f = (1/(2*pi)) * sqrt(k/m)`` (paper equation 4).

    ``stiffness`` (k) and ``mass`` (m) must be positive.  Raises ``ValueError``
    otherwise.
    """
    if stiffness <= 0.0:
        raise ValueError("stiffness must be positive")
    if mass <= 0.0:
        raise ValueError("mass must be positive")
    return math.sqrt(stiffness / mass) / (2.0 * math.pi)


def stiffness_from_frequency(frequency: float, mass: float) -> float:
    """Invert equation (4) for stiffness: ``k = m * (2*pi*f)**2``.

    This is exactly how the paper recovers stiffness in the evaluation stage:
    "after the mass has been obtained from the 3D modeling, the stiffness can be
    calculated through the natural frequency and mass".  ``frequency`` and
    ``mass`` must be positive.
    """
    if frequency <= 0.0:
        raise ValueError("frequency must be positive")
    if mass <= 0.0:
        raise ValueError("mass must be positive")
    omega = 2.0 * math.pi * frequency
    return mass * omega * omega


def mass_from_frequency(frequency: float, stiffness: float) -> float:
    """Invert equation (4) for mass: ``m = k / (2*pi*f)**2``.

    ``frequency`` and ``stiffness`` must be positive.
    """
    if frequency <= 0.0:
        raise ValueError("frequency must be positive")
    if stiffness <= 0.0:
        raise ValueError("stiffness must be positive")
    omega = 2.0 * math.pi * frequency
    return stiffness / (omega * omega)


@dataclass
class WheelModalEvaluation:
    """Result of evaluating a wheel concept's lateral-mode structural response."""

    mass: float
    frequency: float
    stiffness: float
    mode_index: int
    mode_label: str
    meets_stiffness_floor: Optional[bool] = None
    stiffness_floor: Optional[float] = None


def evaluate_wheel(
    mass: float,
    frequency: float,
    mode_index: int = LATERAL_MODE_INDEX,
    stiffness_floor: Optional[float] = None,
) -> WheelModalEvaluation:
    """Evaluate a wheel concept given its mass and predicted modal frequency.

    Recovers the modal stiffness from equation (4), labels the evaluated mode,
    and (optionally) screens the stiffness against a manufacturer's lower-bound
    design constraint ``stiffness_floor`` (Section 5.1).  A concept with
    ``stiffness >= stiffness_floor`` passes (``meets_stiffness_floor is True``).
    """
    stiffness = stiffness_from_frequency(frequency, mass)
    meets: Optional[bool] = None
    if stiffness_floor is not None:
        meets = stiffness >= stiffness_floor
    return WheelModalEvaluation(
        mass=mass,
        frequency=frequency,
        stiffness=stiffness,
        mode_index=mode_index,
        mode_label=mode_label(mode_index),
        meets_stiffness_floor=meets,
        stiffness_floor=stiffness_floor,
    )


def screen_concepts(
    concepts: List[Tuple[float, float]],
    stiffness_floor: float,
    mode_index: int = LATERAL_MODE_INDEX,
) -> List[WheelModalEvaluation]:
    """Evaluate and screen a batch of ``(mass, frequency)`` concepts.

    Returns one :class:`WheelModalEvaluation` per concept (input order
    preserved), each screened against ``stiffness_floor``.  Use
    :func:`passing_concepts` to filter to those that meet the floor.
    """
    return [
        evaluate_wheel(mass, freq, mode_index=mode_index, stiffness_floor=stiffness_floor)
        for (mass, freq) in concepts
    ]


def passing_concepts(evaluations: List[WheelModalEvaluation]) -> List[WheelModalEvaluation]:
    """Filter evaluations to those meeting their stiffness floor, sorted by
    descending stiffness (stiffest concept first), mirroring the paper's practice
    of ranking candidates by stiffness."""
    passing = [e for e in evaluations if e.meets_stiffness_floor]
    return sorted(passing, key=lambda e: e.stiffness, reverse=True)
