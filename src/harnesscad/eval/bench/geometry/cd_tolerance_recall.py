"""Chamfer-Distance Tolerance-Recall curve and AUC-TR (IterCAD).

Mined from *IterCAD: An Iterative Multimodal Agent for Visually-Grounded CAD
Generation and Editing*. IterCAD observes that standard geometric metrics suffer a
"survivor bias": Chamfer distance is averaged **only over samples whose code
executed**, so a method that crashes on hard cases can look artificially precise.
IterCAD fixes this with the **Chamfer Distance Tolerance-Recall (CD-TR)** curve and
its **AUC-TR** summary, which count non-executing samples as failures at every
tolerance.

For a set of generation attempts, each either failed to execute or produced a
Chamfer distance ``cd``:

*   ``recall(tau)`` = fraction of **all** attempts (failures included) with a valid
    ``cd <= tau``.
*   The CD-TR curve is ``recall(tau)`` swept over ``tau in [0, tau_max]``.
*   ``AUC-TR`` is the normalised area under that curve (trapezoidal), in ``[0, 1]``.

Because failures never satisfy any tolerance, a method cannot inflate AUC-TR by
refusing hard cases. Deterministic and stdlib-only.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

__all__ = [
    "Attempt",
    "recall_at_tolerance",
    "cd_tolerance_recall_curve",
    "auc_tr",
]


@dataclass(frozen=True)
class Attempt:
    """One generation attempt.

    ``executed`` marks whether the generated code ran to a valid model. ``cd`` is
    the Chamfer distance when it did (ignored/optional otherwise). A non-executing
    attempt counts as a failure at every tolerance.
    """

    executed: bool
    cd: Optional[float] = None

    def __post_init__(self) -> None:
        if self.executed and self.cd is None:
            raise ValueError("an executed attempt must carry a Chamfer distance")
        if self.cd is not None and self.cd < 0:
            raise ValueError("cd must be non-negative")

    def satisfies(self, tau: float) -> bool:
        return self.executed and self.cd is not None and self.cd <= tau


def recall_at_tolerance(attempts: Sequence[Attempt], tau: float) -> float:
    """Fraction of *all* attempts with a valid Chamfer distance ``<= tau``."""
    if not attempts:
        raise ValueError("need at least one attempt")
    if tau < 0:
        raise ValueError("tau must be non-negative")
    return sum(1 for a in attempts if a.satisfies(tau)) / len(attempts)


def cd_tolerance_recall_curve(
    attempts: Sequence[Attempt], tau_max: float, steps: int = 100
) -> List[Tuple[float, float]]:
    """The CD-TR curve as ``(tau, recall)`` pairs over ``[0, tau_max]``.

    ``steps`` uniform samples yield ``steps + 1`` points (both endpoints included).
    """
    if tau_max <= 0:
        raise ValueError("tau_max must be positive")
    if steps < 1:
        raise ValueError("steps must be >= 1")
    out: List[Tuple[float, float]] = []
    for i in range(steps + 1):
        tau = tau_max * i / steps
        out.append((tau, recall_at_tolerance(attempts, tau)))
    return out


def auc_tr(
    attempts: Sequence[Attempt], tau_max: float, steps: int = 100
) -> float:
    """Normalised area under the CD-TR curve (trapezoidal), in ``[0, 1]``.

    Normalisation divides the raw area by ``tau_max`` so a method that hits recall
    ``1.0`` at ``tau = 0`` scores ``1.0`` and one that never executes scores ``0.0``.
    """
    curve = cd_tolerance_recall_curve(attempts, tau_max, steps)
    area = 0.0
    for (t0, r0), (t1, r1) in zip(curve, curve[1:]):
        area += (r0 + r1) / 2.0 * (t1 - t0)
    return area / tau_max
