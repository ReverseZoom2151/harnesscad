"""Constraint-Aware Editability Benchmark metrics.

Deterministic re-implementation of a constraint-aware, history-based CAD
editability benchmark. The checkable-metric contribution of that benchmark
design is the **Constraint-Aware Editability Benchmark**, which, for each
instance, applies a target parameter edit and reports three diagnostics that
separate *reaching* a valid edited state from *preserving* design intent:

*   **Edit Reachability (ER)** -- fraction of instances whose edited sequence
    reaches a valid CAD state.
*   **Conditional Preserved Constraint Satisfaction Rate (cPCSR)** -- among
    reachable instances, the mean fraction of the preservation-set constraints that
    remain satisfied after the edit.
*   **Overall Editable Success (OES)** = ER x cPCSR -- the strict overall rate.

The module also enumerates 19 explicit constraint types drawn from that design.
Everything is deterministic and stdlib-only.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Mapping, Sequence

__all__ = [
    "CONSTRAINT_TYPES",
    "EditInstance",
    "edit_reachability",
    "conditional_pcsr",
    "overall_editable_success",
    "evaluate",
]

#: The 19 explicit constraint types this benchmark design encodes.
CONSTRAINT_TYPES: tuple = (
    "coincident", "concentric", "collinear", "parallel", "perpendicular",
    "horizontal", "vertical", "tangent", "equal", "symmetric", "midpoint",
    "fix", "offset", "distance", "length", "angle", "radius", "diameter",
    "pattern",
)


@dataclass(frozen=True)
class EditInstance:
    """One benchmark instance: an applied edit and its preservation set.

    ``reached`` is whether the edited sequence executed to a valid state.
    ``preserved`` maps each required constraint to whether it stayed satisfied
    after the edit. When ``reached`` is False the ``preserved`` flags are ignored
    (cPCSR is conditional on reachability).
    """

    reached: bool
    preserved: Mapping[str, bool] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for name in self.preserved:
            if name not in CONSTRAINT_TYPES:
                raise ValueError(f"unknown constraint type {name!r}")

    def satisfaction_fraction(self) -> float:
        """Fraction of preservation-set constraints satisfied (1.0 if the set is empty)."""
        if not self.preserved:
            return 1.0
        sat = sum(1 for ok in self.preserved.values() if ok)
        return sat / len(self.preserved)


def edit_reachability(instances: Sequence[EditInstance]) -> float:
    """ER: fraction of instances that reach a valid edited CAD state."""
    if not instances:
        raise ValueError("need at least one instance")
    return sum(1 for i in instances if i.reached) / len(instances)


def conditional_pcsr(instances: Sequence[EditInstance]) -> float:
    """cPCSR: mean satisfied-constraint fraction over *reachable* instances.

    Returns 0.0 when no instance is reachable (nothing to preserve on).
    """
    reachable = [i for i in instances if i.reached]
    if not reachable:
        return 0.0
    return sum(i.satisfaction_fraction() for i in reachable) / len(reachable)


def overall_editable_success(instances: Sequence[EditInstance]) -> float:
    """OES = ER x cPCSR, the strict overall editability rate."""
    return edit_reachability(instances) * conditional_pcsr(instances)


def evaluate(instances: Sequence[EditInstance]) -> Dict[str, float]:
    """All three metrics as a dict: ``{"ER", "cPCSR", "OES"}``."""
    er = edit_reachability(instances)
    cpcsr = conditional_pcsr(instances)
    return {"ER": er, "cPCSR": cpcsr, "OES": er * cpcsr}
