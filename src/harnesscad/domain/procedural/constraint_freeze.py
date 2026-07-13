"""Freezing marked regions as constraints for a re-optimisation.

From Séquin, *Interactive Procedural Computer-Aided Design*, Section 3.2
(placement & routing). When a small change is made at the periphery of a layout,
"good partial designs in other regions of the layout are lost forever". The paper
proposes a UI in which the designer can:

* mark some areas of the current solution as **intangible**, so "their current
  perimeter must be added as a further constraint for the remaining layout task";
* mark partial features as **highly desirable**, so they "should not get lost"
  during the subsequent stochastic optimisation.

This module implements the deterministic bookkeeping behind those interactions.
A *solution* is a mapping ``variable -> value``. The designer freezes named
regions (sets of variables); this module produces the derived equality
constraints, splits variables into frozen vs. free, and checks whether a proposed
new solution preserves everything that was locked -- reporting exact violations.

Pure stdlib, deterministic.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Mapping, Set, Tuple

Solution = Mapping[str, float]


@dataclass(frozen=True)
class Violation:
    """A frozen variable whose value changed in a proposed solution."""

    variable: str
    expected: float
    actual: float
    kind: str  # "intangible" or "desirable"


@dataclass
class ConstraintSet:
    """Accumulated freeze/desirable constraints derived from designer marks."""

    # variable -> locked value (must not change at all)
    intangible: Dict[str, float] = field(default_factory=dict)
    # variable -> value that should be preserved (soft-but-tracked)
    desirable: Dict[str, float] = field(default_factory=dict)

    def frozen_variables(self) -> Set[str]:
        """All variables that carry any constraint."""
        return set(self.intangible) | set(self.desirable)

    def free_variables(self, all_variables: Iterable[str]) -> Set[str]:
        """Variables free to change during the remaining optimisation."""
        return set(all_variables) - self.frozen_variables()


def freeze_region(
    solution: Solution, region: Iterable[str], *, kind: str = "intangible"
) -> Dict[str, float]:
    """Snapshot the given region's current values as equality constraints.

    This is the "current perimeter must be added as a further constraint" step.
    Raises ``KeyError`` if a region variable is not present in the solution.
    """
    if kind not in ("intangible", "desirable"):
        raise ValueError("kind must be 'intangible' or 'desirable'")
    out: Dict[str, float] = {}
    for var in region:
        if var not in solution:
            raise KeyError(f"variable '{var}' not in solution")
        out[var] = solution[var]
    return out


def build_constraints(
    solution: Solution,
    intangible_regions: Iterable[Iterable[str]] = (),
    desirable_features: Iterable[Iterable[str]] = (),
) -> ConstraintSet:
    """Assemble a :class:`ConstraintSet` from marked regions/features."""
    cs = ConstraintSet()
    for region in intangible_regions:
        cs.intangible.update(freeze_region(solution, region, kind="intangible"))
    for feature in desirable_features:
        cs.desirable.update(freeze_region(solution, feature, kind="desirable"))
    return cs


def check_preserved(
    new_solution: Solution, constraints: ConstraintSet, *, abs_tol: float = 1e-9
) -> Tuple[bool, List[Violation]]:
    """Verify a proposed solution keeps all locked/desirable values.

    Returns ``(ok, violations)``. ``ok`` is True iff there are no *intangible*
    violations (hard constraints); desirable violations are reported but do not
    by themselves make the result invalid.
    """
    violations: List[Violation] = []
    hard_ok = True
    for var, expected in constraints.intangible.items():
        actual = new_solution.get(var)
        if actual is None or abs(actual - expected) > abs_tol:
            hard_ok = False
            violations.append(
                Violation(var, expected, float("nan") if actual is None else actual, "intangible")
            )
    for var, expected in constraints.desirable.items():
        actual = new_solution.get(var)
        if actual is None or abs(actual - expected) > abs_tol:
            violations.append(
                Violation(var, expected, float("nan") if actual is None else actual, "desirable")
            )
    return hard_ok, violations


def project_onto_constraints(
    candidate: Solution, constraints: ConstraintSet
) -> Dict[str, float]:
    """Force the intangible (hard-locked) variables back to their frozen values.

    A convenience "repair" that a re-optimiser can apply to any candidate so the
    frozen perimeter is guaranteed to be respected. Desirable values are *not*
    forced (they remain soft).
    """
    result = dict(candidate)
    result.update(constraints.intangible)
    return result
