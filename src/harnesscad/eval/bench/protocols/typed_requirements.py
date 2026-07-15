"""Typed engineering-requirement checking with partial credit (Self-Improving FEA).

Mined from *Self-Improving CAD Generation Agents with Finite Element Analysis as
Feedback* (Hephaestus-CCX). Instead of grading a generated CAD artifact by
proximity to a gold reference, the paper grades it against a **contract of typed
requirements** -- stress, displacement, modal, buckling, contact and clearance
checks -- each an executable pass/fail predicate. The reported scores are:

*   **mean requirement pass** -- the per-case fraction of typed requirements
    satisfied (partial credit); and
*   **strict pass** -- 1 only if *every* requirement passes.

This module ports that grading. Each :class:`Requirement` names a category, a
comparison operator and a threshold; :func:`check` evaluates a measured value, and
:func:`grade` aggregates a full requirement set. Deterministic, stdlib-only (no FEA
is run; measured values are supplied by the caller).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Mapping, Sequence

__all__ = [
    "REQUIREMENT_CATEGORIES",
    "Requirement",
    "check",
    "mean_requirement_pass",
    "strict_pass",
    "grade",
]

#: The six FEA requirement categories in the Hephaestus-CCX contract.
REQUIREMENT_CATEGORIES: tuple = (
    "stress", "displacement", "modal", "buckling", "contact", "clearance",
)

_OPS: Dict[str, Callable[[float, float], bool]] = {
    "<=": lambda v, t: v <= t,
    "<": lambda v, t: v < t,
    ">=": lambda v, t: v >= t,
    ">": lambda v, t: v > t,
    "==": lambda v, t: v == t,
}


@dataclass(frozen=True)
class Requirement:
    """One typed requirement: ``name`` in ``category`` must satisfy ``op threshold``.

    Example: a stress cap is ``Requirement("max_stress", "stress", "<=", 250e6)``.
    """

    name: str
    category: str
    op: str
    threshold: float

    def __post_init__(self) -> None:
        if self.category not in REQUIREMENT_CATEGORIES:
            raise ValueError(f"unknown category {self.category!r}")
        if self.op not in _OPS:
            raise ValueError(f"unknown operator {self.op!r}")


def check(requirement: Requirement, measured: float) -> bool:
    """True iff ``measured`` satisfies the requirement's predicate."""
    return _OPS[requirement.op](measured, requirement.threshold)


def mean_requirement_pass(
    requirements: Sequence[Requirement], measurements: Mapping[str, float]
) -> float:
    """Fraction of requirements satisfied (partial credit).

    A requirement whose measurement is missing counts as failed.
    """
    if not requirements:
        raise ValueError("need at least one requirement")
    passed = 0
    for r in requirements:
        if r.name in measurements and check(r, measurements[r.name]):
            passed += 1
    return passed / len(requirements)


def strict_pass(
    requirements: Sequence[Requirement], measurements: Mapping[str, float]
) -> bool:
    """True iff every requirement is measured and passes."""
    return mean_requirement_pass(requirements, measurements) == 1.0


def grade(
    requirements: Sequence[Requirement], measurements: Mapping[str, float]
) -> Dict[str, object]:
    """Full scorecard: per-category pass counts, mean pass, and strict pass."""
    by_cat: Dict[str, Dict[str, int]] = {
        c: {"passed": 0, "total": 0} for c in REQUIREMENT_CATEGORIES
    }
    for r in requirements:
        by_cat[r.category]["total"] += 1
        if r.name in measurements and check(r, measurements[r.name]):
            by_cat[r.category]["passed"] += 1
    return {
        "mean_requirement_pass": mean_requirement_pass(requirements, measurements),
        "strict_pass": strict_pass(requirements, measurements),
        "by_category": {c: v for c, v in by_cat.items() if v["total"] > 0},
    }
