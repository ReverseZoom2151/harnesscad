"""Injected-solver geometric impact analysis for one sketch constraint."""

from __future__ import annotations

from dataclasses import dataclass
import math


@dataclass(frozen=True)
class ConstraintImpact:
    valid: bool
    moved: tuple[str, ...]
    max_displacement: float
    dof_before: int
    dof_after: int
    topology_changed: bool
    note: str = ""


def analyze_constraint(before, constraint, solve, *, tolerance=1e-6):
    """``solve(before,constraint)`` returns geometry/dof/valid mapping."""
    after = solve(before, constraint)
    if not after.get("valid", False):
        return ConstraintImpact(False, (), 0.0, int(before.get("dof", 0)),
                                int(after.get("dof", before.get("dof", 0))),
                                False, str(after.get("note", "unsolved")))
    left, right = before.get("geometry", {}), after.get("geometry", {})
    shared = set(left) & set(right)
    distances = {key: math.dist(tuple(left[key]), tuple(right[key])) for key in shared}
    moved = tuple(sorted(key for key, value in distances.items() if value > tolerance))
    return ConstraintImpact(
        True, moved, max(distances.values(), default=0.0),
        int(before.get("dof", 0)), int(after.get("dof", 0)),
        set(left) != set(right), str(after.get("note", "")),
    )
