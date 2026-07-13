"""Hierarchical constraint scopes and branch-pruned solve orchestration.

The module is solver-neutral: callers inject a local solver while this layer
owns scope ordering, local frames, value propagation and original-constraint
revalidation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple


Values = Dict[str, float]
LocalSolver = Callable[["ConstraintScope", Mapping[str, float]], Mapping[str, float]]


@dataclass(frozen=True)
class LocalFrame:
    origin: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    x_axis: Tuple[float, float, float] = (1.0, 0.0, 0.0)
    y_axis: Tuple[float, float, float] = (0.0, 1.0, 0.0)


@dataclass
class ConstraintScope:
    name: str
    frame: LocalFrame = field(default_factory=LocalFrame)
    parameters: Values = field(default_factory=dict)
    constraints: Tuple[str, ...] = ()
    children: List["ConstraintScope"] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("scope name is required")

    def add_child(self, child: "ConstraintScope") -> None:
        if child.name == self.name or any(item.name == child.name for item in self.children):
            raise ValueError(f"duplicate/cyclic child scope {child.name!r}")
        self.children.append(child)

    def walk(self) -> Tuple["ConstraintScope", ...]:
        result = [self]
        for child in sorted(self.children, key=lambda item: item.name):
            result.extend(child.walk())
        return tuple(result)


@dataclass(frozen=True)
class ScopeSolve:
    scope: str
    values: Mapping[str, float]


@dataclass(frozen=True)
class HierarchicalSolveResult:
    scopes: Tuple[ScopeSolve, ...]

    def values_for(self, scope: str) -> Mapping[str, float]:
        for item in self.scopes:
            if item.scope == scope:
                return item.values
        raise KeyError(scope)


def solve_hierarchy(root: ConstraintScope, solver: LocalSolver) -> HierarchicalSolveResult:
    """Solve children before parents and expose child values by qualified name."""
    solved: List[ScopeSolve] = []

    def visit(scope: ConstraintScope) -> Mapping[str, float]:
        inherited: Values = dict(scope.parameters)
        for child in sorted(scope.children, key=lambda item: item.name):
            child_values = visit(child)
            for key, value in child_values.items():
                inherited[f"{child.name}.{key}"] = float(value)
        output = {key: float(value) for key, value in solver(scope, inherited).items()}
        solved.append(ScopeSolve(scope.name, output))
        return output

    visit(root)
    return HierarchicalSolveResult(tuple(solved))


@dataclass(frozen=True)
class EditabilityReport:
    edited_scope: str
    changed_in_scope: int
    changed_outside_scope: int
    unchanged_outside_scope: int

    @property
    def locality(self) -> float:
        total = self.changed_outside_scope + self.unchanged_outside_scope
        return 1.0 if total == 0 else self.unchanged_outside_scope / total


def local_editability(
    before: Mapping[str, Mapping[str, float]],
    after: Mapping[str, Mapping[str, float]],
    edited_scope: str,
    *,
    tolerance: float = 1e-9,
) -> EditabilityReport:
    """Measure whether an edit stays local instead of perturbing sibling scopes."""
    all_scopes = sorted(set(before) | set(after))
    changed_local = changed_other = unchanged_other = 0
    for scope in all_scopes:
        keys = set(before.get(scope, {})) | set(after.get(scope, {}))
        for key in keys:
            a = before.get(scope, {}).get(key)
            b = after.get(scope, {}).get(key)
            changed = (
                a is None or b is None or abs(float(a) - float(b)) > tolerance
            )
            if scope == edited_scope:
                changed_local += int(changed)
            elif changed:
                changed_other += 1
            else:
                unchanged_other += 1
    return EditabilityReport(
        edited_scope, changed_local, changed_other, unchanged_other
    )


@dataclass(frozen=True)
class PrunedBranch:
    label: str
    constraints: Tuple[str, ...]


@dataclass(frozen=True)
class PrunedSolveResult:
    branch: str
    values: Mapping[str, float]
    attempts: int


def solve_pruned_branches(
    branches: Sequence[PrunedBranch],
    solve: Callable[[Tuple[str, ...], Mapping[str, float]], Mapping[str, float]],
    validate_original: Callable[[Mapping[str, float]], bool],
    initial: Optional[Mapping[str, float]] = None,
) -> PrunedSolveResult:
    """Try smooth branch rewrites, accepting only an original-expression solution."""
    if not branches:
        raise ValueError("at least one branch is required")
    seed: Mapping[str, float] = dict(initial or {})
    for attempt, branch in enumerate(branches, 1):
        candidate = {
            key: float(value)
            for key, value in solve(branch.constraints, seed).items()
        }
        if validate_original(candidate):
            return PrunedSolveResult(branch.label, candidate, attempt)
        seed = candidate
    raise ValueError("no pruned branch satisfies the original constraints")
