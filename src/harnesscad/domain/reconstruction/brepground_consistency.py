"""Program + grounding consistency check (FutureCAD / BRepGround).

Li et al., "Towards High-Fidelity CAD Generation via LLM-Driven Program
Generation and Text-Based B-Rep Primitive Grounding" (FutureCAD, 2026),
Sec. 3.1 and Eq. 3-5. A generated CadQuery program is a feature sequence
``F = f_1, ..., f_T``; the kernel executes it while maintaining a transient
B-Rep ``B_i = Phi(B_{i-1}, f_i)``. When a feature needs operands its embedded
text query ``q_i`` must resolve to a *non-empty* set of primitives drawn from
the B-Rep produced by the *preceding* features: ``pi_i subseteq P(B_{i-1})``
(Sec. 3.1). Advanced features (``fillet``/``chamfer``/``shell``) require
``pi_i != {}`` (Sec. 3.1, "typically require pi_i != {}").

The LLM/kernel are external. This module checks, purely from the program's
declared queries and per-step transient primitive sets, whether the
program-and-grounding pair is internally *consistent* -- the deterministic
validity conditions the paper's Invalidity-Ratio metric measures at the
grounding level:

  * every query resolves to at least one primitive (no dangling reference);
  * grounded primitives come from ``P(B_{i-1})``, i.e. the state *before* the
    feature -- never a forward reference to geometry the feature itself creates;
  * a refinement feature that requires operands actually carries a query and
    grounds to a primitive of the type it needs (an edge for fillet/chamfer, a
    face for shell);
  * the query is *specific enough* -- it does not silently ground to a huge set
    when the operation expects one primitive.

Grounding uses :mod:`reconstruction.brepground_grounding`. Pure, deterministic,
stdlib-only.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Tuple

from harnesscad.domain.reconstruction.brepground_grounding import (
    BRepPrimitive,
    ground_all,
    parse_query,
)

# Feature types that select existing B-Rep primitives, and the primitive kind
# each expects for its operand set (Sec. 3.1).
_REFINEMENT_KIND = {
    "fillet": "edge",
    "chamfer": "edge",
    "shell": "face",
}
# Features that require a non-empty operand set to be valid.
_REQUIRES_OPERANDS = frozenset(_REFINEMENT_KIND)


@dataclass(frozen=True)
class FeatureStep:
    """One feature ``f_i`` in the program.

    ``feature``    operation type ("sketch", "extrude", "fillet", ...).
    ``query``      the embedded text reference ``q_i``, or None if the feature
                   takes no primitive operands.
    ``available``  ``P(B_{i-1})`` -- the primitives present *before* this feature
                   executes. Grounding must resolve inside this set.
    ``singular``   whether the feature expects exactly one operand (used for the
                   over-selection check). Defaults to False.
    """

    feature: str
    query: Optional[str] = None
    available: Sequence[BRepPrimitive] = field(default_factory=tuple)
    singular: bool = False


@dataclass(frozen=True)
class StepResult:
    """Outcome of checking a single :class:`FeatureStep`."""

    feature: str
    ok: bool
    grounded: Tuple[int, ...] = ()
    reason: str = ""


@dataclass(frozen=True)
class ConsistencyReport:
    """Whole-program result."""

    ok: bool
    steps: Tuple[StepResult, ...] = ()

    @property
    def failures(self) -> Tuple[StepResult, ...]:
        return tuple(s for s in self.steps if not s.ok)


def check_step(step: FeatureStep) -> StepResult:
    """Check one feature's query/grounding consistency."""
    requires = step.feature in _REQUIRES_OPERANDS

    if step.query is None:
        if requires:
            return StepResult(
                step.feature, False, (),
                "refinement feature '%s' requires an operand query but has none"
                % step.feature,
            )
        return StepResult(step.feature, True, (), "")

    grounded = ground_all(step.query, step.available)
    ids = tuple(sorted(p.index for p in grounded))

    if not grounded:
        return StepResult(
            step.feature, False, (),
            "query %r grounds to no primitive in P(B_{i-1})" % step.query,
        )

    # Forward-reference guard: grounded ids must be a subset of the available
    # set. ground_all already draws only from ``available``, so this is a
    # defensive invariant assertion expressed as a check.
    available_ids = {p.index for p in step.available}
    if not set(ids) <= available_ids:
        return StepResult(
            step.feature, False, ids,
            "grounded primitives escape P(B_{i-1}) (forward reference)",
        )

    # Type check for refinement features.
    expected_kind = _REFINEMENT_KIND.get(step.feature)
    if expected_kind is not None:
        wrong = [p.index for p in grounded if p.kind != expected_kind]
        if wrong:
            return StepResult(
                step.feature, False, ids,
                "feature '%s' expects %s operands but grounded %r"
                % (step.feature, expected_kind, wrong),
            )

    # Over-selection: a singular operation must not resolve to many primitives.
    if step.singular and len(grounded) > 1:
        return StepResult(
            step.feature, False, ids,
            "query %r is ambiguous: %d primitives for a singular operand"
            % (step.query, len(grounded)),
        )

    return StepResult(step.feature, True, ids, "")


def check_program(steps: Sequence[FeatureStep]) -> ConsistencyReport:
    """Check an entire feature sequence; a program is consistent iff every step
    is."""
    results = tuple(check_step(s) for s in steps)
    return ConsistencyReport(ok=all(r.ok for r in results), steps=results)


def query_specificity(step: FeatureStep) -> float:
    """A [0, 1] specificity score for a step's query against its B-Rep.

    1.0 means the query grounds to a single primitive (maximally specific);
    lower values mean it selects a larger fraction of the available set. Useful
    as a soft signal for the paper's "query is specific enough" concern.
    Returns 1.0 for steps without a query.
    """
    if step.query is None:
        return 1.0
    total = len(step.available)
    if total == 0:
        return 0.0
    grounded = ground_all(step.query, step.available)
    if not grounded:
        return 0.0
    return 1.0 - (len(grounded) - 1) / total


def uses_grounding(feature: str) -> bool:
    """Whether ``feature`` is one whose operands come from B-Rep grounding."""
    return feature in _REQUIRES_OPERANDS
