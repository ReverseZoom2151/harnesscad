"""Repair-loop convergence for infeasible CAD command sequences.

Tsuji, Flores Medina, Gupta & Alam, *GenCAD-Self-Repairing* (MIT).

The paper's self-repair pipeline (Fig. 3) is a *loop*: a generated sequence is
handed to the geometry kernel; if the kernel rejects it, the sequence is corrected
and re-checked. This module is the deterministic driver for that detect ->
repair -> re-check loop, built on :func:`gencadrepair_taxonomy.diagnose` and
:func:`gencadrepair_sequence.repair_sequence`.

Because :func:`repair_sequence` is idempotent and always produces a structurally
feasible result, the loop converges in at most a couple of iterations under the
default checker; the driver is written for the general case (a *pluggable*
feasibility checker — e.g. the real OCCT kernel) where each repair may only
resolve part of the infeasibility and progress must be tracked explicitly:

  * stop as soon as the sequence is feasible (``reason="feasible"``);
  * stop when an iteration makes no change or fails to reduce the finding count
    (``reason="no-progress"``) — this is the fixed-point / stall guard;
  * stop at ``max_iterations`` (``reason="max-iterations"``);
  * OPT-IN (``use_error_contract=True``): stop when the checker raises an error
    the structured contract calls unrecoverable (``reason="abstain"``).

THE ABSTAIN GATE (opt-in, default OFF)
--------------------------------------
The stopping rules above all assume the checker ANSWERS. A real kernel-backed
checker also RAISES — and the loop cannot tell "this boolean failed, repair it
and re-check" from "the backend is not installed" / "the brief is ambiguous" /
"the artifact would not load". Iterating on the second kind burns the whole
budget re-asking a question nothing in the loop can change.

:mod:`harnesscad.eval.reliability.error_contract` already makes that call:
:func:`~harnesscad.eval.reliability.error_contract.repair_decision` collapses a
``ToolError`` to ``"retry"`` or ``"abstain"``. ``use_error_contract=True`` wires
it in — a raising checker becomes a ``ToolError``, a ``"retry"`` verdict is just
an infeasible iteration, and an ``"abstain"`` verdict stops the loop with the
error on :attr:`LoopResult.error` so the caller can surface its suggested_action.

It is OFF by default and the default path is unchanged: with the flag off, a
checker that raises propagates its exception to the caller exactly as before.

Pure stdlib, deterministic; no wall-clock, no randomness.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, List, Optional, Sequence, Tuple

from harnesscad.domain.reconstruction.tokens.deepcad_commands import Command
from harnesscad.eval.reliability.error_contract import (
    ToolError,
    from_exception,
    repair_decision,
)
from harnesscad.eval.reliability.sequence_repair import repair_sequence
from harnesscad.eval.reliability.infeasibility_taxonomy import diagnose

FeasibilityChecker = Callable[[Sequence[Command]], bool]


@dataclass(frozen=True)
class LoopStep:
    """One iteration of the repair loop."""

    iteration: int
    findings_before: int
    findings_after: int
    fixes: List[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "iteration": self.iteration,
            "findings_before": self.findings_before,
            "findings_after": self.findings_after,
            "fixes": list(self.fixes),
        }


@dataclass
class LoopResult:
    """Outcome of running the repair loop to convergence."""

    feasible: bool
    converged: bool
    reason: str
    iterations: int
    sequence: List[Command]
    history: List[LoopStep] = field(default_factory=list)
    #: The contract error that ended the loop, set ONLY when the opt-in
    #: ``use_error_contract`` gate abstained (``reason == "abstain"``).
    error: Optional[ToolError] = None

    def to_dict(self) -> dict:
        return {
            "feasible": self.feasible,
            "converged": self.converged,
            "reason": self.reason,
            "iterations": self.iterations,
            "types": [c.type for c in self.sequence],
            "history": [s.to_dict() for s in self.history],
            "error": self.error.to_dict() if self.error is not None else None,
        }


def _probe(
    checker: FeasibilityChecker,
    seq: Sequence[Command],
    use_error_contract: bool,
) -> Tuple[bool, Optional[ToolError]]:
    """Ask the checker, and (opt-in) classify a raise as retry-vs-abstain.

    Returns ``(feasible, abstain_error)``. With the gate OFF this is exactly
    ``checker(seq)`` -- same single call, and an exception propagates to the
    caller untouched, which is what every existing caller already relies on.

    With the gate ON a raise becomes a ``ToolError``: a ``"retry"`` verdict
    reports plain infeasibility (the loop repairs and re-checks, as it would
    for a ``False``), and only an ``"abstain"`` verdict returns the error.
    """
    if not use_error_contract:
        return checker(seq), None
    try:
        return checker(seq), None
    except Exception as exc:  # noqa: BLE001 - the contract classifies it below
        error = from_exception(exc)
        return False, (error if repair_decision(error) == "abstain" else None)


def repair_until_feasible(
    commands: Sequence[Command],
    max_iterations: int = 8,
    checker: Optional[FeasibilityChecker] = None,
    use_error_contract: bool = False,
) -> LoopResult:
    """Iterate detect -> repair -> re-check until the sequence is feasible.

    ``checker`` decides feasibility (defaults to the structural taxonomy check;
    pass an OCCT-backed predicate for the real kernel). Returns a
    :class:`LoopResult` with the final sequence, whether it converged, the
    stopping ``reason`` and a per-iteration ``history``.

    Convergence guarantees:
      * ``max_iterations`` must be >= 1;
      * the loop never runs longer than ``max_iterations`` steps;
      * a repair that neither changes the sequence nor reduces the number of
        structural findings ends the loop as ``no-progress`` (fixed point).

    ``use_error_contract`` (default ``False``) opts into the abstain gate
    described in the module docstring: a checker that RAISES is classified by
    :func:`~harnesscad.eval.reliability.error_contract.repair_decision`, and an
    unrecoverable error stops the loop with ``reason="abstain"`` and the
    ``ToolError`` on :attr:`LoopResult.error` instead of spending the remaining
    budget on a failure no repair can address. LEFT OFF, NOTHING CHANGES: a
    raising checker propagates exactly as it always has.
    """
    if max_iterations < 1:
        raise ValueError("max_iterations must be >= 1")
    if checker is None:
        def checker(seq: Sequence[Command]) -> bool:  # noqa: WPS440 - local default
            return diagnose(seq).feasible

    current: List[Command] = list(commands)
    history: List[LoopStep] = []

    def abstained(error: ToolError, step: int, seq: List[Command]) -> LoopResult:
        return LoopResult(
            feasible=False, converged=True, reason="abstain",
            iterations=step, sequence=seq, history=history, error=error)

    feasible, error = _probe(checker, current, use_error_contract)
    if error is not None:
        return abstained(error, 0, current)
    if feasible:
        return LoopResult(
            feasible=True, converged=True, reason="feasible",
            iterations=0, sequence=current, history=history)

    for step in range(1, max_iterations + 1):
        findings_before = len(diagnose(current).findings)
        outcome = repair_sequence(current)
        repaired = outcome.repaired
        findings_after = len(diagnose(repaired).findings)
        history.append(LoopStep(
            iteration=step,
            findings_before=findings_before,
            findings_after=findings_after,
            fixes=list(outcome.fixes)))

        feasible, error = _probe(checker, repaired, use_error_contract)
        if error is not None:
            return abstained(error, step, repaired)
        if feasible:
            return LoopResult(
                feasible=True, converged=True, reason="feasible",
                iterations=step, sequence=repaired, history=history)

        no_change = repaired == current
        no_reduction = findings_after >= findings_before
        current = repaired
        if no_change or no_reduction:
            feasible, error = _probe(checker, current, use_error_contract)
            if error is not None:
                return abstained(error, step, current)
            return LoopResult(
                feasible=feasible, converged=True,
                reason="no-progress", iterations=step,
                sequence=current, history=history)

    feasible, error = _probe(checker, current, use_error_contract)
    if error is not None:
        return abstained(error, max_iterations, current)
    return LoopResult(
        feasible=feasible, converged=False,
        reason="max-iterations", iterations=max_iterations,
        sequence=current, history=history)
