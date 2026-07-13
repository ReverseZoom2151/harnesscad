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
  * stop at ``max_iterations`` (``reason="max-iterations"``).

Pure stdlib, deterministic; no wall-clock, no randomness.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, List, Optional, Sequence

from harnesscad.domain.reconstruction.tokens.deepcad_commands import Command
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

    def to_dict(self) -> dict:
        return {
            "feasible": self.feasible,
            "converged": self.converged,
            "reason": self.reason,
            "iterations": self.iterations,
            "types": [c.type for c in self.sequence],
            "history": [s.to_dict() for s in self.history],
        }


def repair_until_feasible(
    commands: Sequence[Command],
    max_iterations: int = 8,
    checker: Optional[FeasibilityChecker] = None,
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
    """
    if max_iterations < 1:
        raise ValueError("max_iterations must be >= 1")
    if checker is None:
        def checker(seq: Sequence[Command]) -> bool:  # noqa: WPS440 - local default
            return diagnose(seq).feasible

    current: List[Command] = list(commands)
    history: List[LoopStep] = []

    if checker(current):
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

        if checker(repaired):
            return LoopResult(
                feasible=True, converged=True, reason="feasible",
                iterations=step, sequence=repaired, history=history)

        no_change = repaired == current
        no_reduction = findings_after >= findings_before
        current = repaired
        if no_change or no_reduction:
            return LoopResult(
                feasible=checker(current), converged=True,
                reason="no-progress", iterations=step,
                sequence=current, history=history)

    return LoopResult(
        feasible=checker(current), converged=False,
        reason="max-iterations", iterations=max_iterations,
        sequence=current, history=history)
