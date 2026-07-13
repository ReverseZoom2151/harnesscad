"""Verifier-gated agent termination decisions."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


@dataclass(frozen=True)
class TerminationDecision:
    state: Literal["continue", "complete", "blocked"]
    reason: str = ""


@dataclass(frozen=True)
class TerminationResult:
    accepted: bool
    terminal: bool
    state: str
    diagnostic: str = ""


def gate_termination(decision: TerminationDecision, verifier_ok: bool) -> TerminationResult:
    if decision.state == "complete" and not verifier_ok:
        return TerminationResult(False, False, "continue", "premature-completion")
    if decision.state == "continue":
        return TerminationResult(True, False, "continue")
    return TerminationResult(True, True, decision.state)
