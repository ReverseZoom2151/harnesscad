"""CADSmith Judge escalation & anti-oscillation policy (CADSmith sec. III-E/F).

The Validator Judge receives its own prior feedback each iteration. Two
deterministic policies govern how refinement history steers the loop:

  * Anti-repeat / escalation: "If you gave feedback before and the same issue
    persists, do not repeat the same suggestion. Recommend a fundamentally
    different construction approach." At iteration three and beyond the Refiner
    is explicitly told to reconsider the overall construction strategy rather
    than keep adjusting the same parameters.

  * Anti-oscillation: the Refiner receives the full history of previous attempts
    (what feedback was given, what was tried) to prevent oscillating between the
    same two states.

The *decision to escalate* is deterministic policy — the LLM only phrases the
suggestion. This module factors that policy out so it is testable in isolation:
it tracks feedback/code history, detects a persisting issue and an oscillation
cycle, and emits an :class:`EscalationDirective` telling the Refiner whether to
tweak parameters, try a fundamentally different approach, or break a cycle.

Stdlib only, deterministic.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional, Tuple


# --------------------------------------------------------------------------- #
# Iteration record
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class IterationRecord:
    """One refinement attempt: the issue code the Judge raised and a fingerprint
    of the code that was tried."""

    issue_code: str          # normalised code for the failure (e.g. "bbox_z")
    code_fingerprint: str    # stable hash of the code attempt


def fingerprint_code(code: str) -> str:
    """Stable, deterministic fingerprint of a code attempt (normalised on
    whitespace so cosmetic reformatting is not treated as a new attempt)."""
    normalized = " ".join(code.split())
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()[:16]


# --------------------------------------------------------------------------- #
# Directive
# --------------------------------------------------------------------------- #
class Strategy(Enum):
    ADJUST = "adjust-parameters"                 # normal: tweak the same approach
    ESCALATE = "reconsider-construction"         # try a fundamentally different approach
    BREAK_CYCLE = "break-oscillation"            # detected A<->B oscillation


@dataclass(frozen=True)
class EscalationDirective:
    strategy: Strategy
    reason: str
    persisting_issue: Optional[str] = None
    forbidden_fingerprints: Tuple[str, ...] = ()

    @property
    def escalated(self) -> bool:
        return self.strategy is not Strategy.ADJUST


# --------------------------------------------------------------------------- #
# Policy
# --------------------------------------------------------------------------- #
@dataclass
class EscalationPolicy:
    """Deterministic escalation controller over refinement history.

    Parameters mirror the paper: ``escalate_iteration`` is the iteration index
    (0-based) at/after which persistent failure forces a strategy change (the
    paper's "iteration three and beyond" => index 3); ``persist_threshold`` is
    how many consecutive iterations the *same* issue code must recur before it is
    treated as persisting.
    """

    escalate_iteration: int = 3
    persist_threshold: int = 2
    history: List[IterationRecord] = field(default_factory=list)

    def record(self, issue_code: str, code: str) -> None:
        self.history.append(IterationRecord(issue_code, fingerprint_code(code)))

    # -- detectors --------------------------------------------------------- #
    def _persisting_issue(self) -> Optional[str]:
        """The issue code recurring for the last ``persist_threshold`` records."""
        if len(self.history) < self.persist_threshold:
            return None
        recent = self.history[-self.persist_threshold:]
        codes = {r.issue_code for r in recent}
        if len(codes) == 1:
            return recent[-1].issue_code
        return None

    def _oscillating(self) -> bool:
        """Detect an A<->B<->A oscillation over the last three code attempts."""
        if len(self.history) < 3:
            return False
        a, b, c = (r.code_fingerprint for r in self.history[-3:])
        return a == c and a != b

    def _repeated_attempt(self, next_code: str) -> bool:
        """Would the next code repeat any previously-tried attempt verbatim?"""
        fp = fingerprint_code(next_code)
        return any(r.code_fingerprint == fp for r in self.history)

    # -- decision ---------------------------------------------------------- #
    def directive(self, iteration: int) -> EscalationDirective:
        """Decide the strategy for the given (0-based) iteration index.

        Precedence: oscillation (needs an explicit cycle break) > persistent
        issue or reaching the escalation iteration (reconsider the approach) >
        normal parameter adjustment.
        """
        forbidden = tuple(r.code_fingerprint for r in self.history)
        if self._oscillating():
            return EscalationDirective(
                Strategy.BREAK_CYCLE,
                "code oscillated between two states; break the cycle with a "
                "different construction",
                persisting_issue=self._persisting_issue(),
                forbidden_fingerprints=forbidden,
            )
        persisting = self._persisting_issue()
        if persisting is not None or iteration >= self.escalate_iteration:
            reason = (
                f"issue '{persisting}' persisted across iterations; "
                if persisting else ""
            ) + (
                f"iteration {iteration} at/beyond escalation threshold "
                f"{self.escalate_iteration}"
                if iteration >= self.escalate_iteration
                else "recommend a fundamentally different approach"
            )
            return EscalationDirective(
                Strategy.ESCALATE, reason,
                persisting_issue=persisting,
                forbidden_fingerprints=forbidden,
            )
        return EscalationDirective(
            Strategy.ADJUST, "adjust parameters of the current approach",
            forbidden_fingerprints=forbidden,
        )

    def is_forbidden(self, code: str) -> bool:
        """True if the given code repeats a prior attempt (for the Refiner to
        avoid re-proposing it)."""
        return self._repeated_attempt(code)
