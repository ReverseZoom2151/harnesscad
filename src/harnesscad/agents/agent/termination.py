"""Verifier-gated agent termination decisions, with a claim-vs-evidence gate.

Two independent conditions must hold before an agent may stop and report
success:

  * the **verifier** passed (``verifier_ok``) -- the original gate;
  * the final answer's **claims are backed by evidence** -- the gate
    from cad-cae-copilot and implemented in
    :mod:`harnesscad.agents.agent.claims_gate`: a run whose intent required a
    geometry mutation may not answer until a mutation tool actually succeeded,
    and an answer declaring solver results may not pass unless an approved,
    non-error solver run actually happened.

The claim gate is opt-in and default-safe: :func:`gate_termination` keeps its
original two-argument behaviour, and only engages the claim gate when a caller
supplies the run's resolved intent and/or its evidence.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional, Sequence

from harnesscad.agents.agent.claims_gate import (
    ClaimVerdict,
    RouteIntent,
    RunEvidence,
    gate_claims,
)
from harnesscad.governance.credibility_tier import UNVERIFIED

__all__ = [
    "TerminationDecision",
    "TerminationResult",
    "gate_termination",
]


@dataclass(frozen=True)
class TerminationDecision:
    """The agent's proposed stopping state, plus what it structurally claims.

    ``claims`` is the agent's DECLARED claim set (see
    :data:`~harnesscad.agents.agent.claims_gate.CHECKABLE_CLAIMS`), not a
    reading of its prose. It defaults to empty, so existing callers are
    unaffected; a declared claim can only ever add a requirement.
    """

    state: Literal["continue", "complete", "blocked"]
    reason: str = ""
    claims: Sequence[str] = ()


@dataclass(frozen=True)
class TerminationResult:
    accepted: bool
    terminal: bool
    state: str
    diagnostic: str = ""
    credibility_tier: str = UNVERIFIED
    claim_verdict: Optional[ClaimVerdict] = None


def gate_termination(
    decision: TerminationDecision,
    verifier_ok: bool,
    *,
    intent: Optional[RouteIntent] = None,
    evidence: Optional[RunEvidence] = None,
) -> TerminationResult:
    """Gate a termination decision on the verifier AND on claim evidence.

    The verifier check runs first (an unverified completion is premature
    regardless of what it claims). A completion that passes the verifier is
    then put to :func:`~harnesscad.agents.agent.claims_gate.gate_claims`; a
    rejected claim sends the agent back to ``continue`` with the gate's reason
    as the diagnostic, exactly like a premature completion. ``blocked`` and
    ``continue`` are never claim-gated -- honest failure is always allowed.

    With no ``intent`` and no ``evidence`` this is the original function.
    """
    if decision.state == "complete" and not verifier_ok:
        return TerminationResult(False, False, "continue", "premature-completion")
    if decision.state == "continue":
        return TerminationResult(True, False, "continue")

    verdict = gate_claims(decision.state, tuple(decision.claims),
                          intent=intent, evidence=evidence)
    if not verdict.accepted:
        return TerminationResult(False, False, "continue", verdict.reason,
                                 verdict.credibility_tier, verdict)
    return TerminationResult(True, True, decision.state, "",
                             verdict.credibility_tier, verdict)
