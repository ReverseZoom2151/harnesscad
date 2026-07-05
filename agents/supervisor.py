"""Supervisor — the multi-agent orchestrator (HARNESS_BLUEPRINT sec.12).

A LoopAgent-style loop chaining the role personas from ``agents.roles``:

    Designer -> Modeler -> Verifier -> DFMCritic -> RedTeam -> Reviewer

Each round produces a plan, applies it through the harness, runs the plural
verifier + DFM critic, lets the RedTeam probe for non-manufacturable geometry /
interference (with veto authority), and asks the Reviewer to critique->reflect and
decide. The loop **escalates to stop**: it repeats — feeding the round's diagnostics
back into the Designer — until the stop condition (model verified AND reviewer
approved AND no RedTeam veto) or ``max_rounds`` is exhausted.

Returns a structured :class:`Trajectory` (every round recorded) — the audit trail
the blueprint calls for. Absolute imports, stdlib only.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

from agents.roles import (
    Designer, Modeler, Verifier, DFMCritic, Reviewer, RedTeam,
    DesignPlan, ModelResult, VerifyOutcome, DFMOutcome, RedTeamResult, ReviewResult,
    Finding, findings_from,
)


# --------------------------------------------------------------------------- #
# Trajectory (the structured, replayable record the supervisor returns)
# --------------------------------------------------------------------------- #
@dataclass
class RoundRecord:
    index: int
    plan: DesignPlan
    model: Optional[ModelResult]
    verify: Optional[VerifyOutcome]
    dfm: Optional[DFMOutcome]
    red_team: RedTeamResult
    review: ReviewResult
    stop: bool

    @property
    def approved(self) -> bool:
        return self.review.approved


@dataclass
class Trajectory:
    rounds: List[RoundRecord] = field(default_factory=list)
    approved: bool = False
    stop_reason: str = "not-run"
    digest: str = ""

    @property
    def round_count(self) -> int:
        return len(self.rounds)

    @property
    def final(self) -> Optional[RoundRecord]:
        return self.rounds[-1] if self.rounds else None


# --------------------------------------------------------------------------- #
# Supervisor
# --------------------------------------------------------------------------- #
class Supervisor:
    """Orchestrates the role pipeline with a LoopAgent-style escalate-to-stop loop.

    Only the Designer *needs* configuring (it carries the persona/LLM); the other
    roles default to their mechanical/heuristic forms so a Supervisor(designer) is
    fully runnable. Inject a custom ``red_team`` to change veto behaviour.
    """

    def __init__(
        self,
        designer: Designer,
        modeler: Optional[Modeler] = None,
        verifier: Optional[Verifier] = None,
        dfm: Optional[DFMCritic] = None,
        reviewer: Optional[Reviewer] = None,
        red_team: Optional[RedTeam] = None,
        max_rounds: int = 5,
    ) -> None:
        if max_rounds < 1:
            raise ValueError("max_rounds must be >= 1")
        self.designer = designer
        self.modeler = modeler or Modeler()
        self.verifier = verifier or Verifier()
        self.dfm = dfm or DFMCritic()
        self.reviewer = reviewer or Reviewer()
        self.red_team = red_team or RedTeam()
        self.max_rounds = max_rounds

    def run(self, session, brief: str) -> Trajectory:
        traj = Trajectory()
        diagnostics: Optional[list] = None

        for i in range(self.max_rounds):
            plan = self.designer.design(brief, session.summary(), diagnostics)

            # A Designer that could not produce a plan (bad LLM output): record it
            # as a blocking finding and re-prompt next round.
            if not plan.ok:
                finding = Finding(
                    severity=_ERROR, code="plan-error",
                    message=plan.error or "designer produced no valid plan",
                    source="designer")
                red = self.red_team.attack(session, [finding])
                review = self.reviewer.review(brief, [finding], blocking_ok=False,
                                              veto=red.veto)
                traj.rounds.append(RoundRecord(i, plan, None, None, None, red, review, False))
                diagnostics = [{"severity": "error", "code": "plan-error",
                                "message": plan.error or "planner produced no ops"}]
                continue

            model = self.modeler.model(session, plan)
            verify_o = self.verifier.verify(session)
            dfm_o = self.dfm.critique(session)

            findings: List[Finding] = []
            findings += findings_from(model.diagnostics, "modeler")
            findings += findings_from(verify_o.diagnostics, "verifier")
            findings += findings_from(dfm_o.diagnostics, "dfm-critic")

            red = self.red_team.attack(session, findings)
            findings += [
                Finding(_ERROR, "red-team-veto", r, "red-team")
                for r in red.reasons
            ]

            blocking_ok = model.ok and verify_o.ok and not red.veto
            review = self.reviewer.review(brief, findings,
                                          blocking_ok=blocking_ok, veto=red.veto)
            stop = review.approved
            traj.rounds.append(
                RoundRecord(i, plan, model, verify_o, dfm_o, red, review, stop))

            if stop:
                traj.approved = True
                traj.stop_reason = "verified-and-approved"
                traj.digest = session.digest()
                return traj

            # Escalate: feed this round's diagnostics back to the Designer.
            diagnostics = _feedback(model, verify_o, red)

        traj.approved = False
        traj.stop_reason = "max-rounds-exhausted"
        traj.digest = session.digest()
        return traj


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
from verify import Severity as _Severity  # noqa: E402 - local alias for Finding severity
_ERROR = _Severity.ERROR


def _feedback(model: ModelResult, verify_o: VerifyOutcome, red: RedTeamResult) -> list:
    """Assemble the diagnostics dict list fed back into the next Designer.plan().

    Prefers the model's own apply diagnostics (the concrete block-and-correct /
    verify-failure reasons); adds a RedTeam entry so the next plan avoids the
    vetoed geometry."""
    diags = [d.to_dict() for d in model.diagnostics]
    if not diags:
        diags = [d.to_dict() for d in verify_o.diagnostics
                 if d.severity is _Severity.ERROR]
    for r in red.reasons:
        diags.append({"severity": "error", "code": "red-team-veto", "message": r})
    return diags
