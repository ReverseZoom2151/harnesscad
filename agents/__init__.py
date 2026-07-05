"""agents/ — the multi-agent supervisor layer (HARNESS_BLUEPRINT sec.12).

A thin orchestration layer built *over* the single-agent baseline (the ``agent/``
package, the HarnessSession spine, the plural verifiers, the DFM critic). Distinct
from the singular ``agent/`` package: this one adds role personas, a supervisor loop,
and an async overseer — it does not replace the baseline.

Public API:
  roles      — Designer, Modeler, Verifier, DFMCritic, Reviewer, RedTeam (+ typed I/O)
  supervisor — Supervisor (LoopAgent-style escalate-to-stop) + Trajectory/RoundRecord
  overseer   — AsyncOverseer + Halt
"""

from __future__ import annotations

from agents.roles import (
    Designer, Modeler, Verifier, DFMCritic, Reviewer, RedTeam,
    DesignPlan, ModelResult, VerifyOutcome, DFMOutcome, RedTeamResult, ReviewResult,
    Finding, findings_from, prioritize, default_redteam_probe,
)
from agents.supervisor import Supervisor, Trajectory, RoundRecord
from agents.overseer import AsyncOverseer, Halt

__all__ = [
    "Designer", "Modeler", "Verifier", "DFMCritic", "Reviewer", "RedTeam",
    "DesignPlan", "ModelResult", "VerifyOutcome", "DFMOutcome", "RedTeamResult",
    "ReviewResult", "Finding", "findings_from", "prioritize", "default_redteam_probe",
    "Supervisor", "Trajectory", "RoundRecord",
    "AsyncOverseer", "Halt",
]
