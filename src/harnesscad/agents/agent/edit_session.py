"""Multi-turn, approval-gated editing over a :class:`HarnessSession`.

This module has no model dependency.  A caller supplies proposed typed ops,
reviews the semantic preview, and explicitly approves or rejects it.  Failed
batches are rolled back atomically to their pre-edit checkpoint.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Any, Optional, Sequence

from harnesscad.core.cisp.ops import Op, canonical_json
from harnesscad.eval.quality.edit.diff import OpDiff, op_diff


@dataclass(frozen=True)
class Turn:
    role: str
    content: str
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {"role": self.role, "content": self.content, "metadata": dict(self.metadata)}


@dataclass
class EditProposal:
    id: str
    request: str
    ops: tuple[Op, ...]
    base_digest: str
    base_count: int
    diff: OpDiff
    status: str = "pending"
    result: Any = None

    def preview(self) -> dict:
        return {
            "id": self.id,
            "request": self.request,
            "status": self.status,
            "base_digest": self.base_digest,
            "op_count": len(self.ops),
            "ops": [op.to_dict() for op in self.ops],
            "diff": self.diff.to_dict(),
            "summary": self.diff.render(),
            "requires_approval": self.status == "pending",
        }


class EditSession:
    """Conversation and edit lifecycle around an existing geometry session."""

    def __init__(self, harness_session) -> None:
        self.session = harness_session
        self.turns: list[Turn] = []
        self.proposals: list[EditProposal] = []
        self._sequence = 0

    @property
    def current_ops(self) -> tuple[Op, ...]:
        return tuple(self.session.opdag.ops())

    @property
    def current_digest(self) -> str:
        return self.session.digest()

    def add_turn(self, role: str, content: str, **metadata: Any) -> Turn:
        if role not in {"user", "assistant", "system", "tool"}:
            raise ValueError(f"unsupported role: {role}")
        turn = Turn(role, str(content), dict(metadata))
        self.turns.append(turn)
        return turn

    def propose(self, request: str, ops: Sequence[Op]) -> EditProposal:
        """Stage ops and return a semantic preview without mutating geometry."""
        staged = tuple(ops)
        if not staged:
            raise ValueError("an edit proposal must contain at least one op")
        self._sequence += 1
        base = self.current_ops
        blob = "|".join(canonical_json(op) for op in staged)
        proposal_id = hashlib.sha256(
            f"{self._sequence}|{self.current_digest}|{blob}".encode("utf-8")
        ).hexdigest()[:16]
        proposal = EditProposal(
            id=f"edit-{proposal_id}",
            request=str(request),
            ops=staged,
            base_digest=self.current_digest,
            base_count=len(base),
            diff=op_diff(base, base + staged),
        )
        self.proposals.append(proposal)
        self.add_turn("user", request)
        self.add_turn("assistant", proposal.diff.render(), proposal_id=proposal.id, preview=True)
        return proposal

    def approve(self, proposal_id: str):
        """Apply a pending proposal atomically after explicit approval."""
        proposal = self._find(proposal_id)
        if proposal.status != "pending":
            raise ValueError(f"proposal {proposal_id} is {proposal.status}")
        if self.current_digest != proposal.base_digest or len(self.current_ops) != proposal.base_count:
            proposal.status = "stale"
            self.add_turn("system", "proposal became stale", proposal_id=proposal.id)
            raise RuntimeError("proposal base no longer matches current design")

        checkpoint = f"edit-base-{proposal.id}"
        self.session.checkpoint(checkpoint)
        result = self.session.apply_ops(list(proposal.ops))
        proposal.result = result
        if not result.ok or result.applied != len(proposal.ops):
            # HarnessSession rolls back the rejected op, but earlier ops in the
            # same batch may have succeeded. Restore the entire edit boundary.
            self.session.rollback(checkpoint)
            proposal.status = "failed"
            self.add_turn(
                "tool", "edit failed and was rolled back",
                proposal_id=proposal.id,
                diagnostics=[d.to_dict() for d in result.diagnostics],
            )
            return result

        proposal.status = "applied"
        self.add_turn("tool", "edit applied", proposal_id=proposal.id, digest=result.digest)
        return result

    def reject(self, proposal_id: str, reason: Optional[str] = None) -> EditProposal:
        proposal = self._find(proposal_id)
        if proposal.status != "pending":
            raise ValueError(f"proposal {proposal_id} is {proposal.status}")
        proposal.status = "rejected"
        self.add_turn("user", reason or "edit rejected", proposal_id=proposal.id)
        return proposal

    def history(self) -> dict:
        return {
            "digest": self.current_digest,
            "turns": [turn.to_dict() for turn in self.turns],
            "proposals": [proposal.preview() for proposal in self.proposals],
        }

    def _find(self, proposal_id: str) -> EditProposal:
        for proposal in self.proposals:
            if proposal.id == proposal_id:
                return proposal
        raise KeyError(proposal_id)
