"""Pro-CAD two-agent protocol scaffold -- the deterministic message/critique loop.

Pro-CAD frames text-to-CAD as a conversation between **two roles**:

*   a **Designer** (a.k.a. generator/planner) that proposes a CAD build from the
    brief, and
*   a **Reviewer** (a.k.a. critic/checker) that inspects the proposal, raises
    issues, and either requests a revision or approves.

They exchange messages until the Reviewer approves or a round budget is hit.  The
*proactive ambiguity* half of that idea already lives in
:mod:`harnesscad.domain.spec.clarify_ambiguity`; what was missing is the
**protocol scaffold** -- the turn-taking state machine, the typed message and
critique objects, the termination rules, and a full transcript -- that carries
those two roles' messages back and forth.

This module is *only* the scaffold.  It calls **no model**: the Designer and
Reviewer are injected callables (``Callable`` personas).  Deterministic default
personas are provided so the whole protocol runs, and is tested, offline: the
default Reviewer applies rule-based critiques (empty proposal, unresolved
ambiguity markers, exceeded op budget) and the default Designer applies the
Reviewer's revision requests mechanically.  Swap in an LLM-backed persona and the
same state machine drives a real two-agent Pro-CAD run.

Absolute imports, stdlib only, no network, no wall clock.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Callable, List, Optional, Sequence, Tuple

__all__ = [
    "Role",
    "Verdict",
    "Message",
    "Critique",
    "Proposal",
    "TwoAgentResult",
    "Designer",
    "Reviewer",
    "default_designer",
    "default_reviewer",
    "TwoAgentProtocol",
]


class Role(str, Enum):
    DESIGNER = "designer"
    REVIEWER = "reviewer"


class Verdict(str, Enum):
    APPROVE = "approve"
    REVISE = "revise"
    REJECT = "reject"       # unrecoverable -- stop without approval


# --------------------------------------------------------------------------- #
# typed message vocabulary                                                     #
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Proposal:
    """A Designer's proposed build: a brief plus an ordered list of op strings."""

    brief: str
    ops: Tuple[str, ...] = ()
    revision: int = 0

    def with_ops(self, ops: Sequence[str], revision: int) -> "Proposal":
        return Proposal(brief=self.brief, ops=tuple(ops), revision=revision)


@dataclass(frozen=True)
class Critique:
    """A Reviewer's assessment of a proposal."""

    verdict: Verdict
    issues: Tuple[str, ...] = ()      # human-readable problems
    requests: Tuple[str, ...] = ()    # concrete revision instructions

    @property
    def approved(self) -> bool:
        return self.verdict is Verdict.APPROVE


@dataclass(frozen=True)
class Message:
    """One turn in the transcript."""

    role: Role
    round: int
    proposal: Optional[Proposal] = None
    critique: Optional[Critique] = None


@dataclass(frozen=True)
class TwoAgentResult:
    """Outcome of a full two-agent run."""

    approved: bool
    rounds: int
    final_proposal: Proposal
    final_critique: Critique
    transcript: Tuple[Message, ...]

    @property
    def final_ops(self) -> Tuple[str, ...]:
        return self.final_proposal.ops


# persona callables.
Designer = Callable[[Proposal, Optional[Critique]], Proposal]
Reviewer = Callable[[Proposal], Critique]


# --------------------------------------------------------------------------- #
# deterministic default personas                                              #
# --------------------------------------------------------------------------- #
_AMBIGUOUS_MARKERS = ("?", "TODO", "some", "a few", "roughly", "about")


def default_reviewer(max_ops: int = 32) -> Reviewer:
    """A rule-based Reviewer: rejects empty builds, flags ambiguity markers and
    over-budget op counts, and asks for a concrete revision for each issue."""

    def review(proposal: Proposal) -> Critique:
        issues: List[str] = []
        requests: List[str] = []
        if not proposal.ops:
            issues.append("proposal has no operations")
            requests.append("emit at least one operation")
        for i, op in enumerate(proposal.ops):
            low = op.lower()
            for marker in _AMBIGUOUS_MARKERS:
                if marker.lower() in low:
                    issues.append(f"op {i} contains unresolved marker {marker!r}: {op}")
                    requests.append(f"resolve {marker!r} in op {i}")
        if len(proposal.ops) > max_ops:
            issues.append(f"op budget exceeded ({len(proposal.ops)} > {max_ops})")
            requests.append(f"reduce to at most {max_ops} operations")
        if not issues:
            return Critique(verdict=Verdict.APPROVE)
        return Critique(
            verdict=Verdict.REVISE,
            issues=tuple(issues),
            requests=tuple(requests),
        )

    return review


def default_designer() -> Designer:
    """A mechanical Designer: on a revision request it strips ambiguity markers
    from its ops (satisfying the default Reviewer), preserving order."""

    def design(proposal: Proposal, critique: Optional[Critique]) -> Proposal:
        if critique is None or critique.approved:
            return proposal
        cleaned: List[str] = []
        for op in proposal.ops:
            new_op = op
            for marker in _AMBIGUOUS_MARKERS:
                new_op = new_op.replace(marker, "").replace(marker.lower(), "")
            cleaned.append(" ".join(new_op.split()))
        return proposal.with_ops(cleaned, revision=proposal.revision + 1)

    return design


# --------------------------------------------------------------------------- #
# the protocol                                                                 #
# --------------------------------------------------------------------------- #
@dataclass
class TwoAgentProtocol:
    """Turn-taking scaffold that drives a Designer/Reviewer conversation.

    ``designer`` and ``reviewer`` are injected personas (defaulting to the
    deterministic ones).  :meth:`run` alternates Designer -> Reviewer up to
    ``max_rounds`` and stops on the first ``APPROVE`` or ``REJECT``.
    """

    designer: Designer = field(default_factory=default_designer)
    reviewer: Reviewer = field(default_factory=default_reviewer)
    max_rounds: int = 4

    def run(self, initial: Proposal) -> TwoAgentResult:
        transcript: List[Message] = []
        proposal = initial
        critique: Optional[Critique] = None

        for rnd in range(1, self.max_rounds + 1):
            # Designer turn: first round emits the initial proposal as-is; later
            # rounds revise in response to the previous critique.
            proposal = self.designer(proposal, critique)
            transcript.append(
                Message(role=Role.DESIGNER, round=rnd, proposal=proposal))

            # Reviewer turn.
            critique = self.reviewer(proposal)
            transcript.append(
                Message(role=Role.REVIEWER, round=rnd, critique=critique))

            if critique.verdict in (Verdict.APPROVE, Verdict.REJECT):
                return TwoAgentResult(
                    approved=critique.approved,
                    rounds=rnd,
                    final_proposal=proposal,
                    final_critique=critique,
                    transcript=tuple(transcript),
                )

        # budget exhausted without approval.
        return TwoAgentResult(
            approved=False,
            rounds=self.max_rounds,
            final_proposal=proposal,
            final_critique=critique if critique is not None else Critique(Verdict.REVISE),
            transcript=tuple(transcript),
        )
