"""The Idea-to-CAD collaborative workflow: nested empty-feedback loops.

This is the deterministic orchestration heart of Ocker et al. (2025) — the four
nested loops of Algorithms 1-4 and their *empty-feedback fixpoint* termination
rule, wiring the V-model roles (``agents.idea2cad_roles``) over the shared
blackboard (``agents.idea2cad_blackboard``).

The nesting (outermost to innermost):

  Algorithm 4  Human validation loop   — repeat until ``Fval == empty``
    Algorithm 3  Verification loop      — repeat until ``Fver == empty``
      Algorithm 2  Model design loop    — repeat until a model ``M`` is produced
        (uses ``check(C)`` then ``exec(C)``; retrieves doc-hints from feedback)
  Algorithm 1  Interactive requirements — run once up front, repeat until no
                                          ambiguities remain

Each loop converges when *its own* feedback channel empties — this is the
paper's convergence/consensus protocol, distinct from our generic supervisor's
"reviewer-approves-and-no-veto" stop. The two feedback channels compose as
``design(R, Fval + Fver)`` (validation feedback first), which the blackboard's
``combined_feedback`` owns.

Because the paper's loops are ``while true`` with no explicit bound, we add a
per-loop iteration guard (``max_iters``) so a non-converging VLM cannot hang the
system — mirroring the paper's own warning that "simply increasing the number of
iterations may be insufficient" and can pollute the context. When a guard trips,
the round record carries a non-``converged`` stop reason instead of raising.

Fully deterministic with the default heuristic roles: no VLM, no CAD kernel, no
wall clock. stdlib only, absolute imports.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, List, Optional

from harnesscad.agents.agents.blackboard import DesignBlackboard, VPhase
from harnesscad.agents.agents.vmodel_roles import (
    RequirementsEngineer,
    CadEngineer,
    QualityAssuranceEngineer,
    User,
)


# --------------------------------------------------------------------------- #
# trajectory records
# --------------------------------------------------------------------------- #
@dataclass
class RequirementsTrace:
    rounds: int
    converged: bool
    ambiguity_log: List[List[str]] = field(default_factory=list)
    addendum: Optional[str] = None


@dataclass
class DesignTrace:
    attempts: int
    produced: bool
    check_failures: int = 0


@dataclass
class VerifyRound:
    index: int
    design: DesignTrace
    issues: List[str]
    converged: bool


@dataclass
class VerifyTrace:
    rounds: List[VerifyRound] = field(default_factory=list)
    converged: bool = False

    @property
    def round_count(self) -> int:
        return len(self.rounds)


@dataclass
class ValidateRound:
    index: int
    verify: VerifyTrace
    feedback: List[str]
    accepted: bool


@dataclass
class WorkflowResult:
    requirements: RequirementsTrace
    validation_rounds: List[ValidateRound] = field(default_factory=list)
    accepted: bool = False
    stop_reason: str = "not-run"
    model: Any = None

    @property
    def validation_round_count(self) -> int:
        return len(self.validation_rounds)


# --------------------------------------------------------------------------- #
# the workflow
# --------------------------------------------------------------------------- #
class Idea2CadWorkflow:
    """Runs the four nested loops over a shared :class:`DesignBlackboard`.

    Only the roles need configuring (they carry the injectable VLM behaviours);
    all default to deterministic heuristics, so ``Idea2CadWorkflow().run(...)`` is
    fully runnable and testable. Per-loop iteration guards bound each ``while``.
    """

    def __init__(
        self,
        requirements_engineer: Optional[RequirementsEngineer] = None,
        cad_engineer: Optional[CadEngineer] = None,
        qa_engineer: Optional[QualityAssuranceEngineer] = None,
        user: Optional[User] = None,
        *,
        docs: str = "",
        max_clarify_iters: int = 16,
        max_design_iters: int = 8,
        max_verify_iters: int = 8,
        max_validate_iters: int = 8,
    ) -> None:
        self.re = requirements_engineer or RequirementsEngineer()
        self.cad = cad_engineer or CadEngineer()
        self.qa = qa_engineer or QualityAssuranceEngineer()
        self.user = user or User()
        self.docs = docs
        for name, val in (
            ("max_clarify_iters", max_clarify_iters),
            ("max_design_iters", max_design_iters),
            ("max_verify_iters", max_verify_iters),
            ("max_validate_iters", max_validate_iters),
        ):
            if val < 1:
                raise ValueError(f"{name} must be >= 1")
        self.max_clarify_iters = max_clarify_iters
        self.max_design_iters = max_design_iters
        self.max_verify_iters = max_verify_iters
        self.max_validate_iters = max_validate_iters

    # -- Algorithm 1: interactive requirement specification --------------
    def run_requirements(
        self,
        bb: DesignBlackboard,
        user_reply=None,
    ) -> RequirementsTrace:
        """``while ambiguities != empty: T <- T + user.input()`` (Alg. 1).

        ``user_reply(ambiguities, text)`` is the human proxy supplying the next
        clarification turn. Default: no reply (so the loop terminates once the
        clarifier reports no ambiguities, or when the guard trips).
        """
        bb.enter_phase(VPhase.REQUIREMENTS)
        reply = user_reply or (lambda amb, text: "")
        trace = RequirementsTrace(rounds=0, converged=False)
        for _ in range(self.max_clarify_iters):
            ambiguities = self.re.clarify(bb.sketch, bb.text)
            trace.ambiguity_log.append(list(ambiguities))
            trace.rounds += 1
            if not ambiguities:
                trace.converged = True
                break
            more = reply(ambiguities, bb.text)
            if not more:
                # user offered nothing to resolve the ambiguity; stop (the paper
                # lets the VLM make reasonable assumptions in this case).
                break
            bb.append_text(more)
        # emit the addendum R <- (S, T)
        addendum = self.re.summarise(bb.sketch, bb.text)
        if addendum:
            bb.post_addendum(addendum)
        trace.addendum = addendum
        return trace

    # -- Algorithm 2: model design ---------------------------------------
    def run_design(self, bb: DesignBlackboard) -> DesignTrace:
        """``while not M: C <- code(R, hints); if check(C): M <- exec(C)`` (Alg. 2).

        On the first pass with no feedback a plan is produced; when feedback is
        present, doc-hints are retrieved and folded into code generation.
        """
        bb.enter_phase(VPhase.DESIGN)
        bb.post_docs(self.docs)
        feedback = bb.combined_feedback
        if not feedback and bb.plan is None:
            bb.post_plan(self.cad.plan(bb.specification))

        trace = DesignTrace(attempts=0, produced=False)
        bb.post_model(None)
        for _ in range(self.max_design_iters):
            hints = None
            if feedback:
                hints = self.cad.hints_from_docs(bb.code or "", self.docs, feedback)
                bb.post_hints(hints)
            code = self.cad.generate(bb.specification, hints)
            bb.post_code(code)
            trace.attempts += 1
            if not self.cad.check(code):
                trace.check_failures += 1
                continue
            model = self.cad.execute(code)
            if model is not None:
                bb.post_model(model)
                trace.produced = True
                break
        return trace

    # -- Algorithm 3: VLM-based verification -----------------------------
    def run_verification(self, bb: DesignBlackboard) -> VerifyTrace:
        """``while true: M <- design(R, Fval+Fver); Fver <- qa(R, views);
        if Fver == empty: break`` (Alg. 3)."""
        trace = VerifyTrace()
        for i in range(self.max_verify_iters):
            design = self.run_design(bb)
            bb.enter_phase(VPhase.VERIFICATION)
            report = self.qa.review(bb.specification, bb.model)
            bb.post_verification_feedback(report.issues)
            converged = report.acceptable and design.produced
            trace.rounds.append(
                VerifyRound(index=i, design=design, issues=list(report.issues),
                            converged=converged))
            if converged:
                trace.converged = True
                break
            if not design.produced:
                # cannot verify a model that was never built; stop the loop.
                break
        return trace

    # -- Algorithm 4: human validation -----------------------------------
    def run_validation(self, bb: DesignBlackboard) -> List[ValidateRound]:
        """``while true: M <- verify(R, Fval); output(M); Fval <- user.input();
        if Fval == empty: break`` (Alg. 4)."""
        rounds: List[ValidateRound] = []
        for i in range(self.max_validate_iters):
            verify = self.run_verification(bb)
            bb.enter_phase(VPhase.VALIDATION)
            feedback = self.user.validate(bb.specification, bb.model)
            bb.post_validation_feedback(feedback)
            accepted = not feedback and verify.converged
            rounds.append(ValidateRound(index=i, verify=verify,
                                        feedback=list(feedback), accepted=accepted))
            if not feedback:
                break
        return rounds

    # -- top-level driver -------------------------------------------------
    def run(
        self,
        sketch: Optional[str],
        text: str,
        *,
        user_reply=None,
        blackboard: Optional[DesignBlackboard] = None,
    ) -> WorkflowResult:
        """Run the full nested workflow from raw ``(S, T)`` input to a result.

        Returns a :class:`WorkflowResult` with the requirements trace, every
        validation round (each carrying its verification sub-rounds), and the
        final acceptance / stop reason. Deterministic with the default roles.
        """
        bb = blackboard or DesignBlackboard()
        bb.post_input(sketch, text)

        req = self.run_requirements(bb, user_reply=user_reply)
        val_rounds = self.run_validation(bb)

        accepted = bool(val_rounds) and val_rounds[-1].accepted
        if accepted:
            stop_reason = "accepted"
        elif not any(vr.verify.converged for vr in val_rounds):
            stop_reason = "verification-not-converged"
        else:
            stop_reason = "validation-exhausted"

        return WorkflowResult(
            requirements=req,
            validation_rounds=val_rounds,
            accepted=accepted,
            stop_reason=stop_reason,
            model=bb.model,
        )
