"""Cost-aware progressive escalation policy for a text-to-CAD pipeline.

The module's headline design rule is a
*progressive* pipeline: one generation call on the happy path, a repair attempt
that runs **only after a validation failure**, and a visual (VLM) repair that
runs **only when the user explicitly requests it**. The AgentSCAD README states
it plainly -- "one LLM call generates structured CAD intent and OpenSCAD by
default. Repair runs only after validation failure, and visual repair runs only
when the user requests it."

The transferable, model-free core is the *escalation decision*: given the
outcome of the current stage (a validation report + a cost budget + whether the
user asked for visual repair), decide the next stage deterministically. This
module is that decision as a small state machine, with explicit cost accounting
so the "cost-aware defaults" claim is checkable rather than aspirational.

The state machine (AgentSCAD's runtime):

    INTAKE -> GENERATE -> RENDER -> VALIDATE -> (deliver | escalate)

    escalate, in strict order and each at most once:
        1. one REPAIR attempt, only if VALIDATE found a *critical* failure;
        2. one VISUAL_REPAIR attempt, only if the user requested it AND a
           vision provider is available;
        3. otherwise HUMAN_REVIEW.

Design rules that make this a *harness* piece and not a demo:

*   **Deterministic.** No wall clock, no network, no model. The same inputs
    always pick the same next stage.
*   **Budgeted.** Every stage has a fixed cost; escalation stops the moment the
    remaining budget cannot pay for the cheapest useful next stage. A stage is
    never entered "for free".
*   **Idempotent stages.** REPAIR and VISUAL_REPAIR each fire at most once --
    the policy tracks which escalations were already spent, so a loop that keeps
    failing does not keep paying for the same repair forever.
*   **Uncertainty is not a pass.** A skipped visual check (no provider) is
    recorded as uncertainty and routed to HUMAN_REVIEW, never silently treated
    as success (AgentSCAD: "Missing visual provider support is treated as
    uncertainty, not as a blocking pass").

stdlib-only, absolute imports.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Dict, List, Optional, Sequence, Tuple

__all__ = [
    "Stage",
    "STAGES",
    "ValidationCheck",
    "validation_passed",
    "has_critical_failure",
    "DEFAULT_STAGE_COST",
    "EscalationConfig",
    "PipelineState",
    "decide_next_stage",
    "advance",
    "run_policy",
]

# Ordered pipeline stages (AgentSCAD's execute-cad-job state machine).
INTAKE = "INTAKE"
GENERATE = "GENERATE"
RENDER = "RENDER"
VALIDATE = "VALIDATE"
REPAIR = "REPAIR"
VISUAL_REPAIR = "VISUAL_REPAIR"
DELIVER = "DELIVER"
HUMAN_REVIEW = "HUMAN_REVIEW"

Stage = str

STAGES: Tuple[Stage, ...] = (
    INTAKE,
    GENERATE,
    RENDER,
    VALIDATE,
    REPAIR,
    VISUAL_REPAIR,
    DELIVER,
    HUMAN_REVIEW,
)

# Per-stage nominal cost (unit-free "budget points"; RENDER/VALIDATE are cheap
# deterministic tools, GENERATE/REPAIR spend a model call, VISUAL_REPAIR spends a
# vision call and is the most expensive).
DEFAULT_STAGE_COST: Dict[Stage, float] = {
    INTAKE: 0.0,
    GENERATE: 1.0,
    RENDER: 0.1,
    VALIDATE: 0.1,
    REPAIR: 1.0,
    VISUAL_REPAIR: 2.0,
    DELIVER: 0.0,
    HUMAN_REVIEW: 0.0,
}


@dataclass(frozen=True)
class ValidationCheck:
    """One deterministic validation result (AgentSCAD's ValidationCheck).

    ``passed`` is the outcome; ``is_critical`` marks a check whose failure must
    block delivery and trigger repair (AgentSCAD's mesh/manifold/hole checks are
    critical; drafting hints are not).
    """

    rule_id: str
    passed: bool
    is_critical: bool = False
    message: str = ""


def validation_passed(checks: Sequence[ValidationCheck]) -> bool:
    """True when no *critical* check failed. Non-critical failures do not block."""
    return not has_critical_failure(checks)


def has_critical_failure(checks: Sequence[ValidationCheck]) -> bool:
    """True when at least one critical check failed."""
    return any((not c.passed) and c.is_critical for c in checks)


@dataclass(frozen=True)
class EscalationConfig:
    """Policy configuration.

    ``budget`` is total spend allowed for the whole run. ``visual_repair_requested``
    gates the (expensive) VLM stage -- AgentSCAD never runs it unprompted.
    ``vision_available`` reflects whether a vision provider is configured; when a
    visual repair is wanted but unavailable, the run routes to HUMAN_REVIEW.
    """

    budget: float = 3.0
    visual_repair_requested: bool = False
    vision_available: bool = True
    stage_cost: Dict[Stage, float] = field(default_factory=lambda: dict(DEFAULT_STAGE_COST))

    def cost_of(self, stage: Stage) -> float:
        return float(self.stage_cost.get(stage, 0.0))


@dataclass(frozen=True)
class PipelineState:
    """Immutable snapshot of pipeline progress.

    ``spent`` is cumulative cost; ``repair_used`` / ``visual_repair_used`` make
    each escalation fire at most once; ``checks`` is the latest VALIDATE report.
    """

    stage: Stage = INTAKE
    spent: float = 0.0
    repair_used: bool = False
    visual_repair_used: bool = False
    checks: Tuple[ValidationCheck, ...] = ()
    history: Tuple[Stage, ...] = ()

    def with_checks(self, checks: Sequence[ValidationCheck]) -> "PipelineState":
        return replace(self, checks=tuple(checks))


def _affordable(cfg: EscalationConfig, state: PipelineState, stage: Stage) -> bool:
    return state.spent + cfg.cost_of(stage) <= cfg.budget + 1e-9


def decide_next_stage(state: PipelineState, cfg: EscalationConfig) -> Stage:
    """Pure escalation decision: given the current state, what runs next.

    The linear spine (INTAKE->GENERATE->RENDER->VALIDATE) advances stage by
    stage as long as the budget can pay for the next stage; if it cannot, the
    run stops at HUMAN_REVIEW rather than skipping a stage.

    After VALIDATE (or a repair's re-VALIDATE) the branch logic runs:

    * validation clean            -> DELIVER
    * critical failure, repair
      not yet used, affordable    -> REPAIR
    * else visual repair wanted,
      not used, provider present,
      affordable                  -> VISUAL_REPAIR
    * visual repair wanted but no
      provider                    -> HUMAN_REVIEW (uncertainty, never a pass)
    * otherwise                   -> HUMAN_REVIEW
    """
    stage = state.stage

    if stage == INTAKE:
        return GENERATE if _affordable(cfg, state, GENERATE) else HUMAN_REVIEW
    if stage == GENERATE:
        return RENDER if _affordable(cfg, state, RENDER) else HUMAN_REVIEW
    if stage == RENDER:
        return VALIDATE if _affordable(cfg, state, VALIDATE) else HUMAN_REVIEW

    if stage in (VALIDATE, REPAIR, VISUAL_REPAIR):
        # A repair/visual-repair stage is always followed by a fresh RENDER +
        # VALIDATE in the real pipeline; here the caller feeds the re-validated
        # ``checks`` back in, so we branch directly on them.
        if validation_passed(state.checks):
            return DELIVER

        if has_critical_failure(state.checks):
            if not state.repair_used and _affordable(cfg, state, REPAIR):
                return REPAIR

        if cfg.visual_repair_requested and not state.visual_repair_used:
            if not cfg.vision_available:
                # Wanted but no provider -> uncertainty, not a silent pass.
                return HUMAN_REVIEW
            if _affordable(cfg, state, VISUAL_REPAIR):
                return VISUAL_REPAIR

        return HUMAN_REVIEW

    # Terminal stages have no successor.
    return stage


def advance(state: PipelineState, cfg: EscalationConfig) -> PipelineState:
    """Move to the next stage, charging its cost and marking escalations used."""
    nxt = decide_next_stage(state, cfg)
    if nxt == state.stage:
        return state
    repair_used = state.repair_used or nxt == REPAIR
    visual_used = state.visual_repair_used or nxt == VISUAL_REPAIR
    return replace(
        state,
        stage=nxt,
        spent=state.spent + cfg.cost_of(nxt),
        repair_used=repair_used,
        visual_repair_used=visual_used,
        history=state.history + (nxt,),
    )


def run_policy(
    cfg: EscalationConfig,
    validate_outcomes: Sequence[Sequence[ValidationCheck]],
    *,
    max_steps: int = 32,
) -> PipelineState:
    """Drive the state machine to a terminal stage.

    ``validate_outcomes`` supplies the VALIDATE report to use each time a
    VALIDATE (or post-repair re-VALIDATE) stage is entered, in order. When the
    outcomes are exhausted, the last one is reused (a repair that changes nothing
    fails the same way -- which is exactly why REPAIR fires at most once).

    Returns the terminal :class:`PipelineState` (stage in {DELIVER, HUMAN_REVIEW}).
    """
    state = PipelineState(history=(INTAKE,))
    outcomes = list(validate_outcomes)
    vi = 0
    for _ in range(max_steps):
        if state.stage in (DELIVER, HUMAN_REVIEW):
            return state
        state = advance(state, cfg)
        # VALIDATE and every post-repair re-validation consume the next outcome.
        if state.stage in (VALIDATE, REPAIR, VISUAL_REPAIR):
            picked = outcomes[vi] if vi < len(outcomes) else (outcomes[-1] if outcomes else ())
            vi += 1
            state = state.with_checks(picked)
    return state
