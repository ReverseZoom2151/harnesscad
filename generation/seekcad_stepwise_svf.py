"""Step-wise Visual Feedback (SVF) plan builder for Seek-CAD.

Deterministic re-implementation of the step-wise visual feedback strategy from
"Seek-CAD" (Li et al., ICLR 2026), Section 3.2, and the refinement controller
of Section 3.1(5).

The novelty over prior training-free refiners (3D-Premise, CADCodeVerify) is
that feedback is computed over a *sequence* of step-wise renderings rather than
a single final image, and the VLM judges alignment against the DeepSeek-R1
chain-of-thought (CoT) rather than the raw description.  For an SSR model of n
triples S_1..S_n, the render plan is

    M_I = [R(S_1), R(S1bar (+) S_2), ..., R(S1bar (+) ... (+) S_n)]      (Eq. 4)
    M_U = R(S_1 (+) S_2 (+) ... (+) S_n)                                (Eq. 5)
    M   = [M_I, M_U]                                                    (Eq. 6)

where at intermediate step k the *current* triplet S_k is highlighted while all
prior triplets S_j (j < k) are hidden/dimmed (rendered as Sbar) to avoid
occlusion; M_U renders the complete model with nothing hidden.

The actual rendering (PythonOCC) and the VLM judge (Gemini) are external.  This
module builds the deterministic *plan* — which triplets are visible, which is
highlighted, and how each rendered step pairs with its CoT step t_k — and drives
the refinement loop of Eq. 3 with an injectable judge callable.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, List, Optional, Sequence

# Feedback verdicts F_call (Sec 3.2(2)); L=1 positive, L=0 negative.
POSITIVE = "positive"
NEGATIVE = "negative"


@dataclass(frozen=True)
class StepRender:
    """One entry of the step-wise plan.

    ``visible`` lists the triplet indices drawn; ``highlighted`` is the current
    triplet index emphasised at this step (None for the ultimate render);
    ``hidden`` lists the dimmed prior-triplet indices; ``cot_step`` is the index
    of the paired CoT thought t_k (None when no CoT covers this step).
    """

    kind: str  # "intermediate" or "ultimate"
    step: int
    visible: tuple
    highlighted: Optional[int]
    hidden: tuple
    cot_step: Optional[int]


def build_svf_plan(n_triples: int, cot_len: int = 0) -> List[StepRender]:
    """Construct the M = [M_I, M_U] render plan (Eqs. 4-6).

    ``n_triples`` is n; ``cot_len`` is the number of CoT thoughts available so
    each intermediate step k pairs with t_k when present (Sec 3.2(2)).  The
    intermediate steps come first (in construction order), followed by the
    single ultimate step.
    """
    if n_triples < 1:
        raise ValueError("need at least one triplet")
    if cot_len < 0:
        raise ValueError("cot_len must be non-negative")
    plan: List[StepRender] = []
    for k in range(n_triples):
        visible = tuple(range(k + 1))
        hidden = tuple(range(k))  # priors S_j, j < k are hidden/dimmed
        cot_step = k if k < cot_len else None
        plan.append(
            StepRender(
                kind="intermediate",
                step=k,
                visible=visible,
                highlighted=k,
                hidden=hidden,
                cot_step=cot_step,
            )
        )
    # Ultimate render: whole model, nothing hidden, no single highlight.
    plan.append(
        StepRender(
            kind="ultimate",
            step=n_triples,
            visible=tuple(range(n_triples)),
            highlighted=None,
            hidden=(),
            cot_step=None,
        )
    )
    return plan


def intermediate_steps(plan: Sequence[StepRender]) -> List[StepRender]:
    """The M_I subsequence."""
    return [s for s in plan if s.kind == "intermediate"]


def ultimate_step(plan: Sequence[StepRender]) -> StepRender:
    """The single M_U render."""
    for s in plan:
        if s.kind == "ultimate":
            return s
    raise ValueError("plan has no ultimate step")


@dataclass
class RefinementResult:
    """Outcome of the SVF refinement loop (Eq. 3)."""

    code: object
    rounds: int
    verdicts: List[str] = field(default_factory=list)
    converged: bool = False


def refine_with_svf(
    initial_code: object,
    *,
    n_triples: int,
    judge_fn: Callable[[object, List[StepRender]], str],
    refine_fn: Callable[[object, str], object],
    cot_len: int = 0,
    max_rounds: int = 1,
) -> RefinementResult:
    """Drive the Eq. 3 refinement loop.

    Each round rebuilds the SVF plan for the current code and asks
    ``judge_fn(code, plan) -> verdict`` (POSITIVE/NEGATIVE).  On POSITIVE the
    code is accepted (L=1, no change).  On NEGATIVE, ``refine_fn(code,
    feedback) -> new_code`` produces the next iterate; the paper caps this at
    ``max_rounds`` (N=1 in practice) to avoid the compile-failure inflation
    reported in Table 2.
    """
    if max_rounds < 1:
        raise ValueError("max_rounds must be >= 1")
    code = initial_code
    verdicts: List[str] = []
    rounds = 0
    for _ in range(max_rounds):
        plan = build_svf_plan(n_triples, cot_len)
        verdict = judge_fn(code, plan)
        if verdict not in (POSITIVE, NEGATIVE):
            raise ValueError("judge must return POSITIVE or NEGATIVE")
        verdicts.append(verdict)
        rounds += 1
        if verdict == POSITIVE:
            return RefinementResult(code, rounds, verdicts, converged=True)
        code = refine_fn(code, verdict)
    return RefinementResult(code, rounds, verdicts, converged=False)
