"""Compiler-review refine loop (CRM), mined from cad-judge (arXiv:2508.04002).

CRM is cad-judge's inference-time plug-in: it wraps a text-to-CAD generator in a
``generate -> compile-review -> refine`` loop. The compiler classifies each
failure (format / geometry / extrusion / boolean), renders a feedback message,
and the generator is re-prompted with that diagnostic appended. Unlike a VLM
critic, the reviewer here is the deterministic structural grader in
:mod:`harnesscad.eval.judge.compiler_review`, so the refinement signal is a
checkable property, not a vibe.

This module is the deterministic controller: the caller injects a
``generate(prompt) -> op_sequence`` callable, and the loop drives the review /
re-prompt cycle using :func:`~harnesscad.eval.judge.compiler_review.review_sequence`
and :func:`~harnesscad.eval.judge.compiler_review.feedback_message`. No model
calls live here; it is unit-testable with a scripted generator.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Sequence, Tuple

from harnesscad.eval.judge.compiler_review import (
    ReviewResult,
    feedback_message,
    review_sequence,
)

__all__ = [
    "FEEDBACK_PREFIX",
    "build_refine_prompt",
    "RefineStep",
    "RefineResult",
    "run_refine_loop",
]

#: Prefix inserted before the compiler diagnostic when re-prompting.
FEEDBACK_PREFIX = "\n\n[Compiler feedback] "


def build_refine_prompt(base_prompt: str, result: ReviewResult) -> str:
    """Append the compiler diagnostic to the base prompt for a refine pass."""
    return f"{base_prompt}{FEEDBACK_PREFIX}{feedback_message(result)}"


@dataclass(frozen=True)
class RefineStep:
    """One generate+review iteration of the CRM loop."""

    sequence: Sequence[Dict[str, Any]]
    review: ReviewResult


@dataclass(frozen=True)
class RefineResult:
    """Result of the refine loop: the final sequence and the diagnostic trace."""

    sequence: Sequence[Dict[str, Any]]
    ok: bool
    iters: int
    history: Tuple[RefineStep, ...] = field(default=())


def run_refine_loop(
    prompt: str,
    generate: Callable[[str], Sequence[Dict[str, Any]]],
    *,
    max_iters: int = 1,
) -> RefineResult:
    """Run the CRM generate -> review -> refine loop.

    ``generate(prompt) -> op_sequence`` is the (injected) text-to-CAD generator.
    Each pass reviews the produced sequence with the deterministic structural
    compiler; on a passing review the loop stops, otherwise the diagnostic is
    appended to the prompt and the generator is called again, up to
    ``max_iters`` refinements after the initial generation (``max_iters=0`` is a
    vanilla single pass, ``1`` is the paper default).
    """
    history: List[RefineStep] = []
    current_prompt = prompt
    sequence = generate(current_prompt)
    review = review_sequence(sequence)
    history.append(RefineStep(sequence, review))
    it = 0
    while not review.ok and it < max_iters:
        current_prompt = build_refine_prompt(prompt, review)
        sequence = generate(current_prompt)
        review = review_sequence(sequence)
        history.append(RefineStep(sequence, review))
        it += 1
    return RefineResult(sequence, review.ok, it, tuple(history))
