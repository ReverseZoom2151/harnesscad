"""clarify_scaling -- detect the "scale the sketch" failure mode.

A systematic failure mode: such prompts frequently contain an explicit *scaling
operation* placed **immediately after the 2D sketch construction and before the
3D extrusion**. In a sketch-based CAD scripting API this is invalid -- the
workplane object has no ``scale`` method -- so weak code generators hallucinate
a workplane ``scale(...)`` call and emit non-executable programs. Worse,
the scaling statement admits two conflicting interpretations:

  * **Interpretation A** (literal post-sketch scaling): multiply the sketch by
    the factor, changing the footprint -- which then contradicts the stated
    final dimensions.
  * **Interpretation B** (parameters already in target units): the scaling step
    is redundant and should be ignored.

This module deterministically detects that pattern in an ordered build-step
sequence, decides which interpretation the *stated final dimensions* support,
and rewrites the step list into a safe, unambiguous form (drop the scale when it
is redundant, or fold it into the sketch dimensions when it is literal).

Stdlib-only, no LLM.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

# Build-step kinds we recognise in an ordered instruction list.
SKETCH = "sketch"
SCALE = "scale"
EXTRUDE = "extrude"
ROTATE = "rotate"
TRANSLATE = "translate"

_SCALE_TEXT_RE = re.compile(
    r"scal(?:e|ing)\b.*?(?:factor\s+of\s+)?(-?\d+(?:\.\d+)?)", re.I)
_SKETCH_WORDS = ("sketch", "circle", "rect", "polyline", "segment", "draw",
                 "profile", "loop")
_EXTRUDE_WORDS = ("extrude", "extrusion")


@dataclass
class BuildStep:
    """One ordered build instruction."""

    kind: str
    text: str = ""
    factor: Optional[float] = None  # for SCALE steps


@dataclass(frozen=True)
class ScalingIssue:
    """A detected pre-extrusion scaling ambiguity."""

    index: int                     # position of the scale step
    factor: float
    interpretation: str            # "literal" | "redundant" | "unknown"
    reason: str
    hallucination_risk: bool       # would prompt Workplane.scale(...) ?


def parse_step(text: str) -> BuildStep:
    """Classify a free-text build instruction into a :class:`BuildStep`."""
    low = text.lower()
    m = _SCALE_TEXT_RE.search(low)
    if m and "scal" in low:
        return BuildStep(SCALE, text, float(m.group(1)))
    if any(w in low for w in _EXTRUDE_WORDS):
        return BuildStep(EXTRUDE, text)
    if "rotate" in low:
        return BuildStep(ROTATE, text)
    if "translate" in low:
        return BuildStep(TRANSLATE, text)
    if any(w in low for w in _SKETCH_WORDS):
        return BuildStep(SKETCH, text)
    return BuildStep(SKETCH, text)


def parse_steps(texts: Sequence[str]) -> List[BuildStep]:
    return [parse_step(t) for t in texts]


def _dims_consistent_with_scale(sketch_extent: float, factor: float,
                                final_extent: float, tol: float = 1e-3) -> bool:
    """True if literal scaling reproduces the stated final extent."""
    return abs(sketch_extent * factor - final_extent) <= tol * max(1.0, final_extent)


def detect_scaling(steps: Sequence[BuildStep],
                   *, sketch_extent: Optional[float] = None,
                   final_extent: Optional[float] = None) -> List[ScalingIssue]:
    """Find scale steps sitting between the sketch and the extrusion.

    When ``sketch_extent`` and ``final_extent`` are supplied, decide whether the
    literal interpretation (A) reproduces the stated final dimensions; if not,
    the scale is redundant (B).
    """
    issues: List[ScalingIssue] = []
    # index of the first extrude and last sketch before it.
    extrude_idx = next((i for i, s in enumerate(steps) if s.kind == EXTRUDE),
                       None)
    for i, step in enumerate(steps):
        if step.kind != SCALE:
            continue
        pre_extrude = extrude_idx is None or i < extrude_idx
        has_prior_sketch = any(s.kind == SKETCH for s in steps[:i])
        if not (pre_extrude and has_prior_sketch):
            # scale after the solid exists is a legitimate operation.
            continue
        factor = step.factor if step.factor is not None else 1.0
        interp, reason = _interpret(factor, sketch_extent, final_extent)
        issues.append(ScalingIssue(
            index=i,
            factor=factor,
            interpretation=interp,
            reason=reason,
            hallucination_risk=True,
        ))
    return issues


def _interpret(factor: float, sketch_extent: Optional[float],
               final_extent: Optional[float]) -> Tuple[str, str]:
    if factor == 1.0:
        return ("redundant", "Scale factor 1.0 is a no-op.")
    if sketch_extent is not None and final_extent is not None:
        if _dims_consistent_with_scale(sketch_extent, factor, final_extent):
            return ("literal",
                    "Sketch extent x factor equals the stated final extent.")
        if abs(sketch_extent - final_extent) <= 1e-3 * max(1.0, final_extent):
            return ("redundant",
                    "Sketch extent already equals the final extent; scale "
                    "would contradict the stated dimensions.")
        return ("redundant",
                "Literal scaling contradicts the stated final dimensions.")
    return ("unknown",
            "Ambiguous: pre-extrusion scale with two plausible interpretations.")


def rewrite_steps(steps: Sequence[BuildStep],
                  *, sketch_extent: Optional[float] = None,
                  final_extent: Optional[float] = None) -> List[BuildStep]:
    """Return a safe, unambiguous step list with pre-extrusion scaling removed.

    A *redundant* scale is dropped; a *literal* scale is annotated onto the
    sketch step so the factor is folded into the sketch dimensions rather than
    applied to a workplane (which has no ``scale`` API).
    """
    issues = {iss.index: iss for iss in
              detect_scaling(steps, sketch_extent=sketch_extent,
                             final_extent=final_extent)}
    out: List[BuildStep] = []
    for i, step in enumerate(steps):
        iss = issues.get(i)
        if iss is None:
            out.append(BuildStep(step.kind, step.text, step.factor))
            continue
        if iss.interpretation == "literal":
            out.append(BuildStep(
                SKETCH,
                "Apply scale factor {0} to the sketch dimensions (fold into "
                "sketch geometry, not the workplane).".format(iss.factor),
                iss.factor))
        # redundant / unknown -> drop the scale step.
    return out
