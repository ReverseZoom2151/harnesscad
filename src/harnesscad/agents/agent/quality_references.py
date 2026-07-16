"""State-conditioned reference injection for the agent loop's context builder.

Ported from CadAgent ``agent/references.py`` (CadAgent-main): instead of
relying on the model to load reference docs on demand, the loop's context
builder injects short reference snippets AUTOMATICALLY based on agent state
(iteration number, quality-gate results, error history). Two pieces:

1. **QUALITY_FIX_MAP** -- the source's quality-gate-code -> terse-fix table,
   imported verbatim (unicode ">=0.5mm" rendered as ASCII; wording otherwise
   unchanged). Each entry maps a quality issue code (NO_SOLID, MULTI_SOLID,
   ...) to a concrete repair instruction the model can act on directly.

2. **State-conditioned injector** -- pure functions reproducing the source's
   ``_build_context()`` policy:
     * first iteration -> quick-start pitfall checklist;
     * quality-gate failure -> the per-code fixes for exactly the failed codes;
     * repeated identical errors -> the "change ONE thing at a time" repair
       strategy shift;
     * 3+ failures on the same target -> "undo and take a completely
       different construction approach";
     * approaching the iteration limit -> urgency nudge (simplify, prioritize
       a valid solid over feature completeness);
     * quality passed with warnings -> success-path reinforcement.

Every snippet stays terse (the source budgets ~300 tokens each). The loop
consults :func:`references_for_state` and joins the returned snippets into
its context; nothing here talks to a model or a kernel.

Attribution: CadAgent (agent/references.py). Pure stdlib, deterministic.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

# --------------------------------------------------------------------------- #
# Repair hint mapping (source: QUALITY_FIX_MAP, verbatim modulo ASCII)
# --------------------------------------------------------------------------- #

QUALITY_FIX_MAP: Dict[str, str] = {
    "NO_SOLID": (
        "Use solid primitives (box/cylinder/sphere/torus). "
        "Ensure extrude() has height > 0 and profile is closed. "
        "Do not use open wires or shells."
    ),
    "MULTI_SOLID": (
        "Shapes must physically overlap by >=0.5mm for union/fuse to work. "
        "Translate one shape INTO the other. Example: .translate((0, 0, 2)) "
        "to extend by 2mm into the other shape."
    ),
    "COMPOUND_SHAPE": (
        "Extract the solid before operations: solid = shape.Solids[0]. "
        "Then use solid for boolean ops."
    ),
    "INVALID_SHAPE": (
        "Boolean operation produced invalid geometry. Try: "
        "(1) increase overlap distance, "
        "(2) change boolean order, "
        "(3) simplify geometry."
    ),
    "NEGATIVE_VOLUME": (
        "Inside-out geometry. Reverse construction order or check "
        "that cut subtracts from the larger body."
    ),
    "DIMENSION_SUSPICIOUS": (
        "Check for unit errors. All dimensions are in mm. "
        "Verify dimensions match the requirements."
    ),
    "NO_DOCUMENT": (
        "No active document. Create one: "
        "doc = FreeCAD.newDocument('Model')"
    ),
    "MULTIPLE_OBJECTS": (
        "Multiple shape objects in document -- this is a warning, not a "
        "failure. The display call automatically cleans up previous objects. "
        "If objects persist, fuse all shapes into one solid using .union()."
    ),
}

# --------------------------------------------------------------------------- #
# Phase-aware reference snippets (source constants, ASCII-normalized)
# --------------------------------------------------------------------------- #

REF_FIRST_ITERATION = """\
FIRST EXECUTION CHECKLIST:
- Start from the documented workplane/session entry point; do not re-import
  the kernel yourself.
- Check every primitive's ARGUMENT ORDER against its signature (e.g.
  cylinder takes HEIGHT first, RADIUS second).
- translate((x, y, z)) takes a TUPLE, not separate arguments.
- For fuse/union: shapes MUST physically overlap by >=0.5mm.
- End by binding the finished solid to the required result/display call."""

REF_REPAIR_LOOP = """\
REPAIR STRATEGY -- you have repeated errors, change your approach:
1. Identify the ROOT CAUSE, not just the symptom.
2. Change ONE thing at a time (overlap distance, boolean order, geometry type).
3. If boolean fails: try increasing overlap, or use a different construction
   method.
4. If 3+ failures on same shape: undo the last step and try a completely
   different construction approach.
5. Consider building a simpler version first, then adding complexity."""

REF_DIFFERENT_APPROACH = """\
STOP: 3+ failures on the same construction. Do NOT retry the same recipe.
Undo the last step and rebuild with a DIFFERENT approach (different primitive
decomposition, different boolean order, or a simpler shape that satisfies the
brief). A simpler valid solid beats a complex invalid one."""

REF_ITERATION_URGENCY = """\
WARNING: You are approaching the iteration limit ({max_iter} max, currently
{current}). Focus on producing a valid solid -- simplify if needed. Prioritize
correct geometry over feature completeness."""

REF_QUALITY_PASSED_WARN = """\
Quality check PASSED with warnings. The model is valid but may have issues.
Review warnings above. You may continue adding features or fix warnings.
If done, respond with a summary."""


# --------------------------------------------------------------------------- #
# state-conditioned injector
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class AgentLoopState:
    """The slice of loop state the injector conditions on.

    ``error_history`` is the ordered list of error identifiers (quality codes
    or exception fingerprints) seen so far; repeats are how the injector
    detects a stuck loop, mirroring the source's error-history check.
    """

    iteration: int
    max_iterations: int
    failed_quality_codes: Sequence[str] = ()
    quality_passed_with_warnings: bool = False
    error_history: Sequence[str] = field(default_factory=tuple)

    def repeated_errors(self) -> List[str]:
        """Error ids that occur 2+ times in the history (first-seen order)."""
        seen: Dict[str, int] = {}
        for err in self.error_history:
            seen[err] = seen.get(err, 0) + 1
        return [e for e, n in seen.items() if n >= 2]

    def max_repeat_count(self) -> int:
        counts: Dict[str, int] = {}
        for err in self.error_history:
            counts[err] = counts.get(err, 0) + 1
        return max(counts.values()) if counts else 0


def quality_fix_block(failed_codes: Sequence[str]) -> Optional[str]:
    """Targeted repair advice for exactly the failed quality codes.

    Source: ``_quality_gate_block()`` -- one terse fix line per failed code,
    pulled from QUALITY_FIX_MAP; unknown codes get a generic inspect hint.
    """
    if not failed_codes:
        return None
    lines = ["QUALITY REPAIR GUIDANCE (based on last failure):"]
    for code in failed_codes:
        fix = QUALITY_FIX_MAP.get(
            code, "Inspect the reported issue and adjust the construction.")
        lines.append(f"- {code}: {fix}")
    return "\n".join(lines)


def iteration_urgency_block(iteration: int, max_iterations: int,
                            threshold: float = 0.75) -> Optional[str]:
    """Urgency nudge once ``iteration/max_iterations`` crosses ``threshold``."""
    if max_iterations <= 0 or iteration < 1:
        return None
    if iteration / max_iterations < threshold:
        return None
    return REF_ITERATION_URGENCY.format(max_iter=max_iterations,
                                        current=iteration)


def references_for_state(state: AgentLoopState) -> List[str]:
    """All reference snippets the loop should inject for this state, in order.

    Pure function; the loop joins these into its context block. Ordering
    follows the source: phase snippet first, then targeted quality fixes,
    then strategy shifts, then urgency.
    """
    refs: List[str] = []
    if state.iteration <= 1:
        refs.append(REF_FIRST_ITERATION)
    block = quality_fix_block(state.failed_quality_codes)
    if block:
        refs.append(block)
    if state.max_repeat_count() >= 3:
        refs.append(REF_DIFFERENT_APPROACH)
    elif state.repeated_errors():
        refs.append(REF_REPAIR_LOOP)
    urgency = iteration_urgency_block(state.iteration, state.max_iterations)
    if urgency:
        refs.append(urgency)
    if state.quality_passed_with_warnings and not state.failed_quality_codes:
        refs.append(REF_QUALITY_PASSED_WARN)
    return refs


# --------------------------------------------------------------------------- #
# selfcheck
# --------------------------------------------------------------------------- #

def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="State-conditioned reference injection + QUALITY_FIX_MAP "
                    "(CadAgent agent/references.py port).",
    )
    parser.add_argument("--selfcheck", action="store_true",
                        help="assert the injector's phase, repeat, and "
                             "urgency conditions on synthetic states.")
    args = parser.parse_args(list(argv) if argv is not None else None)
    if not args.selfcheck:
        parser.print_help()
        return 0

    # 1. Map integrity: every fix is a non-empty terse string.
    assert len(QUALITY_FIX_MAP) == 8
    for code, fix in QUALITY_FIX_MAP.items():
        assert fix and len(fix) < 400, code
    print(f"[selfcheck] QUALITY_FIX_MAP: {len(QUALITY_FIX_MAP)} codes")

    # 2. First iteration -> checklist only.
    refs = references_for_state(AgentLoopState(iteration=1, max_iterations=10))
    assert refs == [REF_FIRST_ITERATION], refs
    print("[selfcheck] first-iteration checklist injected")

    # 3. Quality failure -> targeted per-code fixes.
    st = AgentLoopState(iteration=3, max_iterations=10,
                        failed_quality_codes=("MULTI_SOLID", "NO_SOLID"))
    refs = references_for_state(st)
    assert len(refs) == 1 and "MULTI_SOLID" in refs[0] and "NO_SOLID" in refs[0]
    assert QUALITY_FIX_MAP["MULTI_SOLID"] in refs[0]
    print("[selfcheck] quality-failure block targets failed codes only")

    # 4. Repeated error (2x) -> change-ONE-thing strategy; 3x -> different
    #    approach, and the two never co-fire.
    st = AgentLoopState(iteration=4, max_iterations=10,
                        error_history=("INVALID_SHAPE", "INVALID_SHAPE"))
    refs = references_for_state(st)
    assert REF_REPAIR_LOOP in refs and REF_DIFFERENT_APPROACH not in refs
    st = AgentLoopState(iteration=5, max_iterations=10,
                        error_history=("INVALID_SHAPE",) * 3)
    refs = references_for_state(st)
    assert REF_DIFFERENT_APPROACH in refs and REF_REPAIR_LOOP not in refs
    print("[selfcheck] repeat-error strategy shift (2x -> ONE thing, "
          "3x -> different approach)")

    # 5. Urgency near the limit, formatted with real numbers.
    st = AgentLoopState(iteration=8, max_iterations=10)
    refs = references_for_state(st)
    assert any("8" in r and "10" in r and "WARNING" in r for r in refs), refs
    assert iteration_urgency_block(2, 10) is None
    print("[selfcheck] iteration-urgency nudge at >=75% of budget")

    # 6. Passed-with-warnings reinforcement, suppressed when failures exist.
    st = AgentLoopState(iteration=3, max_iterations=10,
                        quality_passed_with_warnings=True)
    assert REF_QUALITY_PASSED_WARN in references_for_state(st)
    st = AgentLoopState(iteration=3, max_iterations=10,
                        quality_passed_with_warnings=True,
                        failed_quality_codes=("NO_SOLID",))
    assert REF_QUALITY_PASSED_WARN not in references_for_state(st)
    print("[selfcheck] passed-with-warnings reinforcement")

    # 7. Determinism.
    st = AgentLoopState(iteration=8, max_iterations=10,
                        failed_quality_codes=("NO_SOLID",),
                        error_history=("a", "a", "b"))
    assert references_for_state(st) == references_for_state(st)
    print("[selfcheck] deterministic")
    print("[selfcheck] OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
