"""Minimal-diff repair discipline: fix ONLY what failed, preserve what passed.

Vendored from **AgentSCAD** (``resources/cad_repos/AgentSCAD-main``,
``src/lib/repair/repair-controller.ts`` -- ``buildRepairPrompt`` and its
"Repair Goal" block). Source repo LICENSE: **MIT** -- the rule text below is
reproduced from that file with attribution.

THE DISCIPLINE
--------------
A naive repair prompt hands the model the failure and asks for a fix; the model
then rewrites the part, and the *passing* checks quietly regress. AgentSCAD's
repair controller closes that by splitting the validation report in two and
constraining the edit against it:

  * **Failed rules** are listed with id, name, criticality and message -- these
    are the ONLY things the repair may address;
  * **Passed rules** are listed too, explicitly, as a do-not-touch inventory
    (the source filters out ``skipped`` messages so a skipped check is never
    presented as a passing one to preserve);
  * the goal block then states the rules: fix only the failed checks, do not
    change dimensions or features that already pass, preserve the required
    features of the original intent.

The failed-vs-passed split is what makes "minimal diff" checkable rather than
aspirational: the prompt names the exact set the repair is licensed to move.

USE IN THE HARNESS
------------------
:func:`build_minimal_diff_prompt` is the general formatter over a
:class:`RuleOutcome` report. :func:`build_refine_prompt` in
:mod:`harnesscad.agents.agent.compiler_refine` -- the harness's repair-loop
prompt formatter -- grows an opt-in ``minimal_diff=True`` that routes the
compiler's own review through :func:`discipline_lines`, so the CRM loop inherits
the discipline without changing its default output.

This module formats text; it selects no tool and calls no model. The prompt is
built from a *validation report* -- structured pass/fail records produced by the
harness's own checkers -- so what the model is told to preserve is what actually
passed, not what it believes passed.

Stdlib-only, deterministic, absolute imports. ``--selfcheck`` proves the
failed/passed split, the skipped-check exclusion, and the preservation rules.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import Optional, Sequence, Tuple

__all__ = [
    "RuleOutcome",
    "REPAIR_GOAL_RULES",
    "discipline_lines",
    "format_rule_lines",
    "build_minimal_diff_prompt",
    "main",
]


@dataclass(frozen=True)
class RuleOutcome:
    """One validation rule's verdict, as a checker reported it.

    ``skipped`` records a check that did not run: it is neither a failure to
    fix nor a pass to preserve, and the source drops it from both lists rather
    than let a model read "skipped" as "fine".
    """

    rule_id: str
    rule_name: str
    passed: bool
    message: str = ""
    is_critical: bool = False
    skipped: bool = False

    def render(self) -> str:
        if self.passed:
            return f"- {self.rule_id} {self.rule_name}: {self.message}"
        severity = "CRITICAL" if self.is_critical else "non-critical"
        return f"- {self.rule_id} {self.rule_name} ({severity}): {self.message}"


#: The repair-goal rules, from AgentSCAD's ``buildRepairPrompt`` (MIT).
REPAIR_GOAL_RULES: Tuple[str, ...] = (
    "Fix ONLY the failed validation checks listed above.",
    "Do NOT change dimensions or features that already pass validation.",
    "Preserve all required features from the CAD intent.",
)


def discipline_lines() -> Tuple[str, ...]:
    """The repair-goal rules, for callers that only want the constraint."""
    return REPAIR_GOAL_RULES


def format_rule_lines(outcomes: Sequence[RuleOutcome], passed: bool) -> str:
    """Render the failed (``passed=False``) or passed half of the report.

    Skipped checks are excluded from BOTH halves. Empty halves render the
    source's explicit placeholders rather than nothing, so the model is never
    left to infer whether a section was empty or omitted.
    """
    rows = [o.render() for o in outcomes
            if not o.skipped and bool(o.passed) is bool(passed)]
    if rows:
        return "\n".join(rows)
    return "(none)" if passed else "(none -- all passed)"


def build_minimal_diff_prompt(
    original_request: str,
    current_code: str,
    outcomes: Sequence[RuleOutcome],
    *,
    intent_block: Optional[str] = None,
    code_language: str = "python",
) -> str:
    """Build a repair prompt constrained to a minimal diff.

    Sections follow the source's order: the original request, the structured
    intent (or an explicit "no structured intent" note -- never a silent gap),
    the current code, the validation results split into **Failed Rules** and
    **Passed Rules**, then the repair goal.
    """
    return "\n".join([
        "## Original Request",
        original_request,
        "",
        "## CAD Intent (from generation)",
        intent_block or "No structured intent available",
        "",
        "## Current Code",
        f"```{code_language}",
        current_code,
        "```",
        "",
        "## Validation Results",
        "### Failed Rules",
        format_rule_lines(outcomes, passed=False),
        "",
        "### Passed Rules",
        format_rule_lines(outcomes, passed=True),
        "",
        "## Repair Goal",
        *REPAIR_GOAL_RULES,
    ])


# --------------------------------------------------------------------------- #
# selfcheck
# --------------------------------------------------------------------------- #

def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Minimal-diff repair prompt with a failed-vs-passed rule "
                    "split (AgentSCAD repair-controller.ts port, MIT).")
    parser.add_argument("--selfcheck", action="store_true",
                        help="prove the failed/passed split, the skipped "
                             "exclusion, and the preservation rules.")
    args = parser.parse_args(list(argv) if argv is not None else None)
    if not args.selfcheck:
        parser.print_help()
        return 0

    outcomes = [
        RuleOutcome("R003", "manifold", False, "mesh is non-manifold",
                    is_critical=True),
        RuleOutcome("R007", "hole_count", False, "expected 4 holes, found 2"),
        RuleOutcome("R001", "bbox", True, "100x60x40 within tolerance"),
        RuleOutcome("R009", "wall_thickness", True, "2.0mm >= 2.0mm minimum"),
        RuleOutcome("R012", "visual", True, "skipped: no reference image",
                    skipped=True),
    ]
    prompt = build_minimal_diff_prompt("a 100x60x40 enclosure with 4 holes",
                                       "box(100, 60, 40)", outcomes)

    failed_block = prompt.split("### Failed Rules")[1].split("### Passed")[0]
    passed_block = prompt.split("### Passed Rules")[1].split("## Repair Goal")[0]

    # 1. Failures land in the failed half only, with criticality.
    assert "R003" in failed_block and "CRITICAL" in failed_block
    assert "R007" in failed_block and "non-critical" in failed_block
    assert "R003" not in passed_block and "R007" not in passed_block
    print("[selfcheck] failed rules listed with criticality, in the failed half")

    # 2. Passing rules are listed for preservation, not dropped.
    assert "R001" in passed_block and "R009" in passed_block
    assert "R001" not in failed_block
    print("[selfcheck] passing rules named as a do-not-touch inventory")

    # 3. A skipped check is in NEITHER half.
    assert "R012" not in failed_block and "R012" not in passed_block
    print("[selfcheck] skipped check excluded from both halves")

    # 4. The discipline is stated.
    for rule in REPAIR_GOAL_RULES:
        assert rule in prompt, rule
    assert "Fix ONLY" in prompt and "Do NOT change" in prompt
    assert "Preserve all required features" in prompt
    print("[selfcheck] repair goal: fix only failed, preserve passing, keep "
          "intent features")

    # 5. Empty halves render explicit placeholders.
    all_pass = [RuleOutcome("R001", "bbox", True, "ok")]
    p = build_minimal_diff_prompt("x", "y", all_pass)
    assert "(none -- all passed)" in p
    p = build_minimal_diff_prompt("x", "y", [RuleOutcome("R1", "n", False, "m")])
    assert "(none)" in p.split("### Passed Rules")[1]
    print("[selfcheck] empty halves are stated explicitly, never omitted")

    # 6. Deterministic + intent block honesty.
    assert build_minimal_diff_prompt("a 100x60x40 enclosure with 4 holes",
                                     "box(100, 60, 40)", outcomes) == prompt
    assert "No structured intent available" in prompt
    assert "part_type: enclosure" in build_minimal_diff_prompt(
        "x", "y", outcomes, intent_block="part_type: enclosure")
    print("[selfcheck] deterministic; a missing intent is stated, not hidden")

    # 7. The compiler-refine loop inherits the discipline opt-in.
    from harnesscad.agents.agent.compiler_refine import build_refine_prompt
    from harnesscad.eval.judge.compiler_review import review_sequence
    review = review_sequence([{"type": "extrude", "depth": 1.0}, {"type": "end"}])
    plain = build_refine_prompt("make a box", review)
    strict = build_refine_prompt("make a box", review, minimal_diff=True)
    assert "Fix ONLY" not in plain and "Fix ONLY" in strict
    assert plain in strict  # default output unchanged, discipline appended
    print("[selfcheck] compiler_refine gains minimal_diff=True without "
          "changing its default output")
    print("[selfcheck] OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
