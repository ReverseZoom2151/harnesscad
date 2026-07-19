import unittest

from harnesscad.agents.agent.compiler_refine import build_refine_prompt
from harnesscad.agents.agent.minimal_diff_repair import (
    REPAIR_GOAL_RULES,
    RuleOutcome,
    build_minimal_diff_prompt,
    discipline_lines,
    format_rule_lines,
    main,
)
from harnesscad.eval.judge.compiler_review import review_sequence

OUTCOMES = [
    RuleOutcome("R003", "manifold", False, "mesh is non-manifold",
                is_critical=True),
    RuleOutcome("R007", "hole_count", False, "expected 4 holes, found 2"),
    RuleOutcome("R001", "bbox", True, "100x60x40 within tolerance"),
    RuleOutcome("R009", "wall_thickness", True, "2.0mm >= 2.0mm minimum"),
    RuleOutcome("R012", "visual", True, "skipped: no reference", skipped=True),
]


def _halves(prompt):
    failed = prompt.split("### Failed Rules")[1].split("### Passed")[0]
    passed = prompt.split("### Passed Rules")[1].split("## Repair Goal")[0]
    return failed, passed


class RuleSplitTests(unittest.TestCase):
    def test_failed_rules_carry_criticality(self):
        failed, _ = _halves(build_minimal_diff_prompt("req", "code", OUTCOMES))
        self.assertIn("R003 manifold (CRITICAL)", failed)
        self.assertIn("R007 hole_count (non-critical)", failed)

    def test_passed_rules_listed_for_preservation(self):
        _, passed = _halves(build_minimal_diff_prompt("req", "code", OUTCOMES))
        self.assertIn("R001", passed)
        self.assertIn("R009", passed)

    def test_halves_are_disjoint(self):
        failed, passed = _halves(build_minimal_diff_prompt("req", "code", OUTCOMES))
        for rid in ("R003", "R007"):
            self.assertNotIn(rid, passed)
        for rid in ("R001", "R009"):
            self.assertNotIn(rid, failed)

    def test_skipped_check_in_neither_half(self):
        failed, passed = _halves(build_minimal_diff_prompt("req", "code", OUTCOMES))
        self.assertNotIn("R012", failed)
        self.assertNotIn("R012", passed)

    def test_empty_halves_use_explicit_placeholders(self):
        p = build_minimal_diff_prompt("r", "c", [RuleOutcome("R1", "n", True, "m")])
        self.assertIn("(none -- all passed)", p)
        p = build_minimal_diff_prompt("r", "c", [RuleOutcome("R1", "n", False, "m")])
        _, passed = _halves(p)
        self.assertIn("(none)", passed)

    def test_format_rule_lines_filters_by_verdict(self):
        self.assertIn("R003", format_rule_lines(OUTCOMES, passed=False))
        self.assertNotIn("R003", format_rule_lines(OUTCOMES, passed=True))


class DisciplineTests(unittest.TestCase):
    def test_goal_states_all_three_rules(self):
        prompt = build_minimal_diff_prompt("req", "code", OUTCOMES)
        for rule in REPAIR_GOAL_RULES:
            self.assertIn(rule, prompt)

    def test_goal_states_the_three_constraints(self):
        joined = " ".join(REPAIR_GOAL_RULES)
        self.assertIn("Address only the validation checks listed as failed",
                      joined)
        self.assertIn("already passes validation", joined)
        self.assertIn("Keep all features that the CAD intent requires", joined)

    def test_discipline_lines_matches_rules(self):
        self.assertEqual(discipline_lines(), REPAIR_GOAL_RULES)

    def test_missing_intent_is_stated_not_hidden(self):
        p = build_minimal_diff_prompt("req", "code", OUTCOMES)
        self.assertIn("No structured intent available", p)
        p = build_minimal_diff_prompt("req", "code", OUTCOMES,
                                      intent_block="part_type: bracket")
        self.assertIn("part_type: bracket", p)
        self.assertNotIn("No structured intent available", p)

    def test_deterministic(self):
        self.assertEqual(build_minimal_diff_prompt("req", "code", OUTCOMES),
                         build_minimal_diff_prompt("req", "code", OUTCOMES))


class CompilerRefineWiringTests(unittest.TestCase):
    def setUp(self):
        self.review = review_sequence([{"type": "extrude", "depth": 1.0},
                                       {"type": "end"}])

    def test_default_output_unchanged(self):
        prompt = build_refine_prompt("base task", self.review)
        self.assertIn("Compiler feedback", prompt)
        self.assertNotIn("Address only the validation checks", prompt)

    def test_minimal_diff_appends_discipline(self):
        plain = build_refine_prompt("base task", self.review)
        strict = build_refine_prompt("base task", self.review, minimal_diff=True)
        self.assertTrue(strict.startswith(plain))
        self.assertIn("[Repair goal]", strict)
        for rule in REPAIR_GOAL_RULES:
            self.assertIn(rule, strict)


class SelfcheckTests(unittest.TestCase):
    def test_selfcheck_exits_zero(self):
        self.assertEqual(main(["--selfcheck"]), 0)


if __name__ == "__main__":
    unittest.main()
