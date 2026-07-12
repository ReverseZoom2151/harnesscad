"""Tests for bench.muse2_rubric_deductions."""

import unittest

from bench.muse2_rubric_deductions import (
    MetricsContext,
    category_breakdown,
    evaluate_rule,
    expected_component_count_from_plan,
    normalize_rubric_weights,
    score_item,
    score_rubric,
    weighted_rubric_score,
)


def _good_ctx(**over):
    base = dict(
        code_valid=True, geometry_valid=True, watertight=True, manifold=True,
        self_intersection_free=True, normal_consistency=True, volume_valid=True,
        bbox_valid=True, occt_valid=True, sandbox_ok=True,
        bbox=(10.0, 20.0, 30.0), solid_count=2, svg_path_count=40,
        svg_component_estimate=2, expected_components=2,
    )
    base.update(over)
    return MetricsContext(**base)


class RulePredicateTests(unittest.TestCase):
    def test_all_pass_on_good_ctx(self):
        ctx = _good_ctx()
        for code in ("code_or_result_missing", "global_geometry_invalid",
                     "bbox_missing_or_collapsed", "component_count_mismatch",
                     "functional_structure_broken", "local_continuity_risk"):
            triggered, _ = evaluate_rule(code, ctx)
            self.assertFalse(triggered, code)

    def test_code_missing_triggers(self):
        ctx = _good_ctx(code_valid=False, sandbox_ok=False)
        triggered, _ = evaluate_rule("code_or_result_missing", ctx)
        self.assertTrue(triggered)

    def test_component_count_mismatch(self):
        ctx = _good_ctx(solid_count=1, expected_components=3)
        triggered, ev = evaluate_rule("component_count_mismatch", ctx)
        self.assertTrue(triggered)
        self.assertIn("expected=3", ev)

    def test_watertight_false_breaks_structure(self):
        ctx = _good_ctx(watertight=False)
        triggered, _ = evaluate_rule("functional_structure_broken", ctx)
        self.assertTrue(triggered)

    def test_geometry_valid_none_counts_as_not_true(self):
        # None (never evaluated) must trigger the same as False.
        ctx = _good_ctx(geometry_valid=None)
        triggered, _ = evaluate_rule("global_geometry_invalid", ctx)
        self.assertTrue(triggered)

    def test_bbox_collapsed(self):
        ctx = _good_ctx(bbox=(5.0, 0.0, 0.0))
        self.assertTrue(ctx.bbox_missing_or_collapsed())
        triggered, _ = evaluate_rule("bbox_missing_or_collapsed", ctx)
        self.assertTrue(triggered)

    def test_unknown_rule_never_triggers(self):
        triggered, ev = evaluate_rule("does_not_exist", _good_ctx())
        self.assertFalse(triggered)
        self.assertEqual(ev, "rule_not_registered")

    def test_actual_components_falls_back_to_svg(self):
        ctx = _good_ctx(solid_count=0, svg_component_estimate=5)
        self.assertEqual(ctx.actual_components(), 5)


class ScoreItemTests(unittest.TestCase):
    def test_perfect_item_no_deductions(self):
        item = {"item_id": "1", "max_points": 4.0, "normalized_weight": 0.5,
                "deduction_rules": [
                    {"rule_code": "component_count_mismatch", "deduction_ratio": 0.5}]}
        out = score_item(item, _good_ctx())
        self.assertEqual(out["score"], 1.0)
        self.assertEqual(out["points"], 4.0)
        self.assertEqual(out["deductions"], [])

    def test_deduction_subtracts_and_clamps(self):
        item = {"item_id": "1", "deduction_rules": [
            {"rule_code": "component_count_mismatch", "deduction_ratio": 0.7},
            {"rule_code": "global_geometry_invalid", "deduction_ratio": 0.7}]}
        ctx = _good_ctx(solid_count=9, geometry_valid=False)
        out = score_item(item, ctx)
        # 1 - 0.7 - 0.7 = -0.4 -> clamped to 0.
        self.assertEqual(out["score"], 0.0)
        self.assertEqual(len(out["deductions"]), 2)

    def test_partial_deduction(self):
        item = {"item_id": "1", "deduction_rules": [
            {"rule_code": "functional_structure_broken", "deduction_ratio": 0.3}]}
        out = score_item(item, _good_ctx(watertight=False))
        self.assertAlmostEqual(out["score"], 0.7)


class AggregationTests(unittest.TestCase):
    def test_weighted_score_and_breakdown(self):
        items = [
            {"item_id": "1", "primary_category": "Functionality", "max_points": 2.0,
             "normalized_weight": 0.5, "deduction_rules": [
                 {"rule_code": "functional_structure_broken", "deduction_ratio": 1.0}]},
            {"item_id": "2", "primary_category": "Assemblability", "max_points": 2.0,
             "normalized_weight": 0.5, "deduction_rules": []},
        ]
        ctx = _good_ctx(watertight=False)
        scored = score_rubric(items, ctx)
        # item1 score 0, item2 score 1 -> weighted 0*0.5 + 1*0.5 = 0.5
        self.assertAlmostEqual(weighted_rubric_score(scored), 0.5)
        bd = category_breakdown(scored)
        self.assertAlmostEqual(bd["Functionality"]["ratio"], 0.0)
        self.assertAlmostEqual(bd["Assemblability"]["ratio"], 1.0)
        self.assertEqual(bd["Functionality"]["item_count"], 1)


class ParserTests(unittest.TestCase):
    def test_component_count_from_marker(self):
        text = "## Planned Assembly Count\n\n3\n\n### A\n### B\n"
        self.assertEqual(expected_component_count_from_plan(text), 3)

    def test_component_count_from_headings(self):
        text = "intro\n### Part A\n### Part B\n### Part C\n"
        self.assertEqual(expected_component_count_from_plan(text), 3)

    def test_normalize_weights_sum_to_one(self):
        items = [
            {"title": "a", "description": "x", "raw_weight": 1.0},
            {"title": "b", "description": "y", "raw_weight": 3.0},
        ]
        out = normalize_rubric_weights(items)
        self.assertAlmostEqual(sum(i["normalized_weight"] for i in out), 1.0)
        self.assertAlmostEqual(out[1]["normalized_weight"], 0.75)

    def test_normalize_dedup_keeps_max_weight(self):
        items = [
            {"title": "a", "description": "x", "raw_weight": 1.0},
            {"title": "a", "description": "x", "raw_weight": 4.0},
        ]
        out = normalize_rubric_weights(items)
        self.assertEqual(len(out), 1)
        self.assertAlmostEqual(out[0]["normalized_weight"], 1.0)

    def test_normalize_all_zero_uniform(self):
        items = [
            {"title": "a", "description": "x", "raw_weight": 0.0},
            {"title": "b", "description": "y", "raw_weight": 0.0},
        ]
        out = normalize_rubric_weights(items)
        self.assertAlmostEqual(out[0]["normalized_weight"], 0.5)


if __name__ == "__main__":
    unittest.main()
