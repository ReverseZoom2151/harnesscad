"""Tests for bench/querycad_query_schema.py."""

from __future__ import annotations

import unittest

from harnesscad.eval.bench.data.qa_query_schema import (
    CadQaQuestion, PropertyFilter, QUESTION_TYPES, VIEWS,
    canonical_aggregation, question_from_dict, filter_from_dict,
)


class TestPropertyFilter(unittest.TestCase):
    def test_ops(self):
        self.assertTrue(PropertyFilter("radius", "eq", 5.0).matches(5.0))
        self.assertFalse(PropertyFilter("radius", "eq", 5.0).matches(4.0))
        self.assertTrue(PropertyFilter("radius", "gt", 5.0).matches(6.0))
        self.assertTrue(PropertyFilter("radius", "lte", 5.0).matches(5.0))
        self.assertTrue(PropertyFilter("radius", "ne", 5.0).matches(4.0))

    def test_approx(self):
        f = PropertyFilter("radius", "approx", 5.0, tol=0.1)
        self.assertTrue(f.matches(5.05))
        self.assertFalse(f.matches(5.2))

    def test_none_never_matches(self):
        self.assertFalse(PropertyFilter("radius", "eq", 5.0).matches(None))

    def test_bad_op(self):
        with self.assertRaises(ValueError):
            PropertyFilter("radius", "between", 5.0)

    def test_neg_tol(self):
        with self.assertRaises(ValueError):
            PropertyFilter("radius", "approx", 5.0, tol=-1.0)


class TestQuestionValidation(unittest.TestCase):
    def test_count(self):
        q = CadQaQuestion(part="hole", question_type="count")
        self.assertIsNone(q.prop)
        self.assertEqual(q.answer_kind(), "int")

    def test_measurement_requires_prop(self):
        with self.assertRaises(ValueError):
            CadQaQuestion(part="hole", question_type="measurement")

    def test_measurement(self):
        q = CadQaQuestion(part="bore", question_type="measurement",
                          prop="Diameter")
        self.assertEqual(q.prop, "diameter")

    def test_bad_measurement_prop(self):
        with self.assertRaises(ValueError):
            CadQaQuestion(part="hole", question_type="measurement",
                          prop="center")

    def test_position(self):
        q = CadQaQuestion(part="shaft", question_type="position", prop="tip")
        self.assertEqual(q.answer_kind(), "vector")

    def test_bad_position_prop(self):
        with self.assertRaises(ValueError):
            CadQaQuestion(part="shaft", question_type="position", prop="radius")

    def test_bad_question_type(self):
        with self.assertRaises(ValueError):
            CadQaQuestion(part="hole", question_type="describe")

    def test_all_question_types_valid(self):
        # every declared type must be constructible (with proper args)
        self.assertIn("count", QUESTION_TYPES)
        self.assertIn("comparison", QUESTION_TYPES)


class TestViews(unittest.TestCase):
    def test_canonical_order_and_dedup(self):
        q = CadQaQuestion(part="hole", question_type="count",
                          views=("right", "top", "right"))
        self.assertEqual(q.views, ("top", "right"))
        self.assertEqual(q.view_set(), frozenset({"top", "right"}))

    def test_bad_view(self):
        with self.assertRaises(ValueError):
            CadQaQuestion(part="hole", question_type="count", views=("side",))

    def test_case_insensitive(self):
        q = CadQaQuestion(part="hole", question_type="count", views=("TOP",))
        self.assertEqual(q.views, ("top",))
        self.assertEqual(len(VIEWS), 6)


class TestComparison(unittest.TestCase):
    def test_comparison_requires_aggregation(self):
        with self.assertRaises(ValueError):
            CadQaQuestion(part="hole", question_type="comparison",
                          prop="diameter")

    def test_aggregation_synonyms(self):
        q = CadQaQuestion(part="hole", question_type="comparison",
                          prop="diameter", aggregation="largest")
        self.assertEqual(q.aggregation, "max")
        self.assertEqual(canonical_aggregation("smallest"), "min")

    def test_bad_aggregation(self):
        with self.assertRaises(KeyError):
            CadQaQuestion(part="hole", question_type="comparison",
                          prop="diameter", aggregation="medium")


class TestWholeModel(unittest.TestCase):
    def test_empty_targets_model(self):
        q = CadQaQuestion(part="", question_type="position", prop="center")
        self.assertTrue(q.targets_whole_model)

    def test_named_part_not_whole(self):
        q = CadQaQuestion(part="hole", question_type="count")
        self.assertFalse(q.targets_whole_model)


class TestFromDict(unittest.TestCase):
    def test_roundtrip(self):
        d = {
            "part": "bore",
            "question_type": "comparison",
            "prop": "diameter",
            "views": ["top"],
            "aggregation": "largest",
            "filters": [{"prop": "diameter", "op": "gt", "value": 2.0}],
        }
        q = question_from_dict(d)
        self.assertEqual(q.question_type, "comparison")
        self.assertEqual(q.aggregation, "max")
        self.assertEqual(len(q.filters), 1)
        self.assertTrue(q.filters[0].matches(3.0))

    def test_filter_from_dict_tol(self):
        f = filter_from_dict({"prop": "radius", "op": "approx",
                              "value": 5.0, "tol": 0.2})
        self.assertTrue(f.matches(5.1))


if __name__ == "__main__":
    unittest.main()
