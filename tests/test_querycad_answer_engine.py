"""Tests for reconstruction/querycad_answer_engine.py."""

from __future__ import annotations

import unittest

from harnesscad.eval.bench.data.querycad_query_schema import CadQaQuestion, PropertyFilter
from harnesscad.agents.rag.querycad_segmentation_grounding import Part
from harnesscad.domain.reconstruction.scene.querycad_answer_engine import Answer, answer_question


def _model():
    return [
        Part(id="h1", feature="hole", attrs={"diameter": 5.0, "depth": 10.0,
             "center": (0.0, 0.0, 0.0)}, visible_views={"top"}, coverage=0.05),
        Part(id="h2", feature="hole", attrs={"diameter": 8.0, "depth": 4.0,
             "center": (10.0, 0.0, 0.0)}, visible_views={"front"},
             coverage=0.05),
        Part(id="h3", feature="hole", attrs={"diameter": 8.0, "depth": 6.0,
             "center": (20.0, 0.0, 0.0)}, visible_views={"top"}, coverage=0.05),
        Part(id="b1", feature="boss", attrs={"diameter": 12.0,
             "position": (5.0, 5.0, 1.0)}, visible_views={"top"}, coverage=0.10),
    ]


class TestCount(unittest.TestCase):
    def test_count_holes(self):
        q = CadQaQuestion(part="hole", question_type="count")
        a = answer_question(_model(), q)
        self.assertEqual(a.value, 3)
        self.assertEqual(a.part_ids, ("h1", "h2", "h3"))

    def test_count_with_view(self):
        q = CadQaQuestion(part="hole", question_type="count", views=("top",))
        a = answer_question(_model(), q)
        self.assertEqual(a.value, 2)
        self.assertEqual(a.part_ids, ("h1", "h3"))

    def test_count_with_filter(self):
        q = CadQaQuestion(part="hole", question_type="count",
                          filters=(PropertyFilter("diameter", "eq", 8.0),))
        a = answer_question(_model(), q)
        self.assertEqual(a.value, 2)


class TestExistence(unittest.TestCase):
    def test_exists(self):
        q = CadQaQuestion(part="boss", question_type="existence")
        self.assertTrue(answer_question(_model(), q).value)

    def test_not_exists(self):
        q = CadQaQuestion(part="slot", question_type="existence")
        a = answer_question(_model(), q)
        self.assertFalse(a.value)
        self.assertEqual(a.part_ids, ())


class TestMeasurement(unittest.TestCase):
    def test_single(self):
        q = CadQaQuestion(part="boss", question_type="measurement",
                          prop="diameter")
        a = answer_question(_model(), q)
        self.assertEqual(a.value, 12.0)
        self.assertEqual(a.part_ids, ("b1",))

    def test_filtered_to_single(self):
        q = CadQaQuestion(part="hole", question_type="measurement",
                          prop="depth",
                          filters=(PropertyFilter("diameter", "eq", 5.0),))
        a = answer_question(_model(), q)
        self.assertEqual(a.value, 10.0)

    def test_multiple_returns_list(self):
        q = CadQaQuestion(part="hole", question_type="measurement",
                          prop="diameter")
        a = answer_question(_model(), q)
        self.assertEqual(a.value, (5.0, 8.0, 8.0))

    def test_missing_property_abstains(self):
        q = CadQaQuestion(part="boss", question_type="measurement",
                          prop="depth")
        a = answer_question(_model(), q)
        self.assertTrue(a.abstained)
        self.assertIsNone(a.value)


class TestPosition(unittest.TestCase):
    def test_center(self):
        q = CadQaQuestion(part="h2", question_type="position", prop="center")
        a = answer_question(_model(), q)
        self.assertEqual(a.value, (10.0, 0.0, 0.0))
        self.assertEqual(a.kind, "vector")

    def test_position_key_fallback(self):
        # boss stores "position", queried via "center" which falls back
        q = CadQaQuestion(part="boss", question_type="position", prop="center")
        a = answer_question(_model(), q)
        self.assertEqual(a.value, (5.0, 5.0, 1.0))

    def test_abstain(self):
        q = CadQaQuestion(part="slot", question_type="position", prop="center")
        a = answer_question(_model(), q)
        self.assertTrue(a.abstained)


class TestComparison(unittest.TestCase):
    def test_largest_diameter(self):
        q = CadQaQuestion(part="hole", question_type="comparison",
                          prop="diameter", aggregation="largest")
        a = answer_question(_model(), q)
        pid, val = a.value
        self.assertEqual(val, 8.0)
        # tie between h2 and h3 broken deterministically by id -> h2
        self.assertEqual(pid, "h2")
        self.assertEqual(a.part_ids, ("h2",))

    def test_smallest(self):
        q = CadQaQuestion(part="hole", question_type="comparison",
                          prop="diameter", aggregation="smallest")
        a = answer_question(_model(), q)
        pid, val = a.value
        self.assertEqual((pid, val), ("h1", 5.0))

    def test_deepest(self):
        q = CadQaQuestion(part="hole", question_type="comparison",
                          prop="depth", aggregation="deepest")
        a = answer_question(_model(), q)
        pid, val = a.value
        self.assertEqual((pid, val), ("h1", 10.0))


class TestWholeModelAndErrors(unittest.TestCase):
    def test_whole_model_position(self):
        model = [Part(id="m", feature="model",
                      attrs={"center": (1.0, 2.0, 3.0)}, coverage=1.0)]
        q = CadQaQuestion(part="", question_type="position", prop="center")
        a = answer_question(model, q)
        self.assertEqual(a.value, (1.0, 2.0, 3.0))

    def test_bad_question(self):
        with self.assertRaises(TypeError):
            answer_question(_model(), "not a question")

    def test_answer_is_frozen(self):
        a = Answer(1, "int", ("h1",))
        with self.assertRaises(Exception):
            a.value = 2


if __name__ == "__main__":
    unittest.main()
