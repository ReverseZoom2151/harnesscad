"""Tests for bench/querycad_eval.py."""

from __future__ import annotations

import unittest

from harnesscad.eval.bench.judges.qa_grade_scale import (
    Grade, grade, aggregate, ERROR_CATEGORIES, CORRECT, PARTIAL, WRONG,
)
from harnesscad.domain.reconstruction.scene.answer_engine import Answer


class TestNumericGrading(unittest.TestCase):
    def test_exact(self):
        g = grade(5.0, 5.0)
        self.assertTrue(g.is_correct)
        self.assertEqual(g.abs_error, 0.0)

    def test_within_abs_tol(self):
        g = grade(5.05, 5.0, abs_tol=0.1)
        self.assertTrue(g.is_correct)

    def test_within_rel_tol(self):
        g = grade(102.0, 100.0, rel_tol=0.05)
        self.assertTrue(g.is_correct)

    def test_out_of_tol(self):
        g = grade(6.0, 5.0, abs_tol=0.1)
        self.assertEqual(g.outcome, WRONG)
        self.assertAlmostEqual(g.abs_error, 1.0)
        self.assertEqual(g.error_category, "reasoning")


class TestIntBoolGrading(unittest.TestCase):
    def test_count_correct(self):
        self.assertTrue(grade(3, 3, kind="int").is_correct)

    def test_count_wrong(self):
        g = grade(2, 3, kind="int")
        self.assertEqual(g.outcome, WRONG)
        self.assertEqual(g.abs_error, 1)

    def test_bool(self):
        self.assertTrue(grade(True, True).is_correct)
        self.assertEqual(grade(True, False).outcome, WRONG)


class TestVectorGrading(unittest.TestCase):
    def test_close(self):
        g = grade((1.0, 2.0, 3.0), (1.0, 2.01, 3.0), abs_tol=0.1)
        self.assertTrue(g.is_correct)

    def test_far(self):
        g = grade((1.0, 2.0, 3.0), (1.0, 9.0, 3.0), abs_tol=0.1)
        self.assertEqual(g.outcome, WRONG)

    def test_length_mismatch(self):
        g = grade((1.0, 2.0), (1.0, 2.0, 3.0))
        self.assertEqual(g.outcome, WRONG)


class TestComparisonGrading(unittest.TestCase):
    def test_both_right(self):
        g = grade(("h2", 8.0), ("h2", 8.0))
        self.assertTrue(g.is_correct)

    def test_value_right_part_wrong(self):
        g = grade(("h3", 8.0), ("h2", 8.0))
        self.assertEqual(g.outcome, PARTIAL)

    def test_both_wrong(self):
        g = grade(("h3", 5.0), ("h2", 8.0), abs_tol=0.01)
        self.assertEqual(g.outcome, WRONG)


class TestSetGrading(unittest.TestCase):
    def test_exact_list(self):
        self.assertTrue(grade((5.0, 8.0), (5.0, 8.0)).is_correct)

    def test_overlap_partial(self):
        g = grade((5.0, 8.0), (5.0, 9.0))
        self.assertEqual(g.outcome, PARTIAL)

    def test_disjoint_wrong(self):
        g = grade((1.0, 2.0), (5.0, 9.0))
        self.assertEqual(g.outcome, WRONG)


class TestAbstain(unittest.TestCase):
    def test_abstained_answer(self):
        a = Answer(None, "number", (), abstained=True)
        g = grade(a, 5.0)
        self.assertEqual(g.outcome, WRONG)
        self.assertEqual(g.error_category, "cad_interface")

    def test_engine_answer_value_extracted(self):
        a = Answer(8.0, "number", ("b1",))
        self.assertTrue(grade(a, 8.0).is_correct)


class TestErrorCategory(unittest.TestCase):
    def test_custom_category(self):
        g = grade(6.0, 5.0, abs_tol=0.1, error_category="masks")
        self.assertEqual(g.error_category, "masks")

    def test_bad_category(self):
        with self.assertRaises(ValueError):
            grade(6.0, 5.0, error_category="nonsense")


class TestAggregate(unittest.TestCase):
    def test_table_one_summary(self):
        grades = [
            Grade(CORRECT, abs_error=0.0),
            Grade(CORRECT, abs_error=0.5),
            Grade(PARTIAL),
            Grade(WRONG, error_category="masks", abs_error=2.0),
            Grade(WRONG, error_category="cad_interface"),
        ]
        s = aggregate(grades)
        self.assertEqual(s["total"], 5)
        self.assertEqual(s["correct"], 2)
        self.assertEqual(s["partial"], 1)
        self.assertEqual(s["wrong"], 2)
        self.assertAlmostEqual(s["accuracy"], 2 / 5)
        self.assertAlmostEqual(s["partial_accuracy"], 3 / 5)
        self.assertEqual(s["errors"]["masks"], 1)
        self.assertEqual(s["errors"]["cad_interface"], 1)
        self.assertEqual(s["errors"]["syntax"], 0)
        # mae over the three grades carrying abs_error: (0 + 0.5 + 2)/3
        self.assertAlmostEqual(s["mae"], (0.0 + 0.5 + 2.0) / 3)

    def test_empty(self):
        s = aggregate([])
        self.assertEqual(s["total"], 0)
        self.assertIsNone(s["accuracy"])
        self.assertIsNone(s["mae"])

    def test_categories_constant(self):
        self.assertEqual(ERROR_CATEGORIES,
                         ("syntax", "reasoning", "masks", "cad_interface"))


if __name__ == "__main__":
    unittest.main()
