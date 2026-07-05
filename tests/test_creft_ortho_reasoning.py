"""Tests for bench.creft_ortho_reasoning."""

import unittest

from bench.creft_ortho_reasoning import (
    COMPOSITE,
    COUNTING,
    RECOGNITION,
    CompositeFormula,
    compute_composites,
    composite_correct,
    overall_accuracy,
    parameter_correct,
    score_sample,
)


class ParameterMatchTest(unittest.TestCase):
    def test_numeric_tolerance(self):
        self.assertTrue(parameter_correct("a", {"a": 1.0000000001}, {"a": 1.0}))
        self.assertFalse(parameter_correct("a", {"a": 1.5}, {"a": 1.0}))

    def test_string_exact(self):
        self.assertTrue(parameter_correct("a", {"a": "x"}, {"a": "x"}))
        self.assertFalse(parameter_correct("a", {"a": "y"}, {"a": "x"}))

    def test_bool_not_confused_with_int(self):
        self.assertFalse(parameter_correct("a", {"a": True}, {"a": 1}))

    def test_missing_prediction(self):
        self.assertFalse(parameter_correct("a", {}, {"a": 1}))

    def test_missing_truth_raises(self):
        with self.assertRaises(KeyError):
            parameter_correct("z", {}, {"a": 1})


class ScoreSampleTest(unittest.TestCase):
    def test_family_accuracy(self):
        truth = {"d1": 5, "d2": 6, "n1": 2, "c1": 11}
        pred = {"d1": 5, "d2": 99, "n1": 2, "c1": 11}
        families = {"d1": RECOGNITION, "d2": RECOGNITION,
                    "n1": COUNTING, "c1": COMPOSITE}
        rep = score_sample(pred, truth, families)
        self.assertEqual(rep.total, 4)
        self.assertEqual(rep.correct, 3)
        self.assertAlmostEqual(rep.family_accuracy(RECOGNITION), 0.5)
        self.assertAlmostEqual(rep.family_accuracy(COUNTING), 1.0)
        self.assertAlmostEqual(rep.family_accuracy(COMPOSITE), 1.0)
        self.assertAlmostEqual(rep.accuracy, 0.75)


class OverallAccuracyTest(unittest.TestCase):
    def test_aggregate(self):
        samples = [
            ({"a": 1, "b": 2}, {"a": 1, "b": 2}),   # 2/2
            ({"a": 1, "b": 9}, {"a": 1, "b": 2}),   # 1/2
        ]
        rep = overall_accuracy(samples)
        self.assertEqual((rep.correct, rep.total), (3, 4))
        self.assertAlmostEqual(rep.accuracy, 0.75)

    def test_empty(self):
        rep = overall_accuracy([])
        self.assertEqual(rep.accuracy, 0.0)

    def test_to_dict(self):
        rep = overall_accuracy([({"a": 1}, {"a": 1})], {"a": RECOGNITION})
        d = rep.to_dict()
        self.assertEqual(d["accuracy"], 1.0)
        self.assertIn(RECOGNITION, d["per_family"])


class CompositeFormulaTest(unittest.TestCase):
    def test_sum_formula(self):
        f = CompositeFormula("spacing", "+", ("pier_dim", "pile_spacing"))
        self.assertEqual(f.compute({"pier_dim": 3, "pile_spacing": 4}), 7.0)

    def test_compute_composites(self):
        formulas = [
            CompositeFormula("s", "+", ("a", "b")),
            CompositeFormula("d", "-", ("a", "b")),
        ]
        out = compute_composites(formulas, {"a": 10, "b": 4})
        self.assertEqual(out, {"s": 14.0, "d": 6.0})

    def test_composite_correct(self):
        f = CompositeFormula("s", "*", ("a", "b"))
        params = {"a": 3, "b": 5}
        self.assertTrue(composite_correct(f, 15.0, params))
        self.assertFalse(composite_correct(f, 14.0, params))

    def test_bad_op(self):
        with self.assertRaises(ValueError):
            CompositeFormula("s", "/", ("a", "b")).compute({"a": 1, "b": 2})

    def test_empty_factors(self):
        with self.assertRaises(ValueError):
            CompositeFormula("s", "+", ()).compute({})


if __name__ == "__main__":
    unittest.main()
