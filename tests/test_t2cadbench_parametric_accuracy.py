"""Tests for bench.t2cadbench_parametric_accuracy."""

import unittest

from harnesscad.eval.bench.t2cadbench_parametric_accuracy import (
    match_parameter,
    mean_parameter_accuracy,
    parameter_accuracy,
    parametric_validity,
)


class MatchParameterTests(unittest.TestCase):
    def test_numeric_within_rel_tol(self):
        self.assertTrue(match_parameter(80.5, 80.0))   # <2%
        self.assertFalse(match_parameter(85.0, 80.0))  # >2%

    def test_categorical(self):
        self.assertTrue(match_parameter("xy", "XY"))
        self.assertFalse(match_parameter("YZ", "XY"))

    def test_missing_never_matches(self):
        self.assertFalse(match_parameter(None, 10.0))


class ParameterAccuracyTests(unittest.TestCase):
    def test_all_correct(self):
        truth = {"length": 80, "width": 50, "height": 30, "plane": "XY"}
        pred = {"length": 80.0, "width": 50.4, "height": 30, "plane": "xy"}
        r = parameter_accuracy(pred, truth)
        self.assertEqual(r["accuracy"], 1.0)
        self.assertEqual(r["matched"], 4)
        self.assertEqual(r["missing"], ())
        self.assertEqual(r["wrong"], ())

    def test_missing_and_wrong(self):
        truth = {"length": 80, "diameter": 15, "radius": 5}
        pred = {"length": 80, "diameter": 99}  # radius missing, diameter wrong
        r = parameter_accuracy(pred, truth)
        self.assertEqual(r["matched"], 1)
        self.assertEqual(r["accuracy"], 1 / 3)
        self.assertEqual(r["missing"], ("radius",))
        self.assertEqual(r["wrong"], ("diameter",))

    def test_extra_params_reported(self):
        r = parameter_accuracy({"a": 1, "b": 2}, {"a": 1})
        self.assertEqual(r["extra"], ("b",))
        self.assertEqual(r["accuracy"], 1.0)

    def test_empty_truth_is_one(self):
        self.assertEqual(parameter_accuracy({}, {})["accuracy"], 1.0)


class ParametricValidityTests(unittest.TestCase):
    def test_valid(self):
        r = parametric_validity(
            {"length": 80, "radius": 5}, required=["length", "radius"],
            ranges={"radius": (0, 100)})
        self.assertTrue(r["valid"])
        self.assertTrue(r["fully_specified"])

    def test_missing_required(self):
        r = parametric_validity({"length": 80}, required=["length", "radius"])
        self.assertFalse(r["fully_specified"])
        self.assertFalse(r["valid"])
        self.assertEqual(r["missing_required"], ("radius",))

    def test_out_of_range(self):
        r = parametric_validity(
            {"radius": -3}, required=["radius"], ranges={"radius": (0, 10)})
        self.assertTrue(r["fully_specified"])
        self.assertFalse(r["valid"])
        self.assertEqual(r["out_of_range"], ("radius",))


class MeanAccuracyTests(unittest.TestCase):
    def test_micro_vs_macro(self):
        ex = [
            ({"a": 1}, {"a": 1}),                       # 1/1
            ({"a": 1, "b": 2, "c": 9}, {"a": 1, "b": 2, "c": 3}),  # 2/3
        ]
        r = mean_parameter_accuracy(ex)
        self.assertEqual(r["n"], 2)
        self.assertAlmostEqual(r["mean_accuracy"], (1.0 + 2 / 3) / 2)
        self.assertAlmostEqual(r["micro_accuracy"], 3 / 4)

    def test_empty_raises(self):
        with self.assertRaises(ValueError):
            mean_parameter_accuracy([])


if __name__ == "__main__":
    unittest.main()
