"""Tests for the OpenECAD generation scoring metric (Table 4, Eq. 1)."""

import unittest

from harnesscad.domain.programs.ast import openecad_script as oe
from harnesscad.eval.bench.sketch import openecad_score as sc


def line():
    return oe.Call(oe.ADD_LINE, (oe.Arg([0, 0], "start"), oe.Arg([1, 0], "end")))


def arc():
    return oe.Call(oe.ADD_ARC, (
        oe.Arg([1, 0], "start"), oe.Arg([0, 1], "end"), oe.Arg([1, 1], "mid")))


def circle():
    return oe.Call(oe.ADD_CIRCLE, (oe.Arg([0, 0], "center"), oe.Arg(1.0, "radius")))


SQUARE = [line(), line(), line(), line()]


class TestComponents(unittest.TestCase):
    def test_curve_accuracy_exact(self):
        self.assertEqual(sc.curve_accuracy(SQUARE, SQUARE), 1.0)

    def test_curve_accuracy_partial(self):
        pred = [line(), arc(), line(), line()]  # one wrong type
        self.assertEqual(sc.curve_accuracy(pred, SQUARE), 0.75)

    def test_curve_accuracy_extra_penalised(self):
        pred = SQUARE + [line()]  # extra curve
        self.assertEqual(sc.curve_accuracy(pred, SQUARE), 4 / 5)

    def test_curve_accuracy_both_empty(self):
        self.assertEqual(sc.curve_accuracy([], []), 1.0)

    def test_absolutely_correct(self):
        self.assertTrue(sc.loop_absolutely_correct(SQUARE, SQUARE))
        self.assertFalse(sc.loop_absolutely_correct([line(), arc()], SQUARE))

    def test_loop_score(self):
        self.assertEqual(sc.loop_score(True, 0.5), 100.0)
        self.assertEqual(sc.loop_score(False, 0.5), 45.0)

    def test_loops_count_accuracy(self):
        self.assertEqual(sc.loops_count_accuracy(2, 2), 1.0)
        self.assertEqual(sc.loops_count_accuracy(1, 2), 0.5)
        self.assertEqual(sc.loops_count_accuracy(0, 0), 1.0)

    def test_types_accepts_strings(self):
        self.assertEqual(
            sc.curve_accuracy(["add_line", "add_arc"], [arc(), arc()]), 0.5)


class TestScoreEquation(unittest.TestCase):
    def test_perfect_score_is_100(self):
        per_loop = [(True, 1.0), (True, 1.0)]
        self.assertAlmostEqual(sc.score(1.0, 1.0, 1.0, per_loop), 100.0)

    def test_zero_score(self):
        self.assertAlmostEqual(sc.score(0.0, 0.0, 0.0, [(False, 0.0)]), 0.0)

    def test_weights_breakdown(self):
        # Only executability correct: 10 points.
        self.assertAlmostEqual(sc.score(1.0, 0.0, 0.0, [(False, 0.0)]), 10.0)
        # Only curves correct: 45 points.
        self.assertAlmostEqual(sc.score(0.0, 1.0, 0.0, [(False, 0.0)]), 45.0)
        # Only loops-count correct: 5 points.
        self.assertAlmostEqual(sc.score(0.0, 0.0, 1.0, [(False, 0.0)]), 5.0)
        # Only loops fully correct: 40 points.
        self.assertAlmostEqual(sc.score(0.0, 0.0, 0.0, [(True, 1.0)]), 40.0)

    def test_loops_term_averages(self):
        # One perfect loop (100), one 90*0.5=45 -> avg 72.5 -> 40*0.725 = 29.0
        val = sc.score(0.0, 0.0, 0.0, [(True, 1.0), (False, 0.5)])
        self.assertAlmostEqual(val, 40.0 * 72.5 / 100.0)

    def test_empty_loops_term_zero(self):
        # 10 + 45 + 5 + 0 (no loops) = 60.
        self.assertAlmostEqual(sc.score(1.0, 1.0, 1.0, []), 60.0)

    def test_out_of_range_raises(self):
        with self.assertRaises(ValueError):
            sc.score(1.5, 0.0, 0.0, [])


class TestEvaluate(unittest.TestCase):
    def test_perfect_prediction(self):
        target = [SQUARE, [circle()]]
        result = sc.evaluate(target, target, executable=1.0)
        self.assertAlmostEqual(result["overall"], 100.0)
        self.assertEqual(result["curves_accuracy"], 1.0)
        self.assertEqual(result["loops_count_accuracy"], 1.0)

    def test_missing_loop(self):
        target = [SQUARE, [circle()]]
        pred = [SQUARE]  # dropped the circle loop
        result = sc.evaluate(pred, target, executable=1.0)
        self.assertLess(result["overall"], 100.0)
        self.assertEqual(result["loops_count_accuracy"], 0.5)
        # Second reference loop unmatched -> not absolutely correct.
        self.assertFalse(result["per_loop"][1][0])

    def test_wrong_curve_type(self):
        target = [SQUARE]
        pred = [[line(), arc(), line(), line()]]
        result = sc.evaluate(pred, target, executable=1.0)
        self.assertEqual(result["per_loop"][0][1], 0.75)
        self.assertFalse(result["per_loop"][0][0])

    def test_non_executable(self):
        target = [SQUARE]
        result = sc.evaluate(target, target, executable=0.0)
        # Loses the 10-point executability term.
        self.assertAlmostEqual(result["overall"], 90.0)


if __name__ == "__main__":
    unittest.main()
