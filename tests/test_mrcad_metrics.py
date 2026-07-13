"""Tests for bench.mrcad_metrics."""

import math
import unittest

from harnesscad.domain.editing.mrcad_schema import Design, arc, circle, line
from harnesscad.domain.editing.mrcad_refinement import RefinementSession
from harnesscad.domain.editing.mrcad_schema import MakeCurve, Message, MoveCurve
from harnesscad.eval.bench.geometry.mrcad_metrics import (
    ConvergenceReport,
    chamfer_asymmetric,
    chamfer_symmetric,
    convergence,
    edit_accuracy,
    exact_match,
    proportional_improvement,
    proportional_improvement_from_distances,
    sample_points,
)


class SampleTest(unittest.TestCase):
    def test_line_endpoints_included(self):
        pts = sample_points(line((0, 0), (10, 0)), n=11)
        self.assertEqual(len(pts), 11)
        self.assertAlmostEqual(pts[0][0], 0.0)
        self.assertAlmostEqual(pts[-1][0], 10.0)
        # evenly spaced
        self.assertAlmostEqual(pts[5][0], 5.0)

    def test_circle_points_on_radius(self):
        pts = sample_points(circle((-2, 0), (2, 0)), n=8)
        self.assertEqual(len(pts), 8)
        for x, y in pts:
            self.assertAlmostEqual(math.hypot(x, y), 2.0)

    def test_arc_endpoints_and_curvature(self):
        # quarter-circle-ish arc through (1,0)-(0,1)-(-1,0) about origin
        pts = sample_points(arc((1, 0), (0, 1), (-1, 0)), n=9)
        self.assertEqual(len(pts), 9)
        self.assertAlmostEqual(pts[0][0], 1.0)
        self.assertAlmostEqual(pts[0][1], 0.0)
        self.assertAlmostEqual(pts[-1][0], -1.0, places=6)
        self.assertAlmostEqual(pts[-1][1], 0.0, places=6)
        for x, y in pts:
            self.assertAlmostEqual(math.hypot(x, y), 1.0, places=6)
        # mid sample should pass near (0,1)
        self.assertAlmostEqual(pts[4][1], 1.0, places=6)

    def test_arc_collinear_fallback(self):
        pts = sample_points(arc((0, 0), (1, 0), (2, 0)), n=5)
        self.assertEqual(len(pts), 5)
        for x, y in pts:
            self.assertAlmostEqual(y, 0.0)


class ChamferTest(unittest.TestCase):
    def test_identical_designs_zero(self):
        d = Design((line((0, 0), (4, 0)), circle((0, 0), (2, 0))))
        self.assertAlmostEqual(chamfer_symmetric(d, d), 0.0)

    def test_symmetry(self):
        a = Design((line((0, 0), (4, 0)),))
        b = Design((line((0, 2), (4, 2)),))
        self.assertAlmostEqual(chamfer_symmetric(a, b), chamfer_symmetric(b, a))

    def test_further_apart_larger(self):
        base = Design((line((0, 0), (4, 0)),))
        near = Design((line((0, 1), (4, 1)),))
        far = Design((line((0, 5), (4, 5)),))
        self.assertLess(
            chamfer_symmetric(base, near), chamfer_symmetric(base, far)
        )

    def test_cap_normalisation(self):
        # A single point maximally far: normalised capped distance is 1.0 each.
        a = Design((line((0, 0), (0, 0.0001)),))  # essentially a point at origin
        b = Design((line((100, 100), (100, 100.0001)),))
        # asymmetric a->b: every sampled src point capped -> sum == n_points
        val = chamfer_asymmetric(a, b, n=10)
        self.assertAlmostEqual(val, 10.0, places=6)

    def test_empty_target(self):
        a = Design((line((0, 0), (4, 0)),))
        # to empty design -> all capped
        self.assertAlmostEqual(chamfer_asymmetric(a, Design.empty(), n=10), 10.0)


class ProportionalImprovementTest(unittest.TestCase):
    def test_from_distances(self):
        self.assertAlmostEqual(proportional_improvement_from_distances(1.0, 0.25), 0.75)
        self.assertAlmostEqual(proportional_improvement_from_distances(0.0, 0.0), 0.0)

    def test_positive_when_closer(self):
        target = Design((line((0, 0), (4, 0)),))
        before = Design((line((0, 5), (4, 5)),))
        after = Design((line((0, 1), (4, 1)),))
        self.assertGreater(proportional_improvement(before, after, target), 0.0)

    def test_negative_when_worse(self):
        # models often made destructive edits during refinement (Sec. 6.3).
        target = Design((line((0, 0), (4, 0)),))
        before = Design((line((0, 1), (4, 1)),))
        after = Design((line((0, 6), (4, 6)),))
        self.assertLess(proportional_improvement(before, after, target), 0.0)


class EditAccuracyTest(unittest.TestCase):
    def test_exact_match(self):
        gold = (MakeCurve(line((0, 0), (1, 0))), MoveCurve(line((0, 0), (1, 0)), (0, 1)))
        self.assertTrue(exact_match(gold, gold))
        self.assertFalse(exact_match(gold[::-1], gold))

    def test_partial_recall(self):
        gold = (MakeCurve(line((0, 0), (1, 0))), MakeCurve(circle((0, 0), (2, 0))))
        pred = (MakeCurve(line((0, 0), (1, 0))),)
        self.assertAlmostEqual(edit_accuracy(pred, gold), 0.5)

    def test_order_insensitive(self):
        gold = (MakeCurve(line((0, 0), (1, 0))), MakeCurve(circle((0, 0), (2, 0))))
        self.assertAlmostEqual(edit_accuracy(gold[::-1], gold), 1.0)

    def test_empty_cases(self):
        self.assertAlmostEqual(edit_accuracy((), ()), 1.0)
        self.assertAlmostEqual(edit_accuracy((MakeCurve(line((0, 0), (1, 0))),), ()), 0.0)


class ConvergenceTest(unittest.TestCase):
    def _build_designs(self):
        target = Design((line((0, 0), (4, 0)),))
        s = RefinementSession()
        s.play_round(Message(text="make far"), [MakeCurve(line((0, 6), (4, 6)))])
        s.play_round(
            Message(text="move down"),
            [MoveCurve(line((0, 6), (4, 6)), (0, -3))],
        )
        s.play_round(
            Message(text="move down more"),
            [MoveCurve(line((0, 3), (4, 3)), (0, -3))],
        )
        return target, s.rollout().designs()

    def test_trajectory_and_monotone(self):
        target, designs = self._build_designs()
        rep = convergence(designs, target)
        self.assertIsInstance(rep, ConvergenceReport)
        self.assertEqual(len(rep.distances), 3)
        self.assertTrue(rep.monotone_nonincreasing)
        self.assertGreater(rep.total_reduction, 0.0)
        self.assertEqual(len(rep.per_round_pi), 3)

    def test_rounds_to_win(self):
        target, designs = self._build_designs()
        rep = convergence(designs, target, threshold=0.001)
        # final round reaches the target exactly
        self.assertEqual(rep.rounds_to_win, 3)
        self.assertAlmostEqual(rep.distances[-1], 0.0)

    def test_no_win_when_threshold_tight_and_unmet(self):
        target = Design((line((0, 0), (4, 0)),))
        designs = (Design((line((0, 9), (4, 9)),)),)
        rep = convergence(designs, target, threshold=0.0001)
        self.assertIsNone(rep.rounds_to_win)


if __name__ == "__main__":
    unittest.main()
