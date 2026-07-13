"""Tests for ComplexGen chain-complex evaluation metrics."""

import unittest

from harnesscad.eval.bench.geometry import complex_matching as m
from harnesscad.domain.reconstruction.brep import chain_complex as cc
from tests.test_complexgen_chain_complex import cube_complex


def _shift(cx, delta):
    def sp(p):
        return (p[0] + delta, p[1] + delta, p[2] + delta)
    corners = [sp(c) for c in cx.corners]
    curves = [cc.Curve(tuple(sp(p) for p in c.points), c.closed) for c in cx.curves]
    patches = [cc.Patch(tuple(sp(p) for p in q.points)) for q in cx.patches]
    return cc.make_complex(corners, curves, patches, cx.curve_corner, cx.patch_curve)


class TestChamfer(unittest.TestCase):
    def test_identical_clouds(self):
        x = [(0.0, 0.0, 0.0), (1.0, 0.0, 0.0)]
        self.assertAlmostEqual(m.chamfer_distance(x, x), 0.0)

    def test_one_sided(self):
        x = [(0.0, 0.0, 0.0)]
        y = [(0.0, 0.0, 0.0), (3.0, 0.0, 0.0)]
        self.assertAlmostEqual(m.chamfer_distance(x, y, "x_to_y"), 0.0)
        self.assertAlmostEqual(m.chamfer_distance(x, y, "y_to_x"), 1.5)
        self.assertAlmostEqual(m.chamfer_distance(x, y, "bi"), 0.75)

    def test_bad_direction(self):
        with self.assertRaises(ValueError):
            m.chamfer_distance([(0.0, 0.0, 0.0)], [(0.0, 0.0, 0.0)], "nope")

    def test_empty(self):
        with self.assertRaises(ValueError):
            m.chamfer_distance([], [(0.0, 0.0, 0.0)])


class TestCurveDistance(unittest.TestCase):
    def test_orientation_invariant(self):
        a = [(0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (2.0, 0.0, 0.0)]
        b = list(reversed(a))
        self.assertAlmostEqual(m.curve_distance(a, b), 0.0)

    def test_offset(self):
        a = [(0.0, 0.0, 0.0), (1.0, 0.0, 0.0)]
        b = [(0.0, 1.0, 0.0), (1.0, 1.0, 0.0)]
        self.assertAlmostEqual(m.curve_distance(a, b), 1.0)

    def test_length_mismatch(self):
        with self.assertRaises(ValueError):
            m.curve_distance([(0.0, 0.0, 0.0)], [(0.0, 0.0, 0.0), (1.0, 0.0, 0.0)])

    def test_closed_curve_shift_invariant(self):
        square = [(0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (1.0, 1.0, 0.0), (0.0, 1.0, 0.0)]
        rolled = square[2:] + square[:2]
        self.assertAlmostEqual(m.closed_curve_distance(square, rolled), 0.0)
        self.assertGreater(m.curve_distance(square, rolled), 0.5)

    def test_closed_curve_reversed(self):
        square = [(0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (1.0, 1.0, 0.0), (0.0, 1.0, 0.0)]
        self.assertAlmostEqual(m.closed_curve_distance(square, list(reversed(square))), 0.0)

    def test_sampled_curve_distance_type_mismatch(self):
        open_c = cc.Curve(((0.0, 0.0, 0.0), (1.0, 0.0, 0.0)), False)
        closed_c = cc.Curve(((0.0, 0.0, 0.0), (1.0, 0.0, 0.0)), True)
        self.assertEqual(m.sampled_curve_distance(open_c, closed_c), m.BIG)
        self.assertAlmostEqual(m.sampled_curve_distance(open_c, open_c), 0.0)


class TestMatching(unittest.TestCase):
    def test_perfect_match(self):
        cost = [[0.0, 5.0], [5.0, 0.0]]
        matches = m.match(cost, 1.0)
        self.assertEqual([(0, 0), (1, 1)], [(p, g) for (p, g, _) in matches])

    def test_threshold_rejects(self):
        cost = [[2.0, 5.0], [5.0, 2.0]]
        self.assertEqual(m.match(cost, 1.0), [])
        self.assertEqual(len(m.match(cost, 2.0)), 2)

    def test_more_predictions_than_gt(self):
        cost = [[0.0], [3.0], [0.05]]
        matches = m.match(cost, 0.1)
        self.assertEqual(len(matches), 1)
        self.assertEqual(matches[0][0], 0)

    def test_empty(self):
        self.assertEqual(m.match([], 1.0), [])
        self.assertEqual(m.match([[]], 1.0), [])

    def test_prf(self):
        prf = m.prf_from_matches(2, 4, 5)
        self.assertAlmostEqual(prf.precision, 0.5)
        self.assertAlmostEqual(prf.recall, 0.4)
        self.assertAlmostEqual(prf.f1, 2 * 0.5 * 0.4 / 0.9)

    def test_prf_empty_both(self):
        prf = m.prf_from_matches(0, 0, 0)
        self.assertAlmostEqual(prf.precision, 1.0)
        self.assertAlmostEqual(prf.recall, 1.0)


class TestEvaluateComplex(unittest.TestCase):
    def setUp(self):
        self.gt = cube_complex()

    def test_identical_complex_scores_perfect(self):
        report = m.evaluate_complex(self.gt, self.gt)
        for key in ("corners", "curves", "patches", "curve_corner", "patch_curve"):
            self.assertAlmostEqual(report[key].f1, 1.0, msg=key)
        self.assertAlmostEqual(report["complex_chamfer"], 0.0)
        self.assertAlmostEqual(report["corner_chamfer"], 0.0)

    def test_shifted_complex_fails_tolerance(self):
        pred = _shift(self.gt, 0.5)
        report = m.evaluate_complex(pred, self.gt, 0.1, 0.1, 0.1)
        self.assertAlmostEqual(report["corners"].f1, 0.0)
        self.assertAlmostEqual(report["curves"].f1, 0.0)
        self.assertGreater(report["complex_chamfer"], 0.1)

    def test_small_shift_still_matches(self):
        pred = _shift(self.gt, 0.01)
        report = m.evaluate_complex(pred, self.gt, 0.1, 0.1, 0.1)
        self.assertAlmostEqual(report["corners"].f1, 1.0)
        self.assertAlmostEqual(report["curves"].f1, 1.0)
        self.assertAlmostEqual(report["patches"].f1, 1.0)
        self.assertGreater(report["corner_chamfer"], 0.0)

    def test_missing_patch_lowers_recall(self):
        gt = self.gt
        pred = cc.make_complex(gt.corners, gt.curves, gt.patches[:-1],
                               gt.curve_corner, gt.patch_curve[:-1])
        report = m.evaluate_complex(pred, gt)
        self.assertAlmostEqual(report["patches"].precision, 1.0)
        self.assertAlmostEqual(report["patches"].recall, 5.0 / 6.0)
        self.assertLess(report["patch_curve"].recall, 1.0)

    def test_topology_errors_detected(self):
        gt = self.gt
        ev = [list(r) for r in gt.curve_corner]
        ev[0][0] = 0
        ev[0][2] = 1                              # wrong corner for curve 0
        pred = cc.make_complex(gt.corners, gt.curves, gt.patches, ev, gt.patch_curve)
        report = m.evaluate_complex(pred, gt)
        self.assertAlmostEqual(report["corners"].f1, 1.0)
        self.assertLess(report["curve_corner"].precision, 1.0)
        self.assertLess(report["curve_corner"].recall, 1.0)
        self.assertAlmostEqual(report["patch_curve"].f1, 1.0)

    def test_deterministic(self):
        pred = _shift(self.gt, 0.02)
        self.assertEqual(m.evaluate_complex(pred, self.gt),
                         m.evaluate_complex(pred, self.gt))


if __name__ == "__main__":
    unittest.main()
