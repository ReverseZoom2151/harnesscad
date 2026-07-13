"""Tests for bench.rastercad_metrics."""

from __future__ import annotations

import unittest

from harnesscad.eval.bench.rastercad_metrics import (
    PRF,
    VectorizationAccuracy,
    match_primitives,
    primitive_distance,
    raster_iou,
    raster_precision_recall_f1,
    stroke_coverage,
    vectorization_accuracy,
)


def _blank(h: int, w: int) -> list[list[int]]:
    return [[0] * w for _ in range(h)]


class TestRasterIoU(unittest.TestCase):
    def test_identical(self) -> None:
        g = [[1, 0], [0, 1]]
        self.assertEqual(raster_iou(g, g), 1.0)

    def test_disjoint(self) -> None:
        a = [[1, 0], [0, 0]]
        b = [[0, 1], [0, 0]]
        self.assertEqual(raster_iou(a, b), 0.0)

    def test_partial(self) -> None:
        a = [[1, 1], [0, 0]]
        b = [[1, 0], [0, 0]]
        # intersection 1, union 2
        self.assertAlmostEqual(raster_iou(a, b), 0.5)

    def test_both_empty_is_one(self) -> None:
        self.assertEqual(raster_iou(_blank(3, 3), _blank(3, 3)), 1.0)

    def test_shape_mismatch(self) -> None:
        with self.assertRaises(ValueError):
            raster_iou([[1]], [[1, 0]])


class TestPRF(unittest.TestCase):
    def test_perfect(self) -> None:
        g = [[1, 0], [1, 0]]
        r = raster_precision_recall_f1(g, g)
        self.assertIsInstance(r, PRF)
        self.assertEqual((r.precision, r.recall, r.f1), (1.0, 1.0, 1.0))

    def test_over_prediction_low_precision(self) -> None:
        pred = [[1, 1], [1, 1]]
        gt = [[1, 0], [0, 0]]
        r = raster_precision_recall_f1(pred, gt)
        self.assertAlmostEqual(r.precision, 0.25)
        self.assertAlmostEqual(r.recall, 1.0)

    def test_under_prediction_low_recall(self) -> None:
        pred = [[1, 0], [0, 0]]
        gt = [[1, 1], [1, 1]]
        r = raster_precision_recall_f1(pred, gt)
        self.assertAlmostEqual(r.precision, 1.0)
        self.assertAlmostEqual(r.recall, 0.25)


class TestStrokeCoverage(unittest.TestCase):
    def test_full_coverage(self) -> None:
        gt = [[1, 1], [0, 0]]
        pred = [[1, 1], [0, 0]]
        self.assertEqual(stroke_coverage(pred, gt), 1.0)

    def test_half_coverage(self) -> None:
        gt = [[1, 1], [0, 0]]
        pred = [[1, 0], [0, 0]]
        self.assertAlmostEqual(stroke_coverage(pred, gt), 0.5)

    def test_empty_gt_is_one(self) -> None:
        self.assertEqual(stroke_coverage(_blank(3, 3), _blank(3, 3)), 1.0)

    def test_tolerance_covers_near_miss(self) -> None:
        gt = _blank(5, 5)
        gt[2][2] = 1
        pred = _blank(5, 5)
        pred[2][3] = 1  # one pixel off
        self.assertEqual(stroke_coverage(pred, gt, tolerance=0), 0.0)
        self.assertEqual(stroke_coverage(pred, gt, tolerance=1), 1.0)

    def test_negative_tolerance(self) -> None:
        with self.assertRaises(ValueError):
            stroke_coverage(_blank(2, 2), _blank(2, 2), tolerance=-1)


class TestPrimitiveDistance(unittest.TestCase):
    def test_different_types_infinite(self) -> None:
        a = {"type": "line", "start": (0, 0), "end": (1, 1)}
        b = {"type": "circle", "center": (0, 0), "radius": 1.0}
        self.assertEqual(primitive_distance(a, b), float("inf"))

    def test_identical_line_zero(self) -> None:
        a = {"type": "line", "start": (0.0, 0.0), "end": (1.0, 0.0)}
        self.assertAlmostEqual(primitive_distance(a, a), 0.0)

    def test_line_reversal_invariant(self) -> None:
        a = {"type": "line", "start": (0.0, 0.0), "end": (1.0, 0.0)}
        b = {"type": "line", "start": (1.0, 0.0), "end": (0.0, 0.0)}
        self.assertAlmostEqual(primitive_distance(a, b), 0.0)

    def test_circle_distance(self) -> None:
        a = {"type": "circle", "center": (0.0, 0.0), "radius": 1.0}
        b = {"type": "circle", "center": (0.0, 0.0), "radius": 1.5}
        self.assertAlmostEqual(primitive_distance(a, b), 0.5)

    def test_arc_reversal_invariant(self) -> None:
        a = {"type": "arc", "start": (0.0, 0.0), "mid": (0.5, 0.5), "end": (1.0, 0.0)}
        b = {"type": "arc", "start": (1.0, 0.0), "mid": (0.5, 0.5), "end": (0.0, 0.0)}
        self.assertAlmostEqual(primitive_distance(a, b), 0.0)


class TestMatchPrimitives(unittest.TestCase):
    def test_greedy_one_to_one(self) -> None:
        pred = [
            {"type": "line", "start": (0.0, 0.0), "end": (1.0, 0.0)},
            {"type": "line", "start": (0.0, 1.0), "end": (1.0, 1.0)},
        ]
        gt = [
            {"type": "line", "start": (0.0, 1.0), "end": (1.0, 1.0)},
            {"type": "line", "start": (0.0, 0.0), "end": (1.0, 0.0)},
        ]
        matches = match_primitives(pred, gt, threshold=0.01)
        self.assertEqual(len(matches), 2)
        # pred0 matches gt1, pred1 matches gt0.
        pairs = {(i, j) for i, j, _ in matches}
        self.assertEqual(pairs, {(0, 1), (1, 0)})

    def test_threshold_excludes_far(self) -> None:
        pred = [{"type": "circle", "center": (0.0, 0.0), "radius": 1.0}]
        gt = [{"type": "circle", "center": (0.9, 0.0), "radius": 1.0}]
        self.assertEqual(match_primitives(pred, gt, threshold=0.5), [])
        self.assertEqual(len(match_primitives(pred, gt, threshold=1.0)), 1)


class TestVectorizationAccuracy(unittest.TestCase):
    def test_perfect(self) -> None:
        prims = [
            {"type": "line", "start": (0.0, 0.0), "end": (1.0, 0.0)},
            {"type": "circle", "center": (0.5, 0.5), "radius": 0.2},
        ]
        acc = vectorization_accuracy(prims, prims, threshold=0.01)
        self.assertIsInstance(acc, VectorizationAccuracy)
        self.assertEqual(acc.num_matched, 2)
        self.assertAlmostEqual(acc.type_accuracy, 1.0)
        self.assertAlmostEqual(acc.f1, 1.0)
        self.assertAlmostEqual(acc.mean_matched_distance, 0.0)

    def test_wrong_type(self) -> None:
        pred = [{"type": "line", "start": (0.0, 0.0), "end": (1.0, 0.0)}]
        gt = [{"type": "circle", "center": (0.5, 0.5), "radius": 0.2}]
        acc = vectorization_accuracy(pred, gt, threshold=0.5)
        self.assertEqual(acc.num_matched, 0)
        self.assertAlmostEqual(acc.type_accuracy, 0.0)
        self.assertAlmostEqual(acc.f1, 0.0)

    def test_both_empty(self) -> None:
        acc = vectorization_accuracy([], [], threshold=0.05)
        self.assertAlmostEqual(acc.type_accuracy, 1.0)
        self.assertAlmostEqual(acc.precision, 1.0)
        self.assertAlmostEqual(acc.recall, 1.0)

    def test_partial_recall(self) -> None:
        line = {"type": "line", "start": (0.0, 0.0), "end": (1.0, 0.0)}
        gt = [line, {"type": "circle", "center": (0.5, 0.5), "radius": 0.2}]
        acc = vectorization_accuracy([line], gt, threshold=0.01)
        self.assertEqual(acc.num_matched, 1)
        self.assertAlmostEqual(acc.recall, 0.5)
        self.assertAlmostEqual(acc.precision, 1.0)

    def test_negative_threshold(self) -> None:
        with self.assertRaises(ValueError):
            vectorization_accuracy([], [], threshold=-1.0)


if __name__ == "__main__":
    unittest.main()
