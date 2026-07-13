"""Tests for bench.cadtransformer_panoptic."""

from __future__ import annotations

import math
import unittest

from harnesscad.eval.bench.cadtransformer_panoptic import (
    instance_from_lengths,
    log_length_weight,
    match_instances,
    panoptic_quality,
    per_class_f1,
    weighted_instance_iou,
)


class TestWeight(unittest.TestCase):
    def test_log_weight(self):
        self.assertAlmostEqual(log_length_weight(0.0), 0.0)
        self.assertAlmostEqual(log_length_weight(math.e - 1), 1.0)

    def test_negative_raises(self):
        with self.assertRaises(ValueError):
            log_length_weight(-1)


class TestWeightedIou(unittest.TestCase):
    def test_identical(self):
        a = {0: 5.0, 1: 3.0}
        self.assertAlmostEqual(weighted_instance_iou(a, a), 1.0, places=4)

    def test_disjoint(self):
        a = {0: 5.0}
        b = {1: 5.0}
        self.assertAlmostEqual(weighted_instance_iou(a, b), 0.0, places=4)

    def test_partial_overlap_weighted(self):
        # shared idx 1 (len e-1 -> w 1); union idx 0,1,2 each w 1 -> 1/3
        a = {0: math.e - 1, 1: math.e - 1}
        b = {1: math.e - 1, 2: math.e - 1}
        self.assertAlmostEqual(weighted_instance_iou(a, b), 1.0 / 3.0, places=4)

    def test_zero_length_shared_contributes_nothing(self):
        # a shared primitive of length 0 has weight 0; overlap counts for 0
        a = {0: 0.0, 1: math.e - 1}
        b = {0: 0.0}
        # intersection {0} weight 0; union {0,1} weight 0 + 1 = 1 -> 0
        self.assertAlmostEqual(weighted_instance_iou(a, b), 0.0, places=4)


class TestMatch(unittest.TestCase):
    def test_same_class_match(self):
        pred = [(1, {0: 5.0, 1: 5.0})]
        gt = [(1, {0: 5.0, 1: 5.0})]
        matches, up, ug = match_instances(pred, gt)
        self.assertEqual(len(matches), 1)
        self.assertEqual(up, [])
        self.assertEqual(ug, [])

    def test_class_mismatch_no_match(self):
        pred = [(2, {0: 5.0, 1: 5.0})]
        gt = [(1, {0: 5.0, 1: 5.0})]
        matches, up, ug = match_instances(pred, gt)
        self.assertEqual(matches, [])
        self.assertEqual(up, [0])
        self.assertEqual(ug, [0])

    def test_below_threshold_no_match(self):
        # iou exactly at 0.5 must NOT match (strictly greater)
        a = {0: math.e - 1, 1: math.e - 1}
        b = {0: math.e - 1, 2: math.e - 1}  # iou = 1/3 < 0.5
        matches, up, ug = match_instances([(1, a)], [(1, b)])
        self.assertEqual(matches, [])

    def test_one_to_one_greedy(self):
        big = {i: 5.0 for i in range(10)}
        overlap = {i: 5.0 for i in range(8)}
        pred = [(1, big), (1, overlap)]
        gt = [(1, big)]
        matches, up, ug = match_instances(pred, gt)
        # pred[0] has higher IoU (identical) -> matched; pred[1] unmatched
        self.assertEqual(len(matches), 1)
        self.assertEqual(matches[0][0], 0)
        self.assertEqual(up, [1])


class TestPerClassF1(unittest.TestCase):
    def test_perfect(self):
        pred = [(1, {0: 5.0}), (2, {1: 5.0})]
        gt = [(1, {0: 5.0}), (2, {1: 5.0})]
        res = per_class_f1(pred, gt)
        self.assertEqual(res["micro"]["f1"], 1.0)
        self.assertEqual(res["per_class"][1]["f1"], 1.0)
        self.assertEqual(res["per_class"][2]["tp"], 1)

    def test_fp_and_fn(self):
        pred = [(1, {0: 5.0})]                 # wrong-class-only extra
        gt = [(2, {1: 5.0})]
        res = per_class_f1(pred, gt)
        self.assertEqual(res["per_class"][1]["fp"], 1)
        self.assertEqual(res["per_class"][2]["fn"], 1)
        self.assertEqual(res["micro"]["f1"], 0.0)

    def test_precision_recall(self):
        # 1 TP, 1 FP, 0 FN -> P=0.5, R=1.0
        pred = [(1, {0: 5.0, 1: 5.0}), (1, {5: 5.0})]
        gt = [(1, {0: 5.0, 1: 5.0})]
        res = per_class_f1(pred, gt)
        self.assertAlmostEqual(res["micro"]["precision"], 0.5)
        self.assertAlmostEqual(res["micro"]["recall"], 1.0)


class TestPanoptic(unittest.TestCase):
    def test_pq_perfect(self):
        pred = [(1, {0: 5.0}), (1, {1: 5.0})]
        res = panoptic_quality(pred, pred)
        self.assertAlmostEqual(res["rq"], 1.0)
        self.assertAlmostEqual(res["sq"], 1.0, places=4)
        self.assertAlmostEqual(res["pq"], 1.0, places=4)

    def test_pq_with_errors(self):
        pred = [(1, {0: 5.0}), (1, {9: 5.0})]  # second is spurious
        gt = [(1, {0: 5.0})]
        res = panoptic_quality(pred, gt)
        self.assertEqual(res["tp"], 1)
        self.assertEqual(res["fp"], 1)
        self.assertEqual(res["fn"], 0)
        self.assertAlmostEqual(res["rq"], 1 / 1.5)

    def test_empty(self):
        res = panoptic_quality([], [])
        self.assertEqual(res["pq"], 0.0)


class TestInstanceFromLengths(unittest.TestCase):
    def test_build(self):
        inst = instance_from_lengths(3, [0, 2, 4], [1.0, 2.0, 3.0])
        self.assertEqual(inst[0], 3)
        self.assertEqual(inst[1], {0: 1.0, 2: 2.0, 4: 3.0})

    def test_length_mismatch(self):
        with self.assertRaises(ValueError):
            instance_from_lengths(1, [0, 1], [1.0])


if __name__ == "__main__":
    unittest.main()
