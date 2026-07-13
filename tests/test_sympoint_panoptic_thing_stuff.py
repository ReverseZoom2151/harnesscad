import math
import unittest

from harnesscad.eval.bench.vision.sympoint_panoptic_thing_stuff import (
    DEFAULT_IGNORE_LABEL,
    IOU_THRESHOLD,
    MIN_OBJ_SCORE,
    accumulate,
    evaluate,
    panoptic_report,
    per_class_quality,
    point_weights,
    weighted_mask_iou,
)


class TestWeights(unittest.TestCase):
    def test_log_weight_rounded(self):
        w = point_weights([0.0, math.e - 1.0])
        self.assertEqual(w[0], 0.0)
        self.assertEqual(w[1], 1.0)

    def test_rounding_to_three_digits(self):
        self.assertEqual(point_weights([1.0]), [round(math.log(2.0), 3)])

    def test_negative_length(self):
        with self.assertRaises(ValueError):
            point_weights([-1.0])


class TestIoU(unittest.TestCase):
    def setUp(self):
        self.w = [1.0, 1.0, 1.0, 1.0]

    def test_identical(self):
        self.assertGreater(weighted_mask_iou([0, 1], [0, 1], self.w), 0.99)

    def test_disjoint(self):
        self.assertAlmostEqual(weighted_mask_iou([0], [1], self.w), 0.0)

    def test_half(self):
        self.assertAlmostEqual(weighted_mask_iou([0, 1], [1, 2], self.w), 1.0 / 3.0, places=4)

    def test_weighting_matters(self):
        w = [10.0, 1.0, 1.0, 1.0]
        iou = weighted_mask_iou([0], [0, 1], w)
        self.assertAlmostEqual(iou, 10.0 / 11.0, places=4)

    def test_index_out_of_range(self):
        with self.assertRaises(IndexError):
            weighted_mask_iou([9], [0], self.w)


class TestAccumulate(unittest.TestCase):
    def setUp(self):
        self.w = [1.0] * 6

    def test_true_positive(self):
        counts = accumulate([(1, 0.9, [0, 1, 2])], [(1, [0, 1, 2])], self.w, num_classes=4)
        self.assertEqual(counts["tp"][1], 1.0)
        self.assertGreater(counts["tp_iou"][1], 0.99)
        self.assertEqual(sum(counts["fn"]), 0.0)
        self.assertEqual(sum(counts["fp"]), 0.0)

    def test_score_gate_drops_prediction(self):
        counts = accumulate([(1, MIN_OBJ_SCORE - 0.01, [0, 1])], [(1, [0, 1])],
                            self.w, num_classes=4)
        self.assertEqual(counts["tp"][1], 0.0)
        self.assertEqual(counts["fn"][1], 1.0)

    def test_wrong_class_overlap_is_fp_for_predicted_class(self):
        counts = accumulate([(2, 0.9, [0, 1, 2])], [(1, [0, 1, 2])], self.w, num_classes=4)
        self.assertEqual(counts["fp"][2], 1.0)
        self.assertEqual(counts["tp"][1], 0.0)
        # GT was "matched" (overlapped) so it is not counted as FN -- reference behaviour
        self.assertEqual(counts["fn"][1], 0.0)

    def test_unmatched_gt_is_fn(self):
        counts = accumulate([(1, 0.9, [3, 4])], [(1, [0, 1])], self.w, num_classes=4)
        self.assertEqual(counts["fn"][1], 1.0)

    def test_gt_can_be_tp_and_spawn_fp(self):
        preds = [(1, 0.9, [0, 1, 2]), (3, 0.8, [0, 1, 2])]
        counts = accumulate(preds, [(1, [0, 1, 2])], self.w, num_classes=4)
        self.assertEqual(counts["tp"][1], 1.0)
        self.assertEqual(counts["fp"][3], 1.0)

    def test_ignore_label_skipped(self):
        counts = accumulate([(DEFAULT_IGNORE_LABEL, 0.9, [0])],
                            [(DEFAULT_IGNORE_LABEL, [0])], [1.0], num_classes=2)
        self.assertEqual(sum(counts["tp"]) + sum(counts["fp"]) + sum(counts["fn"]), 0.0)

    def test_below_threshold_overlap_not_matched(self):
        counts = accumulate([(1, 0.9, [0, 1, 2, 3, 4])], [(1, [4, 5])], self.w + [1.0],
                            num_classes=4)
        self.assertEqual(counts["tp"][1], 0.0)
        self.assertEqual(counts["fn"][1], 1.0)

    def test_accumulator_reuse(self):
        counts = accumulate([(1, 0.9, [0, 1])], [(1, [0, 1])], self.w, num_classes=4)
        counts = accumulate([(1, 0.9, [2, 3])], [(1, [2, 3])], self.w, num_classes=4,
                            counts=counts)
        self.assertEqual(counts["tp"][1], 2.0)


class TestReport(unittest.TestCase):
    def test_perfect_scores(self):
        w = [1.0] * 4
        report = evaluate([(0, 1.0, [0, 1]), (30, 1.0, [2, 3])],
                          [(0, [0, 1]), (30, [2, 3])], [math.e - 1.0] * 4)
        self.assertGreater(report["all"]["pq"], 0.99)
        self.assertGreater(report["thing"]["pq"], 0.99)
        self.assertGreater(report["stuff"]["pq"], 0.99)
        self.assertEqual(len(w), 4)

    def test_thing_stuff_separation(self):
        # thing class 0 correct, stuff class 30 missed
        report = evaluate([(0, 1.0, [0, 1])], [(0, [0, 1]), (30, [2, 3])], [1.0] * 4)
        self.assertGreater(report["thing"]["pq"], 0.9)
        self.assertEqual(report["stuff"]["pq"], 0.0)
        self.assertEqual(report["stuff"]["fn"], 1.0)
        self.assertLess(report["all"]["rq"], 1.0)

    def test_rq_formula(self):
        # 1 TP, 1 FN -> RQ = 1 / (1 + 0.5) = 2/3
        report = evaluate([(0, 1.0, [0, 1])], [(0, [0, 1]), (0, [2, 3])], [1.0] * 4)
        self.assertAlmostEqual(report["all"]["rq"], 2.0 / 3.0, places=4)
        self.assertAlmostEqual(report["all"]["pq"],
                               report["all"]["rq"] * report["all"]["sq"], places=9)

    def test_empty_counts_are_zero(self):
        report = evaluate([], [], [1.0])
        self.assertEqual(report["all"]["pq"], 0.0)
        self.assertEqual(report["all"]["sq"], 0.0)

    def test_per_class_table(self):
        counts = accumulate([(1, 0.9, [0, 1])], [(1, [0, 1])], [1.0, 1.0], num_classes=3)
        table = per_class_quality(counts)
        self.assertEqual(len(table), 3)
        self.assertGreater(table[1]["pq"], 0.9)
        self.assertEqual(table[0]["pq"], 0.0)

    def test_custom_groups(self):
        counts = accumulate([(1, 0.9, [0])], [(1, [0])], [1.0], num_classes=3)
        report = panoptic_report(counts, thing_classes=(0,), stuff_classes=(1,))
        self.assertEqual(report["thing"]["tp"], 0.0)
        self.assertEqual(report["stuff"]["tp"], 1.0)

    def test_threshold_constant(self):
        self.assertEqual(IOU_THRESHOLD, 0.5)

    def test_deterministic(self):
        args = ([(0, 1.0, [0, 1])], [(0, [0, 1]), (0, [2, 3])], [1.0] * 4)
        self.assertEqual(evaluate(*args), evaluate(*args))


if __name__ == "__main__":
    unittest.main()
