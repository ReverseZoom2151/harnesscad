import unittest

from bench.sympoint_pointwise_eval import (
    ConfusionMatrix,
    evaluate,
    per_class_scores,
    point_wise_eval,
)


class TestConfusionMatrix(unittest.TestCase):
    def test_counts(self):
        cm = ConfusionMatrix(num_classes=3, ignore_label=3)
        cm.update([0, 1, 1, 2], [0, 1, 2, 2])
        self.assertEqual(cm.true_positives(), [1, 1, 1])
        self.assertEqual(cm.support(), [1, 1, 2])
        self.assertEqual(cm.predicted_count(), [1, 2, 1])

    def test_ignore_ground_truth_dropped(self):
        cm = ConfusionMatrix(num_classes=2, ignore_label=2)
        cm.update([0, 1], [0, 2])
        self.assertEqual(sum(cm.support()), 1)

    def test_predicted_ignore_absorbed(self):
        cm = ConfusionMatrix(num_classes=2, ignore_label=2)
        cm.update([2], [0])
        self.assertEqual(cm.support(), [1, 0])
        self.assertEqual(cm.true_positives(), [0, 0])
        self.assertEqual(cm.predicted_count(), [0, 0])

    def test_length_mismatch(self):
        cm = ConfusionMatrix(num_classes=2, ignore_label=2)
        with self.assertRaises(ValueError):
            cm.update([0], [0, 1])

    def test_bad_gt_label(self):
        cm = ConfusionMatrix(num_classes=2, ignore_label=5)
        with self.assertRaises(ValueError):
            cm.update([0], [3])

    def test_bad_num_classes(self):
        with self.assertRaises(ValueError):
            ConfusionMatrix(num_classes=0)

    def test_streaming_update(self):
        cm = ConfusionMatrix(num_classes=2, ignore_label=2)
        cm.update([0], [0]).update([1], [1])
        self.assertEqual(cm.true_positives(), [1, 1])


class TestScores(unittest.TestCase):
    def test_perfect(self):
        res = point_wise_eval([0, 1, 1], [0, 1, 1], num_classes=2, ignore_label=2)
        self.assertAlmostEqual(res["miou"], 1.0, places=5)
        self.assertAlmostEqual(res["fwiou"], 1.0, places=5)
        self.assertAlmostEqual(res["pacc"], 1.0, places=5)
        self.assertAlmostEqual(res["macc"], 1.0, places=5)

    def test_absent_class_excluded_not_zero(self):
        # class 1 never appears in ground truth -> excluded from mIoU
        res = point_wise_eval([0, 0], [0, 0], num_classes=2, ignore_label=2)
        self.assertAlmostEqual(res["miou"], 1.0, places=5)
        self.assertIsNone(res["per_class"][1]["iou"])

    def test_miou_vs_fwiou_differ_on_skewed_data(self):
        # 8 points of class 0 all correct; 2 points of class 1 all wrong
        pred = [0] * 8 + [0, 0]
        gt = [0] * 8 + [1, 1]
        res = point_wise_eval(pred, gt, num_classes=2, ignore_label=2)
        self.assertAlmostEqual(res["per_class"][0]["iou"], 0.8, places=4)
        self.assertAlmostEqual(res["per_class"][1]["iou"], 0.0, places=6)
        self.assertAlmostEqual(res["miou"], 0.4, places=4)
        self.assertAlmostEqual(res["fwiou"], 0.8 * 0.8, places=4)
        self.assertGreater(res["fwiou"], res["miou"])
        self.assertAlmostEqual(res["pacc"], 0.8, places=4)
        self.assertAlmostEqual(res["macc"], 0.5, places=4)

    def test_iou_penalises_false_positives(self):
        # class 1 predicted where class 0 truly is -> class 1 IoU degraded
        res = point_wise_eval([1, 1], [0, 1], num_classes=2, ignore_label=2)
        self.assertAlmostEqual(res["per_class"][1]["iou"], 0.5, places=4)
        self.assertAlmostEqual(res["per_class"][0]["iou"], 0.0, places=6)

    def test_ignored_points_do_not_count(self):
        res = point_wise_eval([0, 1, 0], [0, 1, 35], num_classes=2, ignore_label=35)
        self.assertAlmostEqual(res["pacc"], 1.0, places=5)
        self.assertAlmostEqual(res["per_class"][0]["support"], 1.0)

    def test_empty(self):
        res = point_wise_eval([], [], num_classes=3, ignore_label=3)
        self.assertEqual(res["miou"], 0.0)
        self.assertEqual(res["pacc"], 0.0)

    def test_per_class_table_shape(self):
        cm = ConfusionMatrix(num_classes=4, ignore_label=4)
        cm.update([0, 1], [0, 1])
        table = per_class_scores(cm)
        self.assertEqual(len(table), 4)
        self.assertIsNone(table[3]["accuracy"])

    def test_evaluate_matches_oneshot(self):
        cm = ConfusionMatrix(num_classes=2, ignore_label=2)
        cm.update([0, 1], [0, 0])
        self.assertEqual(evaluate(cm), point_wise_eval([0, 1], [0, 0],
                                                       num_classes=2, ignore_label=2))

    def test_deterministic(self):
        a = point_wise_eval([0, 1, 1], [0, 1, 0], num_classes=2, ignore_label=2)
        b = point_wise_eval([0, 1, 1], [0, 1, 0], num_classes=2, ignore_label=2)
        self.assertEqual(a, b)


if __name__ == "__main__":
    unittest.main()
