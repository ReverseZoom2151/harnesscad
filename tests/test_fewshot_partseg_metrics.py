import unittest
from reconstruction.fewshot_partseg_metrics import (
    confusion, per_class_iou, mean_iou, instance_miou, accuracy,
)


class Tests(unittest.TestCase):
    def test_confusion_counts(self):
        conf = confusion([0, 0, 1, 1], [0, 1, 1, 1])
        # class 0: inter=1, union = pred(2)+gt(1)-1 = 2
        self.assertEqual(conf[0], (1, 2))
        # class 1: inter=2, union = pred(2)+gt(3)-2 = 3
        self.assertEqual(conf[1], (2, 3))

    def test_perfect_prediction(self):
        p = [0, 1, 2, 1]
        self.assertEqual(mean_iou(p, p), 1.0)
        for v in per_class_iou(p, p).values():
            self.assertEqual(v, 1.0)

    def test_iou_values(self):
        ious = per_class_iou([0, 0, 1, 1], [0, 1, 1, 1])
        self.assertAlmostEqual(ious[0], 0.5)
        self.assertAlmostEqual(ious[1], 2 / 3)

    def test_mean_iou_average(self):
        m = mean_iou([0, 0, 1, 1], [0, 1, 1, 1])
        self.assertAlmostEqual(m, (0.5 + 2 / 3) / 2)

    def test_fixed_label_space_includes_absent(self):
        # label 2 in neither -> IoU 1.0 by convention, still counted.
        ious = per_class_iou([0, 1], [0, 1], labels=[0, 1, 2])
        self.assertEqual(ious[2], 1.0)
        self.assertEqual(set(ious), {0, 1, 2})

    def test_length_mismatch(self):
        with self.assertRaises(ValueError):
            confusion([0, 1], [0])

    def test_accuracy(self):
        self.assertAlmostEqual(accuracy([0, 1, 1, 0], [0, 1, 0, 0]), 0.75)
        self.assertEqual(accuracy([], []), 1.0)

    def test_instance_miou(self):
        preds = [[0, 0, 1], [1, 1, 1]]
        gts = [[0, 0, 1], [1, 1, 0]]
        # sample 0 perfect -> 1.0 ; sample 1 mIoU averaged
        m = instance_miou(preds, gts)
        self.assertGreater(m, 0.0)
        self.assertLessEqual(m, 1.0)

    def test_instance_miou_mismatch(self):
        with self.assertRaises(ValueError):
            instance_miou([[0]], [[0], [1]])

    def test_empty_defaults(self):
        self.assertEqual(mean_iou([], []), 1.0)
        self.assertEqual(instance_miou([], []), 1.0)


if __name__ == "__main__":
    unittest.main()
