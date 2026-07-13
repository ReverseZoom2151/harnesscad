"""Tests for the Sketch2CAD scene-reconstruction evaluation metrics."""

import unittest

from harnesscad.eval.bench.vision import sketch2cad_scene_metrics as m
from harnesscad.domain.reconstruction.tokens import sketch2cad_scene_descriptor as sd

SHAPES = sd.SHAPE_TYPES


def obj(shape, pos, rot=(0.0, 0.0), size=(1.0, 1.0, 1.0)):
    return sd.SceneObject(shape, pos, rot, size)


class TestPoseAccuracy(unittest.TestCase):
    def test_all_correct(self):
        self.assertEqual(m.pose_accuracy([1, 2, 3], [1, 2, 3]), 1.0)

    def test_half(self):
        self.assertEqual(m.pose_accuracy([1, 9], [1, 2]), 0.5)

    def test_empty(self):
        self.assertEqual(m.pose_accuracy([], []), 0.0)

    def test_length_mismatch(self):
        with self.assertRaises(ValueError):
            m.pose_accuracy([1], [1, 2])


class TestMatching(unittest.TestCase):
    def test_greedy_nearest(self):
        pred = [obj("cube", (0.0, 0.0, 0.0)), obj("cube", (10.0, 0.0, 0.0))]
        gt = [obj("cube", (10.2, 0.0, 0.0)), obj("cube", (0.1, 0.0, 0.0))]
        pairs, up, ug = m.match_objects(pred, gt)
        self.assertEqual(len(pairs), 2)
        self.assertEqual(up, [])
        self.assertEqual(ug, [])
        # nearest pairing: pred(0,0) -> gt(0.1,..), pred(10,..) -> gt(10.2,..)
        for p, g in pairs:
            self.assertLess(m._dist2(p.position, g.position), 1.0)

    def test_unmatched(self):
        pred = [obj("cube", (0.0, 0.0, 0.0))]
        gt = [obj("cube", (0.0, 0.0, 0.0)), obj("pyramid", (5.0, 5.0, 5.0))]
        pairs, up, ug = m.match_objects(pred, gt)
        self.assertEqual(len(pairs), 1)
        self.assertEqual(len(ug), 1)
        self.assertEqual(ug[0].shape, "pyramid")

    def test_deterministic(self):
        pred = [obj("cube", (1.0, 1.0, 1.0)), obj("hip", (2.0, 2.0, 2.0))]
        gt = [obj("hip", (2.1, 2.0, 2.0)), obj("cube", (1.1, 1.0, 1.0))]
        r1 = m.match_objects(pred, gt)
        r2 = m.match_objects(pred, gt)
        self.assertEqual(r1, r2)


class TestErrors(unittest.TestCase):
    def test_position_error(self):
        pred = [obj("cube", (2.0, 4.0, 6.0))]
        gt = [obj("cube", (0.0, 0.0, 0.0))]
        pairs, _, _ = m.match_objects(pred, gt)
        self.assertEqual(m.position_error(pairs), (2.0, 4.0, 6.0))

    def test_size_error(self):
        pred = [obj("cube", (0.0, 0.0, 0.0), size=(3.0, 2.0, 1.0))]
        gt = [obj("cube", (0.0, 0.0, 0.0), size=(1.0, 1.0, 1.0))]
        pairs, _, _ = m.match_objects(pred, gt)
        self.assertEqual(m.size_error(pairs), (2.0, 1.0, 0.0))

    def test_rotation_error_wraps(self):
        pred = [obj("cube", (0.0, 0.0, 0.0), rot=(350.0, 0.0))]
        gt = [obj("cube", (0.0, 0.0, 0.0), rot=(10.0, 0.0))]
        pairs, _, _ = m.match_objects(pred, gt)
        e = m.rotation_error(pairs)
        self.assertAlmostEqual(e[0], 20.0)  # circular distance, not 340
        self.assertAlmostEqual(e[1], 0.0)

    def test_empty_errors_zero(self):
        self.assertEqual(m.position_error([]), (0.0, 0.0, 0.0))
        self.assertEqual(m.rotation_error([]), (0.0, 0.0))


class TestClassificationF1(unittest.TestCase):
    def test_perfect(self):
        pairs = [(obj("cube", (0, 0, 0)), obj("cube", (0, 0, 0)))]
        f1 = m.classification_f1(pairs, [], [], SHAPES)
        self.assertAlmostEqual(f1, 1.0)

    def test_misclassification(self):
        pairs = [(obj("cube", (0, 0, 0)), obj("pyramid", (0, 0, 0)))]
        f1 = m.classification_f1(pairs, [], [], SHAPES)
        self.assertEqual(f1, 0.0)  # cube fp, pyramid fn, no tp

    def test_unmatched_gt_is_false_negative(self):
        pairs = [(obj("cube", (0, 0, 0)), obj("cube", (0, 0, 0)))]
        ug = [obj("cube", (9, 9, 9))]
        f1 = m.classification_f1(pairs, [], ug, SHAPES)
        # cube: tp=1, fn=1 -> f1 = 2/(2+0+1) = 2/3
        self.assertAlmostEqual(f1, 2 / 3)


class TestEvaluateScene(unittest.TestCase):
    def test_full_report_perfect(self):
        pred = [obj("cube", (5.0, 5.0, 0.0), size=(2.0, 2.0, 2.0))]
        gt = [obj("cube", (5.0, 5.0, 0.0), size=(2.0, 2.0, 2.0))]
        rep = m.evaluate_scene(3, 3, pred, gt, SHAPES)
        self.assertEqual(rep.pose_acc, 1.0)
        self.assertAlmostEqual(rep.classification_f1, 1.0)
        self.assertEqual(rep.position_error, (0.0, 0.0, 0.0))
        self.assertEqual(rep.matched, 1)

    def test_wrong_pose(self):
        pred = [obj("cube", (0, 0, 0))]
        gt = [obj("cube", (0, 0, 0))]
        rep = m.evaluate_scene(1, 2, pred, gt, SHAPES)
        self.assertEqual(rep.pose_acc, 0.0)


if __name__ == "__main__":
    unittest.main()
