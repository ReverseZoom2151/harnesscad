"""Tests for bench.rag6d_pose_metrics."""

import math
import unittest

from harnesscad.eval.bench.vision.rag6d_pose_metrics import (
    add,
    add_recall,
    add_s,
    add_s_recall,
    model_diameter,
    pose_accuracy_5cm_5deg,
    rotation_angle_error,
    rotation_angle_error_deg,
    transform_points,
    translation_error,
)

I3 = ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0))


def rot_z(deg):
    c, s = math.cos(math.radians(deg)), math.sin(math.radians(deg))
    return ((c, -s, 0.0), (s, c, 0.0), (0.0, 0.0, 1.0))


CUBE = [
    (1.0, 1.0, 1.0), (1.0, 1.0, -1.0), (1.0, -1.0, 1.0), (1.0, -1.0, -1.0),
    (-1.0, 1.0, 1.0), (-1.0, 1.0, -1.0), (-1.0, -1.0, 1.0), (-1.0, -1.0, -1.0),
]


class TestIdentity(unittest.TestCase):
    def test_add_zero_for_identical_pose(self):
        self.assertAlmostEqual(add(I3, (0, 0, 0), I3, (0, 0, 0), CUBE), 0.0)

    def test_add_s_zero_for_identical_pose(self):
        self.assertAlmostEqual(add_s(I3, (0, 0, 0), I3, (0, 0, 0), CUBE), 0.0)

    def test_rotation_error_zero(self):
        self.assertAlmostEqual(rotation_angle_error(I3, I3), 0.0)

    def test_translation_error_zero(self):
        self.assertAlmostEqual(translation_error((1, 2, 3), (1, 2, 3)), 0.0)


class TestTranslation(unittest.TestCase):
    def test_pure_translation_add(self):
        # every point shifted by (0,0,d) -> ADD = d
        d = add(I3, (0.0, 0.0, 3.0), I3, (0.0, 0.0, 0.0), CUBE)
        self.assertAlmostEqual(d, 3.0)

    def test_translation_error_value(self):
        self.assertAlmostEqual(translation_error((0, 0, 0), (3, 4, 0)), 5.0)


class TestRotation(unittest.TestCase):
    def test_known_angle(self):
        self.assertAlmostEqual(rotation_angle_error_deg(rot_z(30.0), I3), 30.0, places=6)

    def test_angle_symmetry(self):
        a = rotation_angle_error(rot_z(47.0), rot_z(12.0))
        self.assertAlmostEqual(math.degrees(a), 35.0, places=6)

    def test_clamp_no_domain_error(self):
        # A slightly non-orthonormal matrix must not raise from acos.
        R = ((1.0000001, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0))
        self.assertAlmostEqual(rotation_angle_error(R, I3), 0.0, places=3)


class TestAddSSymmetry(unittest.TestCase):
    def test_add_s_le_add(self):
        R = rot_z(90.0)
        a = add(R, (0, 0, 0), I3, (0, 0, 0), CUBE)
        a_s = add_s(R, (0, 0, 0), I3, (0, 0, 0), CUBE)
        self.assertLessEqual(a_s, a + 1e-12)

    def test_add_s_zero_for_symmetric_rotation(self):
        # 90-degree z-rotation maps the cube's point set onto itself, so ADD-S=0.
        R = rot_z(90.0)
        self.assertAlmostEqual(add_s(R, (0, 0, 0), I3, (0, 0, 0), CUBE), 0.0, places=9)


class TestAccuracyFlag(unittest.TestCase):
    def test_pass(self):
        self.assertTrue(pose_accuracy_5cm_5deg(rot_z(3.0), (0.0, 0.0, 0.02),
                                               I3, (0.0, 0.0, 0.0)))

    def test_fail_rotation(self):
        self.assertFalse(pose_accuracy_5cm_5deg(rot_z(9.0), (0.0, 0.0, 0.0),
                                                I3, (0.0, 0.0, 0.0)))

    def test_fail_translation(self):
        self.assertFalse(pose_accuracy_5cm_5deg(I3, (0.0, 0.0, 0.2),
                                                I3, (0.0, 0.0, 0.0)))


class TestRecall(unittest.TestCase):
    def test_diameter(self):
        # cube of side 2 -> space diagonal 2*sqrt(3)
        self.assertAlmostEqual(model_diameter(CUBE), 2.0 * math.sqrt(3.0))

    def test_recall_mixed(self):
        d = model_diameter(CUBE)
        good = (I3, (0.0, 0.0, 0.01 * d), I3, (0.0, 0.0, 0.0))   # ADD tiny
        bad = (I3, (0.0, 0.0, 0.5 * d), I3, (0.0, 0.0, 0.0))     # ADD huge
        r = add_recall([good, bad], CUBE, diameter_fraction=0.1)
        self.assertAlmostEqual(r, 0.5)

    def test_recall_all_pass(self):
        preds = [(I3, (0, 0, 0), I3, (0, 0, 0))] * 4
        self.assertAlmostEqual(add_recall(preds, CUBE), 1.0)
        self.assertAlmostEqual(add_s_recall(preds, CUBE), 1.0)

    def test_recall_empty(self):
        self.assertEqual(add_recall([], CUBE), 0.0)


class TestTransformPoints(unittest.TestCase):
    def test_transform(self):
        pts = transform_points(I3, (1.0, 2.0, 3.0), [(0.0, 0.0, 0.0)])
        self.assertEqual(pts, [(1.0, 2.0, 3.0)])


if __name__ == "__main__":
    unittest.main()
