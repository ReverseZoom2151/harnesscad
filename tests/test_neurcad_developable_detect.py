"""Tests for geometry.neurcad_developable_detect (developability classifier)."""
import math
import unittest

from geometry.neurcad_developable_detect import (
    is_developable, shape_operator_rank, is_tip_point,
    classify_developability, is_doubly_curved,
    segment_developable, developable_fraction,
    shape_operator_from_grad_hess,
)


def sphere_grad_hess(p, r):
    n = tuple(c / r for c in p)
    H = tuple(tuple(((1.0 if i == j else 0.0) - n[i] * n[j]) / r
                    for j in range(3)) for i in range(3))
    return n, H


CYL = ((1.0, 0.0, 0.0),
       ((0.0, 0.0, 0.0), (0.0, 1.0 / 3.0, 0.0), (0.0, 0.0, 0.0)))
CONE = ((1.0, 0.0, 0.0),
        ((0.0, 0.0, 0.0), (0.0, 0.0, 0.0), (0.0, 0.0, 0.9)))
PLANE = ((0.0, 0.0, 1.0), ((0.0, 0.0, 0.0),) * 3)
SADDLE = ((1.0, 0.0, 0.0),
          ((0.0, 0.0, 0.0), (0.0, 2.0, 0.0), (0.0, 0.0, -2.0)))
# Corner: K = pi/2.
s = math.sqrt(math.pi / 2.0)
CORNER = ((1.0, 0.0, 0.0),
          ((0.0, 0.0, 0.0), (0.0, s, 0.0), (0.0, 0.0, s)))


class IsDevelopableTests(unittest.TestCase):
    def test_cylinder_cone_plane_developable(self):
        self.assertTrue(is_developable(*CYL))
        self.assertTrue(is_developable(*CONE))
        self.assertTrue(is_developable(*PLANE))

    def test_sphere_not_developable(self):
        self.assertFalse(is_developable(*sphere_grad_hess((2.0, 0, 0), 2.0)))

    def test_saddle_not_developable(self):
        self.assertFalse(is_developable(*SADDLE))

    def test_doubly_curved_complement(self):
        self.assertFalse(is_doubly_curved(*CYL))
        self.assertTrue(is_doubly_curved(*sphere_grad_hess((0, 3.0, 0), 3.0)))


class RankTests(unittest.TestCase):
    def test_plane_rank_zero(self):
        self.assertEqual(shape_operator_rank(*PLANE), 0)

    def test_cylinder_rank_one(self):
        self.assertEqual(shape_operator_rank(*CYL), 1)

    def test_cone_rank_one(self):
        self.assertEqual(shape_operator_rank(*CONE), 1)

    def test_sphere_rank_two(self):
        self.assertEqual(shape_operator_rank(*sphere_grad_hess((4.0, 0, 0), 4.0)), 2)

    def test_saddle_rank_two(self):
        self.assertEqual(shape_operator_rank(*SADDLE), 2)

    def test_developable_iff_rank_le_one(self):
        for s_ in (PLANE, CYL, CONE):
            self.assertLessEqual(shape_operator_rank(*s_), 1)
            self.assertTrue(is_developable(*s_))


class TipTests(unittest.TestCase):
    def test_corner_is_tip(self):
        self.assertTrue(is_tip_point(*CORNER))

    def test_cylinder_not_tip(self):
        self.assertFalse(is_tip_point(*CYL))

    def test_sphere_not_tip(self):
        self.assertFalse(is_tip_point(*sphere_grad_hess((1.0, 0, 0), 1.0)))


class ClassifyTests(unittest.TestCase):
    def test_labels(self):
        self.assertEqual(classify_developability(*PLANE), "planar")
        self.assertEqual(classify_developability(*CYL), "developable")
        self.assertEqual(classify_developability(*CONE), "developable")
        self.assertEqual(classify_developability(*sphere_grad_hess((2.0, 0, 0), 2.0)),
                         "synclastic")
        self.assertEqual(classify_developability(*SADDLE), "anticlastic")


class SegmentationTests(unittest.TestCase):
    def test_segment_labels(self):
        labels = segment_developable([PLANE, CYL, SADDLE])
        self.assertEqual(labels, ["planar", "developable", "anticlastic"])

    def test_fraction_all_developable(self):
        self.assertEqual(developable_fraction([PLANE, CYL, CONE]), 1.0)

    def test_fraction_mixed(self):
        samples = [CYL, PLANE, sphere_grad_hess((2.0, 0, 0), 2.0), SADDLE]
        self.assertAlmostEqual(developable_fraction(samples), 0.5, places=12)

    def test_fraction_empty_raises(self):
        with self.assertRaises(ValueError):
            developable_fraction([])


class ShapeOperatorAccessTests(unittest.TestCase):
    def test_cylinder_shape_operator_singular(self):
        S = shape_operator_from_grad_hess(*CYL)
        det = S[0][0] * S[1][1] - S[0][1] * S[1][0]
        self.assertAlmostEqual(det, 0.0, places=12)  # rank<=1 -> singular


if __name__ == "__main__":
    unittest.main()
