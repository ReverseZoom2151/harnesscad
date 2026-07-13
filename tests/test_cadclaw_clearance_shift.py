"""Tests for verifiers.cadclaw_clearance_shift.

Deterministic, stdlib-only. Verifies the cheapest-axis clearance-shift
geometry against hand-computed overlaps.
"""
import unittest

from harnesscad.eval.verifiers.cadclaw_clearance_shift import (
    boxes_overlap, suggest_clearance_shift, ClearanceShift,
)


class OverlapTest(unittest.TestCase):

    def test_overlapping_boxes(self):
        a = (0, 0, 0, 10, 10, 10)
        b = (5, 5, 5, 15, 15, 15)
        self.assertTrue(boxes_overlap(a, b))

    def test_disjoint_boxes(self):
        a = (0, 0, 0, 10, 10, 10)
        b = (20, 0, 0, 30, 10, 10)
        self.assertFalse(boxes_overlap(a, b))

    def test_face_contact_not_overlap(self):
        a = (0, 0, 0, 10, 10, 10)
        b = (10, 0, 0, 20, 10, 10)  # touching at x=10
        self.assertFalse(boxes_overlap(a, b))


class ShiftTest(unittest.TestCase):

    def test_simple_x_overlap(self):
        a = (0, 0, 0, 10, 10, 10)
        b = (8, 0, 0, 20, 10, 10)  # 2 mm overlap on x, full on y/z
        s = suggest_clearance_shift(a, b, clearance_mm=1.0)
        self.assertTrue(s.overlaps)
        self.assertEqual(s.axis, "x")
        # push A negative by overlap(2) + clearance(1) = 3
        self.assertAlmostEqual(s.shift_mm, -3.0)

    def test_cheapest_axis_chosen(self):
        a = (0, 0, 0, 10, 10, 10)
        b = (5, 9, 5, 15, 20, 15)  # overlaps x=5, y=1, z=5 -> y cheapest
        s = suggest_clearance_shift(a, b, clearance_mm=1.0)
        self.assertEqual(s.axis, "y")
        self.assertAlmostEqual(s.shift_mm, -2.0)  # overlap 1 + clearance 1

    def test_shift_actually_clears(self):
        a = (0, 0, 0, 10, 10, 10)
        b = (5, 9, 5, 15, 20, 15)
        s = suggest_clearance_shift(a, b, clearance_mm=1.0)
        dx, dy, dz = s.vector
        moved = (a[0] + dx, a[1] + dy, a[2] + dz,
                 a[3] + dx, a[4] + dy, a[5] + dz)
        self.assertFalse(boxes_overlap(moved, b))

    def test_vector_matches_axis(self):
        a = (0, 0, 0, 10, 10, 10)
        b = (8, 0, 0, 20, 10, 10)
        s = suggest_clearance_shift(a, b)
        self.assertEqual(s.vector, (s.shift_mm, 0.0, 0.0))

    def test_no_overlap_zero_shift(self):
        a = (0, 0, 0, 10, 10, 10)
        b = (50, 0, 0, 60, 10, 10)
        s = suggest_clearance_shift(a, b)
        self.assertFalse(s.overlaps)
        self.assertEqual(s.shift_mm, 0.0)

    def test_zero_clearance(self):
        a = (0, 0, 0, 10, 10, 10)
        b = (8, 0, 0, 20, 10, 10)
        s = suggest_clearance_shift(a, b, clearance_mm=0.0)
        self.assertAlmostEqual(s.shift_mm, -2.0)  # just clear the 2 mm overlap

    def test_containment_tie_pushes_away_from_center(self):
        a = (0, 0, 0, 10, 10, 10)
        b = (2, 2, 2, 8, 8, 8)  # B fully inside A, symmetric
        s = suggest_clearance_shift(a, b, clearance_mm=1.0)
        # equal-magnitude moves; center_a >= center_b -> positive
        self.assertGreater(s.shift_mm, 0.0)
        dx, dy, dz = s.vector
        moved = (a[0] + dx, a[1] + dy, a[2] + dz,
                 a[3] + dx, a[4] + dy, a[5] + dz)
        self.assertFalse(boxes_overlap(moved, b))

    def test_deterministic(self):
        a = (0, 0, 0, 10, 10, 10)
        b = (5, 9, 5, 15, 20, 15)
        s1 = suggest_clearance_shift(a, b)
        s2 = suggest_clearance_shift(a, b)
        self.assertEqual(s1, s2)

    def test_negative_clearance_raises(self):
        with self.assertRaises(ValueError):
            suggest_clearance_shift((0, 0, 0, 1, 1, 1),
                                    (0, 0, 0, 1, 1, 1), clearance_mm=-1.0)

    def test_overlap_dims_reported(self):
        a = (0, 0, 0, 10, 10, 10)
        b = (5, 9, 5, 15, 20, 15)
        s = suggest_clearance_shift(a, b)
        self.assertAlmostEqual(s.overlap_dims[0], 5.0)
        self.assertAlmostEqual(s.overlap_dims[1], 1.0)
        self.assertAlmostEqual(s.overlap_dims[2], 5.0)


if __name__ == "__main__":
    unittest.main()
