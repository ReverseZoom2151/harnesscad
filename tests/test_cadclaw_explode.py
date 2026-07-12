"""Tests for geometry.cadclaw_explode.

Deterministic, stdlib-only. Bounding-box centroid, removal-axis, radial
offset and outside-in ordering checked against hand computations.
"""
import math
import unittest

from geometry.cadclaw_explode import (
    bbox_center, assembly_centroid, removal_axis, axis_offset,
    radial_offset, removal_order, RemovalStep,
)


class BasicGeometryTest(unittest.TestCase):

    def test_bbox_center(self):
        self.assertEqual(bbox_center((0, 0, 0, 10, 20, 30)), (5.0, 10.0, 15.0))

    def test_centroid(self):
        c = assembly_centroid([(0, 0, 0), (10, 0, 0), (5, 3, 0)])
        self.assertAlmostEqual(c[0], 5.0)
        self.assertAlmostEqual(c[1], 1.0)
        self.assertAlmostEqual(c[2], 0.0)

    def test_empty_centroid_raises(self):
        with self.assertRaises(ValueError):
            assembly_centroid([])


class RemovalAxisTest(unittest.TestCase):

    def test_dominant_positive_x(self):
        axis, d = removal_axis((10, 1, 0), (0, 0, 0))
        self.assertEqual(axis, "X")
        self.assertEqual(d, 1.0)

    def test_dominant_negative_z(self):
        axis, d = removal_axis((0, 1, -20), (0, 0, 0))
        self.assertEqual(axis, "Z")
        self.assertEqual(d, -1.0)

    def test_tie_prefers_x(self):
        axis, _ = removal_axis((5, 5, 0), (0, 0, 0))
        self.assertEqual(axis, "X")


class AxisOffsetTest(unittest.TestCase):

    def test_positive_y(self):
        self.assertEqual(axis_offset("Y", 1.0, 50.0), (0.0, 50.0, 0.0))

    def test_negative_z(self):
        self.assertEqual(axis_offset("Z", -1.0, 30.0), (0.0, 0.0, -30.0))


class RadialOffsetTest(unittest.TestCase):

    def test_proportional_expansion(self):
        # part at (10,0,0), centroid origin, expansion 0.5 -> +5 on x
        off = radial_offset((10, 0, 0), (0, 0, 0), expansion=0.5)
        self.assertAlmostEqual(off[0], 5.0)
        self.assertAlmostEqual(off[1], 0.0)

    def test_farther_parts_move_more(self):
        near = radial_offset((5, 0, 0), (0, 0, 0), expansion=0.3)
        far = radial_offset((20, 0, 0), (0, 0, 0), expansion=0.3)
        self.assertGreater(abs(far[0]), abs(near[0]))

    def test_centroid_part_nudged_by_min_offset(self):
        off = radial_offset((0, 0, 0), (0, 0, 0), expansion=0.3, min_offset=2.0)
        self.assertAlmostEqual(math.dist(off, (0, 0, 0)), 2.0)

    def test_centroid_part_zero_without_min_offset(self):
        off = radial_offset((0, 0, 0), (0, 0, 0), expansion=0.3)
        self.assertEqual(off, (0.0, 0.0, 0.0))

    def test_negative_expansion_raises(self):
        with self.assertRaises(ValueError):
            radial_offset((1, 0, 0), (0, 0, 0), expansion=-0.1)


class RemovalOrderTest(unittest.TestCase):

    def test_outermost_first_within_tier(self):
        # three parts, same label, increasing distance from centroid
        # centers at x = 0, 10, 40 -> centroid x = 16.67; part 2 is outermost
        bboxes = [
            (-1, -1, -1, 1, 1, 1),      # center x=0
            (9, -1, -1, 11, 1, 1),      # center x=10
            (39, -1, -1, 41, 1, 1),     # center x=40, outermost
        ]
        labels = ["p", "p", "p"]
        order = removal_order(bboxes, labels)
        # outermost (index 2) removed first
        self.assertEqual(order[0].part_index, 2)
        self.assertGreaterEqual(order[0].distance_from_centroid,
                                order[1].distance_from_centroid)

    def test_priority_dominates_distance(self):
        bboxes = [
            (100, 0, 0, 102, 2, 2),   # far, but low-priority label
            (1, 0, 0, 3, 2, 2),       # near, high-priority (belt) removed first
        ]
        labels = ["cbeam", "belt"]
        order = removal_order(bboxes, labels,
                              priority={"belt": 1, "cbeam": 10})
        self.assertEqual(order[0].label, "belt")
        self.assertEqual(order[1].label, "cbeam")

    def test_reverse_is_assembly_order(self):
        bboxes = [(0, 0, 0, 2, 2, 2), (10, 0, 0, 12, 2, 2)]
        labels = ["a", "b"]
        order = removal_order(bboxes, labels)
        assembly = list(reversed(order))
        self.assertEqual([s.part_index for s in assembly],
                         [s.part_index for s in order][::-1])

    def test_deterministic(self):
        bboxes = [(0, 0, 0, 2, 2, 2), (5, 5, 5, 7, 7, 7), (9, 0, 0, 11, 2, 2)]
        labels = ["a", "b", "c"]
        o1 = removal_order(bboxes, labels)
        o2 = removal_order(bboxes, labels)
        self.assertEqual([s.part_index for s in o1],
                         [s.part_index for s in o2])

    def test_step_offset_at(self):
        bboxes = [(10, 0, 0, 12, 2, 2), (-10, 0, 0, -8, 2, 2)]
        labels = ["a", "b"]
        order = removal_order(bboxes, labels)
        step = order[0]
        off = step.offset_at(100.0)
        self.assertEqual(off, step.offset_at(100.0))
        # nonzero translation along the removal axis
        self.assertAlmostEqual(math.dist(off, (0, 0, 0)), 100.0)

    def test_length_mismatch_raises(self):
        with self.assertRaises(ValueError):
            removal_order([(0, 0, 0, 1, 1, 1)], ["a", "b"])

    def test_empty(self):
        self.assertEqual(removal_order([], []), [])


if __name__ == "__main__":
    unittest.main()
