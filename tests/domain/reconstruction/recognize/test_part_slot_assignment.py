"""Tests for PartCrafter deterministic part-slot assignment and overlap validity."""

import unittest

from harnesscad.domain.reconstruction.recognize import part_slot_assignment as ps


def cube(cx, cy, cz, s):
    """8 corner points of an axis-aligned cube centred at (cx,cy,cz), half-size s."""
    return [
        (cx + dx * s, cy + dy * s, cz + dz * s)
        for dx in (-1, 1)
        for dy in (-1, 1)
        for dz in (-1, 1)
    ]


class BBoxTest(unittest.TestCase):
    def test_volume(self):
        bb = ps.part_bbox(cube(0, 0, 0, 1))
        self.assertAlmostEqual(bb.volume, 8.0)

    def test_empty_raises(self):
        with self.assertRaises(ValueError):
            ps.part_bbox([])


class OrderTest(unittest.TestCase):
    def test_descending_volume(self):
        big = cube(0, 0, 0, 2)     # volume 64
        small = cube(5, 5, 5, 1)   # volume 8
        order = ps.canonical_part_order([small, big])
        self.assertEqual(order, [1, 0])  # big first

    def test_largest_always_first_regardless_of_input_order(self):
        a = cube(0, 0, 0, 2)     # volume 64
        b = cube(5, 5, 5, 1)     # volume 8
        c = cube(9, 9, 9, 3)     # volume 216 (largest)
        # in [c, a, b] the largest (c) is at index 0
        self.assertEqual(ps.canonical_part_order([c, a, b])[0], 0)
        # in [a, b, c] the largest (c) is at index 2
        self.assertEqual(ps.canonical_part_order([a, b, c])[0], 2)


class AssignTest(unittest.TestCase):
    def test_slots_contiguous(self):
        parts = [cube(0, 0, 0, 2), cube(5, 5, 5, 1)]
        slots = ps.assign_slots(parts, max_num_parts=8)
        self.assertEqual([s for s, _ in slots], [0, 1])
        self.assertEqual(slots[0][1], 0)  # big part in slot 0

    def test_empty_parts_fallback_to_object(self):
        slots = ps.assign_slots([], max_num_parts=8, object_points=cube(0, 0, 0, 1))
        self.assertEqual(slots, [(0, 0)])

    def test_empty_parts_without_object_raises(self):
        with self.assertRaises(ValueError):
            ps.assign_slots([], max_num_parts=8)

    def test_too_many_parts_rejected(self):
        parts = [cube(i, 0, 0, 1) for i in range(5)]
        with self.assertRaises(ValueError):
            ps.assign_slots(parts, max_num_parts=3)


class IoUTest(unittest.TestCase):
    def test_identical_parts_iou_one(self):
        c = cube(0, 0, 0, 1)
        self.assertAlmostEqual(ps.voxel_iou(c, c), 1.0)

    def test_disjoint_parts_iou_zero(self):
        a = cube(0, 0, 0, 1)
        b = cube(50, 50, 50, 1)
        self.assertEqual(ps.voxel_iou(a, b), 0.0)

    def test_summary_and_validity(self):
        a = cube(0, 0, 0, 1)
        b = cube(50, 50, 50, 1)
        s = ps.iou_summary([a, b])
        self.assertEqual(s["iou_max"], 0.0)
        self.assertTrue(ps.is_valid_decomposition([a, b], max_iou_mean=0.1, max_iou_max=0.1))

    def test_overlapping_rejected(self):
        c = cube(0, 0, 0, 1)
        self.assertFalse(ps.is_valid_decomposition([c, c], max_iou_mean=0.5, max_iou_max=0.5))


if __name__ == "__main__":
    unittest.main()
