"""Tests for assembly.placement_gates (CADCLAW adjacency + floating)."""

import math
import unittest

from harnesscad.domain.assembly.placement_gates import (
    AdjacencyRule,
    Part,
    adjacency_check,
    bbox_distance,
    center_distance,
    floating_check,
)


def _box(label, x, y, z, size=1.0):
    return Part(label, (x, y, z, x + size, y + size, z + size))


class TestDistance(unittest.TestCase):
    def test_overlapping_bboxes_zero(self):
        a = (0, 0, 0, 2, 2, 2)
        b = (1, 1, 1, 3, 3, 3)
        self.assertEqual(bbox_distance(a, b), 0.0)

    def test_gap(self):
        a = (0, 0, 0, 1, 1, 1)
        b = (4, 0, 0, 5, 1, 1)  # 3mm gap in x
        self.assertAlmostEqual(bbox_distance(a, b), 3.0)

    def test_center_distance(self):
        self.assertAlmostEqual(center_distance((0, 0, 0), (0, 3, 4)), 5.0)


class TestAdjacency(unittest.TestCase):
    def test_pass_when_target_near(self):
        parts = [_box("motor", 0, 0, 0), _box("bracket", 2, 0, 0)]
        r = adjacency_check(parts, [AdjacencyRule("motor", "bracket", max_distance=50)])
        self.assertTrue(r.passed)

    def test_fail_when_target_far(self):
        parts = [_box("motor", 0, 0, 0), _box("bracket", 100, 0, 0)]
        r = adjacency_check(parts, [AdjacencyRule("motor", "bracket", max_distance=50)])
        self.assertFalse(r.passed)
        self.assertEqual(len(r.violations), 1)
        self.assertEqual(r.violations[0].source_label, "motor")

    def test_missing_target_infinite(self):
        parts = [_box("motor", 0, 0, 0)]
        r = adjacency_check(parts, [AdjacencyRule("motor", "bracket")])
        self.assertFalse(r.passed)
        self.assertEqual(r.violations[0].nearest_distance, math.inf)

    def test_source_filter(self):
        parts = [_box("motor", 0, 0, 0), _box("motor", 100, 0, 0), _box("bracket", 1, 0, 0)]
        # only check motors at x < 50
        rule = AdjacencyRule("motor", "bracket", max_distance=10,
                             source_filter=lambda p: p.center[0] < 50)
        r = adjacency_check(parts, [rule])
        self.assertTrue(r.passed)


class TestFloating(unittest.TestCase):
    def test_disabled_without_structural(self):
        parts = [_box("idler", 0, 0, 0)]
        r = floating_check(parts, structural_labels=set())
        self.assertTrue(r.passed)
        self.assertEqual(r.checked, 0)

    def test_attached_part_passes(self):
        parts = [_box("cbeam", 0, 0, 0, size=10), _box("idler", 10, 0, 0)]
        r = floating_check(parts, structural_labels={"cbeam"}, max_gap_mm=5.0)
        self.assertTrue(r.passed)
        self.assertEqual(r.checked, 1)

    def test_floating_part_flagged(self):
        parts = [_box("cbeam", 0, 0, 0, size=10), _box("idler", 100, 0, 0)]
        r = floating_check(parts, structural_labels={"cbeam"}, max_gap_mm=5.0)
        self.assertFalse(r.passed)
        self.assertEqual(len(r.floating), 1)
        self.assertEqual(r.floating[0].label, "idler")
        self.assertEqual(r.floating[0].nearest_label, "cbeam")

    def test_exempt_label_skipped(self):
        parts = [_box("cbeam", 0, 0, 0, size=10), _box("belt", 100, 0, 0)]
        r = floating_check(parts, structural_labels={"cbeam"})
        self.assertTrue(r.passed)  # belt exempt by default

    def test_structural_parts_are_anchors_not_candidates(self):
        parts = [_box("cbeam", 0, 0, 0, size=10), _box("cbeam", 100, 0, 0, size=10)]
        r = floating_check(parts, structural_labels={"cbeam"})
        self.assertEqual(r.checked, 0)
        self.assertTrue(r.passed)

    def test_no_structural_present_disabled(self):
        parts = [_box("idler", 0, 0, 0)]
        r = floating_check(parts, structural_labels={"cbeam"})
        self.assertEqual(r.checked, 0)
        self.assertTrue(r.passed)


if __name__ == "__main__":
    unittest.main()
