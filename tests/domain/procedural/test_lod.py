"""Tests for procedural.shapegramm_lod (LOD variable + rule selection)."""

import unittest

from harnesscad.domain.procedural.lod import (
    projected_size, lod_level, select_lod_rule, resolve_lod,
)


class ProjectedSizeTest(unittest.TestCase):
    def test_closer_is_larger(self):
        box = ((0, 0, 0), (1, 1, 1))
        near = projected_size(box[0], box[1], (0.5, 0.5, 20), 500.0)
        far = projected_size(box[0], box[1], (0.5, 0.5, 200), 500.0)
        self.assertGreater(near, far)

    def test_camera_inside_is_infinite(self):
        box = ((0, 0, 0), (10, 10, 10))
        self.assertEqual(projected_size(box[0], box[1], (5, 5, 5), 500.0), float("inf"))

    def test_bigger_box_projects_larger(self):
        small = projected_size((0, 0, 0), (1, 1, 1), (0, 0, 100), 500.0)
        big = projected_size((0, 0, 0), (5, 5, 5), (0, 0, 100), 500.0)
        self.assertGreater(big, small)

    def test_nonpositive_focal_rejected(self):
        with self.assertRaises(ValueError):
            projected_size((0, 0, 0), (1, 1, 1), (0, 0, 10), 0.0)


class LodLevelTest(unittest.TestCase):
    def test_mapping(self):
        thresholds = [100, 20]
        self.assertEqual(lod_level(150, thresholds), 0)
        self.assertEqual(lod_level(100, thresholds), 0)
        self.assertEqual(lod_level(50, thresholds), 1)
        self.assertEqual(lod_level(20, thresholds), 1)
        self.assertEqual(lod_level(5, thresholds), 2)

    def test_infinite_is_lod0(self):
        self.assertEqual(lod_level(float("inf"), [100, 20]), 0)


class SelectRuleTest(unittest.TestCase):
    def setUp(self):
        # object [lod = 0] -> full ; [lod = 1] -> coarse ; [lod >= 1] -> coarsest
        self.conditions = [
            ("=", 0, "full"),
            ("=", 1, "coarse"),
            (">=", 1, "coarsest"),
        ]

    def test_lod0_full(self):
        self.assertEqual(select_lod_rule(self.conditions, 0), "full")

    def test_lod1_takes_first_match(self):
        # both [=1] and [>=1] match; first (coarse) wins
        self.assertEqual(select_lod_rule(self.conditions, 1), "coarse")

    def test_lod2_coarsest(self):
        self.assertEqual(select_lod_rule(self.conditions, 2), "coarsest")

    def test_no_match_returns_none(self):
        self.assertIsNone(select_lod_rule([("=", 5, "x")], 0))

    def test_unknown_operator(self):
        with self.assertRaises(ValueError):
            select_lod_rule([("~", 0, "x")], 0)


class ResolvePipelineTest(unittest.TestCase):
    def test_near_object_full_detail(self):
        conditions = [("=", 0, "full"), (">=", 1, "coarse")]
        box = ((0, 0, 0), (2, 2, 2))
        lod, payload = resolve_lod(box[0], box[1], (1, 1, 8), 500.0, [100, 20], conditions)
        self.assertEqual(lod, 0)
        self.assertEqual(payload, "full")

    def test_far_object_coarse(self):
        conditions = [("=", 0, "full"), (">=", 1, "coarse")]
        box = ((0, 0, 0), (1, 1, 1))
        lod, payload = resolve_lod(box[0], box[1], (0, 0, 5000), 500.0, [100, 20], conditions)
        self.assertGreaterEqual(lod, 1)
        self.assertEqual(payload, "coarse")

    def test_deterministic(self):
        conditions = [("=", 0, "full"), (">=", 1, "coarse")]
        box = ((0, 0, 0), (3, 3, 3))
        a = resolve_lod(box[0], box[1], (1, 1, 50), 400.0, [100, 20], conditions)
        b = resolve_lod(box[0], box[1], (1, 1, 50), 400.0, [100, 20], conditions)
        self.assertEqual(a, b)


if __name__ == "__main__":
    unittest.main()
