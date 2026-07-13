"""Tests for geometry.cqplug_teardrop_profile."""

import math
import unittest

from harnesscad.domain.geometry.features.cqplug_teardrop_profile import (
    TeardropError,
    apex_height,
    bridge_span,
    clip_bounds,
    headroom_saved,
    is_self_supporting,
    junction_point,
    max_overhang_of_profile,
    polygon_area,
    teardrop_profile,
)


class TestApexAndJunction(unittest.TestCase):
    def test_apex_is_r_root_two_at_45(self):
        self.assertAlmostEqual(apex_height(4.0), 4.0 * math.sqrt(2.0), places=9)

    def test_junction_on_the_circle(self):
        x, y = junction_point(3.0)
        self.assertAlmostEqual(math.hypot(x, y), 3.0, places=9)
        self.assertAlmostEqual(x, y, places=9)  # 45 degrees

    def test_steeper_limit_lowers_apex(self):
        self.assertLess(apex_height(5.0, 30.0), apex_height(5.0, 45.0))

    def test_clip_bounds(self):
        lo, hi = clip_bounds(2.0)
        self.assertAlmostEqual(lo, -2.0, places=9)
        self.assertAlmostEqual(hi, 2.0 * math.sqrt(2.0), places=9)

    def test_bad_params(self):
        with self.assertRaises(TeardropError):
            apex_height(0.0)
        with self.assertRaises(TeardropError):
            apex_height(1.0, 90.0)


class TestFullTeardrop(unittest.TestCase):
    def setUp(self):
        self.p = teardrop_profile(4.0, segments=64)

    def test_apex_is_topmost_point(self):
        top = max(self.p.points, key=lambda q: q[1])
        self.assertAlmostEqual(top[0], 0.0, places=9)
        self.assertAlmostEqual(top[1], apex_height(4.0), places=9)

    def test_height_and_width(self):
        self.assertAlmostEqual(self.p.height, 4.0 + apex_height(4.0), places=6)
        # The bore's widest point falls between samples, so the polygon width
        # converges to the diameter from below.
        self.assertLessEqual(self.p.width, 8.0)
        self.assertAlmostEqual(self.p.width, 8.0, delta=0.01)

    def test_area_exceeds_circle_area(self):
        self.assertGreater(polygon_area(self.p.points), math.pi * 16.0 * 0.9)

    def test_arc_endpoints_on_circle(self):
        for pt in self.p.arc:
            self.assertAlmostEqual(math.hypot(*pt), 4.0, places=9)

    def test_two_flanks(self):
        self.assertEqual(len(self.p.flanks), 2)

    def test_self_supporting(self):
        self.assertTrue(is_self_supporting(self.p))

    def test_flanks_are_at_the_overhang_angle(self):
        for a, b in self.p.flanks:
            ang = math.degrees(math.atan2(abs(b[0] - a[0]), abs(b[1] - a[1])))
            self.assertAlmostEqual(ang, 45.0, places=6)


class TestPlainBoreFails(unittest.TestCase):
    def test_circle_roof_is_horizontal(self):
        circle = [(3.0 * math.cos(2 * math.pi * i / 64),
                   3.0 * math.sin(2 * math.pi * i / 64)) for i in range(64)]
        self.assertGreater(max_overhang_of_profile(circle), 80.0)

    def test_teardrop_roof_within_limit(self):
        p = teardrop_profile(3.0, segments=64)
        self.assertLessEqual(max_overhang_of_profile(p.points), 45.0 + 1e-6)


class TestTruncatedTeardrop(unittest.TestCase):
    def test_clip_above_junction_gives_flat_top(self):
        r = 4.0
        p = teardrop_profile(r, clip=r * 1.2, segments=48)
        top = max(q[1] for q in p.points)
        self.assertAlmostEqual(top, r * 1.2, places=9)
        flat = [q for q in p.points if abs(q[1] - r * 1.2) < 1e-9]
        self.assertEqual(len(flat), 2)
        self.assertTrue(is_self_supporting(p))

    def test_clip_zero_is_half_circle(self):
        p = teardrop_profile(4.0, clip=0.0, segments=64)
        self.assertAlmostEqual(max(q[1] for q in p.points), 0.0, places=9)
        self.assertAlmostEqual(polygon_area(p.points),
                               0.5 * math.pi * 16.0, places=1)

    def test_bridge_span_shrinks_as_clip_rises(self):
        r = 4.0
        wide = bridge_span(r, 0.0)
        narrow = bridge_span(r, r * 1.3)
        self.assertAlmostEqual(wide, 2.0 * r, places=9)
        self.assertLess(narrow, wide)

    def test_bridge_span_at_junction_is_full_junction_width(self):
        r = 4.0
        _, yj = junction_point(r)
        self.assertAlmostEqual(bridge_span(r, yj - 1e-9),
                               2.0 * junction_point(r)[0], places=6)

    def test_headroom_saved(self):
        self.assertAlmostEqual(headroom_saved(4.0, 4.0),
                               apex_height(4.0) - 4.0, places=9)

    def test_clip_out_of_range(self):
        with self.assertRaises(TeardropError):
            teardrop_profile(4.0, clip=apex_height(4.0))
        with self.assertRaises(TeardropError):
            teardrop_profile(4.0, clip=-4.0)


class TestRotation(unittest.TestCase):
    def test_rotate_90_moves_apex_to_minus_x(self):
        p = teardrop_profile(2.0, rotate=90.0, segments=32)
        apex = max(p.points, key=lambda q: math.hypot(*q))
        self.assertAlmostEqual(apex[0], -apex_height(2.0), places=6)
        self.assertAlmostEqual(apex[1], 0.0, places=6)

    def test_rotation_preserves_area(self):
        a = polygon_area(teardrop_profile(2.0, segments=32).points)
        b = polygon_area(teardrop_profile(2.0, rotate=37.0, segments=32).points)
        self.assertAlmostEqual(a, b, places=9)


if __name__ == "__main__":
    unittest.main()
