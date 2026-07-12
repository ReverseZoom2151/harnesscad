"""Tests for DeepCAD's arc macro encoding / decoding."""

import math
import unittest

from reconstruction import deepcad2_arc_macro as am


def _close(a, b, tol=1e-9):
    return abs(a - b) <= tol


class TestAngleFromVectorToX(unittest.TestCase):
    def test_axes(self):
        self.assertAlmostEqual(am.angle_from_vector_to_x((1.0, 0.0)), 0.0)
        self.assertAlmostEqual(am.angle_from_vector_to_x((0.0, 1.0)), math.pi / 2)
        self.assertAlmostEqual(am.angle_from_vector_to_x((-1.0, 0.0)), math.pi)
        self.assertAlmostEqual(am.angle_from_vector_to_x((0.0, -1.0)), 3 * math.pi / 2)


class TestAngleQuadrants(unittest.TestCase):
    def test_matches_atan2_mod_2pi(self):
        for k in range(24):
            a = k * (2 * math.pi / 24)
            v = (math.cos(a), math.sin(a))
            got = am.angle_from_vector_to_x(v)
            self.assertAlmostEqual(got, a % (2 * math.pi), places=9)


class TestClockSign(unittest.TestCase):
    def test_ccw_bulge_is_one(self):
        # mid above the chord (0,0)->(2,0): cross(s->m, s->e) = 1*0 - 1*2 < 0 -> 0
        self.assertEqual(am.clock_sign((0, 0), (1, 1), (2, 0)), 0)
        # mid below the chord -> cross > 0 -> 1
        self.assertEqual(am.clock_sign((0, 0), (1, -1), (2, 0)), 1)

    def test_collinear_is_one(self):
        self.assertEqual(am.clock_sign((0, 0), (1, 0), (2, 0)), 1)


class TestArcFromMacro(unittest.TestCase):
    def test_semicircle_flag_zero(self):
        arc = am.arc_from_macro((0.0, 0.0), (2.0, 0.0), math.pi, 0)
        self.assertAlmostEqual(arc.radius, 1.0)
        self.assertTrue(_close(arc.center[0], 1.0) and _close(arc.center[1], 0.0, 1e-9))
        # the mid point bulges upward for flag == 0 on this chord
        self.assertAlmostEqual(arc.mid_point[1], 1.0)

    def test_semicircle_flag_one(self):
        arc = am.arc_from_macro((0.0, 0.0), (2.0, 0.0), math.pi, 1)
        self.assertAlmostEqual(arc.radius, 1.0)
        self.assertAlmostEqual(arc.mid_point[1], -1.0)

    def test_endpoints_lie_on_the_circle(self):
        arc = am.arc_from_macro((0.0, 0.0), (1.0, 2.0), 1.9, 1)
        for p in (arc.start_point, arc.end_point, arc.mid_point):
            d = math.hypot(p[0] - arc.center[0], p[1] - arc.center[1])
            self.assertAlmostEqual(d, abs(arc.radius), places=9)

    def test_quarter_arc_radius_formula(self):
        # chord of a quarter circle of radius 1 is sqrt(2); the flag picks which of
        # the two candidate centres (0,0) / (1,1) the arc bulges around.
        arc0 = am.arc_from_macro((1.0, 0.0), (0.0, 1.0), math.pi / 2, 0)
        self.assertAlmostEqual(arc0.radius, 1.0, places=9)
        self.assertAlmostEqual(arc0.center[0], 1.0, places=9)
        self.assertAlmostEqual(arc0.center[1], 1.0, places=9)
        arc1 = am.arc_from_macro((1.0, 0.0), (0.0, 1.0), math.pi / 2, 1)
        self.assertAlmostEqual(arc1.radius, 1.0, places=9)
        self.assertAlmostEqual(arc1.center[0], 0.0, places=9)
        self.assertAlmostEqual(arc1.center[1], 0.0, places=9)

    def test_degenerate_coincident_endpoints(self):
        with self.assertRaises(ValueError):
            am.arc_from_macro((1.0, 1.0), (1.0, 1.0), math.pi, 1)

    def test_degenerate_sweep(self):
        with self.assertRaises(ValueError):
            am.arc_from_macro((0.0, 0.0), (2.0, 0.0), 0.0, 1)


class TestRoundTrip(unittest.TestCase):
    def test_decode_then_encode_recovers_macro(self):
        for sweep in (0.4, 1.0, math.pi / 2, math.pi, 4.0):
            for flag in (0, 1):
                arc = am.arc_from_macro((0.0, 0.0), (2.0, 1.0), sweep, flag)
                end, back_sweep, back_flag = am.arc_to_macro(
                    arc.start_point, arc.mid_point, arc.end_point, arc.center)
                self.assertEqual(end, (2.0, 1.0))
                # the reference's eps=1e-8 guard in the angle computation costs a
                # few digits of precision -- 1e-5 is well inside the 256-level grid.
                self.assertAlmostEqual(back_sweep, sweep, places=5)
                self.assertEqual(back_flag, flag)


class TestArcBbox(unittest.TestCase):
    def test_semicircle_includes_bulge(self):
        arc = am.arc_from_macro((0.0, 0.0), (2.0, 0.0), math.pi, 0)
        x0, y0, x1, y1 = am.arc_bbox(arc)
        self.assertAlmostEqual(x0, 0.0, places=6)
        self.assertAlmostEqual(x1, 2.0, places=6)
        self.assertAlmostEqual(y0, 0.0, places=6)
        self.assertAlmostEqual(y1, 1.0, places=6)  # the bulge, not the chord

    def test_small_arc_is_endpoint_bounded(self):
        arc = am.arc_from_macro((1.0, 0.0), (0.0, 1.0), math.pi / 2, 0)
        x0, y0, x1, y1 = am.arc_bbox(arc)
        # quarter arc in quadrant 1: no axis extreme point is interior
        self.assertAlmostEqual(x0, 0.0, places=6)
        self.assertAlmostEqual(y0, 0.0, places=6)
        self.assertAlmostEqual(x1, 1.0, places=6)
        self.assertAlmostEqual(y1, 1.0, places=6)

    def test_bbox_contains_sampled_points(self):
        arc = am.arc_from_macro((0.0, 0.0), (1.0, 2.0), 3.0, 1)
        x0, y0, x1, y1 = am.arc_bbox(arc)
        for px, py in am.sample_arc_points(arc, 64):
            self.assertGreaterEqual(px, x0 - 1e-6)
            self.assertLessEqual(px, x1 + 1e-6)
            self.assertGreaterEqual(py, y0 - 1e-6)
            self.assertLessEqual(py, y1 + 1e-6)


class TestSampling(unittest.TestCase):
    def test_arc_samples_hit_both_endpoints(self):
        # sampling always runs counter-clockwise (reference behaviour), so the two
        # extreme samples are the arc's endpoints -- in ccw, not traversal, order.
        arc = am.arc_from_macro((0.0, 0.0), (2.0, 0.0), math.pi, 0)
        pts = am.sample_arc_points(arc, 5)
        self.assertEqual(len(pts), 5)
        ends = sorted([pts[0], pts[-1]])
        self.assertTrue(_close(ends[0][0], 0.0, 1e-6) and _close(ends[0][1], 0.0, 1e-6))
        self.assertTrue(_close(ends[1][0], 2.0, 1e-6) and _close(ends[1][1], 0.0, 1e-6))
        # the mid sample is the bulge apex
        self.assertTrue(_close(pts[2][1], 1.0, 1e-6))

    def test_arc_samples_are_on_the_circle(self):
        arc = am.arc_from_macro((0.0, 0.0), (1.0, 2.0), 2.2, 0)
        for p in am.sample_arc_points(arc, 16):
            d = math.hypot(p[0] - arc.center[0], p[1] - arc.center[1])
            self.assertAlmostEqual(d, abs(arc.radius), places=6)

    def test_circle_sampling_excludes_endpoint(self):
        pts = am.sample_circle_points((0.0, 0.0), 2.0, 4)
        self.assertEqual(len(pts), 4)
        self.assertAlmostEqual(pts[0][0], 2.0)
        self.assertAlmostEqual(pts[1][1], 2.0)
        for p in pts:
            self.assertAlmostEqual(math.hypot(*p), 2.0)

    def test_line_sampling_is_uniform(self):
        pts = am.sample_line_points((0.0, 0.0), (4.0, 0.0), 5)
        self.assertEqual([p[0] for p in pts], [0.0, 1.0, 2.0, 3.0, 4.0])

    def test_too_few_samples_raise(self):
        arc = am.arc_from_macro((0.0, 0.0), (2.0, 0.0), math.pi, 0)
        with self.assertRaises(ValueError):
            am.sample_arc_points(arc, 1)
        with self.assertRaises(ValueError):
            am.sample_line_points((0, 0), (1, 1), 1)


if __name__ == "__main__":
    unittest.main()
