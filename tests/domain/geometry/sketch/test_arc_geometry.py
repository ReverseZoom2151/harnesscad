"""Tests for geometry.gencad2_arc_vector."""

import math
import unittest

from harnesscad.domain.geometry.sketch.arc_geometry import (
    ARGS_DIM,
    TWO_PI,
    angle_from_vector_to_x,
    arc_angles_counterclockwise,
    arc_bbox,
    arc_clock_sign,
    arc_from_vector,
    arc_mid_point,
    arc_sweep_angle,
    circle_bbox,
    circle_end_point,
    circle_start_point,
    dequantize_sweep,
    line_bbox,
    quantize_sweep,
    sample_arc_points,
    sample_circle_points,
    sample_line_points,
)


class TestAngleFromVectorToX(unittest.TestCase):
    def test_axes(self):
        self.assertAlmostEqual(angle_from_vector_to_x((1.0, 0.0)), 0.0)
        self.assertAlmostEqual(angle_from_vector_to_x((0.0, 1.0)), math.pi / 2)
        self.assertAlmostEqual(angle_from_vector_to_x((-1.0, 0.0)), math.pi)
        self.assertAlmostEqual(angle_from_vector_to_x((0.0, -1.0)), math.pi / 2 * 3)

    def test_all_quadrants_match_atan2(self):
        for deg in range(0, 360, 7):
            a = math.radians(deg)
            v = (math.cos(a), math.sin(a))
            got = angle_from_vector_to_x(v)
            self.assertGreaterEqual(got, 0.0)
            self.assertLess(got, TWO_PI + 1e-9)
            self.assertAlmostEqual(got, a % TWO_PI, places=6)

    def test_clamped_against_drift(self):
        # slightly out-of-range component must not raise a domain error
        self.assertAlmostEqual(angle_from_vector_to_x((0.0, 1.0000000001)),
                               math.pi / 2, places=6)


class TestSweepQuantization(unittest.TestCase):
    def test_dequantize(self):
        self.assertAlmostEqual(dequantize_sweep(0), 0.0)
        self.assertAlmostEqual(dequantize_sweep(64), math.pi / 2)
        self.assertAlmostEqual(dequantize_sweep(128), math.pi)

    def test_round_trip(self):
        for q in (1, 13, 64, 128, 200, 255):
            self.assertEqual(quantize_sweep(dequantize_sweep(q)), q)

    def test_floor_of_one(self):
        self.assertEqual(quantize_sweep(0.0), 1)

    def test_clipped_to_range(self):
        self.assertEqual(quantize_sweep(TWO_PI), ARGS_DIM - 1)


class TestArcFromVector(unittest.TestCase):
    def test_quarter_circle_ccw(self):
        # unit circle at origin, (1, 0) -> (0, 1), counter-clockwise (sign 1)
        arc = arc_from_vector((1.0, 0.0), (0.0, 1.0), 64, 1)
        self.assertIsNotNone(arc)
        self.assertAlmostEqual(arc.radius, 1.0, places=9)
        self.assertAlmostEqual(arc.center[0], 0.0, places=9)
        self.assertAlmostEqual(arc.center[1], 0.0, places=9)
        self.assertAlmostEqual(arc.mid_point[0], math.sqrt(0.5), places=9)
        self.assertAlmostEqual(arc.mid_point[1], math.sqrt(0.5), places=9)
        self.assertAlmostEqual(arc.end_angle, math.pi / 2, places=9)

    def test_clock_sign_round_trip(self):
        arc = arc_from_vector((1.0, 0.0), (0.0, 1.0), 64, 1)
        self.assertEqual(
            arc_clock_sign(arc.start_point, arc.mid_point, arc.end_point), 1)

    def test_sign_zero_mirrors_center(self):
        ccw = arc_from_vector((1.0, 0.0), (0.0, 1.0), 64, 1)
        cw = arc_from_vector((1.0, 0.0), (0.0, 1.0), 64, 0)
        # the two centres are reflections across the chord midpoint (0.5, 0.5)
        self.assertAlmostEqual(cw.center[0], 1.0, places=9)
        self.assertAlmostEqual(cw.center[1], 1.0, places=9)
        self.assertAlmostEqual(cw.radius, ccw.radius, places=9)
        self.assertEqual(
            arc_clock_sign(cw.start_point, cw.mid_point, cw.end_point), 0)

    def test_ref_vec_is_unit_and_hits_start_when_ccw(self):
        arc = arc_from_vector((1.0, 0.0), (0.0, 1.0), 64, 1)
        self.assertAlmostEqual(math.hypot(*arc.ref_vec), 1.0, places=9)
        self.assertAlmostEqual(arc.ref_vec[0], 1.0, places=9)
        self.assertAlmostEqual(arc.ref_vec[1], 0.0, places=9)

    def test_endpoints_lie_on_reconstructed_circle(self):
        for q in (16, 64, 100, 128, 180):
            arc = arc_from_vector((2.0, 3.0), (5.0, 1.0), q, 1)
            for p in (arc.start_point, arc.mid_point, arc.end_point):
                d = math.hypot(p[0] - arc.center[0], p[1] - arc.center[1])
                self.assertAlmostEqual(d, abs(arc.radius), places=6)

    def test_degenerate_arc_returns_none(self):
        self.assertIsNone(arc_from_vector((1.0, 1.0), (1.0, 1.0), 64, 1))

    def test_zero_sweep_returns_none(self):
        self.assertIsNone(arc_from_vector((0.0, 0.0), (1.0, 0.0), 0, 1))

    def test_radian_mode(self):
        arc = arc_from_vector((1.0, 0.0), (0.0, 1.0), math.pi / 2, 1,
                              is_numerical=False)
        self.assertAlmostEqual(arc.radius, 1.0, places=9)

    def test_mid_point_helper_matches(self):
        arc = arc_from_vector((1.0, 0.0), (0.0, 1.0), 64, 1)
        mid = arc_mid_point(arc.center, arc.radius, arc.ref_vec,
                            arc.start_angle, arc.end_angle)
        self.assertAlmostEqual(mid[0], arc.mid_point[0], places=9)
        self.assertAlmostEqual(mid[1], arc.mid_point[1], places=9)


class TestArcAngles(unittest.TestCase):
    def test_quarter_arc_angles(self):
        s, m, e = (1.0, 0.0), (math.sqrt(0.5), math.sqrt(0.5)), (0.0, 1.0)
        a_s, a_e = arc_angles_counterclockwise(s, m, e, (0.0, 0.0))
        self.assertLess(a_s, a_e)
        # the reference's 1e-8 normalisation epsilon costs ~1e-4 of angle accuracy
        # near the axes (asin is ill-conditioned there); reproduce it faithfully.
        self.assertAlmostEqual(a_s, 0.0, places=3)
        self.assertAlmostEqual(a_e, math.pi / 2, places=3)

    def test_wrapping_arc_unwraps_branch(self):
        # arc through the +x axis: from -45 deg to +45 deg
        s = (math.cos(-math.pi / 4), math.sin(-math.pi / 4))
        m = (1.0, 0.0)
        e = (math.cos(math.pi / 4), math.sin(math.pi / 4))
        a_s, a_e = arc_angles_counterclockwise(s, m, e, (0.0, 0.0))
        self.assertLess(a_s, 0.0)
        self.assertAlmostEqual(a_e - a_s, math.pi / 2, places=3)

    def test_sweep_angle(self):
        s, m, e = (1.0, 0.0), (math.sqrt(0.5), math.sqrt(0.5)), (0.0, 1.0)
        self.assertAlmostEqual(arc_sweep_angle(s, m, e, (0.0, 0.0)),
                               math.pi / 2, places=3)


class TestBBoxes(unittest.TestCase):
    def test_arc_bbox_includes_top_extreme(self):
        # half circle over the top: (1,0) -> (-1,0) through (0,1)
        s, m, e = (1.0, 0.0), (0.0, 1.0), (-1.0, 0.0)
        box = arc_bbox(s, m, e, (0.0, 0.0), 1.0)
        self.assertAlmostEqual(box[0], -1.0, places=6)
        self.assertAlmostEqual(box[1], 0.0, places=6)
        self.assertAlmostEqual(box[2], 1.0, places=6)
        self.assertAlmostEqual(box[3], 1.0, places=6)  # bulge captured

    def test_arc_bbox_quarter_is_endpoints_only(self):
        s, m, e = (1.0, 0.0), (math.sqrt(0.5), math.sqrt(0.5)), (0.0, 1.0)
        box = arc_bbox(s, m, e, (0.0, 0.0), 1.0)
        self.assertAlmostEqual(box[0], 0.0, places=6)
        self.assertAlmostEqual(box[1], 0.0, places=6)
        self.assertAlmostEqual(box[2], 1.0, places=6)
        self.assertAlmostEqual(box[3], 1.0, places=6)

    def test_arc_bbox_through_positive_x(self):
        s = (math.cos(-math.pi / 4), math.sin(-math.pi / 4))
        m = (1.0, 0.0)
        e = (math.cos(math.pi / 4), math.sin(math.pi / 4))
        box = arc_bbox(s, m, e, (0.0, 0.0), 1.0)
        self.assertAlmostEqual(box[2], 1.0, places=6)  # +x extreme included

    def test_circle_and_line_bbox(self):
        self.assertEqual(circle_bbox((1.0, 2.0), 3.0), (-2.0, -1.0, 4.0, 5.0))
        self.assertEqual(line_bbox((3.0, 1.0), (1.0, 4.0)), (1.0, 1.0, 3.0, 4.0))

    def test_circle_endpoints_convention(self):
        self.assertEqual(circle_start_point((0.0, 0.0), 2.0), (-2.0, 0.0))
        self.assertEqual(circle_end_point((0.0, 0.0), 2.0), (2.0, 0.0))


class TestSampling(unittest.TestCase):
    def test_line_sampling_endpoints(self):
        pts = sample_line_points((0.0, 0.0), (1.0, 2.0), n=5)
        self.assertEqual(len(pts), 5)
        self.assertEqual(pts[0], (0.0, 0.0))
        self.assertAlmostEqual(pts[-1][0], 1.0)
        self.assertAlmostEqual(pts[-1][1], 2.0)
        self.assertAlmostEqual(pts[2][0], 0.5)

    def test_circle_sampling_excludes_endpoint(self):
        pts = sample_circle_points((0.0, 0.0), 1.0, n=4)
        self.assertEqual(len(pts), 4)
        self.assertAlmostEqual(pts[0][0], 1.0, places=9)
        self.assertAlmostEqual(pts[1][1], 1.0, places=9)
        for p in pts:
            self.assertAlmostEqual(math.hypot(*p), 1.0, places=9)

    def test_arc_sampling_on_circle_and_ordered(self):
        s, m, e = (1.0, 0.0), (math.sqrt(0.5), math.sqrt(0.5)), (0.0, 1.0)
        pts = sample_arc_points(s, m, e, (0.0, 0.0), 1.0, n=9)
        self.assertEqual(len(pts), 9)
        for p in pts:
            self.assertAlmostEqual(math.hypot(*p), 1.0, places=9)
        self.assertAlmostEqual(pts[0][0], 1.0, places=6)
        self.assertAlmostEqual(pts[-1][1], 1.0, places=6)

    def test_sampling_is_deterministic(self):
        a = sample_arc_points((1.0, 0.0), (math.sqrt(0.5), math.sqrt(0.5)),
                              (0.0, 1.0), (0.0, 0.0), 1.0, n=7)
        b = sample_arc_points((1.0, 0.0), (math.sqrt(0.5), math.sqrt(0.5)),
                              (0.0, 1.0), (0.0, 0.0), 1.0, n=7)
        self.assertEqual(a, b)

    def test_invalid_n(self):
        with self.assertRaises(ValueError):
            sample_line_points((0.0, 0.0), (1.0, 1.0), n=0)
        with self.assertRaises(ValueError):
            sample_circle_points((0.0, 0.0), 1.0, n=0)


if __name__ == "__main__":
    unittest.main()
