"""Tests for geometry.arcs_chord_tolerance."""

import math
import unittest

from harnesscad.domain.geometry.arcs_chord_tolerance import (
    approximate_arc,
    approximate_circle,
    chord_error,
    chord_length,
    sagitta,
    segment_angle_for_tolerance,
    segments_for_tolerance,
)
from harnesscad.domain.geometry.arcs_closest_point import Arc2D


class TestSagitta(unittest.TestCase):
    def test_semicircle_sagitta_is_the_radius(self):
        self.assertAlmostEqual(sagitta(10.0, math.pi), 10.0)

    def test_zero_angle_is_zero(self):
        self.assertAlmostEqual(sagitta(10.0, 0.0), 0.0)

    def test_sign_of_angle_is_irrelevant(self):
        self.assertAlmostEqual(
            sagitta(3.0, -1.1), sagitta(3.0, 1.1)
        )

    def test_bad_radius(self):
        with self.assertRaises(ValueError):
            sagitta(0.0, 1.0)

    def test_chord_length_semicircle_is_diameter(self):
        self.assertAlmostEqual(chord_length(5.0, math.pi), 10.0)


class TestSegmentCount(unittest.TestCase):
    def test_segment_angle_inverts_sagitta(self):
        theta = segment_angle_for_tolerance(100.0, 1.0)
        self.assertAlmostEqual(sagitta(100.0, theta), 1.0)

    def test_minimum_of_two_segments(self):
        # a tiny sweep still yields the reference implementation's floor of 2
        self.assertEqual(segments_for_tolerance(100.0, 0.01, 0.001), 2)

    def test_tighter_tolerance_needs_more_segments(self):
        coarse = segments_for_tolerance(100.0, math.pi, 5.0)
        fine = segments_for_tolerance(100.0, math.pi, 0.05)
        self.assertLess(coarse, fine)

    def test_degenerate_tolerance_collapses_to_one_chord(self):
        self.assertEqual(segments_for_tolerance(10.0, math.pi, 0.0), 1)
        self.assertEqual(segments_for_tolerance(10.0, math.pi, 50.0), 1)

    def test_chord_error_of_count_respects_tolerance(self):
        radius, sweep, tolerance = 100.0, math.pi / 2.0, 0.5
        n = segments_for_tolerance(radius, sweep, tolerance)
        self.assertLessEqual(chord_error(radius, sweep, n), tolerance + 1e-12)
        # one fewer segment would violate it (n is minimal, given the floor of 2)
        if n > 2:
            self.assertGreater(chord_error(radius, sweep, n - 1), tolerance)

    def test_chord_error_bad_count(self):
        with self.assertRaises(ValueError):
            chord_error(1.0, 1.0, 0)


class TestApproximateArc(unittest.TestCase):
    def test_endpoints_are_exact(self):
        arc = Arc2D((0.0, 0.0), 100.0, 0.0, math.pi / 2.0)
        pts = approximate_arc(arc, 10.0)
        self.assertAlmostEqual(pts[0][0], arc.start()[0])
        self.assertAlmostEqual(pts[0][1], arc.start()[1])
        self.assertAlmostEqual(pts[-1][0], arc.end()[0])
        self.assertAlmostEqual(pts[-1][1], arc.end()[1])

    def test_every_vertex_lies_on_the_arc(self):
        arc = Arc2D((3.0, -2.0), 100.0, 0.4, math.pi / 2.0)
        pts = approximate_arc(arc, 10.0)
        for p in pts:
            r = math.hypot(p[0] - arc.centre[0], p[1] - arc.centre[1])
            self.assertAlmostEqual(r, arc.radius, places=9)

    def test_chord_midpoints_stay_within_tolerance(self):
        arc = Arc2D((0.0, 0.0), 50.0, 0.0, math.radians(270.0))
        tolerance = 0.25
        pts = approximate_arc(arc, tolerance)
        for i in range(1, len(pts)):
            mid = (
                (pts[i - 1][0] + pts[i][0]) / 2.0,
                (pts[i - 1][1] + pts[i][1]) / 2.0,
            )
            deviation = arc.radius - math.hypot(mid[0], mid[1])
            self.assertLessEqual(deviation, tolerance + 1e-9)

    def test_clockwise_arc_is_reversed_but_valid(self):
        arc = Arc2D((0.0, 0.0), 10.0, math.pi, -math.pi / 2.0)
        pts = approximate_arc(arc, 0.1)
        self.assertGreater(len(pts), 2)
        self.assertAlmostEqual(pts[-1][0], 0.0, places=9)
        self.assertAlmostEqual(pts[-1][1], 10.0, places=9)

    def test_degenerate_tolerance_is_a_single_chord(self):
        arc = Arc2D((0.0, 0.0), 10.0, 0.0, math.pi)
        self.assertEqual(len(approximate_arc(arc, 0.0)), 2)
        self.assertEqual(len(approximate_arc(arc, 100.0)), 2)

    def test_bad_radius(self):
        with self.assertRaises(ValueError):
            approximate_arc(Arc2D((0.0, 0.0), -1.0, 0.0, 1.0), 0.1)

    def test_deterministic(self):
        arc = Arc2D((1.0, 1.0), 20.0, 0.3, 2.0)
        self.assertEqual(approximate_arc(arc, 0.05), approximate_arc(arc, 0.05))


class TestApproximateCircle(unittest.TestCase):
    def test_ring_is_open_and_on_the_circle(self):
        ring = approximate_circle((0.0, 0.0), 10.0, 0.05)
        self.assertGreater(len(ring), 8)
        for p in ring:
            self.assertAlmostEqual(math.hypot(p[0], p[1]), 10.0, places=9)
        # first point is not duplicated at the end
        self.assertNotAlmostEqual(ring[0][1], ring[-1][1], places=6)

    def test_finer_tolerance_gives_more_vertices(self):
        coarse = approximate_circle((0.0, 0.0), 10.0, 1.0)
        fine = approximate_circle((0.0, 0.0), 10.0, 0.01)
        self.assertLess(len(coarse), len(fine))


if __name__ == "__main__":
    unittest.main()
