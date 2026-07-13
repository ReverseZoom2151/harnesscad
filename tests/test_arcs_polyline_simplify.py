"""Tests for geometry.arcs_polyline_simplify."""

import math
import unittest

from harnesscad.domain.geometry.arcs_polyline_simplify import (
    max_deviation,
    perpendicular_distance,
    polyline_length,
    simplify,
    simplify_indices,
)


class TestPerpendicularDistance(unittest.TestCase):
    def test_point_above_horizontal_line(self):
        d = perpendicular_distance((0.0, 0.0), (10.0, 0.0), (5.0, 3.0))
        self.assertAlmostEqual(d, 3.0)

    def test_point_on_line_is_zero(self):
        d = perpendicular_distance((0.0, 0.0), (10.0, 10.0), (4.0, 4.0))
        self.assertAlmostEqual(d, 0.0)

    def test_infinite_line_semantics_beyond_the_end(self):
        # the projection falls outside the segment, but we measure to the line
        d = perpendicular_distance((0.0, 0.0), (10.0, 0.0), (50.0, 2.0))
        self.assertAlmostEqual(d, 2.0)

    def test_degenerate_segment_falls_back_to_start_distance(self):
        d = perpendicular_distance((1.0, 2.0), (1.0, 2.0), (4.0, 6.0))
        self.assertAlmostEqual(d, 5.0)


class TestSimplify(unittest.TestCase):
    def test_empty(self):
        self.assertEqual(simplify([], 1.0), [])

    def test_single_point(self):
        self.assertEqual(simplify([(0.0, 0.0)], 1.0), [(0.0, 0.0)])

    def test_two_points_unchanged(self):
        pts = [(0.0, 0.0), (10.0, 2.0)]
        self.assertEqual(simplify(pts, 1.0), pts)

    def test_straight_line_collapses_to_endpoints(self):
        pts = [(float(i), 0.0) for i in range(100)]
        self.assertEqual(simplify(pts, 0.1), [pts[0], pts[99]])

    def test_small_vertical_jitter_collapses(self):
        max_jitter = 0.1
        pts = [
            (float(i), max_jitter * math.sin(i / 100.0 * math.pi))
            for i in range(100)
        ]
        got = simplify(pts, max_jitter * 2.0)
        self.assertEqual(got, [pts[0], pts[99]])

    def test_realistic_line_matches_reference(self):
        # the `simplify_more_realistic_line` case from arcs-core
        line = [
            (-43.0, 8.0),
            (-24.0, 19.0),
            (-13.0, 23.0),
            (-8.0, 36.0),
            (7.0, 40.0),
            (24.0, 12.0),
            (44.0, -6.0),
            (57.0, 2.0),
            (70.0, 7.0),
        ]
        got = simplify(line, 10.0)
        self.assertEqual(got, [line[0], line[4], line[6], line[8]])

    def test_zero_tolerance_keeps_non_collinear_points(self):
        line = [(0.0, 0.0), (1.0, 1.0), (2.0, 0.0), (3.0, 1.0)]
        self.assertEqual(simplify(line, 0.0), line)

    def test_indices_are_ascending_and_consistent(self):
        line = [
            (-43.0, 8.0),
            (-24.0, 19.0),
            (-13.0, 23.0),
            (-8.0, 36.0),
            (7.0, 40.0),
            (24.0, 12.0),
            (44.0, -6.0),
            (57.0, 2.0),
            (70.0, 7.0),
        ]
        idx = simplify_indices(line, 10.0)
        self.assertEqual(idx, sorted(idx))
        self.assertEqual([line[i] for i in idx], simplify(line, 10.0))

    def test_output_is_a_subsequence_and_honours_tolerance(self):
        pts = [
            (float(i), math.sin(i / 4.0) * 5.0 + math.cos(i / 7.0))
            for i in range(60)
        ]
        tolerance = 1.5
        got = simplify(pts, tolerance)
        self.assertLess(len(got), len(pts))
        self.assertEqual(got[0], pts[0])
        self.assertEqual(got[-1], pts[-1])
        for p in got:
            self.assertIn(p, pts)
        self.assertLessEqual(max_deviation(pts, got), tolerance + 1e-9)

    def test_deterministic(self):
        pts = [(float(i), (i % 3) * 0.5) for i in range(40)]
        self.assertEqual(simplify(pts, 0.4), simplify(pts, 0.4))


class TestPolylineLength(unittest.TestCase):
    def test_length(self):
        self.assertAlmostEqual(
            polyline_length([(0.0, 0.0), (3.0, 4.0), (3.0, 5.0)]), 6.0
        )

    def test_empty_is_zero(self):
        self.assertEqual(polyline_length([]), 0.0)


if __name__ == "__main__":
    unittest.main()
