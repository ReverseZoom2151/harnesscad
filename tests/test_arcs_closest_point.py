"""Tests for geometry.arcs_closest_point."""

import math
import unittest

from harnesscad.domain.geometry.arcs_closest_point import (
    INFINITE,
    MANY,
    ONE,
    Arc2D,
    Closest,
    closest_point_on_arc,
    closest_point_on_polyline,
    closest_point_on_segment,
    distance_to_arc,
    distance_to_segment,
)


class TestSegment(unittest.TestCase):
    def test_point_on_the_line(self):
        got = closest_point_on_segment((1.0, 2.0), (3.0, 10.0), (2.0, 6.0))
        self.assertEqual(got.kind, ONE)
        self.assertAlmostEqual(got.single()[0], 2.0)
        self.assertAlmostEqual(got.single()[1], 6.0)

    def test_zero_length_line(self):
        got = closest_point_on_segment((1.0, 2.0), (1.0, 2.0), (10.0, 0.0))
        self.assertEqual(got.single(), (1.0, 2.0))

    def test_above_the_line(self):
        got = closest_point_on_segment((0.0, 0.0), (10.0, 0.0), (5.0, 5.0))
        self.assertAlmostEqual(got.single()[0], 5.0)
        self.assertAlmostEqual(got.single()[1], 0.0)

    def test_past_the_end_clamps(self):
        got = closest_point_on_segment((0.0, 0.0), (10.0, 0.0), (15.0, 5.0))
        self.assertEqual(got.single(), (10.0, 0.0))

    def test_before_the_start_clamps(self):
        got = closest_point_on_segment((0.0, 0.0), (10.0, 0.0), (-5.0, 5.0))
        self.assertEqual(got.single(), (0.0, 0.0))

    def test_distance_helper(self):
        self.assertAlmostEqual(
            distance_to_segment((0.0, 0.0), (10.0, 0.0), (5.0, -4.0)), 4.0
        )


class TestArcGeometry(unittest.TestCase):
    def test_start_and_end(self):
        arc = Arc2D((5.0, 100.0), 10.0, 0.0, math.pi / 2.0)
        self.assertAlmostEqual(arc.start()[0], 15.0)
        self.assertAlmostEqual(arc.start()[1], 100.0)
        self.assertAlmostEqual(arc.end()[0], 5.0)
        self.assertAlmostEqual(arc.end()[1], 110.0)
        self.assertAlmostEqual(arc.end_angle, math.pi / 2.0)
        self.assertTrue(arc.is_anticlockwise)
        self.assertTrue(arc.is_minor_arc)

    def test_length(self):
        arc = Arc2D((0.0, 0.0), 2.0, 0.0, math.pi)
        self.assertAlmostEqual(arc.length(), 2.0 * math.pi)

    def test_contains_angle_quadrant(self):
        arc = Arc2D((0.0, 0.0), 1.0, 0.0, math.pi / 2.0)
        self.assertTrue(arc.contains_angle(math.radians(45.0)))
        self.assertTrue(arc.contains_angle(0.0))
        self.assertTrue(arc.contains_angle(math.radians(90.0)))
        self.assertFalse(arc.contains_angle(math.radians(120.0)))

    def test_contains_angle_outside(self):
        arc = Arc2D((0.0, 0.0), 1.0, 0.0, math.pi / 4.0)
        self.assertFalse(arc.contains_angle(math.radians(90.0)))

    def test_contains_angle_reverse_arc(self):
        arc = Arc2D((0.0, 0.0), 1.0, math.pi / 2.0, -math.pi / 4.0)
        self.assertTrue(arc.contains_angle(math.radians(45.0)))
        self.assertFalse(arc.contains_angle(math.radians(80.0) + 1.0))

    def test_contains_angle_across_branch_cut(self):
        # sweeps from 150 degrees through 180 to 210 degrees
        arc = Arc2D((0.0, 0.0), 1.0, math.radians(150.0), math.radians(60.0))
        self.assertTrue(arc.contains_angle(math.pi))
        self.assertTrue(arc.contains_angle(math.radians(-150.0)))  # == 210
        self.assertFalse(arc.contains_angle(0.0))

    def test_full_circle_contains_everything(self):
        arc = Arc2D((0.0, 0.0), 1.0, 0.0, 2.0 * math.pi)
        self.assertTrue(arc.contains_angle(1.234))
        self.assertTrue(arc.is_major_arc)


class TestArcClosestPoint(unittest.TestCase):
    def test_centre_is_infinite(self):
        arc = Arc2D((0.0, 0.0), 10.0, 0.0, math.pi)
        got = closest_point_on_arc(arc, (0.0, 0.0))
        self.assertEqual(got.kind, INFINITE)
        self.assertTrue(got.is_infinite)
        self.assertEqual(got.points, ())
        self.assertAlmostEqual(distance_to_arc(arc, (0.0, 0.0)), 10.0)

    def test_start_point(self):
        arc = Arc2D((0.0, 0.0), 10.0, 0.0, math.pi)
        got = closest_point_on_arc(arc, arc.start())
        self.assertEqual(got.kind, ONE)
        self.assertAlmostEqual(got.single()[0], arc.start()[0])
        self.assertAlmostEqual(got.single()[1], arc.start()[1])

    def test_radial_projection(self):
        arc = Arc2D((0.0, 0.0), 10.0, 0.0, math.pi)
        got = closest_point_on_arc(arc, (0.0, 3.0))
        self.assertAlmostEqual(got.single()[0], 0.0)
        self.assertAlmostEqual(got.single()[1], 10.0)
        self.assertAlmostEqual(distance_to_arc(arc, (0.0, 3.0)), 7.0)

    def test_midway_between_end_points_is_a_tie(self):
        arc = Arc2D((0.0, 0.0), 10.0, 0.0, math.pi)
        got = closest_point_on_arc(arc, (0.0, -10.0))
        self.assertEqual(got.kind, MANY)
        self.assertEqual(len(got.points), 2)
        xs = sorted(round(p[0], 6) for p in got.points)
        self.assertEqual(xs, [-10.0, 10.0])
        with self.assertRaises(ValueError):
            got.single()

    def test_nearer_endpoint_wins(self):
        arc = Arc2D((0.0, 0.0), 10.0, 0.0, math.pi)
        # bearing is -45 degrees, outside the arc, closer to the start (+x)
        got = closest_point_on_arc(arc, (5.0, -5.0))
        self.assertEqual(got.kind, ONE)
        self.assertAlmostEqual(got.single()[0], 10.0)
        self.assertAlmostEqual(got.single()[1], 0.0, places=6)

    def test_nearer_end_endpoint_wins(self):
        arc = Arc2D((0.0, 0.0), 10.0, 0.0, math.pi)
        got = closest_point_on_arc(arc, (-5.0, -5.0))
        self.assertAlmostEqual(got.single()[0], -10.0)


class TestPolyline(unittest.TestCase):
    def test_single_point(self):
        got = closest_point_on_polyline([(1.0, 1.0)], (5.0, 5.0))
        self.assertEqual(got.single(), (1.0, 1.0))

    def test_nearest_segment(self):
        poly = [(0.0, 0.0), (10.0, 0.0), (10.0, 10.0)]
        got = closest_point_on_polyline(poly, (11.0, 6.0))
        self.assertAlmostEqual(got.single()[0], 10.0)
        self.assertAlmostEqual(got.single()[1], 6.0)

    def test_symmetric_tie_reports_many(self):
        poly = [(-10.0, 0.0), (0.0, 0.0), (0.0, -10.0)]
        # equidistant from the horizontal and vertical legs
        got = closest_point_on_polyline(poly, (-4.0, -4.0))
        self.assertEqual(got.kind, MANY)
        self.assertEqual(len(got.points), 2)

    def test_empty_raises(self):
        with self.assertRaises(ValueError):
            closest_point_on_polyline([], (0.0, 0.0))


class TestClosestConstructors(unittest.TestCase):
    def test_helpers(self):
        self.assertEqual(Closest.one((1.0, 2.0)).points, ((1.0, 2.0),))
        self.assertEqual(Closest.many([(0.0, 0.0)]).kind, MANY)
        self.assertTrue(Closest.infinite().is_infinite)


if __name__ == "__main__":
    unittest.main()
