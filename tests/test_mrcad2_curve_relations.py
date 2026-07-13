import math
import unittest

from harnesscad.domain.editing.mrcad_schema import arc, circle, line
from harnesscad.domain.geometry.mrcad2_curve_relations import (
    concentric,
    curve_center,
    curve_radius,
    endpoints,
    meeting_ends,
    parallel,
    parallel_distance,
    perpendicular,
    point_to_curve_distance,
)


class TestLineRelations(unittest.TestCase):
    def test_parallel_horizontal(self):
        self.assertTrue(parallel(line((0, 0), (4, 0)), line((0, 2), (4, 2))))

    def test_parallel_vertical_pair(self):
        self.assertTrue(parallel(line((1, 0), (1, 5)), line((3, -2), (3, 9))))

    def test_not_parallel(self):
        self.assertFalse(parallel(line((0, 0), (4, 0)), line((0, 0), (4, 4))))

    def test_perpendicular_axis_aligned(self):
        self.assertTrue(perpendicular(line((0, 0), (4, 0)), line((2, 0), (2, 5))))

    def test_perpendicular_diagonal(self):
        self.assertTrue(perpendicular(line((0, 0), (1, 1)), line((0, 0), (1, -1))))

    def test_not_perpendicular(self):
        self.assertFalse(perpendicular(line((0, 0), (4, 0)), line((0, 0), (4, 1))))


class TestParallelDistance(unittest.TestCase):
    def test_overlapping_gap(self):
        d = parallel_distance(line((0, 0), (10, 0)), line((2, 3), (8, 3)))
        self.assertAlmostEqual(d, 3.0)

    def test_non_overlapping_returns_none(self):
        # Parallel but shifted entirely past the end -> no overlap.
        self.assertIsNone(parallel_distance(line((0, 0), (4, 0)), line((10, 3), (14, 3))))

    def test_not_parallel_returns_none(self):
        self.assertIsNone(parallel_distance(line((0, 0), (4, 0)), line((0, 0), (4, 4))))


class TestCrossCurve(unittest.TestCase):
    def test_meeting_ends_line_line(self):
        self.assertTrue(meeting_ends(line((0, 0), (1, 1)), line((1, 1), (2, 0))))

    def test_meeting_ends_line_arc_uses_arc_endpoints(self):
        # Arc endpoints are p0 and p2; the mid point (5,5) must NOT count.
        a = arc((0, 0), (5, 5), (10, 0))
        self.assertTrue(meeting_ends(line((10, 0), (12, 0)), a))
        self.assertFalse(meeting_ends(line((5, 5), (6, 6)), a))

    def test_endpoints_kinds(self):
        self.assertEqual(len(endpoints(line((0, 0), (1, 0)))), 2)
        self.assertEqual(len(endpoints(arc((0, 0), (1, 1), (2, 0)))), 2)
        self.assertEqual(endpoints(circle((0, 0), (2, 0))), ())

    def test_concentric_circle_circle(self):
        self.assertTrue(concentric(circle((-2, 0), (2, 0)), circle((0, -3), (0, 3))))

    def test_concentric_circle_arc(self):
        # Arc on the unit circle centred at origin; circle also centred at origin.
        a = arc((1, 0), (0, 1), (-1, 0))
        self.assertTrue(concentric(a, circle((-5, 0), (5, 0))))

    def test_line_never_concentric(self):
        self.assertFalse(concentric(line((0, 0), (1, 1)), circle((-1, 0), (1, 0))))


class TestCenterRadius(unittest.TestCase):
    def test_circle_center_radius(self):
        c = circle((-3, 0), (3, 0))
        self.assertAlmostEqual(curve_center(c)[0], 0.0)
        self.assertAlmostEqual(curve_radius(c), 3.0)

    def test_arc_center_radius(self):
        a = arc((1, 0), (0, 1), (-1, 0))
        cx, cy = curve_center(a)
        self.assertAlmostEqual(cx, 0.0)
        self.assertAlmostEqual(cy, 0.0)
        self.assertAlmostEqual(curve_radius(a), 1.0)


class TestPointToCurveDistance(unittest.TestCase):
    def test_line_perpendicular_foot(self):
        d = point_to_curve_distance(line((0, 0), (10, 0)), (5, 4))
        self.assertAlmostEqual(d, 4.0)

    def test_line_beyond_endpoint(self):
        d = point_to_curve_distance(line((0, 0), (10, 0)), (-3, 4))
        self.assertAlmostEqual(d, 5.0)

    def test_circle_distance(self):
        c = circle((-2, 0), (2, 0))  # radius 2 at origin
        self.assertAlmostEqual(point_to_curve_distance(c, (5, 0)), 3.0)
        self.assertAlmostEqual(point_to_curve_distance(c, (0, 0)), 2.0)

    def test_arc_inside_sweep(self):
        # Upper unit semicircle; target straight up is on the arc's bearing.
        a = arc((1, 0), (0, 1), (-1, 0))
        self.assertAlmostEqual(point_to_curve_distance(a, (0, 3)), 2.0)

    def test_arc_outside_sweep_uses_endpoint(self):
        # Target below the x-axis: bearing not in the upper semicircle sweep,
        # so distance is to the nearer endpoint (1,0) or (-1,0).
        a = arc((1, 0), (0, 1), (-1, 0))
        d = point_to_curve_distance(a, (1, -1))
        self.assertAlmostEqual(d, 1.0)

    def test_arc_collinear_falls_back_to_segment(self):
        a = arc((0, 0), (1, 0), (2, 0))  # collinear control points
        self.assertAlmostEqual(point_to_curve_distance(a, (1, 3)), 3.0)


if __name__ == "__main__":
    unittest.main()
