import unittest

from harnesscad.domain.geometry.solidtype_integer_geometry import (
    NANO_PER_MM,
    MAX_COORD,
    CoordRangeError,
    mm_to_nano,
    nano_to_mm,
    vec_to_int,
    vec_to_float,
    add_i,
    sub_i,
    dot_i,
    cross_i,
    equals_i,
    length_squared_i,
    segment_intersection_2i,
    line_line_closest_points_3i,
    plane_plane_intersection,
    clip_line_to_polygon_3i,
    VertexRegistry,
)


class TestConversions(unittest.TestCase):
    def test_mm_to_nano_exact(self):
        self.assertEqual(mm_to_nano(1.0), NANO_PER_MM)
        self.assertEqual(mm_to_nano(2.5), 2_500_000)
        self.assertEqual(mm_to_nano(-0.001), -1_000)

    def test_round_trip(self):
        self.assertAlmostEqual(nano_to_mm(mm_to_nano(3.14159)), 3.14159, places=6)

    def test_vec_conversion(self):
        self.assertEqual(vec_to_int((1.0, 2.0, 3.0)), (1_000_000, 2_000_000, 3_000_000))
        self.assertEqual(vec_to_float((1_000_000, 2_000_000)), (1.0, 2.0))

    def test_range_guard(self):
        with self.assertRaises(CoordRangeError):
            mm_to_nano(MAX_COORD)  # * NANO_PER_MM overflows the guard


class TestIntegerVectorOps(unittest.TestCase):
    def test_add_sub(self):
        self.assertEqual(add_i((1, 2, 3), (4, 5, 6)), (5, 7, 9))
        self.assertEqual(sub_i((4, 5, 6), (1, 2, 3)), (3, 3, 3))

    def test_dot_cross(self):
        self.assertEqual(dot_i((1, 2, 3), (4, 5, 6)), 32)
        self.assertEqual(cross_i((1, 0, 0), (0, 1, 0)), (0, 0, 1))

    def test_equals_and_length(self):
        self.assertTrue(equals_i((1, 2, 3), (1, 2, 3)))
        self.assertFalse(equals_i((1, 2, 3), (1, 2, 4)))
        self.assertEqual(length_squared_i((3, 4)), 25)


class TestSegmentIntersection(unittest.TestCase):
    def test_crossing(self):
        # Two segments crossing at the origin.
        p = segment_intersection_2i((-10, 0), (10, 0), (0, -10), (0, 10))
        self.assertEqual(p, (0, 0))

    def test_snapped_off_grid(self):
        # Cross at (5, 5): still exact integer.
        p = segment_intersection_2i((0, 0), (10, 10), (0, 10), (10, 0))
        self.assertEqual(p, (5, 5))

    def test_parallel_returns_none(self):
        self.assertIsNone(
            segment_intersection_2i((0, 0), (10, 0), (0, 5), (10, 5))
        )

    def test_out_of_range_returns_none(self):
        # Lines would cross far outside both spans.
        self.assertIsNone(
            segment_intersection_2i((0, 0), (1, 0), (5, -5), (5, 5))
        )

    def test_negative_cross_branch(self):
        # Swap orientation so the determinant is negative; must still find it.
        p = segment_intersection_2i((0, -10), (0, 10), (-10, 0), (10, 0))
        self.assertEqual(p, (0, 0))

    def test_shared_vertex_is_bit_exact(self):
        # Same crossing computed from two different orderings snaps identically.
        a = segment_intersection_2i((0, 0), (8, 6), (0, 6), (8, 0))
        b = segment_intersection_2i((8, 0), (0, 6), (8, 6), (0, 0))
        self.assertEqual(a, b)


class TestLineLineClosest(unittest.TestCase):
    def test_intersecting_lines(self):
        r = line_line_closest_points_3i((0, 0, 0), (1, 0, 0), (5, 0, 0), (0, 1, 0))
        self.assertIsNotNone(r)
        pa, pb = r
        self.assertEqual(pa, pb)  # single welded point
        self.assertEqual(pa, (5, 0, 0))

    def test_parallel_returns_none(self):
        self.assertIsNone(
            line_line_closest_points_3i((0, 0, 0), (1, 0, 0), (0, 5, 0), (1, 0, 0))
        )

    def test_skew_lines_average(self):
        # Skew lines: midpoint of the closest approach, both get same point.
        r = line_line_closest_points_3i((0, 0, 0), (1, 0, 0), (0, 0, 10), (0, 1, 0))
        self.assertIsNotNone(r)
        pa, pb = r
        self.assertEqual(pa, pb)
        self.assertEqual(pa, (0, 0, 5))


class TestPlanePlaneIntersection(unittest.TestCase):
    def test_orthogonal_planes(self):
        # xy-plane (normal +z through origin) and xz-plane (normal +y through origin).
        r = plane_plane_intersection((0, 0, 1), (0, 0, 0), (0, 1, 0), (0, 0, 0))
        self.assertIsNotNone(r)
        point, direction = r
        # Intersection line is the x-axis: direction parallel to x, point on axis.
        self.assertEqual(direction, (0 * 0 - 1 * 1, 1 * 0 - 0 * 0, 0 * 1 - 0 * 0))
        self.assertEqual(point[1], 0)
        self.assertEqual(point[2], 0)

    def test_parallel_planes_none(self):
        self.assertIsNone(
            plane_plane_intersection((0, 0, 1), (0, 0, 0), (0, 0, 1), (0, 0, 5))
        )

    def test_offset_plane_point_on_line(self):
        # xy-plane at z=0 and a plane x=4 (normal +x through (4,0,0)).
        r = plane_plane_intersection((0, 0, 1), (0, 0, 0), (1, 0, 0), (4, 0, 0))
        self.assertIsNotNone(r)
        point, direction = r
        self.assertEqual(point[0], 4)  # line lies at x=4
        self.assertEqual(point[2], 0)  # and z=0


class TestClipLineToPolygon(unittest.TestCase):
    def test_line_through_square(self):
        square = [(0, 0, 0), (10, 0, 0), (10, 10, 0), (0, 10, 0)]
        # Horizontal line through the middle at y=5.
        spans = clip_line_to_polygon_3i((-5, 5, 0), (1, 0, 0), square)
        self.assertEqual(len(spans), 1)
        _, _, start, end = spans[0]
        pts = {start, end}
        self.assertIn((0, 5, 0), pts)
        self.assertIn((10, 5, 0), pts)

    def test_degenerate_polygon(self):
        self.assertEqual(clip_line_to_polygon_3i((0, 0, 0), (1, 0, 0), [(0, 0, 0)]), [])


class TestVertexRegistry(unittest.TestCase):
    def test_interning_welds_coincident(self):
        reg = VertexRegistry()
        a = reg.intern((1_000, 2_000, 3_000))
        b = reg.intern((1_000, 2_000, 3_000))
        c = reg.intern((1_000, 2_000, 3_001))
        self.assertEqual(a, b)
        self.assertNotEqual(a, c)
        self.assertEqual(len(reg), 2)

    def test_coord_and_get(self):
        reg = VertexRegistry()
        vid = reg.intern((5, 5, 5))
        self.assertEqual(reg.coord(vid), (5, 5, 5))
        self.assertEqual(reg.get((5, 5, 5)), vid)
        self.assertIsNone(reg.get((5, 5, 6)))
        self.assertIn((5, 5, 5), reg)
        self.assertNotIn((9, 9, 9), reg)

    def test_intern_mm_quantises(self):
        reg = VertexRegistry()
        # 1.0000004 mm and 1.0000001 mm both snap to 1_000_000 nm.
        a = reg.intern_mm((1.0000004, 0.0, 0.0))
        b = reg.intern_mm((1.0000001, 0.0, 0.0))
        self.assertEqual(a, b)
        self.assertEqual(len(reg), 1)

    def test_intersection_feeds_registry(self):
        # Two segment intersections that land on the same grid point weld to one id.
        reg = VertexRegistry()
        p1 = segment_intersection_2i((0, 0), (10, 10), (0, 10), (10, 0))
        p2 = segment_intersection_2i((5, 0), (5, 10), (0, 5), (10, 5))
        self.assertEqual(p1, (5, 5))
        self.assertEqual(p2, (5, 5))
        self.assertEqual(reg.intern(p1), reg.intern(p2))
        self.assertEqual(len(reg), 1)


if __name__ == "__main__":
    unittest.main()
