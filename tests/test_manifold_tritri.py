"""Tests for geometry.manifold_tritri."""

import math
import unittest

from harnesscad.domain.geometry.mesh.manifold_tritri import (
    triangles_intersect,
    triangle_triangle_segment,
    segment_plane_intersection,
    segment_segment_2d,
    point_side_of_plane,
)


class TestPointSide(unittest.TestCase):
    def setUp(self):
        self.tri = ((0, 0, 0), (1, 0, 0), (0, 1, 0))  # z = 0 plane

    def test_above_below_on(self):
        self.assertNotEqual(point_side_of_plane((0.2, 0.2, 1.0), self.tri), 0)
        self.assertEqual(
            point_side_of_plane((0.2, 0.2, 1.0), self.tri),
            -point_side_of_plane((0.2, 0.2, -1.0), self.tri),
        )
        self.assertEqual(point_side_of_plane((0.2, 0.2, 0.0), self.tri), 0)


class TestSegmentPlane(unittest.TestCase):
    def test_crossing(self):
        tri = ((0, 0, 0), (1, 0, 0), (0, 1, 0))
        p = segment_plane_intersection((0.3, 0.3, -1.0), (0.3, 0.3, 1.0), tri)
        self.assertIsNotNone(p)
        self.assertAlmostEqual(p[2], 0.0)
        self.assertAlmostEqual(p[0], 0.3)

    def test_parallel_none(self):
        tri = ((0, 0, 0), (1, 0, 0), (0, 1, 0))
        self.assertIsNone(
            segment_plane_intersection((0, 0, 1), (1, 0, 1), tri)
        )

    def test_not_reaching(self):
        tri = ((0, 0, 0), (1, 0, 0), (0, 1, 0))
        self.assertIsNone(
            segment_plane_intersection((0.3, 0.3, 1.0), (0.3, 0.3, 2.0), tri)
        )


class TestSegSeg2D(unittest.TestCase):
    def test_proper_cross(self):
        p = segment_segment_2d((0, 0), (2, 2), (0, 2), (2, 0))
        self.assertIsNotNone(p)
        self.assertAlmostEqual(p[0], 1.0)
        self.assertAlmostEqual(p[1], 1.0)

    def test_no_cross(self):
        self.assertIsNone(segment_segment_2d((0, 0), (1, 0), (0, 1), (1, 1)))

    def test_collinear_none(self):
        self.assertIsNone(segment_segment_2d((0, 0), (2, 0), (1, 0), (3, 0)))


class TestTriTriIntersect(unittest.TestCase):
    def test_transversal_crossing(self):
        # Triangle in z=0 plane and a vertical triangle piercing it.
        t1 = ((0, 0, 0), (4, 0, 0), (0, 4, 0))
        t2 = ((1, 1, -1), (2, 1, 1), (1, 2, 1))
        self.assertTrue(triangles_intersect(t1, t2))

    def test_disjoint_above(self):
        t1 = ((0, 0, 0), (4, 0, 0), (0, 4, 0))
        t2 = ((1, 1, 1), (2, 1, 2), (1, 2, 2))  # entirely above z=0
        self.assertFalse(triangles_intersect(t1, t2))

    def test_disjoint_same_plane(self):
        t1 = ((0, 0, 0), (1, 0, 0), (0, 1, 0))
        t2 = ((5, 5, 0), (6, 5, 0), (5, 6, 0))  # coplanar, far apart
        self.assertFalse(triangles_intersect(t1, t2))

    def test_coplanar_overlap(self):
        t1 = ((0, 0, 0), (4, 0, 0), (0, 4, 0))
        t2 = ((1, 1, 0), (3, 1, 0), (1, 3, 0))  # coplanar, overlapping
        self.assertTrue(triangles_intersect(t1, t2))

    def test_shared_vertex(self):
        t1 = ((0, 0, 0), (4, 0, 0), (0, 4, 0))
        t2 = ((0, 0, 0), (0, 0, 3), (2, -1, 1))
        self.assertTrue(triangles_intersect(t1, t2))

    def test_segment_endpoints_known(self):
        # z=0 triangle; vertical triangle crosses it along a known chord.
        t1 = ((-5, -5, 0), (5, -5, 0), (0, 5, 0))
        t2 = ((0, 0, -1), (4, 0, -1), (2, 0, 3))
        seg = triangle_triangle_segment(t1, t2)
        self.assertIsNotNone(seg)
        a, b = seg
        # Both endpoints lie in z=0 and on y=0 (t2 lies in the y=0 plane).
        self.assertAlmostEqual(a[2], 0.0)
        self.assertAlmostEqual(b[2], 0.0)
        self.assertAlmostEqual(a[1], 0.0)
        self.assertAlmostEqual(b[1], 0.0)
        xs = sorted([a[0], b[0]])
        # Intersection chord spans x in [0, 4] clipped by triangle t1 interior.
        self.assertGreaterEqual(xs[0], -0.001)
        self.assertLessEqual(xs[1], 4.001)

    def test_segment_none_when_disjoint(self):
        t1 = ((0, 0, 0), (4, 0, 0), (0, 4, 0))
        t2 = ((1, 1, 1), (2, 1, 2), (1, 2, 2))
        self.assertIsNone(triangle_triangle_segment(t1, t2))


if __name__ == "__main__":
    unittest.main()
