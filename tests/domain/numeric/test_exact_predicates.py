"""Tests for numeric.manifold_predicates."""

import unittest

from harnesscad.domain.numeric.exact_predicates import orient2d, orient3d, incircle, insphere


class TestOrient2d(unittest.TestCase):
    def test_ccw_positive(self):
        self.assertEqual(orient2d((0, 0), (1, 0), (0, 1)), 1)

    def test_cw_negative(self):
        self.assertEqual(orient2d((0, 0), (0, 1), (1, 0)), -1)

    def test_collinear_zero(self):
        self.assertEqual(orient2d((0, 0), (1, 1), (2, 2)), 0)

    def test_near_collinear_exact(self):
        # Points collinear in exact arithmetic but prone to float error.
        a = (0.0, 0.0)
        b = (1.0, 1.0)
        c = (3.0, 3.0)
        self.assertEqual(orient2d(a, b, c), 0)

    def test_tiny_positive_offset(self):
        a = (0.5, 0.5)
        b = (12.0, 12.0)
        c = (24.0, 24.0 + 1e-9)
        # c is a hair above the line -> CCW positive.
        self.assertEqual(orient2d(a, b, c), 1)


class TestOrient3d(unittest.TestCase):
    def test_sign_known(self):
        # Standard basis tetrahedron.
        a = (0, 0, 0)
        b = (1, 0, 0)
        c = (0, 1, 0)
        d_below = (0, 0, -1)
        d_above = (0, 0, 1)
        s1 = orient3d(a, b, c, d_below)
        s2 = orient3d(a, b, c, d_above)
        self.assertEqual(s1, -s2)
        self.assertNotEqual(s1, 0)

    def test_coplanar_zero(self):
        self.assertEqual(orient3d((0, 0, 0), (1, 0, 0), (0, 1, 0), (1, 1, 0)), 0)

    def test_near_coplanar_exact(self):
        a = (0.0, 0.0, 0.0)
        b = (1.0, 0.0, 0.0)
        c = (0.0, 1.0, 0.0)
        d = (0.3, 0.3, 0.0)
        self.assertEqual(orient3d(a, b, c, d), 0)

    def test_tiny_offset_detected(self):
        a = (0.0, 0.0, 0.0)
        b = (1.0, 0.0, 0.0)
        c = (0.0, 1.0, 0.0)
        d = (0.3, 0.3, 1e-10)
        self.assertNotEqual(orient3d(a, b, c, d), 0)


class TestIncircle(unittest.TestCase):
    def setUp(self):
        # Unit square corners CCW define a circle of radius sqrt(2)/2 centred at (0.5,0.5)
        self.a = (0.0, 0.0)
        self.b = (1.0, 0.0)
        self.c = (1.0, 1.0)

    def test_center_inside(self):
        self.assertEqual(incircle(self.a, self.b, self.c, (0.5, 0.5)), 1)

    def test_far_outside(self):
        self.assertEqual(incircle(self.a, self.b, self.c, (10.0, 10.0)), -1)

    def test_on_circle_zero(self):
        # (0,1) lies on the circumcircle of the unit square.
        self.assertEqual(incircle(self.a, self.b, self.c, (0.0, 1.0)), 0)


class TestInsphere(unittest.TestCase):
    def setUp(self):
        # Positively oriented tetrahedron.
        self.a = (0.0, 0.0, 0.0)
        self.b = (1.0, 0.0, 0.0)
        self.c = (0.0, 1.0, 0.0)
        self.d = (0.0, 0.0, 1.0)
        from harnesscad.domain.numeric.exact_predicates import orient3d as o3
        if o3(self.a, self.b, self.c, self.d) < 0:
            self.b, self.c = self.c, self.b
        assert o3(self.a, self.b, self.c, self.d) > 0

    def test_centroid_inside(self):
        e = (0.25, 0.25, 0.25)
        self.assertEqual(insphere(self.a, self.b, self.c, self.d, e), 1)

    def test_far_outside(self):
        e = (5.0, 5.0, 5.0)
        self.assertEqual(insphere(self.a, self.b, self.c, self.d, e), -1)

    def test_on_sphere_zero(self):
        # Sphere through the 4 unit points also passes through (1,1,0).
        e = (1.0, 1.0, 0.0)
        self.assertEqual(insphere(self.a, self.b, self.c, self.d, e), 0)


if __name__ == "__main__":
    unittest.main()
