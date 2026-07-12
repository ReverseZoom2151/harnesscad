"""Tests for the PPA parametric sketch-primitive representation."""

import math
import unittest

from reconstruction import ppa_primitive as pp


class TestConstructors(unittest.TestCase):
    def test_line_row(self):
        l = pp.line((1.0, 2.0), (3.0, 4.0), flag=True)
        self.assertEqual(l.ptype, pp.LINE)
        self.assertEqual(l.params, (1.0, 2.0, 3.0, 4.0, 0.0, 0.0, 0.0))
        self.assertEqual(l.to_row(), (pp.LINE, 1, (1.0, 2.0, 3.0, 4.0, 0.0, 0.0, 0.0)))
        self.assertEqual(l.meaningful, 4)

    def test_circle_row_radius_in_last_slot(self):
        c = pp.circle((5.0, 6.0), 7.0, flag=False)
        # Table I: circle radius occupies the 7th slot, slots x2..y3 are padding.
        self.assertEqual(c.params, (5.0, 6.0, 0.0, 0.0, 0.0, 0.0, 7.0))
        self.assertEqual(c.to_row()[1], 0)  # flag encoded as 0
        self.assertEqual(c.radius, 7.0)
        self.assertEqual(c.meaningful, 3)

    def test_arc_row(self):
        a = pp.arc((0, 0), (1, 1), (2, 0))
        self.assertEqual(a.params, (0, 0, 1, 1, 2, 0, 0.0))
        self.assertEqual(a.meaningful, 6)

    def test_point_row(self):
        pt = pp.point((9.0, 8.0))
        self.assertEqual(pt.params, (9.0, 8.0, 0.0, 0.0, 0.0, 0.0, 0.0))
        self.assertEqual(pt.meaningful, 2)

    def test_type_codes_stable(self):
        self.assertEqual(pp.TYPE_CODE[pp.LINE], 0)
        self.assertEqual(pp.circle((0, 0), 1).type_code, 1)

    def test_bad_type_and_length(self):
        with self.assertRaises(ValueError):
            pp.Primitive("spline", True, (0,) * 7)
        with self.assertRaises(ValueError):
            pp.Primitive(pp.LINE, True, (0, 0, 0))
        with self.assertRaises(ValueError):
            pp.point((0, 0)).radius


class TestRowRoundTrip(unittest.TestCase):
    def test_from_row_inverts_to_row(self):
        for prim in (pp.line((1, 2), (3, 4)), pp.circle((0, 0), 2.5, flag=False),
                     pp.arc((0, 0), (1, 2), (3, 1)), pp.point((7, 7))):
            t, f, params = prim.to_row()
            self.assertEqual(pp.Primitive.from_row(t, f, params), prim)

    def test_control_points(self):
        self.assertEqual(pp.line((1, 2), (3, 4)).control_points(), ((1, 2), (3, 4)))
        self.assertEqual(pp.circle((5, 6), 1).control_points(), ((5, 6),))
        self.assertEqual(pp.arc((0, 0), (1, 1), (2, 0)).control_points(),
                         ((0, 0), (1, 1), (2, 0)))


class TestSketchSetSemantics(unittest.TestCase):
    def test_order_invariant_equality(self):
        a = pp.line((0, 0), (1, 0))
        b = pp.circle((2, 2), 1)
        s1 = pp.Sketch([a, b])
        s2 = pp.Sketch([b, a])
        self.assertEqual(s1, s2)
        self.assertEqual(hash(s1), hash(s2))
        self.assertEqual(len(s1), 2)

    def test_distinct_sketches_differ(self):
        s1 = pp.Sketch([pp.line((0, 0), (1, 0))])
        s2 = pp.Sketch([pp.line((0, 0), (2, 0))])
        self.assertNotEqual(s1, s2)

    def test_iteration(self):
        prims = [pp.point((0, 0)), pp.point((1, 1))]
        self.assertEqual(list(pp.Sketch(prims)), prims)


class TestSampling(unittest.TestCase):
    def test_line_endpoints_included(self):
        pts = pp.sample_primitive(pp.line((0, 0), (10, 0)), n=11)
        self.assertEqual(len(pts), 11)
        self.assertEqual(pts[0], (0, 0))
        self.assertAlmostEqual(pts[-1][0], 10.0)
        self.assertAlmostEqual(pts[5][0], 5.0)

    def test_circle_points_on_radius(self):
        pts = pp.sample_primitive(pp.circle((0, 0), 3.0), n=8)
        self.assertEqual(len(pts), 8)
        for x, y in pts:
            self.assertAlmostEqual(math.hypot(x, y), 3.0)

    def test_arc_passes_through_midpoint(self):
        p1, p2, p3 = (1, 0), (0, 1), (-1, 0)  # upper semicircle, radius 1
        pts = pp.sample_primitive(pp.arc(p1, p2, p3), n=32)
        # every sample lies on the unit circle centred at origin
        for x, y in pts:
            self.assertAlmostEqual(math.hypot(x, y), 1.0, places=6)
        # arc stays in the upper half-plane (through (0,1)), never dips below
        self.assertTrue(all(y >= -1e-9 for _, y in pts))

    def test_point_single_sample(self):
        self.assertEqual(pp.sample_primitive(pp.point((4, 5))), ((4, 5),))

    def test_circumcircle(self):
        c, r = pp.circumcircle((1, 0), (0, 1), (-1, 0))
        self.assertAlmostEqual(c[0], 0.0)
        self.assertAlmostEqual(c[1], 0.0)
        self.assertAlmostEqual(r, 1.0)
        self.assertEqual(pp.circumcircle((0, 0), (1, 0), (2, 0))[0], None)


if __name__ == "__main__":
    unittest.main()
