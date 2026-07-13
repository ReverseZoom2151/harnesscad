"""Tests for PHT-CAD Efficient Hybrid Parametrization (EHP)."""

import math
import unittest

from harnesscad.domain.reconstruction.tokens import pht_ehp as ehp


class NormaliseCoordTest(unittest.TestCase):
    def test_endpoints(self):
        self.assertEqual(ehp.normalise_coord(0.0, 0.0, 10.0), 0.0)
        self.assertEqual(ehp.normalise_coord(5.0, 0.0, 10.0), 500.0)

    def test_upper_bound_exclusive(self):
        v = ehp.normalise_coord(10.0, 0.0, 10.0)
        self.assertLess(v, ehp.NORM_RANGE)
        self.assertGreater(v, 999.0)

    def test_clamp_and_degenerate(self):
        self.assertEqual(ehp.normalise_coord(-5.0, 0.0, 10.0), 0.0)
        self.assertEqual(ehp.normalise_coord(3.0, 5.0, 5.0), 0.0)


class LineTest(unittest.TestCase):
    def test_direction_inferred(self):
        l = ehp.make_line(0, 0, 10, 0)
        self.assertAlmostEqual(l.direction_deg(), 0.0)
        self.assertAlmostEqual(l.length(), 10.0)

    def test_direction_diagonal(self):
        l = ehp.make_line(0, 0, 5, 5)
        self.assertAlmostEqual(l.direction_deg(), 45.0)

    def test_validity_and_tokens(self):
        l = ehp.make_line(1, 2, 3, 4, v=0)
        self.assertEqual(l.tokens(), [1.0, 2.0, 3.0, 4.0, 0.0])

    def test_zero_length_rejected(self):
        with self.assertRaises(ValueError):
            ehp.make_line(1, 1, 1, 1)

    def test_bad_validity_rejected(self):
        with self.assertRaises(ValueError):
            ehp.make_line(0, 0, 1, 1, v=2)


class CircleArcTest(unittest.TestCase):
    def test_circle_positive_radius(self):
        c = ehp.make_circle(5, 5, 2)
        self.assertEqual(c.tokens(), [5.0, 5.0, 2.0])
        with self.assertRaises(ValueError):
            ehp.make_circle(0, 0, 0)

    def test_arc_sweep(self):
        a = ehp.make_arc(0, 0, 1, 0, 90)
        self.assertAlmostEqual(a.sweep_deg(), 90.0)

    def test_arc_endpoints(self):
        a = ehp.make_arc(0, 0, 2, 0, 90)
        sx, sy = a.endpoint_start()
        ex, ey = a.endpoint_end()
        self.assertAlmostEqual(sx, 2.0)
        self.assertAlmostEqual(sy, 0.0)
        self.assertAlmostEqual(ex, 0.0, places=6)
        self.assertAlmostEqual(ey, 2.0)

    def test_arc_full_sweep(self):
        a = ehp.Arc(0, 0, 1, 10, 10)
        self.assertEqual(a.sweep_deg(), 360.0)

    def test_arc_empty_sweep_rejected(self):
        with self.assertRaises(ValueError):
            ehp.make_arc(0, 0, 1, 30, 30)


class EfficiencyTest(unittest.TestCase):
    def test_field_counts(self):
        prims = [ehp.Point(0, 0), ehp.make_line(0, 0, 1, 0),
                 ehp.make_circle(0, 0, 1), ehp.make_arc(0, 0, 1, 0, 90)]
        # EHP: 2 + 5 + 3 + 5 = 15
        self.assertEqual(ehp.field_count(prims), 15)
        # baseline: 2 + 10 + 9 + 11 = 32
        self.assertEqual(ehp.baseline_field_count(prims), 32)

    def test_efficiency_report(self):
        prims = [ehp.make_line(0, 0, 1, 0)]
        rep = ehp.efficiency(prims)
        self.assertEqual(rep.ehp_fields, 5)
        self.assertEqual(rep.baseline_fields, 10)
        self.assertEqual(rep.saved_fields, 5)
        self.assertAlmostEqual(rep.compression_ratio, 0.5)
        self.assertAlmostEqual(rep.reduction, 0.5)

    def test_empty(self):
        rep = ehp.efficiency([])
        self.assertEqual(rep.reduction, 0.0)
        self.assertEqual(rep.compression_ratio, 0.0)


class TokensTest(unittest.TestCase):
    def test_flatten(self):
        prims = [ehp.Point(1, 2), ehp.make_circle(3, 4, 5)]
        self.assertEqual(ehp.to_tokens(prims), [1.0, 2.0, 3.0, 4.0, 5.0])


if __name__ == "__main__":
    unittest.main()
