"""Tests for deterministic least-squares PPA primitive fitting."""

import math
import unittest

from harnesscad.domain.reconstruction.sketch import primitives as pp
from harnesscad.domain.geometry.sketch import primitive_fit as fit


class TestFitPoint(unittest.TestCase):
    def test_centroid(self):
        prim, res = fit.fit_point([(0, 0), (2, 0), (1, 3)])
        self.assertEqual(prim.ptype, pp.POINT)
        self.assertAlmostEqual(prim.params[0], 1.0)
        self.assertAlmostEqual(prim.params[1], 1.0)
        self.assertGreater(res, 0.0)

    def test_empty_raises(self):
        with self.assertRaises(ValueError):
            fit.fit_point([])


class TestFitLine(unittest.TestCase):
    def test_horizontal_line(self):
        pts = [(x, 5.0) for x in range(6)]
        prim, res = fit.fit_line(pts)
        self.assertEqual(prim.ptype, pp.LINE)
        self.assertLess(res, 1e-9)
        (x1, y1), (x2, y2) = prim.control_points()
        self.assertAlmostEqual(y1, 5.0)
        self.assertAlmostEqual(y2, 5.0)
        self.assertAlmostEqual(min(x1, x2), 0.0)
        self.assertAlmostEqual(max(x1, x2), 5.0)

    def test_vertical_line(self):
        pts = [(3.0, y) for y in range(6)]
        prim, res = fit.fit_line(pts)
        self.assertLess(res, 1e-9)
        for (x, _) in prim.control_points():
            self.assertAlmostEqual(x, 3.0)

    def test_diagonal_residual_small(self):
        pts = [(t, 2 * t + 1) for t in range(10)]
        _, res = fit.fit_line(pts)
        self.assertLess(res, 1e-9)

    def test_noisy_line_residual_positive(self):
        pts = [(0, 0), (1, 0.1), (2, -0.1), (3, 0.05)]
        _, res = fit.fit_line(pts)
        self.assertGreater(res, 0.0)

    def test_too_few(self):
        with self.assertRaises(ValueError):
            fit.fit_line([(0, 0)])


class TestFitCircle(unittest.TestCase):
    def test_recover_unit_circle(self):
        pts = [(math.cos(a), math.sin(a)) for a in
               [i * math.pi / 6 for i in range(12)]]
        prim, res = fit.fit_circle(pts)
        self.assertEqual(prim.ptype, pp.CIRCLE)
        cx, cy = prim.control_points()[0]
        self.assertAlmostEqual(cx, 0.0, places=6)
        self.assertAlmostEqual(cy, 0.0, places=6)
        self.assertAlmostEqual(prim.radius, 1.0, places=6)
        self.assertLess(res, 1e-6)

    def test_offset_circle(self):
        cx0, cy0, r0 = 3.0, -2.0, 4.0
        pts = [(cx0 + r0 * math.cos(a), cy0 + r0 * math.sin(a))
               for a in [i * math.pi / 8 for i in range(16)]]
        prim, _ = fit.fit_circle(pts)
        cx, cy = prim.control_points()[0]
        self.assertAlmostEqual(cx, cx0, places=6)
        self.assertAlmostEqual(cy, cy0, places=6)
        self.assertAlmostEqual(prim.radius, r0, places=6)

    def test_too_few(self):
        with self.assertRaises(ValueError):
            fit.fit_circle([(0, 0), (1, 1)])


class TestFitArc(unittest.TestCase):
    def test_semicircle_arc_on_circle(self):
        # upper half of unit circle
        pts = [(math.cos(a), math.sin(a)) for a in
               [i * math.pi / 10 for i in range(11)]]  # 0 .. pi
        prim, res = fit.fit_arc(pts)
        self.assertEqual(prim.ptype, pp.ARC)
        self.assertLess(res, 1e-6)
        # all three control points on the unit circle
        for (x, y) in prim.control_points():
            self.assertAlmostEqual(math.hypot(x, y), 1.0, places=6)
        # midpoint should be near the top (0, 1)
        mid = prim.control_points()[1]
        self.assertAlmostEqual(mid[0], 0.0, places=5)
        self.assertAlmostEqual(mid[1], 1.0, places=5)

    def test_arc_samples_stay_on_arc(self):
        pts = [(math.cos(a), math.sin(a)) for a in
               [i * math.pi / 10 for i in range(11)]]
        prim, _ = fit.fit_arc(pts)
        sampled = pp.sample_primitive(prim, n=40)
        for (x, y) in sampled:
            self.assertAlmostEqual(math.hypot(x, y), 1.0, places=5)
            self.assertGreaterEqual(y, -1e-5)  # stays in upper half


class TestFitBest(unittest.TestCase):
    def test_prefers_line_for_collinear(self):
        pts = [(t, t) for t in range(5)]
        prim, res = fit.fit_best(pts)
        self.assertEqual(prim.ptype, pp.LINE)
        self.assertLess(res, 1e-9)

    def test_prefers_circle_for_ring(self):
        pts = [(math.cos(a), math.sin(a)) for a in
               [i * math.pi / 6 for i in range(12)]]
        prim, res = fit.fit_best(pts)
        self.assertIn(prim.ptype, (pp.CIRCLE, pp.ARC))
        self.assertLess(res, 1e-6)

    def test_single_point(self):
        prim, _ = fit.fit_best([(2, 2)])
        self.assertEqual(prim.ptype, pp.POINT)

    def test_deterministic(self):
        pts = [(t, 3 * t) for t in range(6)]
        self.assertEqual(fit.fit_best(pts)[0], fit.fit_best(pts)[0])


if __name__ == "__main__":
    unittest.main()
