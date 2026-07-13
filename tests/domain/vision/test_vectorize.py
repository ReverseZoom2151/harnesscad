"""Tests for vision.rastercad_vectorize."""

from __future__ import annotations

import math
import unittest

from harnesscad.domain.vision.vectorize import (
    ArcFit,
    CircleFit,
    LineFit,
    classify_component,
    connected_components,
    fit_circle,
    fit_line,
    vectorize,
)


def _blank(h: int, w: int) -> list[list[int]]:
    return [[0] * w for _ in range(h)]


class TestConnectedComponents(unittest.TestCase):
    def test_empty(self) -> None:
        self.assertEqual(connected_components(_blank(4, 4)), [])
        self.assertEqual(connected_components([]), [])

    def test_single_component(self) -> None:
        g = _blank(5, 5)
        g[1][1] = g[1][2] = g[2][2] = 1
        comps = connected_components(g, connectivity=4)
        self.assertEqual(len(comps), 1)
        self.assertEqual(set(comps[0]), {(1, 1), (1, 2), (2, 2)})

    def test_two_separate_components(self) -> None:
        g = _blank(5, 7)
        g[0][0] = 1
        g[4][6] = 1
        comps = connected_components(g, connectivity=8)
        self.assertEqual(len(comps), 2)

    def test_diagonal_needs_8_connectivity(self) -> None:
        g = _blank(3, 3)
        g[0][0] = 1
        g[1][1] = 1
        self.assertEqual(len(connected_components(g, connectivity=4)), 2)
        self.assertEqual(len(connected_components(g, connectivity=8)), 1)

    def test_bad_connectivity(self) -> None:
        with self.assertRaises(ValueError):
            connected_components(_blank(2, 2), connectivity=6)

    def test_deterministic_order(self) -> None:
        g = _blank(4, 4)
        g[3][3] = 1
        g[0][0] = 1
        comps = connected_components(g)
        # Row-major scan: (0,0) component discovered before (3,3).
        self.assertEqual(comps[0][0], (0, 0))
        self.assertEqual(comps[1][0], (3, 3))


class TestFitLine(unittest.TestCase):
    def test_horizontal_line(self) -> None:
        pts = [(0.0, 0.5), (0.25, 0.5), (0.5, 0.5), (1.0, 0.5)]
        fit = fit_line(pts)
        self.assertIsInstance(fit, LineFit)
        self.assertAlmostEqual(fit.residual, 0.0, places=9)
        xs = sorted([fit.start[0], fit.end[0]])
        self.assertAlmostEqual(xs[0], 0.0, places=9)
        self.assertAlmostEqual(xs[1], 1.0, places=9)
        self.assertAlmostEqual(fit.start[1], 0.5, places=9)
        self.assertAlmostEqual(fit.end[1], 0.5, places=9)

    def test_diagonal_line_zero_residual(self) -> None:
        pts = [(0.0, 0.0), (0.2, 0.2), (0.4, 0.4), (0.9, 0.9)]
        fit = fit_line(pts)
        self.assertAlmostEqual(fit.residual, 0.0, places=9)

    def test_single_point(self) -> None:
        fit = fit_line([(0.3, 0.7)])
        self.assertEqual(fit.start, fit.end)

    def test_empty_raises(self) -> None:
        with self.assertRaises(ValueError):
            fit_line([])


class TestFitCircle(unittest.TestCase):
    def test_unit_circle_recovered(self) -> None:
        cx, cy, r = 0.5, 0.5, 0.3
        pts = [
            (cx + r * math.cos(t), cy + r * math.sin(t))
            for t in [i * math.pi / 8 for i in range(16)]
        ]
        fit = fit_circle(pts)
        self.assertIsNotNone(fit)
        assert fit is not None
        self.assertAlmostEqual(fit.center[0], cx, places=6)
        self.assertAlmostEqual(fit.center[1], cy, places=6)
        self.assertAlmostEqual(fit.radius, r, places=6)
        self.assertLess(fit.residual, 1e-6)

    def test_too_few_points(self) -> None:
        self.assertIsNone(fit_circle([(0.0, 0.0), (1.0, 1.0)]))

    def test_collinear_returns_none(self) -> None:
        pts = [(0.0, 0.0), (0.1, 0.1), (0.2, 0.2), (0.3, 0.3)]
        self.assertIsNone(fit_circle(pts))


class TestClassifyAndVectorize(unittest.TestCase):
    def test_line_component_classified_as_line(self) -> None:
        g = _blank(16, 16)
        for c in range(2, 14):
            g[8][c] = 1
        prims = vectorize(g)
        self.assertEqual(len(prims), 1)
        self.assertIsInstance(prims[0], LineFit)

    def test_circle_component_classified_as_circle(self) -> None:
        g = _blank(32, 32)
        cx, cy, r = 15.5, 15.5, 10.0
        for k in range(720):
            t = k * math.pi / 360.0
            rr = int(round(cy + r * math.sin(t)))
            cc = int(round(cx + r * math.cos(t)))
            if 0 <= rr < 32 and 0 <= cc < 32:
                g[rr][cc] = 1
        prims = vectorize(g)
        self.assertEqual(len(prims), 1)
        self.assertIsInstance(prims[0], CircleFit)
        prim = prims[0]
        assert isinstance(prim, CircleFit)
        # Center should be near the middle of the normalised canvas.
        self.assertAlmostEqual(prim.center[0], 0.5, places=1)
        self.assertAlmostEqual(prim.center[1], 0.5, places=1)

    def test_arc_component_classified_as_arc(self) -> None:
        g = _blank(32, 32)
        cx, cy, r = 15.5, 15.5, 10.0
        # Quarter arc only (0..90 degrees) -> open arc, low coverage.
        for k in range(200):
            t = k * (math.pi / 2.0) / 200.0
            rr = int(round(cy + r * math.sin(t)))
            cc = int(round(cx + r * math.cos(t)))
            if 0 <= rr < 32 and 0 <= cc < 32:
                g[rr][cc] = 1
        prims = vectorize(g)
        self.assertEqual(len(prims), 1)
        self.assertIsInstance(prims[0], ArcFit)

    def test_two_lines_two_primitives(self) -> None:
        g = _blank(20, 20)
        for c in range(2, 18):
            g[3][c] = 1
        for c in range(2, 18):
            g[16][c] = 1
        prims = vectorize(g)
        self.assertEqual(len(prims), 2)
        self.assertTrue(all(isinstance(p, LineFit) for p in prims))

    def test_min_size_filters_noise(self) -> None:
        g = _blank(10, 10)
        g[0][0] = 1  # single-pixel noise
        for c in range(2, 8):
            g[5][c] = 1
        prims = vectorize(g, min_size=2)
        self.assertEqual(len(prims), 1)

    def test_empty_canvas(self) -> None:
        self.assertEqual(vectorize(_blank(8, 8)), [])

    def test_classify_empty_raises(self) -> None:
        with self.assertRaises(ValueError):
            classify_component([], 8, 8)


if __name__ == "__main__":
    unittest.main()
