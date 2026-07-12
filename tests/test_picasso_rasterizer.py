"""Tests for drawings.picasso_rasterizer."""

from __future__ import annotations

import math
import unittest

from drawings.picasso_rasterizer import (
    Arc,
    Circle,
    Dot,
    Line,
    binarize,
    circumcircle,
    foreground_pixels,
    rasterize,
)


class TestDistanceHelpers(unittest.TestCase):
    def test_circumcircle_unit(self):
        cc = circumcircle((1.0, 0.0), (0.0, 1.0), (-1.0, 0.0))
        self.assertIsNotNone(cc)
        (cx, cy), r = cc
        self.assertAlmostEqual(cx, 0.0, places=6)
        self.assertAlmostEqual(cy, 0.0, places=6)
        self.assertAlmostEqual(r, 1.0, places=6)

    def test_circumcircle_collinear_none(self):
        self.assertIsNone(circumcircle((0.0, 0.0), (1.0, 1.0), (2.0, 2.0)))


class TestRasterizeBasics(unittest.TestCase):
    def test_empty_is_blank(self):
        img = rasterize([], width=16, height=16)
        self.assertEqual(len(img), 16)
        self.assertEqual(len(img[0]), 16)
        self.assertEqual(sum(sum(row) for row in img), 0.0)

    def test_values_in_range(self):
        img = rasterize(
            [Line((0.1, 0.1), (0.9, 0.9)), Circle((0.5, 0.5), 0.3)],
            width=32,
            height=32,
        )
        for row in img:
            for v in row:
                self.assertGreaterEqual(v, 0.0)
                self.assertLessEqual(v, 1.0)

    def test_deterministic(self):
        prims = [Line((0.2, 0.3), (0.8, 0.7)), Arc((0.2, 0.5), (0.5, 0.2), (0.8, 0.5))]
        a = rasterize(prims, width=24, height=24)
        b = rasterize(prims, width=24, height=24)
        self.assertEqual(a, b)

    def test_reject_tiny(self):
        with self.assertRaises(ValueError):
            rasterize([], width=1, height=8)


class TestLineRendering(unittest.TestCase):
    def test_endpoints_are_ink(self):
        img = rasterize([Line((0.0, 0.0), (1.0, 1.0))], width=32, height=32,
                        stroke_width=2.0)
        self.assertGreater(img[0][0], 0.5)
        self.assertGreater(img[31][31], 0.5)

    def test_off_line_is_blank(self):
        # A horizontal line across the middle; a corner should be untouched.
        img = rasterize([Line((0.0, 0.5), (1.0, 0.5))], width=32, height=32,
                        stroke_width=1.5, aa=1.0)
        self.assertEqual(img[0][0], 0.0)
        # The middle row is inked.
        mid = 16
        self.assertGreater(max(img[mid]), 0.9)

    def test_diagonal_covers_diagonal(self):
        img = rasterize([Line((0.0, 0.0), (1.0, 1.0))], width=20, height=20,
                        stroke_width=2.0)
        for i in range(20):
            # The pixel on the diagonal should be strongly inked.
            self.assertGreater(img[i][i], 0.5)


class TestCircleRendering(unittest.TestCase):
    def test_center_is_blank_ring_is_ink(self):
        img = rasterize([Circle((0.5, 0.5), 0.3)], width=40, height=40,
                        stroke_width=1.5)
        # Centre pixel far from the ring -> blank.
        self.assertEqual(img[20][20], 0.0)
        # A pixel on the ring (radius 0.3 -> ~11.7px from centre 19.5).
        r_px = 0.3 * 39.0
        cx = 0.5 * 39.0
        col = int(round(cx + r_px))
        self.assertGreater(img[20][col] if col < 40 else 0.0, 0.4)

    def test_symmetry(self):
        img = rasterize([Circle((0.5, 0.5), 0.35)], width=41, height=41)
        # Left/right symmetric about column 20.
        for y in range(41):
            for x in range(20):
                self.assertAlmostEqual(img[y][x], img[y][40 - x], places=6)


class TestArcRendering(unittest.TestCase):
    def test_arc_passes_through_mid(self):
        arc = Arc((0.2, 0.5), (0.5, 0.2), (0.8, 0.5))
        img = rasterize([arc], width=48, height=48, stroke_width=2.0)
        # Mid point (0.5, 0.2) should be inked.
        mx, my = int(round(0.5 * 47)), int(round(0.2 * 47))
        self.assertGreater(img[my][mx], 0.4)

    def test_arc_excludes_far_side(self):
        # Upper semicircle: the point diametrically opposite mid (below) is off-arc.
        arc = Arc((0.2, 0.5), (0.5, 0.2), (0.8, 0.5))
        img = rasterize([arc], width=48, height=48, stroke_width=2.0)
        # A point on the *lower* half of that circle should be blank.
        # Circle centre ~ (0.5, 0.5) canvas; lower point (0.5, 0.8).
        lx, ly = int(round(0.5 * 47)), int(round(0.8 * 47))
        self.assertEqual(img[ly][lx], 0.0)


class TestDotAndUtilities(unittest.TestCase):
    def test_dot_renders(self):
        img = rasterize([Dot((0.5, 0.5))], width=21, height=21, stroke_width=3.0)
        self.assertGreater(img[10][10], 0.9)
        self.assertEqual(img[0][0], 0.0)

    def test_binarize_and_foreground(self):
        img = rasterize([Line((0.0, 0.5), (1.0, 0.5))], width=16, height=16)
        b = binarize(img, threshold=0.5)
        fg = foreground_pixels(img, threshold=0.5)
        self.assertEqual(sum(sum(r) for r in b), len(fg))
        self.assertTrue(all(0 <= y < 16 and 0 <= x < 16 for y, x in fg))
        self.assertGreater(len(fg), 0)

    def test_max_union(self):
        # Union coverage is at least each individual coverage.
        p1 = [Line((0.0, 0.5), (1.0, 0.5))]
        p2 = [Line((0.5, 0.0), (0.5, 1.0))]
        a = rasterize(p1, width=24, height=24)
        b = rasterize(p2, width=24, height=24)
        both = rasterize(p1 + p2, width=24, height=24)
        for y in range(24):
            for x in range(24):
                self.assertGreaterEqual(both[y][x] + 1e-9, max(a[y][x], b[y][x]))


if __name__ == "__main__":
    unittest.main()
