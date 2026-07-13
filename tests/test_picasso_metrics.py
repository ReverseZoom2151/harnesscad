"""Tests for drawings.picasso_metrics."""

from __future__ import annotations

import math
import unittest

from harnesscad.domain.drawings.picasso_rasterizer import Circle, Line, rasterize
from harnesscad.domain.drawings.picasso_metrics import (
    chamfer_distance,
    foreground_iou,
    img_mse,
    pixel_accuracy,
    render_eval,
)


def _const(h, w, v):
    return [[v for _ in range(w)] for _ in range(h)]


class TestImgMse(unittest.TestCase):
    def test_zero_identical(self):
        img = rasterize([Line((0.1, 0.1), (0.9, 0.9))], 20, 20)
        self.assertAlmostEqual(img_mse(img, img), 0.0)

    def test_two_terms(self):
        # Target has a single foreground pixel; pred all zeros.
        target = [[0.0, 0.0], [0.0, 1.0]]
        pred = _const(2, 2, 0.0)
        # fg term: 1 fg pixel, err^2=1 -> 1/(2*1)=0.5
        # global term: sum err^2=1 over 4 px -> 1/(2*4)=0.125
        self.assertAlmostEqual(img_mse(pred, target), 0.625)

    def test_no_foreground(self):
        target = _const(3, 3, 0.0)
        pred = [[0.0, 0.0, 0.0], [0.0, 0.6, 0.0], [0.0, 0.0, 0.0]]
        # fg term 0; global term = 0.36 / (2*9)
        self.assertAlmostEqual(img_mse(pred, target), 0.36 / 18.0)

    def test_worse_pred_higher(self):
        target = rasterize([Circle((0.5, 0.5), 0.3)], 24, 24)
        good = rasterize([Circle((0.5, 0.5), 0.31)], 24, 24)
        bad = rasterize([Circle((0.2, 0.2), 0.1)], 24, 24)
        self.assertLess(img_mse(good, target), img_mse(bad, target))


class TestChamfer(unittest.TestCase):
    def test_zero_identical(self):
        img = rasterize([Line((0.2, 0.2), (0.8, 0.8))], 24, 24)
        self.assertAlmostEqual(chamfer_distance(img, img), 0.0)

    def test_both_empty(self):
        self.assertEqual(chamfer_distance(_const(4, 4, 0.0), _const(4, 4, 0.0)), 0.0)

    def test_one_empty_inf(self):
        a = rasterize([Line((0.0, 0.5), (1.0, 0.5))], 16, 16)
        b = _const(16, 16, 0.0)
        self.assertEqual(chamfer_distance(a, b), math.inf)

    def test_shifted_positive(self):
        a = rasterize([Line((0.0, 0.4), (1.0, 0.4))], 24, 24)
        b = rasterize([Line((0.0, 0.6), (1.0, 0.6))], 24, 24)
        self.assertGreater(chamfer_distance(a, b), 0.0)

    def test_symmetric(self):
        a = rasterize([Circle((0.5, 0.5), 0.3)], 24, 24)
        b = rasterize([Circle((0.55, 0.5), 0.3)], 24, 24)
        self.assertAlmostEqual(chamfer_distance(a, b), chamfer_distance(b, a), places=9)


class TestAggregates(unittest.TestCase):
    def test_pixel_accuracy_perfect(self):
        img = rasterize([Line((0.1, 0.1), (0.9, 0.9))], 20, 20)
        self.assertEqual(pixel_accuracy(img, img), 1.0)

    def test_pixel_accuracy_range(self):
        a = rasterize([Line((0.0, 0.0), (0.5, 0.0))], 16, 16)
        b = rasterize([Line((0.5, 1.0), (1.0, 1.0))], 16, 16)
        acc = pixel_accuracy(a, b)
        self.assertTrue(0.0 <= acc <= 1.0)

    def test_foreground_iou_identical(self):
        img = rasterize([Circle((0.5, 0.5), 0.3)], 24, 24)
        self.assertEqual(foreground_iou(img, img), 1.0)

    def test_foreground_iou_both_empty(self):
        self.assertEqual(foreground_iou(_const(4, 4, 0.0), _const(4, 4, 0.0)), 1.0)

    def test_render_eval_keys(self):
        a = rasterize([Line((0.1, 0.5), (0.9, 0.5))], 16, 16)
        b = rasterize([Line((0.1, 0.6), (0.9, 0.6))], 16, 16)
        rep = render_eval(a, b)
        self.assertEqual(
            set(rep), {"img_mse", "chamfer", "pixel_accuracy", "foreground_iou"}
        )
        for v in rep.values():
            self.assertIsInstance(v, float)


if __name__ == "__main__":
    unittest.main()
