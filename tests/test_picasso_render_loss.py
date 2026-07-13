"""Tests for drawings.picasso_render_loss."""

from __future__ import annotations

import unittest

from harnesscad.domain.drawings.picasso_rasterizer import Circle, Line, rasterize
from harnesscad.domain.drawings.picasso_render_loss import (
    bce_loss,
    distance_field_l2,
    distance_transform,
    downsample,
    image_pyramid,
    iou_loss,
    l2_loss,
    mse,
    multiscale_l2_loss,
    raster_iou,
)


def _const(h, w, v):
    return [[v for _ in range(w)] for _ in range(h)]


class TestPyramid(unittest.TestCase):
    def test_downsample_halves(self):
        img = _const(8, 8, 1.0)
        d = downsample(img)
        self.assertEqual(len(d), 4)
        self.assertEqual(len(d[0]), 4)
        self.assertEqual(d[0][0], 1.0)

    def test_downsample_averages(self):
        img = [[0.0, 1.0], [1.0, 0.0]]
        d = downsample(img)
        self.assertEqual(d, [[0.5]])

    def test_pyramid_levels(self):
        img = _const(16, 16, 0.5)
        pyr = image_pyramid(img, levels=5)
        self.assertEqual(len(pyr), 5)
        self.assertEqual((len(pyr[0]), len(pyr[0][0])), (16, 16))
        self.assertEqual((len(pyr[4]), len(pyr[4][0])), (1, 1))

    def test_pyramid_stops_early(self):
        img = _const(4, 4, 1.0)
        pyr = image_pyramid(img, levels=8)
        # 4 -> 2 -> 1, cannot go further.
        self.assertEqual(len(pyr), 3)


class TestSingleScaleLosses(unittest.TestCase):
    def test_l2_zero_identical(self):
        img = _const(6, 6, 0.7)
        self.assertEqual(l2_loss(img, img), 0.0)

    def test_l2_value(self):
        a = _const(2, 2, 1.0)
        b = _const(2, 2, 0.0)
        self.assertEqual(l2_loss(a, b), 4.0)
        self.assertEqual(mse(a, b), 1.0)

    def test_shape_mismatch(self):
        with self.assertRaises(ValueError):
            l2_loss(_const(2, 2, 0.0), _const(2, 3, 0.0))

    def test_bce_low_when_matching(self):
        a = _const(4, 4, 0.9)
        low = bce_loss(a, _const(4, 4, 1.0))
        high = bce_loss(a, _const(4, 4, 0.0))
        self.assertLess(low, high)


class TestMultiscale(unittest.TestCase):
    def test_zero_identical(self):
        img = rasterize([Line((0.1, 0.1), (0.9, 0.9))], width=16, height=16)
        self.assertAlmostEqual(multiscale_l2_loss(img, img), 0.0)

    def test_monotone_with_disagreement(self):
        target = rasterize([Circle((0.5, 0.5), 0.3)], width=32, height=32)
        near = rasterize([Circle((0.52, 0.5), 0.3)], width=32, height=32)
        far = rasterize([Circle((0.2, 0.2), 0.1)], width=32, height=32)
        self.assertLess(
            multiscale_l2_loss(near, target),
            multiscale_l2_loss(far, target),
        )

    def test_greater_than_single_scale(self):
        target = rasterize([Line((0.0, 0.5), (1.0, 0.5))], width=16, height=16)
        pred = rasterize([Line((0.0, 0.6), (1.0, 0.6))], width=16, height=16)
        ms = multiscale_l2_loss(pred, target)
        single = l2_loss(pred, target)
        self.assertGreaterEqual(ms, single)


class TestIoU(unittest.TestCase):
    def test_identical_iou_one(self):
        img = rasterize([Line((0.1, 0.1), (0.9, 0.9))], width=24, height=24)
        self.assertAlmostEqual(raster_iou(img, img), 1.0)
        self.assertAlmostEqual(iou_loss(img, img), 0.0)

    def test_empty_both(self):
        z = _const(4, 4, 0.0)
        self.assertEqual(raster_iou(z, z), 1.0)

    def test_disjoint_low(self):
        a = rasterize([Line((0.0, 0.0), (0.3, 0.0))], width=32, height=32)
        b = rasterize([Line((0.7, 1.0), (1.0, 1.0))], width=32, height=32)
        self.assertLess(raster_iou(a, b), 0.2)


class TestDistanceField(unittest.TestCase):
    def test_distance_transform_zero_on_fg(self):
        img = [[0.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 0.0]]
        dt = distance_transform(img)
        self.assertEqual(dt[1][1], 0.0)
        self.assertAlmostEqual(dt[0][0], 2.0 ** 0.5)

    def test_distance_transform_empty(self):
        img = _const(3, 3, 0.0)
        dt = distance_transform(img)
        # All equal to grid diagonal.
        self.assertTrue(all(all(v == dt[0][0] for v in row) for row in dt))

    def test_distance_field_l2_zero_identical(self):
        img = rasterize([Circle((0.5, 0.5), 0.3)], width=20, height=20)
        self.assertAlmostEqual(distance_field_l2(img, img), 0.0)

    def test_distance_field_l2_positive(self):
        a = rasterize([Line((0.0, 0.5), (1.0, 0.5))], width=20, height=20)
        b = rasterize([Line((0.0, 0.2), (1.0, 0.2))], width=20, height=20)
        self.assertGreater(distance_field_l2(a, b), 0.0)


if __name__ == "__main__":
    unittest.main()
