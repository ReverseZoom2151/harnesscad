"""Tests for bench.magic3d_silhouette_iou (Magic3DSketch Eq. 1 and Eq. 2)."""

import unittest

from harnesscad.eval.bench.vision.silhouette_iou import (
    soft_intersection,
    soft_union,
    soft_iou,
    iou_loss,
    downsample,
    multiscale_iou_loss,
)


class SoftIoUTest(unittest.TestCase):
    def test_identical_binary_masks(self):
        m = [[1, 0, 1], [0, 1, 0]]
        self.assertAlmostEqual(soft_iou(m, m), 1.0)
        self.assertAlmostEqual(iou_loss(m, m), 0.0)

    def test_disjoint_binary_masks_zero_iou(self):
        # Note from prompt: soft-IoU is 0 exactly for disjoint BINARY masks.
        a = [[1, 0], [1, 0]]
        b = [[0, 1], [0, 1]]
        self.assertEqual(soft_intersection(a, b), 0.0)
        self.assertAlmostEqual(soft_iou(a, b), 0.0)
        self.assertAlmostEqual(iou_loss(a, b), 1.0)

    def test_partial_overlap_jaccard(self):
        # A = 3 cells, B = 3 cells, intersection = 2, union = 4 -> 0.5
        a = [[1, 1, 1], [0, 0, 0]]
        b = [[1, 1, 0], [1, 0, 0]]
        self.assertEqual(soft_intersection(a, b), 2.0)
        self.assertEqual(soft_union(a, b), 4.0)
        self.assertAlmostEqual(soft_iou(a, b), 0.5)
        self.assertAlmostEqual(iou_loss(a, b), 0.5)

    def test_soft_values(self):
        a = [[0.5, 0.5]]
        b = [[0.5, 0.5]]
        # inter = 0.5, union = 1.0 + 1.0 - 0.5 = 1.5 -> 1/3
        self.assertAlmostEqual(soft_iou(a, b), 1.0 / 3.0)

    def test_both_empty_is_perfect(self):
        z = [[0, 0], [0, 0]]
        self.assertAlmostEqual(soft_iou(z, z), 1.0)
        self.assertAlmostEqual(iou_loss(z, z), 0.0)

    def test_symmetry(self):
        a = [[1, 0.3], [0.7, 1]]
        b = [[0.2, 1], [1, 0.1]]
        self.assertAlmostEqual(soft_iou(a, b), soft_iou(b, a))

    def test_shape_mismatch_raises(self):
        with self.assertRaises(ValueError):
            soft_iou([[1, 0]], [[1], [0]])

    def test_iou_in_unit_interval(self):
        a = [[0.9, 0.1, 0.4], [0.2, 0.8, 0.6]]
        b = [[0.3, 0.7, 0.5], [0.9, 0.2, 0.1]]
        v = soft_iou(a, b)
        self.assertGreaterEqual(v, 0.0)
        self.assertLessEqual(v, 1.0)


class DownsampleTest(unittest.TestCase):
    def test_factor_one_copies(self):
        m = [[1, 2], [3, 4]]
        d = downsample(m, 1)
        self.assertEqual(d, [[1, 2], [3, 4]])
        d[0][0] = 99
        self.assertEqual(m[0][0], 1)  # copy, not alias

    def test_block_mean(self):
        m = [[1, 1, 2, 2], [1, 1, 2, 2], [3, 3, 4, 4], [3, 3, 4, 4]]
        self.assertEqual(downsample(m, 2), [[1.0, 2.0], [3.0, 4.0]])

    def test_partial_block_uses_actual_area(self):
        m = [[1, 2, 3]]  # 1x3, factor 2 -> block0 mean(1,2)=1.5, block1 mean(3)=3
        self.assertEqual(downsample(m, 2), [[1.5, 3.0]])

    def test_bad_factor(self):
        with self.assertRaises(ValueError):
            downsample([[1]], 0)


class MultiScaleTest(unittest.TestCase):
    def test_single_scale_equals_iou_loss(self):
        a = [[1, 1, 1], [0, 0, 0]]
        b = [[1, 1, 0], [1, 0, 0]]
        self.assertAlmostEqual(
            multiscale_iou_loss(a, b, (1,), (1.0,)), iou_loss(a, b)
        )

    def test_weighted_sum(self):
        a = [[1, 1], [0, 0]]
        b = [[1, 0], [0, 0]]
        # scale 1: inter=1, union=2 -> loss 0.5
        # scale 2: both pool to single cell a=0.5,b=0.25;
        #   inter=0.125 union=0.5+0.25-0.125=0.625 -> iou=0.2 loss=0.8
        expected = 0.6 * 0.5 + 0.4 * 0.8
        self.assertAlmostEqual(
            multiscale_iou_loss(a, b, (1, 2), (0.6, 0.4)), expected
        )

    def test_identical_masks_zero_loss_when_binary_at_each_scale(self):
        # 2x2-uniform blocks stay binary under factor-1 and factor-2 pooling,
        # so identical masks give exactly zero loss at those scales.
        m = [[1, 1, 0, 0], [1, 1, 0, 0], [0, 0, 1, 1], [0, 0, 1, 1]]
        self.assertAlmostEqual(
            multiscale_iou_loss(m, m, (1, 2), (0.6, 0.4)), 0.0
        )

    def test_identical_soft_mask_not_zero_after_pooling(self):
        # Downsampling produces soft (0.5) values whose self-soft-IoU is < 1.
        m = [[1, 0], [0, 0]]
        loss = multiscale_iou_loss(m, m, (2,), (1.0,))
        self.assertGreater(loss, 0.0)

    def test_length_mismatch_raises(self):
        with self.assertRaises(ValueError):
            multiscale_iou_loss([[1]], [[1]], (1, 2), (1.0,))

    def test_empty_scales_raises(self):
        with self.assertRaises(ValueError):
            multiscale_iou_loss([[1]], [[1]], (), ())


if __name__ == "__main__":
    unittest.main()
