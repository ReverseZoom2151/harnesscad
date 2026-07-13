"""Tests for geometry.shapeit_transforms."""

import math
import unittest

from harnesscad.domain.geometry.features.shapeit_heightfield import HeightField
from harnesscad.domain.geometry.features import shapeit_transforms as tf


def _single(rows, cols, r, c, val=1.0):
    hf = HeightField(rows, cols, 0.0, 1.0)
    hf.set(r, c, val)
    return hf


class TestTranslate(unittest.TestCase):
    def test_shift_right_down(self):
        hf = _single(5, 5, 1, 1)
        out = tf.translate(hf, d_row=1, d_col=2)
        self.assertEqual(out.get(2, 3), 1.0)
        self.assertEqual(out.get(1, 1), 0.0)
        self.assertEqual(out.raised_cells(), 1)

    def test_shift_off_grid_reads_floor(self):
        hf = _single(3, 3, 0, 0)
        out = tf.translate(hf, d_row=-1, d_col=0)
        self.assertEqual(out.raised_cells(), 0)

    def test_source_unchanged(self):
        hf = _single(3, 3, 1, 1)
        tf.translate(hf, 1, 1)
        self.assertEqual(hf.get(1, 1), 1.0)


class TestScale(unittest.TestCase):
    def test_shrink_keeps_center(self):
        hf = HeightField(5, 5, 0.0, 1.0)
        hf.fill(1.0)
        out = tf.scale(hf, 0.5)
        # center stays raised
        self.assertEqual(out.get(2, 2), 1.0)

    def test_enlarge_spreads(self):
        hf = HeightField(5, 5, 0.0, 1.0)
        hf.set(2, 2, 1.0)  # centre pin
        out = tf.scale(hf, 3.0)
        # enlarging a single central pin covers neighbours
        self.assertGreater(out.raised_cells(), 1)

    def test_bad_factor(self):
        hf = HeightField(3, 3)
        with self.assertRaises(ValueError):
            tf.scale(hf, 0.0)

    def test_per_axis(self):
        hf = HeightField(5, 5, 0.0, 1.0)
        hf.set(2, 2, 1.0)
        out = tf.scale(hf, factor_row=1.0, factor_col=3.0)
        # stretched horizontally only
        bb = out.bounding_box()
        self.assertIsNotNone(bb)
        min_r, min_c, max_r, max_c = bb
        self.assertGreater(max_c - min_c, max_r - min_r)


class TestRotate(unittest.TestCase):
    def test_rotate_90_matches_rotate90(self):
        hf = HeightField(5, 5, 0.0, 1.0)
        hf.set(0, 2, 1.0)
        a = tf.rotate(hf, math.pi / 2.0)
        b = tf.rotate90(hf, 1)
        self.assertEqual(a.to_rows(), b.to_rows())

    def test_full_turn_identity(self):
        hf = HeightField(5, 5, 0.0, 1.0)
        hf.set(1, 3, 1.0)
        out = tf.rotate(hf, 2.0 * math.pi)
        self.assertEqual(out.to_rows(), hf.to_rows())


class TestRotate90(unittest.TestCase):
    def test_ccw_corner(self):
        # 2x3 grid, mark top-left; CCW 90 -> bottom-left of a 3x2 grid
        hf = HeightField.from_rows([[1.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
        out = tf.rotate90(hf, 1)
        self.assertEqual(out.rows, 3)
        self.assertEqual(out.cols, 2)
        self.assertEqual(out.get(2, 0), 1.0)

    def test_four_rotations_identity(self):
        hf = HeightField.from_rows([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]])
        out = hf
        for _ in range(4):
            out = tf.rotate90(out, 1)
        self.assertEqual(out.to_rows(), hf.to_rows())

    def test_negative_k(self):
        hf = HeightField.from_rows([[1.0, 0.0], [0.0, 0.0]])
        self.assertEqual(
            tf.rotate90(hf, -1).to_rows(), tf.rotate90(hf, 3).to_rows()
        )

    def test_180(self):
        hf = HeightField.from_rows([[1.0, 0.0], [0.0, 0.0]])
        out = tf.rotate90(hf, 2)
        self.assertEqual(out.get(1, 1), 1.0)


class TestAmplitude(unittest.TestCase):
    def test_gain_amplifies(self):
        hf = HeightField(1, 2, 0.0, 10.0, [1.0, 2.0])
        out = tf.scale_amplitude(hf, 2.0)
        self.assertEqual(out.get(0, 0), 2.0)
        self.assertEqual(out.get(0, 1), 4.0)

    def test_gain_about_floor(self):
        hf = HeightField(1, 1, 2.0, 10.0, [4.0])
        out = tf.scale_amplitude(hf, 0.5)
        # floor 2 + 0.5*(4-2) = 3
        self.assertEqual(out.get(0, 0), 3.0)

    def test_gain_clamped(self):
        hf = HeightField(1, 1, 0.0, 1.0, [0.5])
        out = tf.scale_amplitude(hf, 100.0)
        self.assertEqual(out.get(0, 0), 1.0)

    def test_negative_gain(self):
        hf = HeightField(1, 1)
        with self.assertRaises(ValueError):
            tf.scale_amplitude(hf, -1.0)


class TestVerticalOffset(unittest.TestCase):
    def test_raise(self):
        hf = HeightField(1, 2, 0.0, 1.0, [0.2, 0.3])
        out = tf.vertical_offset(hf, 0.4)
        self.assertAlmostEqual(out.get(0, 0), 0.6)
        self.assertAlmostEqual(out.get(0, 1), 0.7)

    def test_lower_clamps(self):
        hf = HeightField(1, 1, 0.0, 1.0, [0.2])
        out = tf.vertical_offset(hf, -1.0)
        self.assertEqual(out.get(0, 0), 0.0)


if __name__ == "__main__":
    unittest.main()
