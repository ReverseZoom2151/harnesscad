"""Tests for geometry.shapeit_primitives."""

import math
import unittest

from harnesscad.domain.geometry.features.height_field import HeightField
from harnesscad.domain.geometry.features import height_patterns as prim


class TestRectangle(unittest.TestCase):
    def test_fills_region(self):
        hf = HeightField(5, 5, 0.0, 1.0)
        prim.draw_rectangle(hf, top=1, left=1, height_rows=2, width_cols=3, value=0.8)
        self.assertEqual(hf.get(1, 1), 0.8)
        self.assertEqual(hf.get(2, 3), 0.8)
        self.assertEqual(hf.get(0, 0), 0.0)
        self.assertEqual(hf.raised_cells(), 6)

    def test_clips_to_grid(self):
        hf = HeightField(3, 3, 0.0, 1.0)
        prim.draw_rectangle(hf, top=2, left=2, height_rows=5, width_cols=5, value=1.0)
        self.assertEqual(hf.raised_cells(), 1)
        self.assertEqual(hf.get(2, 2), 1.0)

    def test_additive(self):
        hf = HeightField.filled(2, 2, 0.3)
        prim.draw_rectangle(hf, 0, 0, 2, 2, 0.4, additive=True)
        self.assertAlmostEqual(hf.get(0, 0), 0.7)

    def test_negative_extent_raises(self):
        hf = HeightField(2, 2)
        with self.assertRaises(ValueError):
            prim.draw_rectangle(hf, 0, 0, -1, 2, 1.0)


class TestDisc(unittest.TestCase):
    def test_center_and_radius(self):
        hf = HeightField(7, 7, 0.0, 1.0)
        prim.draw_disc(hf, 3, 3, radius=2.0, value=1.0)
        self.assertEqual(hf.get(3, 3), 1.0)
        self.assertEqual(hf.get(3, 5), 1.0)   # distance 2 == radius
        self.assertEqual(hf.get(3, 6), 0.0)   # distance 3 > radius
        # symmetric about the centre
        self.assertEqual(hf.get(1, 3), hf.get(5, 3))

    def test_radius_zero(self):
        hf = HeightField(3, 3, 0.0, 1.0)
        prim.draw_disc(hf, 1, 1, radius=0.0, value=1.0)
        self.assertEqual(hf.raised_cells(), 1)

    def test_negative_radius_raises(self):
        hf = HeightField(3, 3)
        with self.assertRaises(ValueError):
            prim.draw_disc(hf, 1, 1, -1.0, 1.0)


class TestLine(unittest.TestCase):
    def test_horizontal(self):
        hf = HeightField(3, 5, 0.0, 1.0)
        prim.draw_line(hf, 1, 0, 1, 4, value=1.0)
        self.assertEqual([hf.get(1, c) for c in range(5)], [1.0] * 5)

    def test_diagonal(self):
        hf = HeightField(4, 4, 0.0, 1.0)
        prim.draw_line(hf, 0, 0, 3, 3, value=1.0)
        self.assertTrue(all(hf.get(i, i) == 1.0 for i in range(4)))
        self.assertEqual(hf.raised_cells(), 4)

    def test_single_point(self):
        hf = HeightField(3, 3, 0.0, 1.0)
        prim.draw_line(hf, 1, 1, 1, 1, value=1.0)
        self.assertEqual(hf.raised_cells(), 1)


class TestGradient(unittest.TestCase):
    def test_row_ramp(self):
        hf = HeightField(3, 2, 0.0, 1.0)
        prim.draw_linear_gradient(hf, 0.0, 1.0, axis="row")
        self.assertEqual(hf.get(0, 0), 0.0)
        self.assertAlmostEqual(hf.get(1, 0), 0.5)
        self.assertEqual(hf.get(2, 0), 1.0)

    def test_col_ramp(self):
        hf = HeightField(2, 5, 0.0, 1.0)
        prim.draw_linear_gradient(hf, 0.0, 1.0, axis="col")
        self.assertEqual(hf.get(0, 0), 0.0)
        self.assertEqual(hf.get(0, 4), 1.0)
        self.assertAlmostEqual(hf.get(0, 2), 0.5)

    def test_single_col_span(self):
        hf = HeightField(1, 1, 0.0, 1.0)
        prim.draw_linear_gradient(hf, 0.2, 0.9, axis="col")
        self.assertAlmostEqual(hf.get(0, 0), 0.2)

    def test_bad_axis(self):
        hf = HeightField(2, 2)
        with self.assertRaises(ValueError):
            prim.draw_linear_gradient(hf, 0.0, 1.0, axis="diag")


class TestCone(unittest.TestCase):
    def test_peak_and_falloff(self):
        hf = HeightField(7, 7, 0.0, 1.0)
        prim.draw_cone(hf, 3, 3, radius=3.0, peak_value=1.0, base_value=0.0)
        self.assertEqual(hf.get(3, 3), 1.0)
        # distance 3 -> base 0
        self.assertAlmostEqual(hf.get(3, 6), 0.0)
        # monotone decreasing outward along a row
        self.assertGreater(hf.get(3, 3), hf.get(3, 4))
        self.assertGreater(hf.get(3, 4), hf.get(3, 5))

    def test_bad_radius(self):
        hf = HeightField(3, 3)
        with self.assertRaises(ValueError):
            prim.draw_cone(hf, 1, 1, 0.0, 1.0)


class TestWave(unittest.TestCase):
    def test_cosine_values(self):
        hf = HeightField(1, 5, -1.0, 1.0)
        prim.draw_wave(hf, amplitude=1.0, wavelength=4.0, axis="col", offset=0.0)
        # cos(2pi*x/4): x=0 ->1, x=1 ->0, x=2 ->-1, x=3 ->0, x=4 ->1
        self.assertAlmostEqual(hf.get(0, 0), 1.0)
        self.assertAlmostEqual(hf.get(0, 1), 0.0)
        self.assertAlmostEqual(hf.get(0, 2), -1.0)
        self.assertAlmostEqual(hf.get(0, 4), 1.0)

    def test_offset_and_clamp(self):
        hf = HeightField(1, 4, 0.0, 1.0)
        prim.draw_wave(hf, amplitude=0.5, wavelength=4.0, offset=0.5)
        self.assertAlmostEqual(hf.get(0, 0), 1.0)   # 0.5 + 0.5
        self.assertAlmostEqual(hf.get(0, 2), 0.0)   # 0.5 - 0.5

    def test_additive_superposition(self):
        hf = HeightField.filled(1, 4, 0.5)
        hf.min_height = -1.0
        prim.draw_wave(hf, amplitude=0.5, wavelength=4.0, additive=True)
        self.assertAlmostEqual(hf.get(0, 0), 1.0)

    def test_phase_shift(self):
        hf = HeightField(1, 4, -1.0, 1.0)
        prim.draw_wave(hf, 1.0, 4.0, phase=math.pi)
        self.assertAlmostEqual(hf.get(0, 0), -1.0)

    def test_bad_wavelength(self):
        hf = HeightField(2, 2, -1.0, 1.0)
        with self.assertRaises(ValueError):
            prim.draw_wave(hf, 1.0, 0.0)


if __name__ == "__main__":
    unittest.main()
