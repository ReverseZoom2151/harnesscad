"""Tests for geometry.shapeit_heightfield."""

import unittest

from harnesscad.domain.geometry.features import height_field as hfmod
from harnesscad.domain.geometry.features.height_field import HeightField


class TestConstruction(unittest.TestCase):
    def test_default_fill_is_floor(self):
        hf = HeightField(3, 4, min_height=0.0, max_height=1.0)
        self.assertEqual(hf.rows, 3)
        self.assertEqual(hf.cols, 4)
        self.assertEqual(len(hf.heights), 12)
        self.assertTrue(all(h == 0.0 for h in hf.heights))

    def test_rejects_bad_dims(self):
        with self.assertRaises(ValueError):
            HeightField(0, 4)
        with self.assertRaises(ValueError):
            HeightField(4, 0)

    def test_rejects_inverted_stroke(self):
        with self.assertRaises(ValueError):
            HeightField(2, 2, min_height=1.0, max_height=0.0)

    def test_initial_heights_clamped(self):
        hf = HeightField(1, 3, 0.0, 1.0, [-5.0, 0.5, 9.0])
        self.assertEqual(hf.heights, [0.0, 0.5, 1.0])

    def test_wrong_length_raises(self):
        with self.assertRaises(ValueError):
            HeightField(2, 2, heights=[1.0, 2.0, 3.0])

    def test_filled_classmethod(self):
        hf = HeightField.filled(2, 2, 0.7)
        self.assertTrue(all(abs(h - 0.7) < 1e-12 for h in hf.heights))

    def test_from_rows(self):
        hf = HeightField.from_rows([[0.0, 0.2], [0.4, 0.6]])
        self.assertEqual(hf.rows, 2)
        self.assertEqual(hf.cols, 2)
        self.assertEqual(hf.get(1, 0), 0.4)

    def test_from_rows_ragged_raises(self):
        with self.assertRaises(ValueError):
            HeightField.from_rows([[0.0, 0.2], [0.4]])


class TestAccess(unittest.TestCase):
    def test_get_set_clamps(self):
        hf = HeightField(2, 2, 0.0, 1.0)
        hf.set(0, 1, 5.0)
        self.assertEqual(hf.get(0, 1), 1.0)
        hf.set(0, 1, -2.0)
        self.assertEqual(hf.get(0, 1), 0.0)

    def test_add_clamps(self):
        hf = HeightField(1, 1, 0.0, 1.0)
        hf.add(0, 0, 0.4)
        hf.add(0, 0, 0.4)
        self.assertAlmostEqual(hf.get(0, 0), 0.8)
        hf.add(0, 0, 0.9)
        self.assertEqual(hf.get(0, 0), 1.0)

    def test_out_of_bounds(self):
        hf = HeightField(2, 2)
        self.assertFalse(hf.in_bounds(2, 0))
        self.assertTrue(hf.in_bounds(1, 1))
        with self.assertRaises(IndexError):
            hf.get(2, 0)

    def test_to_rows_roundtrip(self):
        rows = [[0.0, 0.25], [0.5, 1.0]]
        hf = HeightField.from_rows(rows)
        self.assertEqual(hf.to_rows(), rows)

    def test_copy_is_independent(self):
        hf = HeightField(2, 2)
        c = hf.copy()
        c.set(0, 0, 1.0)
        self.assertEqual(hf.get(0, 0), 0.0)
        self.assertEqual(c.get(0, 0), 1.0)


class TestWholeField(unittest.TestCase):
    def test_apply_clamps(self):
        hf = HeightField.filled(2, 2, 0.5)
        hf.apply(lambda h: h * 10.0)
        self.assertTrue(all(h == 1.0 for h in hf.heights))

    def test_normalized(self):
        hf = HeightField(1, 3, 10.0, 20.0, [10.0, 15.0, 20.0])
        self.assertEqual(hf.normalized(), [0.0, 0.5, 1.0])

    def test_normalized_zero_span(self):
        hf = HeightField(1, 2, 5.0, 5.0)
        self.assertEqual(hf.normalized(), [0.0, 0.0])


class TestSummaries(unittest.TestCase):
    def test_min_max_mean(self):
        hf = HeightField.from_rows([[0.0, 1.0], [0.5, 0.5]])
        self.assertEqual(hf.max(), 1.0)
        self.assertEqual(hf.min(), 0.0)
        self.assertAlmostEqual(hf.mean(), 0.5)

    def test_total_travel(self):
        hf = HeightField(1, 3, 2.0, 10.0, [2.0, 4.0, 7.0])
        # travel above floor 2.0: 0 + 2 + 5 = 7
        self.assertAlmostEqual(hf.total_travel(), 7.0)

    def test_raised_cells(self):
        hf = HeightField.from_rows([[0.0, 0.3], [0.0, 0.9]])
        self.assertEqual(hf.raised_cells(), 2)
        self.assertEqual(hf.raised_cells(threshold=0.5), 1)

    def test_bounding_box(self):
        hf = HeightField(4, 4, 0.0, 1.0)
        hf.set(1, 1, 1.0)
        hf.set(2, 3, 1.0)
        self.assertEqual(hf.bounding_box(), (1, 1, 2, 3))

    def test_bounding_box_empty(self):
        hf = HeightField(3, 3)
        self.assertIsNone(hf.bounding_box())


class TestMetrics(unittest.TestCase):
    def test_mae_identical_is_zero(self):
        a = HeightField.from_rows([[0.0, 1.0], [0.5, 0.2]])
        b = a.copy()
        self.assertEqual(hfmod.mean_absolute_error(a, b), 0.0)

    def test_mae_value(self):
        a = HeightField.from_rows([[0.0, 0.0]])
        b = HeightField.from_rows([[0.4, 0.6]])
        self.assertAlmostEqual(hfmod.mean_absolute_error(a, b), 0.5)

    def test_rmse_value(self):
        a = HeightField.from_rows([[0.0, 0.0]])
        b = HeightField.from_rows([[0.3, 0.4]])
        # rms of (0.3, 0.4) = sqrt((0.09+0.16)/2) = sqrt(0.125)
        self.assertAlmostEqual(hfmod.root_mean_square_error(a, b), 0.125 ** 0.5)

    def test_match_ratio(self):
        a = HeightField.from_rows([[0.0, 0.5, 1.0, 0.2]])
        b = HeightField.from_rows([[0.0, 0.5, 0.9, 0.9]])
        self.assertEqual(hfmod.match_ratio(a, b, tolerance=0.0), 0.5)
        self.assertEqual(hfmod.match_ratio(a, b, tolerance=0.11), 0.75)

    def test_metric_shape_mismatch(self):
        a = HeightField(2, 2)
        b = HeightField(2, 3)
        with self.assertRaises(ValueError):
            hfmod.mean_absolute_error(a, b)

    def test_match_ratio_negative_tol(self):
        a = HeightField(2, 2)
        b = HeightField(2, 2)
        with self.assertRaises(ValueError):
            hfmod.match_ratio(a, b, tolerance=-0.1)


if __name__ == "__main__":
    unittest.main()
