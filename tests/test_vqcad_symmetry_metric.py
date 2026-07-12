"""Tests for bench.vqcad_symmetry_metric (VQ-CAD Eq. 13 symmetry metric)."""

from __future__ import annotations

import unittest

from bench.vqcad_symmetry_metric import (
    horizontal_symmetry,
    intensity_centroid,
    is_more_symmetric,
    symmetry_score,
    vertical_symmetry,
)


class TestCentroid(unittest.TestCase):
    def test_geometric_fallback_for_blank_image(self):
        img = [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]
        o_x, o_y = intensity_centroid(img)
        self.assertAlmostEqual(o_x, 0.5, places=12)  # (2-1)/2
        self.assertAlmostEqual(o_y, 1.0, places=12)  # (3-1)/2

    def test_single_bright_pixel(self):
        img = [[0.0, 0.0, 0.0], [0.0, 5.0, 0.0]]
        o_x, o_y = intensity_centroid(img)
        self.assertAlmostEqual(o_x, 1.0, places=12)
        self.assertAlmostEqual(o_y, 1.0, places=12)

    def test_symmetric_intensity_centroid_centered(self):
        img = [[1.0, 0.0, 1.0]]
        o_x, o_y = intensity_centroid(img)
        self.assertAlmostEqual(o_y, 1.0, places=12)

    def test_negative_intensity_raises(self):
        with self.assertRaises(ValueError):
            intensity_centroid([[-1.0, 0.0]])


class TestSymmetryScore(unittest.TestCase):
    def test_perfectly_symmetric_scores_zero(self):
        # symmetric about both axes: mirror in rows and columns
        img = [
            [1.0, 2.0, 1.0],
            [3.0, 4.0, 3.0],
            [1.0, 2.0, 1.0],
        ]
        self.assertAlmostEqual(symmetry_score(img), 0.0, places=12)

    def test_horizontal_symmetric_has_zero_horizontal(self):
        img = [
            [1.0, 9.0, 1.0],
            [2.0, 8.0, 2.0],
        ]
        self.assertAlmostEqual(horizontal_symmetry(img), 0.0, places=12)

    def test_asymmetric_scores_positive(self):
        # two unequal-weight pixels pull the centroid off both, so neither the
        # horizontal nor vertical reflection lines them up.
        img = [
            [9.0, 0.0, 0.0, 1.0],
            [0.0, 0.0, 0.0, 0.0],
        ]
        self.assertGreater(symmetry_score(img), 0.0)

    def test_vertical_flip_detects_row_asymmetry(self):
        # single column, mass concentrated at top with a light tail -> centroid
        # sits near the top, so the vertical reflection misaligns the two pixels.
        img = [
            [9.0],
            [0.0],
            [0.0],
            [1.0],
        ]
        self.assertGreater(vertical_symmetry(img), 0.0)

    def test_lone_point_is_symmetric_about_itself(self):
        img = [
            [0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0],
            [7.0, 0.0, 0.0, 0.0],
        ]
        # centroid sits on the bright pixel, so reflection maps it to itself;
        # score is 0 because a lone point is trivially symmetric about itself.
        self.assertAlmostEqual(symmetry_score(img), 0.0, places=12)

    def test_two_pixels_asymmetric_about_centroid(self):
        # two unequal-weight pixels: centroid is pulled toward the heavier one,
        # so the reflection does not line them up -> positive score
        img = [
            [9.0, 0.0, 0.0, 1.0],
        ]
        self.assertGreater(symmetry_score(img), 0.0)


class TestComparison(unittest.TestCase):
    def test_is_more_symmetric(self):
        sym = [
            [1.0, 2.0, 1.0],
            [1.0, 2.0, 1.0],
        ]
        asym = [
            [9.0, 0.0, 0.0, 1.0],
            [0.0, 0.0, 0.0, 0.0],
        ]
        self.assertTrue(is_more_symmetric(sym, asym))
        self.assertFalse(is_more_symmetric(asym, sym))


class TestValidation(unittest.TestCase):
    def test_empty_image_raises(self):
        with self.assertRaises(ValueError):
            symmetry_score([])

    def test_non_rectangular_raises(self):
        with self.assertRaises(ValueError):
            symmetry_score([[1.0, 2.0], [3.0]])

    def test_empty_row_raises(self):
        with self.assertRaises(ValueError):
            symmetry_score([[]])


if __name__ == "__main__":
    unittest.main()
