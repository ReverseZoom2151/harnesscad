"""Tests for geometry.dlwheel_edge (paper 112 edge extraction)."""

import math
import unittest

from harnesscad.domain.geometry.views import edge_detection as de


class GradientTests(unittest.TestCase):
    def test_gradient_x_step(self):
        img = [[0.0, 0.0, 10.0, 10.0]]
        gx = de.gradient_x(img)
        # differences: 0, 10, 0, 0 (last col padded)
        self.assertEqual(gx[0], [0.0, 10.0, 0.0, 0.0])

    def test_gradient_y_step(self):
        img = [[0.0], [0.0], [5.0]]
        gy = de.gradient_y(img)
        self.assertEqual([r[0] for r in gy], [0.0, 5.0, 0.0])

    def test_gradient_magnitude(self):
        gx = [[3.0]]
        gy = [[4.0]]
        self.assertAlmostEqual(de.gradient_magnitude(gx, gy)[0][0], 5.0)

    def test_shape_mismatch(self):
        with self.assertRaises(ValueError):
            de.gradient_magnitude([[1.0, 2.0]], [[1.0]])

    def test_non_rectangular(self):
        with self.assertRaises(ValueError):
            de.gradient_x([[1.0, 2.0], [3.0]])

    def test_empty(self):
        with self.assertRaises(ValueError):
            de.gradient_x([])


class SobelTests(unittest.TestCase):
    def test_sobel_vertical_edge(self):
        # Left half 0, right half 100 -> strong horizontal gradient at boundary.
        img = [[0.0, 0.0, 100.0, 100.0] for _ in range(3)]
        mag = de.sobel_magnitude(img)
        # Column 1 (at the edge) should have larger response than column 0.
        self.assertGreater(mag[1][1], mag[1][0])

    def test_sobel_uniform_zero(self):
        img = [[7.0] * 4 for _ in range(4)]
        mag = de.sobel_magnitude(img)
        for row in mag:
            for v in row:
                self.assertAlmostEqual(v, 0.0)

    def test_sobel_x_known(self):
        # A single bright center pixel.
        img = [[0, 0, 0], [0, 9, 0], [0, 0, 0]]
        sx = de.sobel_x(img)
        # center of Sx over symmetric input is 0
        self.assertAlmostEqual(sx[1][1], 0.0)


class ThresholdTests(unittest.TestCase):
    def test_threshold(self):
        mag = [[1.0, 5.0], [10.0, 0.0]]
        binm = de.threshold_edges(mag, 5.0)
        self.assertEqual(binm, [[0, 1], [1, 0]])

    def test_edge_coordinates(self):
        edge = [[0, 1], [1, 0]]
        coords = de.edge_coordinates(edge)
        self.assertEqual(coords, [(1, 0), (0, 1)])


if __name__ == "__main__":
    unittest.main()
