"""Tests for Hough line and circle detection (drawings->CAD)."""

import math
import unittest

from harnesscad.domain.drawings import hough_primitives as hp


class HoughLineTest(unittest.TestCase):
    def test_detects_horizontal_line(self):
        # points along y = 0
        pts = [(float(x), 0.0) for x in range(0, 10)]
        lines = hp.hough_lines(pts, rho_res=0.5, threshold=9, max_lines=1)
        self.assertEqual(len(lines), 1)
        line = lines[0]
        # a horizontal line y=0 has theta = pi/2 (normal points +y), rho ~= 0
        self.assertAlmostEqual(line.theta_rad, math.pi / 2, places=1)
        self.assertAlmostEqual(line.rho, 0.0, places=6)

    def test_all_points_support_the_line(self):
        pts = [(float(x), 3.0) for x in range(0, 6)]
        lines = hp.hough_lines(pts, threshold=6, max_lines=1)
        self.assertEqual(lines[0].votes, 6)

    def test_empty_points(self):
        self.assertEqual(hp.hough_lines([]), ())

    def test_point_line_distance(self):
        # line y=0 is rho=0, theta=pi/2 ; point (5,2) distance 2
        self.assertAlmostEqual(hp.point_line_distance(5.0, 2.0, 0.0, math.pi / 2), 2.0)


class HoughCircleTest(unittest.TestCase):
    def test_detects_circle_center(self):
        cx, cy, r = 0.0, 0.0, 5.0
        pts = [
            (r * math.cos(2 * math.pi * k / 24), r * math.sin(2 * math.pi * k / 24))
            for k in range(24)
        ]
        circles = hp.hough_circles(pts, [5.0], threshold=5, max_circles=1)
        self.assertTrue(len(circles) >= 1)
        best = circles[0]
        self.assertAlmostEqual(best.cx, 0.0, places=0)
        self.assertAlmostEqual(best.cy, 0.0, places=0)
        self.assertEqual(best.radius, 5.0)

    def test_empty(self):
        self.assertEqual(hp.hough_circles([], [1.0]), ())
        self.assertEqual(hp.hough_circles([(0, 0)], []), ())


if __name__ == "__main__":
    unittest.main()
