"""Tests for domain.procedural.exterior_completion."""

import math
import unittest

from harnesscad.domain.procedural.exterior_completion import (
    bounding_box,
    footprint_violation,
    generate_gable_roof,
    opening_preservation,
)

FOOTPRINT = [(0.0, 0.0), (10.0, 0.0), (10.0, 6.0), (0.0, 6.0)]


class BoundingBoxTest(unittest.TestCase):
    def test_box(self):
        self.assertEqual(bounding_box(FOOTPRINT), ((0.0, 0.0), (10.0, 6.0)))

    def test_empty_raises(self):
        with self.assertRaises(ValueError):
            bounding_box([])


class RoofTest(unittest.TestCase):
    def test_ridge_along_longer_axis(self):
        roof = generate_gable_roof(FOOTPRINT, wall_height=3.0, pitch_deg=45.0)
        # longer axis is x (span 10 > 6); ridge endpoints share y at mid (3.0)
        self.assertAlmostEqual(roof.ridge[0][1], 3.0)
        self.assertAlmostEqual(roof.ridge[1][1], 3.0)

    def test_rise_at_45deg(self):
        roof = generate_gable_roof(FOOTPRINT, wall_height=3.0, pitch_deg=45.0)
        # half span of shorter axis (6/2=3) * tan(45)=3
        self.assertAlmostEqual(roof.height, 3.0)
        self.assertAlmostEqual(roof.ridge[0][2], 6.0)  # 3 wall + 3 rise

    def test_eaves_on_footprint(self):
        roof = generate_gable_roof(FOOTPRINT, wall_height=3.0, pitch_deg=30.0)
        for (x, y, z) in roof.eaves:
            self.assertAlmostEqual(z, 3.0)

    def test_bad_pitch(self):
        with self.assertRaises(ValueError):
            generate_gable_roof(FOOTPRINT, 3.0, 90.0)


class ViolationTest(unittest.TestCase):
    def test_all_inside(self):
        pts = [(1.0, 1.0), (9.0, 5.0), (5.0, 3.0)]
        self.assertEqual(footprint_violation(pts, FOOTPRINT), 0.0)

    def test_some_outside(self):
        pts = [(1.0, 1.0), (11.0, 3.0)]  # second outside
        self.assertAlmostEqual(footprint_violation(pts, FOOTPRINT), 0.5)


class OpeningTest(unittest.TestCase):
    def test_all_preserved(self):
        req = [(0.0, 2.0), (10.0, 4.0)]
        gen = [(0.01, 2.0), (10.0, 4.01)]
        self.assertEqual(opening_preservation(gen, req, tol=0.1), 1.0)

    def test_missing_opening(self):
        req = [(0.0, 2.0), (10.0, 4.0)]
        gen = [(0.0, 2.0)]
        self.assertAlmostEqual(opening_preservation(gen, req, tol=0.1), 0.5)

    def test_empty_required(self):
        with self.assertRaises(ValueError):
            opening_preservation([(0, 0)], [], tol=0.1)


if __name__ == "__main__":
    unittest.main()
