"""Tests for procedural.autocad_array."""

import math
import unittest

from harnesscad.domain.procedural.array_patterns import (
    Placement,
    linear_array,
    fit_linear_array,
    rectangular_array,
    polar_array,
)


class TestLinear(unittest.TestCase):
    def test_count_and_step(self):
        ps = linear_array((0.0, 0.0), 3, (10.0, 0.0))
        self.assertEqual([p.x for p in ps], [0.0, 10.0, 20.0])
        self.assertTrue(all(p.y == 0.0 for p in ps))

    def test_zero_count(self):
        self.assertEqual(linear_array((0.0, 0.0), 0, (1.0, 1.0)), [])

    def test_negative_raises(self):
        with self.assertRaises(ValueError):
            linear_array((0.0, 0.0), -1, (1.0, 0.0))


class TestFitLinear(unittest.TestCase):
    def test_matches_floor_division(self):
        # total 100, pitch 10 -> 10 copies (repeat_block_horizontally behaviour)
        ps = fit_linear_array((0.0, 0.0), 100.0, 10.0)
        self.assertEqual(len(ps), 10)
        self.assertAlmostEqual(ps[-1].x, 90.0)

    def test_non_divisible(self):
        ps = fit_linear_array((0.0, 0.0), 95.0, 10.0)
        self.assertEqual(len(ps), 9)

    def test_direction_normalized(self):
        ps = fit_linear_array((0.0, 0.0), 30.0, 10.0, direction=(0.0, 5.0))
        self.assertAlmostEqual(ps[1].y, 10.0)
        self.assertAlmostEqual(ps[1].x, 0.0)

    def test_bad_pitch(self):
        with self.assertRaises(ValueError):
            fit_linear_array((0.0, 0.0), 10.0, 0.0)


class TestRectangular(unittest.TestCase):
    def test_grid(self):
        ps = rectangular_array((0.0, 0.0), 2, 3, (0.0, 5.0), (4.0, 0.0))
        self.assertEqual(len(ps), 6)
        # row 0: y=0, x=0,4,8 ; row 1: y=5
        self.assertAlmostEqual(ps[0].as_tuple()[0], 0.0)
        self.assertAlmostEqual(ps[2].x, 8.0)
        self.assertAlmostEqual(ps[3].y, 5.0)

    def test_empty(self):
        self.assertEqual(rectangular_array((0.0, 0.0), 0, 5, (1.0, 0.0), (0.0, 1.0)), [])


class TestPolar(unittest.TestCase):
    def test_full_circle_spacing(self):
        ps = polar_array((0.0, 0.0), 1.0, 4)
        angs = [p.rotation for p in ps]
        self.assertAlmostEqual(angs[1], math.pi / 2)
        # 4 distinct positions, not duplicating the start
        self.assertAlmostEqual(ps[0].x, 1.0)
        self.assertAlmostEqual(ps[0].y, 0.0, places=9)

    def test_partial_sweep_includes_endpoints(self):
        ps = polar_array((0.0, 0.0), 2.0, 3, start_angle=0.0,
                         sweep=math.pi, rotate_items=False)
        self.assertAlmostEqual(ps[0].rotation, 0.0)
        # last copy at angle pi -> position (-2, 0)
        self.assertAlmostEqual(ps[-1].x, -2.0)
        self.assertAlmostEqual(ps[-1].y, 0.0, places=9)

    def test_rotate_items_outward(self):
        ps = polar_array((0.0, 0.0), 1.0, 4, rotate_items=True)
        self.assertAlmostEqual(ps[1].rotation, math.pi / 2)

    def test_single_and_zero(self):
        self.assertEqual(len(polar_array((0.0, 0.0), 1.0, 1)), 1)
        self.assertEqual(polar_array((0.0, 0.0), 1.0, 0), [])


if __name__ == "__main__":
    unittest.main()
