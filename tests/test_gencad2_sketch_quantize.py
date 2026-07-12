"""Tests for reconstruction.gencad2_sketch_quantize."""

import math
import unittest

from reconstruction.gencad2_sketch_quantize import (
    ARGS_DIM,
    NORM_FACTOR,
    SKETCH_DIM,
    bbox_size_from_bbox,
    check_extent_range,
    dequantize_angle,
    dequantize_coord_system,
    dequantize_sketch_size,
    dequantize_unit,
    denormalize_sketch_point,
    max_quantization_error,
    normalize_shape_point,
    normalize_sketch_length,
    normalize_sketch_point,
    quantization_step,
    quantize_angle,
    quantize_coord_system,
    quantize_coordinate,
    quantize_radius,
    quantize_sketch_size,
    quantize_unit,
    shape_normalize_scale,
    sketch_denormalize_scale,
    sketch_normalize_scale,
)


class TestConstants(unittest.TestCase):
    def test_norm_factor(self):
        self.assertEqual(NORM_FACTOR, 0.75)
        self.assertEqual(ARGS_DIM, 256)
        self.assertEqual(SKETCH_DIM, 256)


class TestShapeNormalize(unittest.TestCase):
    def test_scale_puts_shape_in_075_cube(self):
        bbox = [(2.0, 1.0, -0.5), (-2.0, -1.0, 0.5)]
        scale = shape_normalize_scale(bbox)
        self.assertAlmostEqual(scale, 0.75 / 2.0, places=12)
        p = normalize_shape_point((2.0, 1.0, -0.5), scale)
        self.assertAlmostEqual(p[0], 0.75, places=12)
        self.assertLessEqual(max(abs(v) for v in p), 0.75 + 1e-12)

    def test_size_argument(self):
        bbox = [(1.0, 0.0, 0.0), (-1.0, 0.0, 0.0)]
        self.assertAlmostEqual(shape_normalize_scale(bbox, size=2.0), 1.5, places=12)

    def test_degenerate_bbox_raises(self):
        with self.assertRaises(ValueError):
            shape_normalize_scale([(0.0, 0.0, 0.0), (0.0, 0.0, 0.0)])


class TestSketchNormalize(unittest.TestCase):
    def test_scale_formula(self):
        # (256 / 2 * 0.75 - 1) / bbox_size = 95 / bbox_size
        self.assertAlmostEqual(sketch_normalize_scale(1.0), 95.0, places=9)
        self.assertAlmostEqual(sketch_normalize_scale(5.0), 19.0, places=9)

    def test_start_point_maps_to_center(self):
        p = normalize_sketch_point((3.0, 4.0), (3.0, 4.0), 2.0)
        self.assertAlmostEqual(p[0], 128.0, places=9)
        self.assertAlmostEqual(p[1], 128.0, places=9)

    def test_extreme_point_stays_in_raster(self):
        # a point one bbox_size away lands 95 units from the centre -> inside 0..255
        p = normalize_sketch_point((2.0, 0.0), (0.0, 0.0), 2.0)
        self.assertAlmostEqual(p[0], 128.0 + 95.0, places=9)
        self.assertLess(p[0], 256.0)

    def test_round_trip(self):
        start, bsize = (1.5, -2.5), 3.0
        for pt in ((1.5, -2.5), (4.5, -2.5), (0.0, 0.0)):
            norm = normalize_sketch_point(pt, start, bsize)
            back = denormalize_sketch_point(norm, bsize)
            self.assertAlmostEqual(back[0], pt[0] - start[0], places=9)
            self.assertAlmostEqual(back[1], pt[1] - start[1], places=9)

    def test_scales_are_inverse(self):
        self.assertAlmostEqual(
            sketch_normalize_scale(7.0) * sketch_denormalize_scale(7.0), 1.0, places=12)

    def test_length_has_no_translation(self):
        self.assertAlmostEqual(normalize_sketch_length(1.0, 1.0), 95.0, places=9)

    def test_zero_bbox_size_raises(self):
        with self.assertRaises(ValueError):
            sketch_normalize_scale(0.0)

    def test_bbox_size_from_bbox(self):
        # start at (0,0), box spanning (-1,-3)..(2,1) -> max offset is 3
        self.assertAlmostEqual(
            bbox_size_from_bbox((-1.0, -3.0, 2.0, 1.0), (0.0, 0.0)), 3.0, places=9)


class TestQuantization(unittest.TestCase):
    def test_coordinate_clipping(self):
        self.assertEqual(quantize_coordinate(127.6), 128)
        self.assertEqual(quantize_coordinate(-4.0), 0)
        self.assertEqual(quantize_coordinate(999.0), 255)

    def test_radius_floor_is_one(self):
        self.assertEqual(quantize_radius(0.2), 1)
        self.assertEqual(quantize_radius(0.0), 1)
        self.assertEqual(quantize_radius(30.4), 30)

    def test_unit_offsets(self):
        self.assertEqual(quantize_unit(-1.0), 0)
        self.assertEqual(quantize_unit(0.0), 128)
        self.assertEqual(quantize_unit(1.0), 255)  # 256 clipped to n-1

    def test_unit_round_trip_within_half_bucket(self):
        for v in (-0.99, -0.5, -0.125, 0.0, 0.37, 0.75):
            back = dequantize_unit(quantize_unit(v))
            self.assertLessEqual(abs(back - v), max_quantization_error() + 1e-12)

    def test_angle_offsets(self):
        self.assertEqual(quantize_angle(-math.pi), 0)
        self.assertEqual(quantize_angle(0.0), 128)
        self.assertEqual(quantize_angle(math.pi / 2), 192)

    def test_angle_round_trip(self):
        for a in (-3.0, -1.0, 0.0, 0.5, 2.9):
            back = dequantize_angle(quantize_angle(a))
            self.assertLessEqual(abs(back - a), math.pi / ARGS_DIM + 1e-9)

    def test_sketch_size_has_no_offset(self):
        self.assertEqual(quantize_sketch_size(0.0), 0)
        self.assertEqual(quantize_sketch_size(1.0), 128)
        self.assertEqual(quantize_sketch_size(2.0), 255)

    def test_sketch_size_round_trip(self):
        for s in (0.0, 0.25, 1.0, 1.5):
            self.assertAlmostEqual(
                dequantize_sketch_size(quantize_sketch_size(s)), s, delta=2.0 / 256)

    def test_size_and_unit_differ(self):
        # the classic bug: quantising a size with the unit formula doubles it
        self.assertNotEqual(quantize_sketch_size(0.5), quantize_unit(0.5))

    def test_step_and_error(self):
        self.assertAlmostEqual(quantization_step(), 2.0 / 256, places=12)
        self.assertAlmostEqual(max_quantization_error(), 1.0 / 256, places=12)


class TestExtentAndCoordSystem(unittest.TestCase):
    def test_extent_range_ok(self):
        self.assertEqual(check_extent_range(1.5), 1.5)
        self.assertEqual(check_extent_range(-2.0), -2.0)

    def test_extent_out_of_range_raises(self):
        with self.assertRaises(ValueError):
            check_extent_range(2.5)

    def test_coord_system_round_trip(self):
        origin = (0.25, -0.5, 0.0)
        theta, phi, gamma = 1.0, -2.0, 0.5
        levels = quantize_coord_system(origin, theta, phi, gamma)
        self.assertEqual(len(levels), 6)
        o, t, p, g = dequantize_coord_system(levels)
        for got, want in zip(o, origin):
            self.assertLessEqual(abs(got - want), max_quantization_error() + 1e-12)
        for got, want in zip((t, p, g), (theta, phi, gamma)):
            self.assertLessEqual(abs(got - want), math.pi / ARGS_DIM + 1e-9)

    def test_coord_system_levels_are_ints_in_range(self):
        levels = quantize_coord_system((1.0, -1.0, 0.0), math.pi, -math.pi, 0.0)
        for v in levels:
            self.assertIsInstance(v, int)
            self.assertGreaterEqual(v, 0)
            self.assertLessEqual(v, 255)

    def test_bad_level_count_raises(self):
        with self.assertRaises(ValueError):
            dequantize_coord_system((1, 2, 3))

    def test_deterministic(self):
        a = quantize_coord_system((0.1, 0.2, 0.3), 0.4, 0.5, 0.6)
        b = quantize_coord_system((0.1, 0.2, 0.3), 0.4, 0.5, 0.6)
        self.assertEqual(a, b)


if __name__ == "__main__":
    unittest.main()
