"""Tests for the exact DeepCAD normalisation / quantisation numerics."""

import math
import unittest

from harnesscad.domain.reconstruction.tokens import deepcad_quantize as dn


class TestUnitFamily(unittest.TestCase):
    def test_endpoints(self):
        self.assertEqual(dn.numericalize_unit(-1.0), 0)
        self.assertEqual(dn.numericalize_unit(0.0), 128)
        # +1.0 maps to 256 then clips to 255 -- the reference's lossy top end.
        self.assertEqual(dn.numericalize_unit(1.0), 255)

    def test_clipping_out_of_range(self):
        self.assertEqual(dn.numericalize_unit(-5.0), 0)
        self.assertEqual(dn.numericalize_unit(5.0), 255)

    def test_inverse_is_exact_on_levels(self):
        for level in (0, 1, 64, 128, 255):
            value = dn.denumericalize_unit(level)
            self.assertEqual(dn.numericalize_unit(value), level)

    def test_denumericalize_formula(self):
        self.assertAlmostEqual(dn.denumericalize_unit(0), -1.0)
        self.assertAlmostEqual(dn.denumericalize_unit(128), 0.0)
        self.assertAlmostEqual(dn.denumericalize_unit(255), 255 / 256 * 2 - 1)


class TestAngleFamily(unittest.TestCase):
    def test_endpoints(self):
        self.assertEqual(dn.numericalize_angle(-math.pi), 0)
        self.assertEqual(dn.numericalize_angle(0.0), 128)
        self.assertEqual(dn.numericalize_angle(math.pi), 255)

    def test_roundtrip(self):
        for level in (0, 30, 128, 200, 255):
            angle = dn.denumericalize_angle(level)
            self.assertTrue(-math.pi <= angle <= math.pi)
            self.assertEqual(dn.numericalize_angle(angle), level)


class TestSizeFamily(unittest.TestCase):
    def test_forward(self):
        self.assertEqual(dn.numericalize_size(0.0), 0)
        self.assertEqual(dn.numericalize_size(1.0), 128)
        self.assertEqual(dn.numericalize_size(2.0), 255)  # clipped

    def test_roundtrip(self):
        for level in (0, 5, 128, 255):
            self.assertEqual(dn.numericalize_size(dn.denumericalize_size(level)), level)


class TestPixelAndRadius(unittest.TestCase):
    def test_pixel_clip(self):
        self.assertEqual(dn.numericalize_pixel(-3.2), 0)
        self.assertEqual(dn.numericalize_pixel(127.4), 127)
        self.assertEqual(dn.numericalize_pixel(999.0), 255)

    def test_radius_min_is_one(self):
        self.assertEqual(dn.numericalize_radius(0.2), 1)
        self.assertEqual(dn.numericalize_radius(0.0), 1)
        self.assertEqual(dn.numericalize_radius(40.6), 41)


class TestSweepFamily(unittest.TestCase):
    def test_quarter_turn(self):
        self.assertEqual(dn.numericalize_sweep(math.pi / 2), 64)
        self.assertEqual(dn.numericalize_sweep(math.pi), 128)
        self.assertEqual(dn.numericalize_sweep(2 * math.pi), 255)  # clipped

    def test_roundtrip(self):
        for level in (1, 64, 128, 255):
            self.assertEqual(dn.numericalize_sweep(dn.denumericalize_sweep(level)), level)


class TestShapeNormalisation(unittest.TestCase):
    def test_scale_uses_norm_factor(self):
        bbox = [(2.0, 1.0, 0.5), (-1.0, -1.0, -0.5)]
        self.assertAlmostEqual(dn.shape_scale(bbox), 0.75 / 2.0)

    def test_normalised_shape_fits_in_075_cube(self):
        bbox = [(2.0, 1.0, 0.5), (-1.0, -1.0, -0.5)]
        pts = dn.normalize_shape([(2.0, 1.0, 0.5), (-1.0, -1.0, -0.5)], bbox)
        peak = max(abs(c) for p in pts for c in p)
        self.assertAlmostEqual(peak, 0.75)

    def test_degenerate_bbox_raises(self):
        with self.assertRaises(ValueError):
            dn.shape_scale([(0.0, 0.0, 0.0)])


class TestSketchNormalisation(unittest.TestCase):
    def test_bbox_size_is_relative_to_start(self):
        pts = [(0.0, 0.0), (4.0, 0.0), (4.0, 1.0), (0.0, 1.0)]
        # corners (0,0) and (4,1); deviations from start (0,0) -> max 4.
        self.assertAlmostEqual(dn.sketch_bbox_size(pts, (0.0, 0.0)), 4.0)
        # from a different start point the size grows.
        self.assertAlmostEqual(dn.sketch_bbox_size(pts, (-1.0, 0.0)), 5.0)

    def test_start_point_lands_at_canvas_centre(self):
        pts = [(0.0, 0.0), (4.0, 0.0), (4.0, 1.0), (0.0, 1.0)]
        out = dn.normalize_sketch(pts)
        self.assertAlmostEqual(out[0][0], 128.0)
        self.assertAlmostEqual(out[0][1], 128.0)

    def test_normalised_sketch_fits_canvas_with_headroom(self):
        pts = [(0.0, 0.0), (4.0, 0.0), (4.0, 4.0), (0.0, 4.0)]
        out = dn.normalize_sketch(pts)
        # scale = (128*0.75 - 1)/4 = 23.75 -> far corner at 128 + 4*23.75 = 223
        self.assertAlmostEqual(out[2][0], 223.0)
        self.assertTrue(all(0 <= c <= 256 for p in out for c in p))

    def test_denormalize_inverts_normalize(self):
        pts = [(0.0, 0.0), (3.0, 0.0), (3.0, 2.0), (0.0, 2.0)]
        size = dn.sketch_bbox_size(pts, pts[0])
        back = dn.denormalize_sketch(dn.normalize_sketch(pts), size)
        for original, restored in zip(pts, back):
            self.assertAlmostEqual(restored[0], original[0] - pts[0][0], places=9)
            self.assertAlmostEqual(restored[1], original[1] - pts[0][1], places=9)

    def test_zero_bbox_size_raises(self):
        with self.assertRaises(ValueError):
            dn.sketch_normalize_scale(0.0)


class TestExtrudeBlock(unittest.TestCase):
    def _params(self):
        return {"theta": 0.0, "phi": math.pi / 2, "gamma": -math.pi,
                "px": 0.0, "py": 0.25, "pz": -0.5,
                "s": 1.0, "e1": 0.5, "e2": -0.5, "b": 1, "u": 0}

    def test_each_field_uses_its_own_family(self):
        q = dn.numericalize_extrude(self._params())
        self.assertEqual(q["theta"], 128)
        self.assertEqual(q["phi"], 192)
        self.assertEqual(q["gamma"], 0)
        self.assertEqual(q["px"], 128)
        self.assertEqual(q["pz"], 64)
        self.assertEqual(q["s"], 128)
        self.assertEqual(q["e1"], 192)
        self.assertEqual(q["b"], 1)
        self.assertEqual(q["u"], 0)

    def test_roundtrip_on_quantised_grid(self):
        q = dn.numericalize_extrude(self._params())
        again = dn.numericalize_extrude(dn.denumericalize_extrude(q))
        self.assertEqual(q, again)

    def test_extent_out_of_range_raises(self):
        params = self._params()
        params["e1"] = 3.0
        with self.assertRaises(ValueError):
            dn.numericalize_extrude(params)


if __name__ == "__main__":
    unittest.main()
