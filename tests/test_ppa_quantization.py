"""Tests for PPA coordinate normalisation + 6-bit quantisation."""

import unittest

from harnesscad.domain.reconstruction.sketch import ppa_primitive as pp
from harnesscad.domain.reconstruction.tokens import ppa_quantization as pq


class TestQuantLattice(unittest.TestCase):
    def test_six_bit_levels(self):
        self.assertEqual(pq.levels(6), 64)
        self.assertEqual(pq.levels(1), 2)

    def test_endpoints(self):
        self.assertEqual(pq.quantize(0.0), 0)
        self.assertEqual(pq.quantize(1.0), 63)

    def test_clamping(self):
        self.assertEqual(pq.quantize(-5.0), 0)
        self.assertEqual(pq.quantize(2.0), 63)

    def test_roundtrip_is_near_identity(self):
        for level in range(64):
            v = pq.dequantize(level)
            self.assertEqual(pq.quantize(v), level)

    def test_max_error_half_level(self):
        # a value exactly between two levels should be within half a level
        v = 0.5 / 63  # halfway to level 1... but clamped rounding, still small
        self.assertLessEqual(abs(pq.dequantize(pq.quantize(v)) - v), 0.5 / 63 + 1e-12)

    def test_bad_bits(self):
        with self.assertRaises(ValueError):
            pq.levels(0)


class TestNormalization(unittest.TestCase):
    def test_bbox(self):
        s = pp.Sketch([pp.line((0, 0), (10, 4)), pp.point((2, -3))])
        box = pq.sketch_bbox(s)
        self.assertEqual((box.xmin, box.ymin, box.xmax, box.ymax), (0, -3, 10, 4))

    def test_circle_radius_expands_bbox(self):
        s = pp.Sketch([pp.circle((5, 5), 5.0)])
        box = pq.sketch_bbox(s)
        self.assertEqual((box.xmin, box.ymin, box.xmax, box.ymax), (0, 0, 10, 10))

    def test_normalize_into_unit_and_back(self):
        s = pp.Sketch([pp.line((0, 0), (10, 10)), pp.circle((5, 5), 2.5)])
        norm, box = pq.normalize_sketch(s)
        for prim in norm:
            for (x, y) in prim.control_points():
                self.assertGreaterEqual(x, -1e-9)
                self.assertLessEqual(x, 1.0 + 1e-9)
                self.assertGreaterEqual(y, -1e-9)
                self.assertLessEqual(y, 1.0 + 1e-9)
        back = pq.denormalize_sketch(norm, box)
        self.assertEqual(back, s)

    def test_normalized_radius_scaled(self):
        s = pp.Sketch([pp.circle((5, 5), 5.0)])  # extent 10
        norm, box = pq.normalize_sketch(s)
        self.assertEqual(box.extent, 10.0)
        self.assertAlmostEqual(list(norm)[0].radius, 0.5)


class TestQuantizePrimitive(unittest.TestCase):
    def test_padding_stays_zero(self):
        norm, _ = pq.normalize_sketch(pp.Sketch([pp.point((0, 0))]))
        q = pq.quantize_primitive(list(norm)[0])
        self.assertEqual(len(q), 7)
        self.assertEqual(q[2:], (0, 0, 0, 0, 0))

    def test_dequantize_inverts_type_flag(self):
        norm, _ = pq.normalize_sketch(pp.Sketch([pp.circle((5, 5), 2.5, flag=False)]))
        prim = list(norm)[0]
        q = pq.quantize_primitive(prim)
        rebuilt = pq.dequantize_primitive(prim.ptype, prim.flag, q)
        self.assertEqual(rebuilt.ptype, pp.CIRCLE)
        self.assertFalse(rebuilt.flag)

    def test_quantization_error_bounded(self):
        s = pp.Sketch([pp.line((0, 0), (10, 7)), pp.arc((0, 0), (5, 5), (10, 0))])
        norm, _ = pq.normalize_sketch(s)
        err = pq.quantization_error(norm)
        self.assertGreater(err["count"], 0)
        self.assertLessEqual(err["max"], 0.5 / 63 + 1e-9)
        self.assertLessEqual(err["mean"], err["max"] + 1e-12)

    def test_empty_sketch_error_zero(self):
        err = pq.quantization_error(pp.Sketch([]))
        self.assertEqual(err, {"mean": 0.0, "max": 0.0, "count": 0})


if __name__ == "__main__":
    unittest.main()
