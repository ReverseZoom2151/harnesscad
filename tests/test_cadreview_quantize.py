import unittest

import cadreview_quantize as q


class TestQuantize(unittest.TestCase):
    def test_range_and_bounds(self):
        self.assertEqual(q.quantize(0.0, 0.0, 256.0), 0)
        self.assertEqual(q.quantize(256.0, 0.0, 256.0), 255)
        # clamping
        self.assertEqual(q.quantize(-100.0, 0.0, 256.0), 0)
        self.assertEqual(q.quantize(1e9, 0.0, 256.0), 255)

    def test_eight_bit_levels(self):
        levels = {q.quantize(v, 0.0, 256.0) for v in range(257)}
        self.assertLessEqual(max(levels), 255)
        self.assertGreaterEqual(min(levels), 0)

    def test_roundtrip_monotonic(self):
        prev = -1.0
        for lvl in range(256):
            val = q.dequantize(lvl, 0.0, 256.0)
            self.assertGreater(val, prev)
            prev = val

    def test_snap_idempotent(self):
        s1 = q.snap(123.7, 0.0, 256.0)
        s2 = q.snap(s1, 0.0, 256.0)
        self.assertAlmostEqual(s1, s2, places=6)

    def test_bad_range_raises(self):
        with self.assertRaises(ValueError):
            q.quantize(1.0, 5.0, 5.0)

    def test_quantize_program(self):
        src = "translate([10, 0, 128]) cube([3, 3, 3]);"
        out = q.quantize_program(src, 0.0, 256.0)
        # still parseable-ish text with numbers snapped to the grid
        self.assertIn("translate", out)
        self.assertIn("cube", out)

    def test_sgo_weights(self):
        tokens = ["translate", "10", "cube", "3.5", "x"]
        weights = q.sgo_token_weights(tokens)
        self.assertEqual(weights, [1.0, 2.0, 1.0, 2.0, 1.0])

    def test_is_numeric_token(self):
        self.assertTrue(q.is_numeric_token("42"))
        self.assertTrue(q.is_numeric_token("-3.14"))
        self.assertFalse(q.is_numeric_token("cube"))


if __name__ == "__main__":
    unittest.main()
