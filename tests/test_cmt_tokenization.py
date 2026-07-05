import unittest

from reconstruction.cmt_tokenization import (
    bounding_box, edge_token, surface_token, order_tokens,
    quantize, dequantize, quantize_token,
)


class TestBoundingBox(unittest.TestCase):
    def test_box_spans_points(self):
        box = bounding_box(((0.0, 1.0, 2.0), (3.0, -1.0, 0.5)))
        self.assertEqual(box, (0.0, -1.0, 0.5, 3.0, 1.0, 2.0))

    def test_empty_raises(self):
        with self.assertRaises(ValueError):
            bounding_box(())


class TestEdgeToken(unittest.TestCase):
    def test_layout_embeds_vertices_no_vertex_token(self):
        tok = edge_token((0.0, 0.0, 0.0), (1.0, 0.0, 0.0))
        # bbox(6) + start(3) + end(3) = 12 values, box derived from endpoints
        self.assertEqual(len(tok), 12)
        self.assertEqual(tok[:6], (0.0, 0.0, 0.0, 1.0, 0.0, 0.0))
        self.assertEqual(tok[6:9], (0.0, 0.0, 0.0))
        self.assertEqual(tok[9:12], (1.0, 0.0, 0.0))

    def test_features_and_explicit_box_appended(self):
        tok = edge_token((0.0, 0.0, 0.0), (1.0, 1.0, 1.0),
                         box=(0.0, 0.0, 0.0, 1.0, 1.0, 1.0), features=(0.5, 0.25))
        self.assertEqual(len(tok), 14)
        self.assertEqual(tok[-2:], (0.5, 0.25))

    def test_bad_box(self):
        with self.assertRaises(ValueError):
            edge_token((0, 0, 0), (1, 1, 1), box=(0, 0, 0))


class TestSurfaceToken(unittest.TestCase):
    def test_layout(self):
        tok = surface_token((0.0, 0.0, 0.0, 2.0, 2.0, 0.0), features=(0.1,))
        self.assertEqual(len(tok), 7)
        self.assertEqual(tok[6], 0.1)

    def test_bad_box(self):
        with self.assertRaises(ValueError):
            surface_token((0.0, 1.0))


class TestOrdering(unittest.TestCase):
    def test_ascending_by_leading_box_coords(self):
        a = surface_token((1.0, 0.0, 0.0, 2.0, 1.0, 1.0))
        b = surface_token((0.0, 0.0, 0.0, 1.0, 1.0, 1.0))
        c = surface_token((0.0, 0.0, 0.0, 0.5, 1.0, 1.0))
        ordered = order_tokens((a, b, c))
        # b and c share x1,y1,z1; c has smaller x2 so it comes first
        self.assertEqual(ordered, (c, b, a))

    def test_stable_deterministic(self):
        toks = (surface_token((0.0, 0.0, 0.0, 1.0, 1.0, 1.0)),
                surface_token((0.0, 0.0, 0.0, 1.0, 1.0, 1.0)))
        self.assertEqual(order_tokens(toks), order_tokens(toks))


class TestQuantize(unittest.TestCase):
    def test_endpoints(self):
        self.assertEqual(quantize(0.0, 4), 0)
        self.assertEqual(quantize(1.0, 4), 15)

    def test_clamped(self):
        self.assertEqual(quantize(-5.0, 4), 0)
        self.assertEqual(quantize(5.0, 4), 15)

    def test_roundtrip_close(self):
        for bits in (4, 6, 8):
            level = quantize(0.5, bits)
            self.assertAlmostEqual(dequantize(level, bits), 0.5, delta=1.0 / ((1 << bits) - 1))

    def test_quantize_token(self):
        levels = quantize_token((0.0, 0.5, 1.0), 4)
        self.assertEqual(levels, (0, 8, 15))

    def test_bad_args(self):
        with self.assertRaises(ValueError):
            quantize(0.5, 0)
        with self.assertRaises(ValueError):
            quantize(0.5, 4, lo=1.0, hi=1.0)


if __name__ == "__main__":
    unittest.main()
