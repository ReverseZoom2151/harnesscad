import unittest

from harnesscad.domain.reconstruction.tokens.skexgen_quantize import (
    BIT, CURVE_END, FACE_END, LOOP_END, PAD, PIX_OFFSET, SE_END,
    center_vertices, command_vocab_size, coord_vocab_size, curve_points,
    curve_vertices, dequantize, encode_sketch, merge_se, normalize_scale,
    pixel_from_xy, pixel_vocab_size, quantize, shift_stream, sketch_center_scale,
    split_on, strip_padding, xy_from_pixel,
)


def _square(size=1.0):
    pts = [(-size, -size), (size, -size), (size, size), (-size, size)]
    loop = []
    for i in range(4):
        loop.append({"type": "line", "start": pts[i], "end": pts[(i + 1) % 4]})
    return [[loop]]


class TestQuantize(unittest.TestCase):
    def test_endpoints(self):
        self.assertEqual(quantize(-1.0), 0)
        self.assertEqual(quantize(1.0), 63)
        self.assertEqual(quantize(-5.0), 0)
        self.assertEqual(quantize(5.0), 63)

    def test_roundtrip_is_close(self):
        for level in (0, 17, 63):
            self.assertEqual(quantize(dequantize(level)), level)

    def test_bit_changes_range(self):
        self.assertEqual(quantize(1.0, bit=8), 255)
        self.assertAlmostEqual(dequantize(0, bit=8), -1.0)

    def test_scale_range_variant(self):
        # extrude scale uses [0, SCALE_R]
        self.assertEqual(quantize(0.0, BIT, 0.0, 1.4), 0)
        self.assertEqual(quantize(0.7, BIT, 0.0, 1.4), 31)
        # truncation (not rounding), matching numpy's astype('int32')
        self.assertEqual(quantize(1.4, BIT, 0.0, 1.4), 62)
        self.assertEqual(quantize(2.0, BIT, 0.0, 1.4), 63)


class TestPixel(unittest.TestCase):
    def test_flatten(self):
        self.assertEqual(pixel_from_xy(0, 0), 0)
        self.assertEqual(pixel_from_xy(3, 2), 2 * 64 + 3)
        self.assertEqual(pixel_from_xy(63, 63), 4095)

    def test_roundtrip(self):
        for x, y in ((0, 0), (5, 9), (63, 1), (12, 63)):
            self.assertEqual(xy_from_pixel(pixel_from_xy(x, y)), (x, y))

    def test_out_of_range(self):
        self.assertRaises(ValueError, pixel_from_xy, 64, 0)
        self.assertRaises(ValueError, xy_from_pixel, 4096)

    def test_vocab_sizes(self):
        self.assertEqual(pixel_vocab_size(6), 4096 + 5)
        self.assertEqual(coord_vocab_size(6), 64 + 5)
        self.assertEqual(command_vocab_size(), 7)


class TestNormalise(unittest.TestCase):
    def test_center_and_scale(self):
        verts = [(0.0, 0.0), (2.0, 0.0), (2.0, 2.0), (0.0, 2.0)]
        self.assertEqual(center_vertices(verts), (1.0, 1.0))
        centered = [(v[0] - 1.0, v[1] - 1.0) for v in verts]
        self.assertAlmostEqual(normalize_scale(centered), 0.5 * (8.0 ** 0.5))

    def test_sketch_center_scale(self):
        center, scale = sketch_center_scale(_square())
        self.assertEqual(center, (0.0, 0.0))
        self.assertAlmostEqual(scale, 0.5 * (8.0 ** 0.5))

    def test_curve_points_vs_vertices(self):
        line = {"type": "line", "start": (0.0, 0.0), "end": (1.0, 0.0)}
        arc = {"type": "arc", "start": (0.0, 0.0), "mid": (1.0, 1.0), "end": (2.0, 0.0)}
        circ = {"type": "circle", "pt1": (0.0, 1.0), "pt2": (2.0, 1.0),
                "pt3": (1.0, 0.0), "pt4": (1.0, 2.0)}
        self.assertEqual(len(curve_points(line)), 1)
        self.assertEqual(len(curve_vertices(line)), 2)
        self.assertEqual(len(curve_points(arc)), 2)
        self.assertEqual(len(curve_vertices(arc)), 3)
        self.assertEqual(len(curve_points(circ)), 4)
        self.assertRaises(ValueError, curve_points, {"type": "spline"})


class TestEncodeSketch(unittest.TestCase):
    def test_square_streams(self):
        enc = encode_sketch(_square())
        # 4 lines -> 4 * (1 pixel + curve-end) + loop/face/sketch ends
        self.assertEqual(len(enc["pix"]), 4 * 2 + 3)
        self.assertEqual(len(enc["cmd"]), 4 + 3)
        self.assertEqual(len(enc["xy"]), len(enc["pix"]))
        # raw sentinels after PIX_PAD: curve=3 loop=2 face=1 sketch=0
        self.assertEqual(enc["pix"][1], 3)
        self.assertEqual(enc["pix"][-3], 2)
        self.assertEqual(enc["pix"][-2], 1)
        self.assertEqual(enc["pix"][-1], 0)
        # commands are line (0 + CMD_PAD)
        self.assertEqual(enc["cmd"][0], 3)
        self.assertEqual(enc["cmd"][-3:], [2, 1, 0])

    def test_pixel_matches_xy(self):
        enc = encode_sketch(_square())
        for pixel, (x, y) in zip(enc["pix"], enc["xy"]):
            if pixel < 4:  # sentinel
                continue
            self.assertEqual(pixel - 4, pixel_from_xy(x - 4, y - 4))

    def test_deterministic(self):
        self.assertEqual(encode_sketch(_square()), encode_sketch(_square()))

    def test_arc_and_circle_lengths(self):
        arc_loop = [
            {"type": "arc", "start": (0.0, 0.0), "mid": (1.0, 1.0), "end": (2.0, 0.0)},
            {"type": "line", "start": (2.0, 0.0), "end": (0.0, 0.0)},
        ]
        enc = encode_sketch([[arc_loop]])
        self.assertEqual(len(enc["pix"]), (2 + 1) + (1 + 1) + 3)
        self.assertEqual(enc["cmd"][0], 1 + 3)

        circ = [{"type": "circle", "pt1": (-1.0, 0.0), "pt2": (1.0, 0.0),
                 "pt3": (0.0, -1.0), "pt4": (0.0, 1.0)}]
        enc = encode_sketch([[circ]])
        self.assertEqual(len(enc["pix"]), 4 + 1 + 3)
        self.assertEqual(enc["cmd"][0], 2 + 3)


class TestMerge(unittest.TestCase):
    def test_merge_and_strip(self):
        enc = encode_sketch(_square())
        ext = [1] * 18 + [0]
        merged = merge_se([enc["pix"]], [ext])
        self.assertEqual(merged[-1], PAD)
        self.assertEqual(merged[len(enc["pix"]) - 1], SE_END)  # sketch end
        self.assertEqual(merged[-2], SE_END)                   # extrude end
        self.assertEqual(strip_padding(merged), merged[:-1])
        self.assertGreaterEqual(min(t for t in merged if t >= PIX_OFFSET), PIX_OFFSET)

    def test_shift(self):
        self.assertEqual(shift_stream([0, 3, 4]), [1, 4, 5])

    def test_mismatch(self):
        self.assertRaises(ValueError, merge_se, [[1]], [])

    def test_split_on(self):
        toks = [5, 6, CURVE_END, 7, CURVE_END]
        self.assertEqual(split_on(toks, CURVE_END), [[5, 6, 4], [7, 4]])
        self.assertRaises(ValueError, split_on, [5, 6], CURVE_END)

    def test_structural_split_chain(self):
        enc = encode_sketch(_square())
        sketch = shift_stream(enc["pix"])
        self.assertEqual(sketch[-1], SE_END)
        faces = split_on(sketch[:-1], FACE_END)
        self.assertEqual(len(faces), 1)
        loops = split_on(faces[0][:-1], LOOP_END)
        self.assertEqual(len(loops), 1)
        curves = split_on(loops[0][:-1], CURVE_END)
        self.assertEqual(len(curves), 4)


if __name__ == "__main__":
    unittest.main()
