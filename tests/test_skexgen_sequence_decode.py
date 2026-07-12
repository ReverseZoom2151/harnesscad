import unittest

from reconstruction.skexgen_extrude_tokens import encode_extrude
from reconstruction.skexgen_sequence_decode import (
    SkexGenParseError, VertexTable, circle_from_rim, circumcenter, decode_sketch,
    invalid_percent, is_valid, obj_records, parse_tokens, split_se,
)
from reconstruction.skexgen_token_format import encode_sketch, merge_se

EXT_KW = dict(
    extrude_value=(0.5, 0.0),
    origin=(0.0, 0.0, 0.0),
    x_axis=(1.0, 0.0, 0.0),
    y_axis=(0.0, 1.0, 0.0),
    z_axis=(0.0, 0.0, 1.0),
    set_op="NewBodyFeatureOperation",
    scale=0.7,
    offset=(0.0, 0.0),
)


def _square_sketch():
    pts = [(-1.0, -1.0), (1.0, -1.0), (1.0, 1.0), (-1.0, 1.0)]
    loop = [{"type": "line", "start": pts[i], "end": pts[(i + 1) % 4]}
            for i in range(4)]
    return [[loop]]


def _circle_sketch():
    loop = [{"type": "circle", "pt1": (-1.0, 0.0), "pt2": (1.0, 0.0),
             "pt3": (0.0, -1.0), "pt4": (0.0, 1.0)}]
    return [[loop]]


def _stream(sketch):
    enc = encode_sketch(sketch)
    ext = encode_extrude(**EXT_KW)
    return merge_se([enc["pix"]], [ext])


class TestPrimitives(unittest.TestCase):
    def test_circumcenter(self):
        c = circumcenter((-1.0, 0.0), (0.0, 1.0), (1.0, 0.0))
        self.assertAlmostEqual(c[0], 0.0)
        self.assertAlmostEqual(c[1], 0.0)

    def test_circumcenter_collinear(self):
        self.assertRaises(SkexGenParseError, circumcenter,
                          (0.0, 0.0), (1.0, 0.0), (2.0, 0.0))

    def test_circle_from_rim(self):
        center, radius = circle_from_rim((-1.0, 0.0), (1.0, 0.0),
                                         (0.0, -1.0), (0.0, 1.0))
        self.assertEqual(center, (0.0, 0.0))
        self.assertAlmostEqual(radius, 1.0)

    def test_vertex_table_dedup(self):
        t = VertexTable()
        self.assertEqual(t.save(1.0, 2.0), 0)
        self.assertEqual(t.save(1.0, 2.0), 0)
        self.assertEqual(t.save(3.0, 4.0), 1)
        self.assertEqual(t.save(1.0, 2.0, "r"), 2)
        self.assertEqual(len(t.vertices()), 3)
        self.assertEqual(t.obj_lines()[0], "v 1.0 2.0")


class TestSplit(unittest.TestCase):
    def test_split_pairs(self):
        pairs = split_se(_stream(_square_sketch()))
        self.assertEqual(len(pairs), 1)
        sketch, ext = pairs[0]
        self.assertEqual(len(ext), 19)
        self.assertEqual(sketch[-1], 1)

    def test_empty(self):
        self.assertRaises(SkexGenParseError, split_se, [0])

    def test_odd_groups(self):
        self.assertRaises(SkexGenParseError, split_se, [5, 4, 3, 2, 1, 0])

    def test_two_se(self):
        enc = encode_sketch(_square_sketch())
        ext = encode_extrude(**EXT_KW)
        merged = merge_se([enc["pix"], enc["pix"]], [ext, ext])
        self.assertEqual(len(split_se(merged)), 2)


class TestParse(unittest.TestCase):
    def test_square_roundtrip(self):
        ses = parse_tokens(_stream(_square_sketch()))
        self.assertEqual(len(ses), 1)
        faces = ses[0]["faces"]
        self.assertEqual(len(faces), 1)
        self.assertEqual(len(faces[0]), 1)
        loop = faces[0][0]
        self.assertEqual(len(loop), 4)
        self.assertTrue(all(c["type"] == "line" for c in loop))
        self.assertTrue(all(c["is_outer"] for c in loop))
        # closed loop: each end == next start
        for i, c in enumerate(loop):
            nxt = loop[(i + 1) % 4]
            self.assertAlmostEqual(c["end"][0], nxt["start"][0])
            self.assertAlmostEqual(c["end"][1], nxt["start"][1])

    def test_geometry_recovered(self):
        # unit square, scale 0.7, offset 0 -> corners near +/-(0.7/sqrt(2))
        ses = parse_tokens(_stream(_square_sketch()))
        xs = [c["start"][0] for c in ses[0]["faces"][0][0]]
        self.assertAlmostEqual(max(xs), 0.7 / (2.0 ** 0.5), places=1)

    def test_circle(self):
        ses = parse_tokens(_stream(_circle_sketch()))
        circ = ses[0]["faces"][0][0][0]
        self.assertEqual(circ["type"], "circle")
        self.assertGreater(circ["radius"], 0.0)

    def test_arc_loop(self):
        loop = [
            {"type": "arc", "start": (-1.0, 0.0), "mid": (0.0, 1.0), "end": (1.0, 0.0)},
            {"type": "line", "start": (1.0, 0.0), "end": (-1.0, 0.0)},
        ]
        ses = parse_tokens(_stream([[loop]]))
        curves = ses[0]["faces"][0][0]
        self.assertEqual([c["type"] for c in curves], ["arc", "line"])
        self.assertIn("center", curves[0])

    def test_obj_records(self):
        ses = parse_tokens(_stream(_square_sketch()))
        verts, curves = obj_records(ses[0]["faces"])
        self.assertEqual(curves[0], "face")
        self.assertEqual(curves[1], "out")
        self.assertEqual(len(verts), 4)          # square shares corner vertices
        self.assertTrue(curves[2].startswith("l "))

    def test_extrude_recovered(self):
        ses = parse_tokens(_stream(_square_sketch()))
        self.assertEqual(ses[0]["extrude"]["op_name"], "add")
        self.assertAlmostEqual(ses[0]["extrude"]["scale"], 0.7, places=1)

    def test_deterministic(self):
        a = parse_tokens(_stream(_square_sketch()))
        b = parse_tokens(_stream(_square_sketch()))
        self.assertEqual(a[0]["curves"], b[0]["curves"])


class TestValidity(unittest.TestCase):
    def test_valid(self):
        self.assertTrue(is_valid(_stream(_square_sketch())))

    def test_bad_curve_length(self):
        toks = _stream(_square_sketch())
        # make a curve group with 3 pixel tokens: duplicate a pixel token
        bad = toks[:1] + [toks[0], toks[0] + 1] + toks[1:]
        self.assertFalse(is_valid(bad))

    def test_truncated_extrude(self):
        toks = _stream(_square_sketch())
        self.assertFalse(is_valid(toks[:-3] + [0]))

    def test_no_terminator(self):
        self.assertFalse(is_valid([5, 6, 7]))

    def test_invalid_percent(self):
        good = _stream(_square_sketch())
        self.assertEqual(invalid_percent([good, good]), 0.0)
        self.assertEqual(invalid_percent([good, [5, 6, 7]]), 50.0)
        self.assertEqual(invalid_percent([]), 0.0)

    def test_zero_length_line_rejected(self):
        self.assertRaises(SkexGenParseError, decode_sketch,
                          [10, 4, 10, 4, 3, 2, 1], 1.0, (0.0, 0.0))


if __name__ == "__main__":
    unittest.main()
