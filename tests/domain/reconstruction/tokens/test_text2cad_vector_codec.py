"""Tests for the Text2CAD CAD-vec codec (reconstruction.t2c3_cad_vec_codec)."""

import unittest

from harnesscad.domain.reconstruction.tokens.text2cad_vector_codec import (
    BOOLEAN_OFFSET,
    CAD_CLASS_INFO,
    COORD_OFFSET,
    END_CURVE,
    END_EXTRUSION,
    END_FACE,
    END_LOOP,
    END_SKETCH,
    EXT_BLOCK_LENGTH,
    FLAG_PAD,
    MAX_CAD_SEQUENCE_LENGTH,
    PADDING,
    START,
    CadVecError,
    boolean_name,
    decode_extrusion,
    decode_loop,
    decode_model,
    encode_curve,
    encode_extrusion,
    encode_loop,
    encode_model,
    split_tokens,
)


def _square_part():
    loop = [
        {"type": "line", "start": (0, 0)},
        {"type": "line", "start": (100, 0)},
        {"type": "line", "start": (100, 100)},
        {"type": "line", "start": (0, 100)},
    ]
    return {
        "sketch": [[loop]],
        "extrusion": {
            "extent_one": 200,
            "extent_two": 128,
            "origin": (128, 128, 128),
            "euler": (128, 0, 255),
            "boolean": 1,
            "sketch_size": 64,
        },
    }


def _circle_part():
    loop = [{"type": "circle", "center": (128, 128), "pt1": (128, 200)}]
    return {
        "sketch": [[loop]],
        "extrusion": {
            "extent_one": 10,
            "extent_two": 0,
            "origin": (1, 2, 3),
            "euler": (4, 5, 6),
            "boolean": 2,
            "sketch_size": 255,
        },
    }


class TestConstants(unittest.TestCase):
    def test_class_info(self):
        self.assertEqual(CAD_CLASS_INFO["one_hot_size"], 267)
        self.assertEqual(CAD_CLASS_INFO["index_size"], 11)
        self.assertEqual(CAD_CLASS_INFO["flag_size"], 12)
        self.assertEqual(COORD_OFFSET, 11)
        self.assertEqual(BOOLEAN_OFFSET, 7)

    def test_boolean_name(self):
        self.assertEqual(boolean_name(0), "NewBodyFeatureOperation")
        self.assertEqual(boolean_name(2), "CutFeatureOperation")
        with self.assertRaises(CadVecError):
            boolean_name(9)


class TestCurveTokens(unittest.TestCase):
    def test_line_emits_start_only(self):
        toks = encode_curve({"type": "line", "start": (3, 4)})
        self.assertEqual(toks, [(3 + COORD_OFFSET, 4 + COORD_OFFSET), (END_CURVE, 0)])

    def test_arc_emits_start_and_mid(self):
        toks = encode_curve({"type": "arc", "start": (0, 0), "mid": (5, 5)})
        self.assertEqual(len(toks), 3)
        self.assertEqual(toks[-1], (END_CURVE, 0))

    def test_circle_emits_center_and_pt1(self):
        toks = encode_curve({"type": "circle", "center": (10, 10), "pt1": (10, 20)})
        self.assertEqual(toks[0], (21, 21))
        self.assertEqual(toks[1], (21, 31))

    def test_unquantised_coordinate_rejected(self):
        with self.assertRaises(CadVecError):
            encode_curve({"type": "line", "start": (0.5, 3)})
        with self.assertRaises(CadVecError):
            encode_curve({"type": "line", "start": (256, 3)})

    def test_unknown_type_rejected(self):
        with self.assertRaises(CadVecError):
            encode_curve({"type": "spline", "start": (0, 0)})

    def test_circle_must_be_alone_in_loop(self):
        with self.assertRaises(CadVecError):
            encode_loop([
                {"type": "circle", "center": (1, 1), "pt1": (1, 2)},
                {"type": "line", "start": (0, 0)},
            ])


class TestLoopChaining(unittest.TestCase):
    def test_n_curves_cost_n_coordinate_tokens(self):
        loop = [{"type": "line", "start": (i, i)} for i in range(4)]
        toks = encode_loop(loop)
        # 4 curves x (1 coord + END_CURVE) + END_LOOP
        self.assertEqual(len(toks), 9)
        self.assertEqual(toks[-1], (END_LOOP, 0))

    def test_decode_closes_the_loop(self):
        loop = [
            {"type": "line", "start": (0, 0)},
            {"type": "line", "start": (10, 0)},
            {"type": "line", "start": (10, 10)},
        ]
        toks = encode_loop(loop)[:-1]  # drop END_LOOP
        curves = decode_loop(toks)
        self.assertEqual(len(curves), 3)
        self.assertEqual(curves[0]["end"], (10, 0))
        self.assertEqual(curves[2]["end"], (0, 0))  # wraps to the first start

    def test_arc_round_trip(self):
        loop = [
            {"type": "arc", "start": (0, 0), "mid": (5, 7)},
            {"type": "line", "start": (10, 0)},
        ]
        curves = decode_loop(encode_loop(loop)[:-1])
        self.assertEqual(curves[0]["type"], "arc")
        self.assertEqual(curves[0]["mid"], (5, 7))
        self.assertEqual(curves[0]["end"], (10, 0))
        self.assertEqual(curves[1]["type"], "line")
        self.assertEqual(curves[1]["end"], (0, 0))

    def test_circle_radius_recovered(self):
        loop = [{"type": "circle", "center": (100, 100), "pt1": (100, 130)}]
        curves = decode_loop(encode_loop(loop)[:-1])
        self.assertEqual(curves[0]["type"], "circle")
        self.assertAlmostEqual(curves[0]["radius"], 30.0)

    def test_bad_group_length_rejected(self):
        bad = [(20, 20), (21, 21), (22, 22), (END_CURVE, 0), (30, 30), (END_CURVE, 0)]
        with self.assertRaises(CadVecError):
            decode_loop(bad)


class TestExtrusionBlock(unittest.TestCase):
    def test_block_is_eleven_tokens_in_order(self):
        toks = encode_extrusion(_square_part()["extrusion"])
        self.assertEqual(len(toks), EXT_BLOCK_LENGTH)
        self.assertEqual(toks[-1], (END_EXTRUSION, 0))
        self.assertEqual(toks[0], (200 + COORD_OFFSET, 0))
        self.assertEqual(toks[8], (1 + BOOLEAN_OFFSET, 0))   # boolean uses END_PAD only
        self.assertEqual(toks[9], (64 + COORD_OFFSET, 0))

    def test_round_trip(self):
        ext = _square_part()["extrusion"]
        self.assertEqual(decode_extrusion(encode_extrusion(ext)), ext)

    def test_invalid_boolean(self):
        ext = dict(_square_part()["extrusion"], boolean=4)
        with self.assertRaises(CadVecError):
            encode_extrusion(ext)

    def test_short_block_rejected(self):
        with self.assertRaises(CadVecError):
            decode_extrusion([(20, 0)] * 5)


class TestModelStreams(unittest.TestCase):
    def test_start_wrapper_and_structure(self):
        vecs = encode_model([_square_part()])
        self.assertEqual(vecs.cad_vec[0], (START, 0))
        self.assertEqual(vecs.cad_vec[-1], (START, 0))
        ids = [t[0] for t in vecs.cad_vec]
        self.assertIn(END_SKETCH, ids)
        self.assertIn(END_FACE, ids)
        self.assertIn(END_EXTRUSION, ids)

    def test_flag_and_index_vectors(self):
        vecs = encode_model([_square_part(), _circle_part()])
        self.assertEqual(len(vecs.flag_vec), len(vecs.cad_vec))
        self.assertEqual(len(vecs.index_vec), len(vecs.cad_vec))
        # extrusion flags are [1, 1..10] per part
        self.assertEqual(vecs.flag_vec[-12:-1], [1, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        self.assertEqual(set(vecs.index_vec), {0, 1})
        self.assertEqual(vecs.index_vec[0], 0)
        self.assertEqual(vecs.index_vec[-1], 1)

    def test_padding(self):
        vecs = encode_model([_square_part()], padding=True)
        self.assertEqual(len(vecs.cad_vec), MAX_CAD_SEQUENCE_LENGTH)
        self.assertEqual(vecs.cad_vec[-1], (PADDING, PADDING))
        self.assertEqual(vecs.flag_vec[-1], FLAG_PAD)
        self.assertEqual(vecs.index_vec[-1], 1)

    def test_padding_overflow(self):
        with self.assertRaises(CadVecError):
            encode_model([_square_part()], padding=True, max_cad_seq_len=10)

    def test_model_round_trip(self):
        model = [_square_part(), _circle_part()]
        decoded = decode_model(encode_model(model, padding=True).cad_vec)
        self.assertEqual(len(decoded), 2)
        for original, got in zip(model, decoded):
            self.assertEqual(got["extrusion"], original["extrusion"])
        first_loop = decoded[0]["sketch"][0][0]
        self.assertEqual([c["type"] for c in first_loop], ["line"] * 4)
        self.assertEqual(first_loop[0]["start"], (0, 0))
        self.assertEqual(decoded[1]["sketch"][0][0][0]["type"], "circle")

    def test_multi_face_multi_loop_round_trip(self):
        outer = [{"type": "line", "start": (0, 0)},
                 {"type": "line", "start": (50, 0)},
                 {"type": "line", "start": (50, 50)}]
        inner = [{"type": "circle", "center": (20, 20), "pt1": (20, 25)}]
        part = _square_part()
        part["sketch"] = [[outer, inner], [outer]]
        decoded = decode_model(encode_model([part]).cad_vec)[0]
        self.assertEqual(len(decoded["sketch"]), 2)
        self.assertEqual(len(decoded["sketch"][0]), 2)
        self.assertEqual(decoded["sketch"][0][1][0]["type"], "circle")

    def test_decode_requires_start_wrapper(self):
        with self.assertRaises(CadVecError):
            decode_model([(20, 20), (END_CURVE, 0)])

    def test_empty_model_rejected(self):
        with self.assertRaises(CadVecError):
            encode_model([])


class TestSplitTokens(unittest.TestCase):
    def test_split_drops_the_terminator(self):
        toks = [(20, 20), (5, 0), (21, 21), (5, 0)]
        self.assertEqual(split_tokens(toks, 5), [[(20, 20)], [(21, 21)]])

    def test_trailing_remainder_is_not_a_chunk(self):
        self.assertEqual(split_tokens([(20, 20)], 5), [])


if __name__ == "__main__":
    unittest.main()
