"""Tests for reconstruction.vitruvion_primitive_tokens."""

import unittest

from geometry.vitruvion_sketch_norm import VArc, VCircle, VLine, VPoint, entity_from_params
from reconstruction.vitruvion_primitive_tokens import (
    COORD_TOKEN_MAP,
    DEFAULT_NUM_BINS,
    GATHER_MAP,
    NON_COORD_TOKEN,
    Token,
    construction_tokens,
    coordinate_token_range,
    dequantize_params,
    entities_from_tokens,
    pad_or_truncate,
    param_seq_from_tokens,
    quantize_params,
    tokenize_sketch,
    vocabulary_size,
)


class TestVocabulary(unittest.TestCase):
    def test_control_tokens(self):
        self.assertEqual(len(Token), 7)
        self.assertEqual(int(Token.Pad), 0)
        self.assertEqual(int(Token.Stop), 2)
        self.assertEqual(int(Token.Point), 6)

    def test_coord_slot_ids_are_contiguous_in_type_order(self):
        self.assertEqual(COORD_TOKEN_MAP[VArc], [2, 3, 4, 5, 6, 7])
        self.assertEqual(COORD_TOKEN_MAP[VCircle], [8, 9, 10])
        self.assertEqual(COORD_TOKEN_MAP[VLine], [11, 12, 13, 14])
        self.assertEqual(COORD_TOKEN_MAP[VPoint], [15, 16])

    def test_flag_tokens_sit_above_the_bins(self):
        self.assertEqual(construction_tokens(64), {True: 71, False: 72})
        self.assertEqual(coordinate_token_range(64), (7, 70))
        self.assertEqual(vocabulary_size(64), 73)


class TestQuantization(unittest.TestCase):
    def test_truncating_not_rounding(self):
        # 0.0 sits exactly on a bin boundary with an even bin count.
        self.assertEqual(quantize_params([0.0], 64), [32])
        # A value just below the boundary must floor DOWN (a rounding quantiser,
        # like DeepCAD's, would return 32 here).
        self.assertEqual(quantize_params([-0.0001], 64), [31])
        self.assertEqual(quantize_params([0.0078], 64), [32])

    def test_domain_edges(self):
        self.assertEqual(quantize_params([-0.5], 64), [0])
        self.assertEqual(quantize_params([0.5], 64), [63])  # clamped from bin 64

    def test_out_of_domain_raises(self):
        with self.assertRaises(ValueError):
            quantize_params([0.6], 64)
        with self.assertRaises(ValueError):
            quantize_params([-0.5001], 64)

    def test_ten_decimal_rounding_tolerates_float_noise(self):
        self.assertEqual(quantize_params([0.5 + 1e-12], 64), [63])

    def test_dequantize_lands_on_bin_center(self):
        # Bin centres, NOT bin edges: bin 0 -> -0.5 + 1/128.
        self.assertAlmostEqual(dequantize_params([0], 64)[0], -0.5 + 1.0 / 128)
        self.assertAlmostEqual(dequantize_params([63], 64)[0], 0.5 - 1.0 / 128)
        self.assertAlmostEqual(dequantize_params([32], 64)[0], 1.0 / 128)

    def test_roundtrip_error_is_bounded_by_half_a_bin(self):
        half_bin = 0.5 / DEFAULT_NUM_BINS
        for i in range(101):
            value = -0.5 + i / 100.0
            back = dequantize_params(quantize_params([value]))[0]
            self.assertLessEqual(abs(back - value), half_bin + 1e-9)

    def test_dequantize_rejects_out_of_range_bins(self):
        with self.assertRaises(ValueError):
            dequantize_params([64], 64)
        with self.assertRaises(ValueError):
            dequantize_params([-1], 64)


class TestTokenizeSketch(unittest.TestCase):
    def setUp(self):
        self.entities = [
            VCircle(xCenter=0.0, yCenter=0.0, radius=0.25),
            entity_from_params([-0.5, -0.5, 0.5, 0.5]),
        ]

    def test_stream_lengths_agree(self):
        streams, _ = tokenize_sketch(self.entities, 64)
        self.assertEqual(len(streams["val"]), len(streams["coord"]))
        self.assertEqual(len(streams["val"]), len(streams["pos"]))
        # Start + (type + 3 params + flag) + (type + 4 params + flag) + Stop
        self.assertEqual(len(streams["val"]), 1 + 5 + 6 + 1)

    def test_stream_contents(self):
        streams, _ = tokenize_sketch(self.entities, 64)
        val, coord, pos = streams["val"], streams["coord"], streams["pos"]
        self.assertEqual(val[0], int(Token.Start))
        self.assertEqual(val[1], int(Token.Circle))
        self.assertEqual(val[-1], int(Token.Stop))
        # Circle params: centre (0, 0) -> bin 32 -> token 39; radius 0.25 -> bin 48.
        self.assertEqual(val[2:5], [39, 39, 48 + len(Token)])
        self.assertEqual(val[5], construction_tokens(64)[False])
        self.assertEqual(coord[0], NON_COORD_TOKEN)
        self.assertEqual(coord[2:5], COORD_TOKEN_MAP[VCircle])
        self.assertEqual(coord[5], NON_COORD_TOKEN)
        # pos: 1 for Start, 2 for every token of the first primitive, 3 for the second.
        self.assertEqual(pos, [1, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 4])

    def test_construction_flag_token(self):
        circle = VCircle(radius=0.25, is_construction=True)
        streams, _ = tokenize_sketch([circle], 64)
        self.assertEqual(streams["val"][5], construction_tokens(64)[True])

    def test_gather_idxs_point_at_the_val_stream(self):
        streams, gather = tokenize_sketch(self.entities, 64)
        self.assertEqual(gather[0], 0)
        # Circle: entity token at index 1, centre.x at index 2.
        self.assertEqual(gather[1:3], [1, 2])
        # Line: entity token at index 6, start.x at 7, end.x at 9.
        self.assertEqual(gather[3:6], [6, 7, 9])
        self.assertEqual(streams["val"][gather[1]], int(Token.Circle))
        self.assertEqual(streams["val"][gather[3]], int(Token.Line))

    def test_gather_map_offsets(self):
        self.assertEqual(GATHER_MAP[VArc], [0, 1, 3, 5])
        self.assertEqual(GATHER_MAP[VLine], [0, 1, 3])

    def test_padding_and_truncation(self):
        streams, _ = tokenize_sketch(self.entities, 64, max_length=20)
        self.assertEqual(len(streams["val"]), 20)
        self.assertEqual(streams["val"][-1], int(Token.Pad))
        short, _ = tokenize_sketch(self.entities, 64, max_length=4)
        self.assertEqual(len(short["pos"]), 4)

    def test_pad_or_truncate_identity(self):
        self.assertEqual(pad_or_truncate([1, 2, 3]), [1, 2, 3])

    def test_no_stop_no_construction(self):
        streams, _ = tokenize_sketch(
            [VPoint(x=0.1, y=0.2)], 64, include_construction=False, include_stop=False
        )
        # Start + type + 2 params, with no flag token and no Stop.
        self.assertEqual(len(streams["val"]), 4)

    def test_out_of_domain_params_are_clipped_not_raised(self):
        streams, _ = tokenize_sketch([VPoint(x=0.9, y=-0.9)], 64)
        self.assertEqual(streams["val"][2:4], [63 + len(Token), 0 + len(Token)])

    def test_unsupported_entity_raises(self):
        with self.assertRaises(ValueError):
            tokenize_sketch([object()], 64)


class TestDecoding(unittest.TestCase):
    def test_param_seq_splits_by_type_token(self):
        entities = [VCircle(radius=0.25), VPoint(x=0.1, y=0.1, is_construction=True)]
        streams, _ = tokenize_sketch(entities, 64)
        seq = param_seq_from_tokens(streams["val"], 64)
        self.assertEqual(len(seq), 2)
        self.assertEqual(len(seq[0][0]), 3)
        self.assertFalse(seq[0][1])
        self.assertEqual(len(seq[1][0]), 2)
        self.assertTrue(seq[1][1])

    def test_decoding_stops_at_pad(self):
        entities = [VCircle(radius=0.25)]
        streams, _ = tokenize_sketch(entities, 64, max_length=32)
        self.assertEqual(len(param_seq_from_tokens(streams["val"], 64)), 1)

    def test_roundtrip_entities(self):
        entities = [
            VCircle(xCenter=0.1, yCenter=-0.1, radius=0.25),
            entity_from_params([-0.25, -0.25, 0.25, 0.25]),
            VPoint(x=0.0, y=0.375),
        ]
        streams, _ = tokenize_sketch(entities, 64)
        rebuilt = entities_from_tokens(streams["val"], 64)
        self.assertEqual([type(e) for e in rebuilt], [VCircle, VLine, VPoint])
        half_bin = 0.5 / 64
        self.assertLessEqual(abs(rebuilt[0].radius - 0.25), half_bin)
        self.assertLessEqual(abs(rebuilt[2].y - 0.375), half_bin)

    def test_arc_roundtrip(self):
        arc = VArc(xCenter=0.0, yCenter=0.0, radius=0.25, startParam=0.0, endParam=2.0)
        streams, _ = tokenize_sketch([arc], 64)
        rebuilt = entities_from_tokens(streams["val"], 64)
        self.assertIsInstance(rebuilt[0], VArc)
        self.assertLessEqual(abs(rebuilt[0].radius - 0.25), 0.05)


if __name__ == "__main__":
    unittest.main()
