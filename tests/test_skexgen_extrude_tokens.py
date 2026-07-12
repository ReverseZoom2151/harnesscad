import unittest

from reconstruction.skexgen_extrude_tokens import (
    EXT_FLAGS, EXT_SEQ_LEN, OP_ADD, OP_CUT, OP_INTERSECT, decode_extrude,
    encode_extrude, extrude_flags, extrude_vocab_size, flag_field_positions,
    is_valid_extrude_block, op_token,
)
from reconstruction.skexgen_token_format import shift_stream

KWARGS = dict(
    extrude_value=(0.5, -0.25),
    origin=(0.0, 0.25, -0.5),
    x_axis=(1.0, 0.0, 0.0),
    y_axis=(0.0, 1.0, 0.0),
    z_axis=(0.0, 0.0, 1.0),
    set_op="NewBodyFeatureOperation",
    scale=0.7,
    offset=(0.1, -0.2),
)


class TestEncode(unittest.TestCase):
    def test_length_and_end(self):
        block = encode_extrude(**KWARGS)
        self.assertEqual(len(block), EXT_SEQ_LEN)
        self.assertEqual(block[-1], 0)

    def test_rotation_trits(self):
        block = encode_extrude(**KWARGS)
        self.assertEqual(block[5:14], [3, 2, 2, 2, 3, 2, 2, 2, 3])

    def test_rotation_clipped_and_rounded(self):
        kw = dict(KWARGS, x_axis=(-0.9, 4.0, 0.2))
        block = encode_extrude(**kw)
        self.assertEqual(block[5:8], [1, 3, 2])

    def test_op_tokens(self):
        self.assertEqual(op_token("JoinFeatureOperation"), OP_ADD)
        self.assertEqual(op_token("NewBodyFeatureOperation"), OP_ADD)
        self.assertEqual(op_token("CutFeatureOperation"), OP_CUT)
        self.assertEqual(op_token("IntersectFeatureOperation"), OP_INTERSECT)
        self.assertRaises(ValueError, op_token, "FilletOperation")

    def test_bad_shapes(self):
        self.assertRaises(ValueError, encode_extrude, **dict(KWARGS, extrude_value=(1.0,)))
        self.assertRaises(ValueError, encode_extrude, **dict(KWARGS, origin=(0.0, 0.0)))
        self.assertRaises(ValueError, encode_extrude, **dict(KWARGS, offset=(0.0, 0.0, 0.0)))

    def test_all_tokens_nonneg(self):
        self.assertTrue(all(t >= 0 for t in encode_extrude(**KWARGS)))

    def test_deterministic(self):
        self.assertEqual(encode_extrude(**KWARGS), encode_extrude(**KWARGS))


class TestDecode(unittest.TestCase):
    def test_roundtrip_shifted(self):
        block = shift_stream(encode_extrude(**KWARGS))
        got = decode_extrude(block)
        self.assertAlmostEqual(got["value"][0], 0.5, places=1)
        self.assertAlmostEqual(got["value"][1], -0.25, places=1)
        self.assertEqual(got["x_axis"], [1, 0, 0])
        self.assertEqual(got["z_axis"], [0, 0, 1])
        self.assertEqual(got["op_name"], "add")
        self.assertAlmostEqual(got["scale"], 0.7, places=1)
        self.assertAlmostEqual(got["offset"][0], 0.1, places=1)

    def test_roundtrip_raw(self):
        block = encode_extrude(**KWARGS)
        got = decode_extrude(block, shifted=False)
        self.assertEqual(got["op"], OP_ADD)

    def test_cut_op(self):
        block = shift_stream(encode_extrude(**dict(KWARGS, set_op="CutFeatureOperation")))
        self.assertEqual(decode_extrude(block)["op_name"], "cut")

    def test_bad_length(self):
        self.assertRaises(ValueError, decode_extrude, [1] * 18)

    def test_bad_end_token(self):
        block = shift_stream(encode_extrude(**KWARGS))
        block[-1] = 5
        self.assertRaises(ValueError, decode_extrude, block)

    def test_bad_op(self):
        block = shift_stream(encode_extrude(**KWARGS))
        block[14] = 9
        self.assertRaises(ValueError, decode_extrude, block)


class TestFlags(unittest.TestCase):
    def test_flag_stream(self):
        self.assertEqual(len(EXT_FLAGS), EXT_SEQ_LEN)
        flags = extrude_flags(2)
        self.assertEqual(len(flags), 2 * EXT_SEQ_LEN + 1)
        self.assertEqual(flags[-1], 0)
        self.assertEqual(flags[:2], [1, 1])
        self.assertRaises(ValueError, extrude_flags, -1)

    def test_field_positions(self):
        self.assertEqual(flag_field_positions(1), [0, 1])
        self.assertEqual(flag_field_positions(3), list(range(5, 14)))
        self.assertEqual(flag_field_positions(4), [14])
        self.assertEqual(flag_field_positions(7), [18])


class TestValidity(unittest.TestCase):
    def test_valid(self):
        self.assertTrue(is_valid_extrude_block(shift_stream(encode_extrude(**KWARGS))))

    def test_invalid_rotation(self):
        block = shift_stream(encode_extrude(**KWARGS))
        block[5] = 40
        self.assertFalse(is_valid_extrude_block(block))

    def test_invalid_truncated(self):
        self.assertFalse(is_valid_extrude_block([1, 2, 3]))

    def test_vocab_size(self):
        self.assertEqual(extrude_vocab_size(6), 64 + 2)


if __name__ == "__main__":
    unittest.main()
