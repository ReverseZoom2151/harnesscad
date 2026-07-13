import unittest

from harnesscad.domain.reconstruction.tokens.cadparser_schema import (
    COMMAND_TYPES, N_COMMAND_TYPES, PARAM_LEN, PARAM_ONEHOT_DIM, UNUSED,
    Command, command, command_onehot, dequantize, from_vector, pad_sequence,
    param_index, param_names, param_onehot, quantize, sequence_matrix, to_vector,
    NC_DEFAULT, SOS, EOS, PAD,
)


class TestSchema(unittest.TestCase):
    def test_vocab_sizes(self):
        self.assertEqual(PARAM_LEN, 19)
        self.assertEqual(PARAM_ONEHOT_DIM, 257)
        self.assertEqual(N_COMMAND_TYPES, len(set(COMMAND_TYPES)))

    def test_aliases_share_param_footprint(self):
        self.assertEqual(param_names("Ec"), param_names("E"))
        self.assertEqual(param_names("Rc"), param_names("R"))
        self.assertEqual(param_names("Cf"), param_names("F"))

    def test_to_vector_unused_is_minus_one(self):
        vec = to_vector(command("L", x=0.5, y=-0.25))
        self.assertEqual(len(vec), PARAM_LEN)
        self.assertEqual(vec[0], 0.5)
        self.assertEqual(vec[1], -0.25)
        # every other slot is the -1 sentinel
        self.assertTrue(all(v == UNUSED for v in vec[2:]))

    def test_vector_roundtrip(self):
        cmd = command("E", tx=0.1, ty=0.2, tz=0.3, theta=0.0, gamma=0.0,
                      delta=0.0, s=0.5, e1=0.4, e2=-0.4)
        vec = to_vector(cmd)
        back = from_vector("E", vec)
        self.assertEqual(back.get("e1"), 0.4)
        self.assertEqual(back.get("e2"), -0.4)
        self.assertEqual(to_vector(back), vec)

    def test_reject_unknown_param(self):
        with self.assertRaises(ValueError):
            Command("L", (("r", 1.0),))
        with self.assertRaises(ValueError):
            Command("nope")

    def test_pad_sequence_terminators(self):
        seq = pad_sequence([command("L", x=0.0, y=0.0)], nc=8)
        self.assertEqual(len(seq), 8)
        self.assertEqual(seq[0].type, SOS)
        self.assertEqual(seq[2].type, EOS)
        self.assertTrue(all(c.type == PAD for c in seq[3:]))

    def test_pad_overflow_raises(self):
        with self.assertRaises(ValueError):
            pad_sequence([command("L", x=0.0, y=0.0)] * 10, nc=3)

    def test_sequence_matrix_shape(self):
        types, matrix = sequence_matrix([command("C", x=0.0, y=0.0, r=0.5)], nc=NC_DEFAULT)
        self.assertEqual(len(types), NC_DEFAULT)
        self.assertEqual(len(matrix), NC_DEFAULT)
        self.assertTrue(all(len(row) == PARAM_LEN for row in matrix))

    def test_command_onehot(self):
        oh = command_onehot("R")
        self.assertEqual(len(oh), N_COMMAND_TYPES)
        self.assertEqual(sum(oh), 1)

    def test_param_index_sentinel_and_levels(self):
        self.assertEqual(param_index(UNUSED), 0)          # -1 -> index 0
        self.assertEqual(param_index(-1.0, low=-1.0), 0)  # sentinel wins
        self.assertEqual(param_index(1.0), 256)           # top level -> 256
        # a genuine low-end value maps to level 0 -> index 1 (not the sentinel)
        self.assertEqual(param_index(-0.999), 1)
        oh = param_onehot(0.0)
        self.assertEqual(len(oh), PARAM_ONEHOT_DIM)
        self.assertEqual(sum(oh), 1)

    def test_quantize_roundtrip_monotone(self):
        self.assertEqual(quantize(-1.0), 0)
        self.assertEqual(quantize(1.0), 255)
        self.assertAlmostEqual(dequantize(quantize(0.5)), 0.5, places=2)


if __name__ == "__main__":
    unittest.main()
