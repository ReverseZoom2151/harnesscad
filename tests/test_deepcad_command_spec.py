"""Tests for the exact DeepCAD command-sequence spec."""

import unittest

from harnesscad.domain.reconstruction.tokens import deepcad_command_spec as spec


class TestVocabulary(unittest.TestCase):
    def test_exactly_six_types(self):
        self.assertEqual(spec.N_COMMAND_TYPES, 6)
        self.assertEqual(
            spec.COMMAND_TYPES,
            (spec.SOL, spec.LINE, spec.ARC, spec.CIRCLE, spec.EXT, spec.EOS))

    def test_sixteen_param_slots_exact_order(self):
        self.assertEqual(spec.PARAM_LEN, 16)
        self.assertEqual(
            spec.PARAM_SLOTS,
            ("x", "y", "alpha", "f", "r", "theta", "phi", "gamma",
             "px", "py", "pz", "s", "e1", "e2", "b", "u"))

    def test_param_names_per_type(self):
        self.assertEqual(spec.param_names(spec.LINE), ("x", "y"))
        self.assertEqual(spec.param_names(spec.ARC), ("x", "y", "alpha", "f"))
        self.assertEqual(spec.param_names(spec.CIRCLE), ("x", "y", "r"))
        self.assertEqual(len(spec.param_names(spec.EXT)), 11)
        self.assertEqual(spec.param_names(spec.SOL), ())
        self.assertEqual(spec.param_names(spec.EOS), ())


class TestVectorPacking(unittest.TestCase):
    def test_unused_slots_are_minus_one(self):
        vec = spec.to_vector(spec.command(spec.LINE, x=0.5, y=-0.25))
        self.assertEqual(len(vec), 16)
        self.assertEqual(vec[spec.PARAM_INDEX["x"]], 0.5)
        self.assertEqual(vec[spec.PARAM_INDEX["y"]], -0.25)
        self.assertEqual(vec[spec.PARAM_INDEX["r"]], spec.UNUSED)
        self.assertTrue(all(v == spec.UNUSED for v in
                            (vec[spec.PARAM_INDEX["theta"]], vec[spec.PARAM_INDEX["u"]])))

    def test_sol_and_eos_all_sentinel(self):
        self.assertTrue(all(v == spec.UNUSED for v in spec.to_vector(spec.Command(spec.SOL))))
        self.assertTrue(all(v == spec.UNUSED for v in spec.to_vector(spec.Command(spec.EOS))))

    def test_roundtrip_vector(self):
        cmd = spec.command(spec.EXT, theta=0.1, phi=0.2, gamma=0.3,
                           px=0.4, py=-0.5, pz=0.6, s=0.7, e1=0.8, e2=-0.9, b=1, u=2)
        back = spec.from_vector(spec.EXT, spec.to_vector(cmd))
        self.assertEqual(back, cmd)

    def test_bad_param_rejected(self):
        with self.assertRaises(ValueError):
            spec.Command(spec.LINE, (("r", 0.1),))

    def test_unknown_type_rejected(self):
        with self.assertRaises(ValueError):
            spec.Command("Fillet")


class TestSequencePacking(unittest.TestCase):
    def test_pad_to_nc(self):
        seq = spec.pad_sequence([spec.command(spec.LINE, x=0.1, y=0.2)])
        self.assertEqual(len(seq), spec.NC_DEFAULT)
        self.assertEqual(seq[0].type, spec.LINE)
        self.assertTrue(all(c.type == spec.EOS for c in seq[1:]))

    def test_overflow_raises(self):
        with self.assertRaises(ValueError):
            spec.pad_sequence([spec.Command(spec.SOL)] * 3, nc=2)

    def test_vector_representation_roundtrip(self):
        cmds = [spec.Command(spec.SOL),
                spec.command(spec.LINE, x=0.1, y=0.2),
                spec.command(spec.ARC, x=0.3, y=0.4, alpha=0.5, f=1),
                spec.command(spec.EXT, theta=0.0, phi=0.0, gamma=0.0,
                             px=0, py=0, pz=0, s=1, e1=0.5, e2=0, b=0, u=0)]
        types, matrix = spec.vector_representation(cmds)
        self.assertEqual(len(types), spec.NC_DEFAULT)
        self.assertEqual(len(matrix[0]), 16)
        self.assertEqual(spec.commands_from_vectors(types, matrix), cmds)


class TestQuantization(unittest.TestCase):
    def test_levels_and_onehot_width(self):
        self.assertEqual(spec.N_QUANT_LEVELS, 256)
        self.assertEqual(spec.PARAM_ONEHOT_DIM, 257)

    def test_quantize_bounds(self):
        self.assertEqual(spec.quantize(-1.0), 0)
        self.assertEqual(spec.quantize(1.0), 255)
        self.assertEqual(spec.quantize(-5.0), 0)  # clamped

    def test_quantize_dequantize_close(self):
        for v in (-0.9, -0.3, 0.0, 0.42, 0.99):
            q = spec.quantize(v)
            self.assertLess(abs(spec.dequantize(q) - v), 2.0 / 255 + 1e-9)

    def test_sentinel_maps_to_zero(self):
        self.assertEqual(spec.param_index(spec.UNUSED), 0)
        self.assertEqual(spec.param_onehot(spec.UNUSED)[0], 1)

    def test_param_index_offset_by_one(self):
        self.assertEqual(spec.param_index(-1.0 + 2e-9), spec.quantize(-1.0 + 2e-9) + 1)

    def test_command_onehot(self):
        oh = spec.command_onehot(spec.ARC)
        self.assertEqual(sum(oh), 1)
        self.assertEqual(oh[spec.COMMAND_INDEX[spec.ARC]], 1)


if __name__ == "__main__":
    unittest.main()
