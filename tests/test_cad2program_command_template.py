import unittest

from harnesscad.domain.reconstruction.translate.shape_program import (
    ShapeProgram, make_instance, PrimitiveInstance,
)
from harnesscad.domain.reconstruction.tokens.cad2program import (
    quantize_value, dequantize_value, quantize_angle, dequantize_angle,
    encode_command, decode_command, encode_program, quantization_error,
    Command, DEFAULT_RESOLUTION, DEFAULT_N_BINS,
)


class QuantizeValueTest(unittest.TestCase):
    def test_on_grid(self):
        self.assertEqual(quantize_value(9.0, resolution=3.0), 3)
        self.assertEqual(dequantize_value(3, resolution=3.0), 9.0)

    def test_round_to_nearest(self):
        self.assertEqual(quantize_value(10.0, resolution=3.0), 3)   # 10/3=3.33
        self.assertEqual(quantize_value(11.0, resolution=3.0), 4)   # 11/3=3.67

    def test_clamp_high(self):
        self.assertEqual(quantize_value(1e9, resolution=3.0, n_bins=1500), 1499)

    def test_clamp_low(self):
        self.assertEqual(quantize_value(-50.0, resolution=3.0), 0)

    def test_bad_resolution(self):
        with self.assertRaises(ValueError):
            quantize_value(1.0, resolution=0)


class QuantizeAngleTest(unittest.TestCase):
    def test_four_bins(self):
        self.assertEqual(quantize_angle(0), 0)
        self.assertEqual(quantize_angle(90), 1)
        self.assertEqual(quantize_angle(180), 2)
        self.assertEqual(quantize_angle(270), 3)
        self.assertEqual(quantize_angle(360), 0)

    def test_dequantize(self):
        self.assertEqual(dequantize_angle(1), 90.0)
        self.assertEqual(dequantize_angle(3), 270.0)


class CommandTest(unittest.TestCase):
    def test_encode_decode_on_grid(self):
        inst = make_instance("m1", (6, 9, 12, 3, 6, 9, 90))
        cmd = encode_command(inst)
        self.assertIsInstance(cmd, Command)
        self.assertEqual(cmd.model_id, "model_m1")
        self.assertEqual(len(cmd.param_bins), 6)
        dec = decode_command(cmd)
        self.assertEqual(dec.bbox.position, (6, 9, 12))
        self.assertEqual(dec.bbox.size, (3, 6, 9))
        self.assertEqual(dec.bbox.angle_z, 90.0)

    def test_model_specific_params_dropped(self):
        inst = make_instance("m1", (3, 3, 3, 3, 3, 3, 0), {"N": 2, "BT": 9})
        dec = decode_command(encode_command(inst))
        self.assertEqual(dec.params, ())

    def test_encode_program(self):
        prog = ShapeProgram([make_instance("a", (3, 3, 3, 3, 3, 3, 0)),
                             make_instance("b", (6, 6, 6, 6, 6, 6, 0))])
        cmds = encode_program(prog)
        self.assertEqual(len(cmds), 2)


class QuantizationErrorTest(unittest.TestCase):
    def test_zero_on_grid(self):
        prog = ShapeProgram([make_instance("m", (3, 6, 9, 12, 15, 18, 0))])
        self.assertEqual(quantization_error(prog), 0.0)

    def test_nonzero_off_grid(self):
        # 407.5 with 3mm resolution: nearest bin center is 408 -> error 0.5.
        prog = ShapeProgram([make_instance("m", (407.5, 0, 0, 3, 3, 3, 0))])
        err = quantization_error(prog)
        self.assertAlmostEqual(err, 0.5)

    def test_text_would_be_lossless(self):
        # The paper's argument: the continuous value 407.5 is exact as text but
        # incurs a >0 quantization error under the command template.
        prog = ShapeProgram([make_instance("m", (407.5, 0, 0, 3, 3, 3, 0))])
        self.assertGreater(quantization_error(prog), 0.0)


if __name__ == "__main__":
    unittest.main()
