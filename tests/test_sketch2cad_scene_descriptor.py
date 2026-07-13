"""Tests for the Sketch2CAD scene-descriptor token codec."""

import random
import unittest

from harnesscad.domain.reconstruction.tokens import sketch2cad_scene as sd


class TestQuantize(unittest.TestCase):
    def test_endpoints(self):
        self.assertEqual(sd.quantize(0.0, 0.0, 200.0, 200), 0)
        self.assertEqual(sd.quantize(200.0, 0.0, 200.0, 200), 199)

    def test_clamp(self):
        self.assertEqual(sd.quantize(-10.0, 0.0, 200.0, 200), 0)
        self.assertEqual(sd.quantize(500.0, 0.0, 200.0, 200), 199)

    def test_round_half_up(self):
        # midpoint of a 4-bin [0,360] lattice: bin width 120; 60 -> 0.5 level -> 1
        self.assertEqual(sd.quantize(60.0, 0.0, 360.0, 4), 1)

    def test_single_bin(self):
        self.assertEqual(sd.quantize(5.0, 0.0, 10.0, 1), 0)

    def test_dequantize_centres(self):
        for lvl in range(4):
            v = sd.dequantize(lvl, 0.0, 360.0, 4)
            self.assertEqual(sd.quantize(v, 0.0, 360.0, 4), lvl)

    def test_bad_bins(self):
        with self.assertRaises(ValueError):
            sd.quantize(1.0, 0.0, 1.0, 0)


class TestVocabLayout(unittest.TestCase):
    def test_total_vocab_size_formula(self):
        c = sd.DescriptorConfig(
            n_cam_pose=60, n_bin_pos=200, n_bin_rot=4, n_bin_size=60
        )
        expected = 60 + len(sd.SHAPE_TYPES) + 200 + 4 + 60
        self.assertEqual(c.vocab_size, expected)

    def test_blocks_are_contiguous_and_disjoint(self):
        c = sd.DescriptorConfig()
        self.assertEqual(c.off_cam, 0)
        self.assertEqual(c.off_shape, c.n_cam_pose)
        self.assertEqual(c.off_pos, c.off_shape + c.n_shape_type)
        self.assertEqual(c.off_rot, c.off_pos + c.n_bin_pos)
        self.assertEqual(c.off_size, c.off_rot + c.n_bin_rot)

    def test_seven_shapes(self):
        self.assertEqual(len(sd.SHAPE_TYPES), 7)
        self.assertIn("mansard", sd.SHAPE_INDEX)


class TestObjectRow(unittest.TestCase):
    def setUp(self):
        self.codec = sd.SceneDescriptorCodec()

    def test_row_length(self):
        obj = sd.SceneObject("cube", (10.0, 20.0, 5.0), (90.0, 0.0), (4.0, 4.0, 3.0))
        row = self.codec.encode_object(obj)
        self.assertEqual(len(row), sd.TOKENS_PER_OBJECT)

    def test_tokens_in_expected_blocks(self):
        c = self.codec.config
        obj = sd.SceneObject("cylinder", (10.0, 20.0, 5.0), (90.0, 0.0), (4.0, 4.0, 3.0))
        row = self.codec.encode_object(obj)
        # shape token in shape block
        self.assertTrue(c.off_shape <= row[0] < c.off_pos)
        for i in (1, 2, 3):
            self.assertTrue(c.off_pos <= row[i] < c.off_rot)
        for i in (4, 5):
            self.assertTrue(c.off_rot <= row[i] < c.off_size)
        for i in (6, 7, 8):
            self.assertTrue(c.off_size <= row[i] < c.vocab_size)

    def test_object_roundtrip(self):
        obj = sd.SceneObject("pyramid", (100.0, 50.0, 0.0), (180.0, 0.0), (30.0, 30.0, 20.0))
        row = self.codec.encode_object(obj)
        back = self.codec.decode_object(row)
        self.assertEqual(back.shape, "pyramid")
        for a, b in zip(back.position, obj.position):
            self.assertAlmostEqual(a, b, delta=1.0)  # within one pos bin
        for a, b in zip(back.size, obj.size):
            self.assertAlmostEqual(a, b, delta=1.0)

    def test_unknown_shape_rejected(self):
        with self.assertRaises(ValueError):
            sd.SceneObject("dome", (0, 0, 0), (0, 0), (1, 1, 1))


class TestScene(unittest.TestCase):
    def setUp(self):
        self.codec = sd.SceneDescriptorCodec()
        self.objs = [
            sd.SceneObject("cube", (10.0, 20.0, 5.0), (0.0, 0.0), (4.0, 4.0, 3.0)),
            sd.SceneObject("cylinder", (100.0, 30.0, 0.0), (90.0, 0.0), (5.0, 5.0, 8.0)),
        ]

    def test_scene_prefix_is_pose(self):
        seq = self.codec.encode_scene(7, self.objs)
        self.assertEqual(seq[0], 7)  # off_cam == 0
        self.assertEqual(len(seq), 1 + 2 * sd.TOKENS_PER_OBJECT)

    def test_scene_roundtrip(self):
        seq = self.codec.encode_scene(42, self.objs)
        pose, objs = self.codec.decode_scene(seq)
        self.assertEqual(pose, 42)
        self.assertEqual(len(objs), 2)
        self.assertEqual([o.shape for o in objs], ["cube", "cylinder"])

    def test_pose_out_of_range(self):
        with self.assertRaises(ValueError):
            self.codec.encode_scene(999, self.objs)

    def test_bad_body_length(self):
        with self.assertRaises(ValueError):
            self.codec.decode_scene([0, 1, 2, 3])

    def test_random_ordering_deterministic(self):
        rng1 = random.Random(123)
        rng2 = random.Random(123)
        s1 = sd.serialize_scene(self.codec, 3, self.objs, rng1)
        s2 = sd.serialize_scene(self.codec, 3, self.objs, rng2)
        self.assertEqual(s1, s2)
        self.assertEqual(s1[0], 3)  # pose stays first
        # decodes to the same set of shapes
        _, objs = self.codec.decode_scene(s1)
        self.assertEqual({o.shape for o in objs}, {"cube", "cylinder"})


if __name__ == "__main__":
    unittest.main()
