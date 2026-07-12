import unittest

from drawings.s2cadsig_block_codec import (
    BlockFormatError,
    BlockShape,
    CURVE_BASE,
    CURVE_OFFSET,
    CURVE_PROFILE,
    NUM_INPUT_CHANNELS,
    NUM_LABEL_CHANNELS,
    NUM_RAW_CHANNELS,
    build_raw_block,
    channel_index,
    cook_raw_inputs,
    decode_block,
    encode_block,
    inflate_block,
    pixel_slice,
    split_block,
)


def _sample_shape():
    return BlockShape(2, 3)


def _sample_values(shape):
    # deterministic ramp
    return [float(i % 7) / 7.0 for i in range(shape.size)]


class TestBlockShape(unittest.TestCase):
    def test_sizes(self):
        s = _sample_shape()
        self.assertEqual(s.pixels, 6)
        self.assertEqual(s.channels, NUM_RAW_CHANNELS)
        self.assertEqual(s.size, 6 * 17)

    def test_bad_dims(self):
        with self.assertRaises(BlockFormatError):
            BlockShape(0, 4)

    def test_channel_index(self):
        self.assertEqual(channel_index("user_stroke"), 0)
        self.assertEqual(channel_index("operation_type"), 16)
        with self.assertRaises(BlockFormatError):
            channel_index("nope")


class TestCodec(unittest.TestCase):
    def test_roundtrip(self):
        shape = _sample_shape()
        vals = _sample_values(shape)
        stream = encode_block(vals, shape)
        out = inflate_block(stream, shape)
        self.assertEqual(len(out), shape.size)
        for a, b in zip(vals, out):
            self.assertAlmostEqual(a, b, places=6)

    def test_deterministic_stream(self):
        shape = _sample_shape()
        vals = _sample_values(shape)
        self.assertEqual(encode_block(vals, shape), encode_block(vals, shape))

    def test_length_mismatch_on_encode(self):
        shape = _sample_shape()
        with self.assertRaises(BlockFormatError):
            encode_block([0.0, 1.0], shape)

    def test_shape_mismatch_on_inflate(self):
        shape = _sample_shape()
        stream = encode_block(_sample_values(shape), shape)
        with self.assertRaises(BlockFormatError):
            inflate_block(stream, BlockShape(3, 3))

    def test_bad_stream(self):
        with self.assertRaises(BlockFormatError):
            inflate_block(b"not-zlib", _sample_shape())

    def test_pixel_slice(self):
        shape = _sample_shape()
        vals = _sample_values(shape)
        px = pixel_slice(vals, shape, 1, 2)
        self.assertEqual(len(px), NUM_RAW_CHANNELS)
        self.assertEqual(px, vals[5 * 17:6 * 17])
        with self.assertRaises(BlockFormatError):
            pixel_slice(vals, shape, 5, 0)


class TestSplit(unittest.TestCase):
    def test_channel_counts(self):
        shape = _sample_shape()
        inp, lab = split_block(_sample_values(shape), shape)
        self.assertEqual(len(inp), shape.pixels * NUM_INPUT_CHANNELS)
        self.assertEqual(len(lab), shape.pixels * NUM_LABEL_CHANNELS)

    def test_input_channels_copied(self):
        shape = BlockShape(1, 1)
        vals = build_raw_block(
            {
                "user_stroke": [1.0],
                "scaffold_lines": [0.5],
                "context_normal_x": [0.0],
                "context_normal_y": [0.0],
                "context_normal_z": [1.0],
                "context_depth": [2.5],
            },
            shape,
        )
        inp, _ = split_block(vals, shape)
        self.assertEqual(inp, [1.0, 0.5, 0.0, 0.0, 1.0, 2.5])

    def test_curve_conflict_priority(self):
        shape = BlockShape(1, 4)
        # pixel0: base only; p1: base+offset+profile; p2: offset+profile; p3: none
        vals = build_raw_block(
            {
                "base_curve": [1.0, 1.0, 0.0, 0.0],
                "offset_curve": [0.0, 1.0, 1.0, 0.0],
                "profile_curve": [0.0, 1.0, 1.0, 0.0],
            },
            shape,
        )
        _, lab = split_block(vals, shape)

        def cls(i):
            b = i * NUM_LABEL_CHANNELS
            return (lab[b + 11], lab[b + 12], lab[b + 13], lab[b + 14])

        self.assertEqual(cls(0), (1.0, 0.0, 0.0, float(CURVE_BASE)))
        self.assertEqual(cls(1), (1.0, 0.0, 0.0, float(CURVE_BASE)))
        self.assertEqual(cls(2), (0.0, 1.0, 0.0, float(CURVE_OFFSET)))
        self.assertEqual(cls(3), (0.0, 0.0, 0.0, float(CURVE_BASE)))

    def test_profile_only_gets_class_two(self):
        shape = BlockShape(1, 1)
        vals = build_raw_block({"profile_curve": [1.0]}, shape)
        _, lab = split_block(vals, shape)
        self.assertEqual(lab[13], 1.0)
        self.assertEqual(lab[14], float(CURVE_PROFILE))


class TestCook(unittest.TestCase):
    def test_named_maps(self):
        shape = BlockShape(1, 2)
        vals = build_raw_block(
            {
                "user_stroke": [1.0, 0.0],
                "context_depth": [3.0, 4.0],
                "context_normal_z": [1.0, 1.0],
                "stitching_face": [0.2, 0.9],
                "offset_distance": [0.0, 1.5],
                "offset_direction_x": [0.0, 1.0],
                "offset_sign": [0.0, -1.0],
                "operation_type": [2.0, 2.0],
                "base_curve": [0.0, 1.0],
            },
            shape,
        )
        s = cook_raw_inputs(vals, shape)
        self.assertEqual(s.user_stroke, [1.0, 0.0])
        self.assertEqual(s.context_depth, [3.0, 4.0])
        self.assertEqual(s.context_normal, [(0.0, 0.0, 1.0), (0.0, 0.0, 1.0)])
        self.assertEqual(s.stitching_face, [0.2, 0.9])
        self.assertEqual(s.offset_distance, [0.0, 1.5])
        self.assertEqual(s.offset_direction[1], (1.0, 0.0, 0.0))
        self.assertEqual(s.offset_sign, [0.0, -1.0])
        self.assertEqual(s.operation_label, 2)
        self.assertEqual(s.curve_class[1], (1.0, 0.0, 0.0))
        self.assertEqual(s.stroke_mask(), [0.0, 1.0])
        self.assertIn("shape_mask", s.as_dict())

    def test_decode_block_end_to_end(self):
        shape = BlockShape(2, 2)
        vals = build_raw_block({"operation_type": [1.0] * 4}, shape)
        inp, lab = decode_block(encode_block(vals, shape), shape)
        self.assertEqual(len(inp), 4 * NUM_INPUT_CHANNELS)
        self.assertEqual(lab[10], 1.0)

    def test_build_raw_block_validates(self):
        with self.assertRaises(BlockFormatError):
            build_raw_block({"user_stroke": [1.0]}, BlockShape(2, 2))


if __name__ == "__main__":
    unittest.main()
