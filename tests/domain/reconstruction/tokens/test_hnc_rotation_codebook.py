import unittest

from harnesscad.domain.reconstruction.tokens.hnc_rotation_codebook import (
    ROTATION_FRAMES,
    NUM_FRAMES,
    clip_axis,
    clip_orientation,
    quantize_orientation,
    is_known_orientation,
    frame_pattern,
    frame_axes,
    frame_matrix,
)


class TestRotationCodebook(unittest.TestCase):
    def test_exactly_25_distinct_frames(self):
        self.assertEqual(NUM_FRAMES, 25)
        self.assertEqual(len(set(ROTATION_FRAMES)), 25)
        for f in ROTATION_FRAMES:
            self.assertEqual(len(f), 9)
            for c in f:
                self.assertIn(c, (-1, 0, 1))

    def test_clip_axis_rounds_and_clips(self):
        self.assertEqual(clip_axis([0.4, -0.6, 1.9]), (0, -1, 1))
        self.assertEqual(clip_axis([-2.0, 3.0, 0.0]), (-1, 1, 0))
        # round-half-to-even like numpy.rint
        self.assertEqual(clip_axis([0.5, 1.5, 2.5]), (0, 1, 1))

    def test_identity_frame_roundtrip(self):
        # Frame 20 is the world-aligned identity (x=+x, y=+y, z=+z).
        idx = quantize_orientation((1, 0, 0), (0, 1, 0), (0, 0, 1))
        self.assertEqual(frame_pattern(idx), (1, 0, 0, 0, 1, 0, 0, 0, 1))
        self.assertEqual(frame_axes(idx), ((1, 0, 0), (0, 1, 0), (0, 0, 1)))
        self.assertEqual(frame_matrix(idx), [[1, 0, 0], [0, 1, 0], [0, 0, 1]])

    def test_every_frame_quantizes_to_its_own_index(self):
        for i, f in enumerate(ROTATION_FRAMES):
            tx, ty, tz = f[0:3], f[3:6], f[6:9]
            self.assertEqual(quantize_orientation(tx, ty, tz), i)
            self.assertTrue(is_known_orientation(tx, ty, tz))

    def test_near_integer_floats_snap_to_frame(self):
        # noisy axis vectors still clip onto identity frame
        idx = quantize_orientation((0.98, 0.02, -0.01), (0.03, 1.01, 0.0),
                                   (0.0, -0.02, 0.99))
        self.assertEqual(frame_pattern(idx), (1, 0, 0, 0, 1, 0, 0, 0, 1))

    def test_unknown_orientation_rejected(self):
        # all-zero axes clip to a length-9 zero pattern not in the table
        self.assertFalse(is_known_orientation((0, 0, 0), (0, 0, 0), (0, 0, 0)))
        with self.assertRaises(ValueError):
            quantize_orientation((0, 0, 0), (0, 0, 0), (0, 0, 0))

    def test_frame_pattern_bounds(self):
        with self.assertRaises(IndexError):
            frame_pattern(25)
        with self.assertRaises(IndexError):
            frame_pattern(-1)


if __name__ == "__main__":
    unittest.main()
