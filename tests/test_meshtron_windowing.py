"""Tests for Meshtron hourglass windowing / truncation."""

import unittest

from formats.meshtron_windowing import (
    EOS,
    PAD,
    SOS,
    frame_sequence,
    group_coords_to_vertices,
    group_vertices_to_faces,
    hierarchy_levels,
    hourglass_stage_lengths,
    pad_batch,
    pad_to_length,
    receptive_field_size,
    shortened_length,
    shortening_indices,
    sliding_windows,
    truncate_segments,
)


class FramingTest(unittest.TestCase):
    def test_frame_adds_nine_each_side(self):
        coords = list(range(9))
        framed = frame_sequence(coords)
        self.assertEqual(framed[:9], [SOS] * 9)
        self.assertEqual(framed[-9:], [EOS] * 9)
        self.assertEqual(framed[9:-9], coords)
        self.assertEqual(len(framed), 9 + 9 + 9)

    def test_pad_to_length(self):
        self.assertEqual(pad_to_length([1, 2], 5), [1, 2, PAD, PAD, PAD])

    def test_pad_too_long_raises(self):
        with self.assertRaises(ValueError):
            pad_to_length([1, 2, 3], 2)

    def test_pad_batch(self):
        out = pad_batch([[1], [1, 2, 3], [1, 2]])
        self.assertTrue(all(len(s) == 3 for s in out))
        self.assertEqual(out[0], [1, PAD, PAD])

    def test_pad_batch_empty(self):
        self.assertEqual(pad_batch([]), [])


class GroupingTest(unittest.TestCase):
    def test_coords_to_vertices(self):
        verts = group_coords_to_vertices(list(range(9)))
        self.assertEqual(verts, [[0, 1, 2], [3, 4, 5], [6, 7, 8]])

    def test_coords_bad_length(self):
        with self.assertRaises(ValueError):
            group_coords_to_vertices([1, 2])

    def test_vertices_to_faces(self):
        faces = group_vertices_to_faces(list(range(9)))
        self.assertEqual(len(faces), 3)

    def test_hierarchy_levels(self):
        # 2 triangles -> 18 coords, 6 vertices, 2 faces
        self.assertEqual(hierarchy_levels(list(range(18))), (18, 6, 2))

    def test_hierarchy_bad(self):
        with self.assertRaises(ValueError):
            hierarchy_levels(list(range(10)))


class ShorteningTest(unittest.TestCase):
    def test_shortening_indices_factor3(self):
        self.assertEqual(shortening_indices(9, 3), [2, 5, 8])

    def test_shortened_length(self):
        self.assertEqual(shortened_length(9, 3), 3)

    def test_stage_lengths_two_factor3(self):
        # coordinate -> vertex -> face levels
        self.assertEqual(hourglass_stage_lengths(81, (3, 3)), [81, 27, 9])

    def test_bad_factor(self):
        with self.assertRaises(ValueError):
            shortening_indices(9, 0)


class TruncateTest(unittest.TestCase):
    def test_segments_aligned_and_padded(self):
        tokens = list(range(9 * 5))  # 45 tokens, window 18 (two faces)
        segs = truncate_segments(tokens, 18)
        self.assertEqual(len(segs), 3)          # 18, 18, 9->padded to 18
        self.assertTrue(all(len(s) == 18 for s in segs))
        self.assertEqual(segs[-1][9:], [PAD] * 9)

    def test_window_must_align(self):
        with self.assertRaises(ValueError):
            truncate_segments(list(range(18)), 10)  # 10 not multiple of 9

    def test_no_pad(self):
        segs = truncate_segments(list(range(27)), 18, pad=False)
        self.assertEqual(len(segs[-1]), 9)


class SlidingWindowTest(unittest.TestCase):
    def test_spans_grow_then_roll(self):
        spans = sliding_windows(list(range(6)), window=3)
        self.assertEqual(spans, [(0, 1), (0, 2), (0, 3), (1, 4), (2, 5), (3, 6)])

    def test_span_width_capped(self):
        spans = sliding_windows(list(range(10)), window=4)
        for start, end in spans:
            self.assertLessEqual(end - start, 4)

    def test_step(self):
        spans = sliding_windows(list(range(10)), window=4, step=2)
        self.assertEqual(spans[0], (0, 1))
        self.assertEqual(spans[1], (0, 3))

    def test_receptive_field(self):
        self.assertEqual(receptive_field_size(0, 5), 1)
        self.assertEqual(receptive_field_size(3, 5), 4)
        self.assertEqual(receptive_field_size(10, 5), 5)

    def test_bad_window(self):
        with self.assertRaises(ValueError):
            sliding_windows([1, 2], 0)


if __name__ == "__main__":
    unittest.main()
