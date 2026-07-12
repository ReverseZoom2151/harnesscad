"""Tests for reconstruction.tar3d_part_sequence."""

import unittest

from reconstruction.tar3d_part_sequence import (
    PLANE_ORDER,
    TriplaneIndexGrid,
    build_sequence,
    detokenize,
    is_valid_sequence,
    next_part_targets,
    prefill,
    sequence_positions,
    teacher_forcing_accuracy,
)


def _grid(h, w, k, fill):
    planes = {p: [[fill(p, r, c) for c in range(w)] for r in range(h)]
              for p in PLANE_ORDER}
    return TriplaneIndexGrid(planes, h, w, k)


class TestGridValidation(unittest.TestCase):
    def test_rejects_wrong_shape(self):
        with self.assertRaises(ValueError):
            TriplaneIndexGrid({"XY": [[0]], "YZ": [[0]], "XZ": [[0, 1]]}, 1, 1, 4)

    def test_rejects_out_of_codebook(self):
        with self.assertRaises(ValueError):
            _grid(1, 1, 2, lambda p, r, c: 5)

    def test_rejects_missing_plane(self):
        with self.assertRaises(ValueError):
            TriplaneIndexGrid({"XY": [[0]], "YZ": [[0]]}, 1, 1, 4)


class TestPositions(unittest.TestCase):
    def test_length_and_interleave(self):
        pos = sequence_positions(1, 2)
        self.assertEqual(len(pos), 6)
        # (row,col) raster, planes adjacent at each cell.
        self.assertEqual(pos, [
            ("XY", 0, 0), ("YZ", 0, 0), ("XZ", 0, 0),
            ("XY", 0, 1), ("YZ", 0, 1), ("XZ", 0, 1),
        ])

    def test_raster_within_plane(self):
        pos = sequence_positions(2, 2)
        xy_cells = [(r, c) for (p, r, c) in pos if p == "XY"]
        self.assertEqual(xy_cells, [(0, 0), (0, 1), (1, 0), (1, 1)])


class TestSequenceRoundTrip(unittest.TestCase):
    def test_build_then_detokenize(self):
        # Encode plane identity into the value so we can verify placement.
        base = {"XY": 0, "YZ": 1, "XZ": 2}
        g = _grid(2, 3, 100, lambda p, r, c: base[p] * 10 + r * 3 + c)
        seq = build_sequence(g)
        self.assertEqual(len(seq), 3 * 2 * 3)
        back = detokenize(seq, 2, 3, 100)
        self.assertEqual(back.planes, g.planes)

    def test_first_triple_is_cell_00(self):
        g = _grid(2, 2, 100, lambda p, r, c: {"XY": 7, "YZ": 8, "XZ": 9}[p])
        seq = build_sequence(g)
        self.assertEqual(seq[:3], [7, 8, 9])

    def test_detokenize_length_check(self):
        with self.assertRaises(ValueError):
            detokenize([0, 1, 2], 2, 2, 4)


class TestValidity(unittest.TestCase):
    def test_valid(self):
        self.assertTrue(is_valid_sequence([0, 1, 2], 1, 1, 4))

    def test_bad_length(self):
        self.assertFalse(is_valid_sequence([0, 1], 1, 1, 4))

    def test_out_of_range(self):
        self.assertFalse(is_valid_sequence([0, 1, 9], 1, 1, 4))


class TestNextPart(unittest.TestCase):
    def test_targets_without_prompt(self):
        pairs = next_part_targets([5, 6, 7])
        self.assertEqual(pairs, [
            ((), 5),
            ((5,), 6),
            ((5, 6), 7),
        ])

    def test_targets_with_prompt(self):
        pairs = next_part_targets([5, 6], prompt=[99])
        self.assertEqual(pairs[0], ((99,), 5))
        self.assertEqual(pairs[1], ((99, 5), 6))

    def test_prompt_never_a_target(self):
        pairs = next_part_targets([1, 2], prompt=[8, 9])
        targets = [t for _, t in pairs]
        self.assertEqual(targets, [1, 2])

    def test_prefill(self):
        self.assertEqual(prefill([8, 9], [1, 2]), [8, 9, 1, 2])


class TestMetric(unittest.TestCase):
    def test_perfect(self):
        self.assertEqual(teacher_forcing_accuracy([1, 2, 3], [1, 2, 3]), 1.0)

    def test_half(self):
        self.assertAlmostEqual(teacher_forcing_accuracy([1, 0, 3, 0], [1, 2, 3, 4]), 0.5)

    def test_empty(self):
        self.assertEqual(teacher_forcing_accuracy([], []), 1.0)

    def test_length_mismatch(self):
        with self.assertRaises(ValueError):
            teacher_forcing_accuracy([1], [1, 2])


if __name__ == "__main__":
    unittest.main()
