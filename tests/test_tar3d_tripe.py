"""Tests for geometry.tar3d_tripe."""

import math
import unittest

from harnesscad.domain.geometry.views.tar3d_tripe import (
    apply_rope,
    rope_frequencies,
    rope_vector,
    tripe_encoding,
    tripe_position_ids,
)


class TestPositionIds(unittest.TestCase):
    def test_length(self):
        ids = tripe_position_ids(2, 3)
        self.assertEqual(len(ids), 3 * 2 * 3)

    def test_trip2d_repeats_cell_thrice(self):
        ids = tripe_position_ids(1, 2)
        cells = [c for c, _ in ids]
        self.assertEqual(cells, [0, 0, 0, 1, 1, 1])

    def test_trip1d_cycles_planes(self):
        ids = tripe_position_ids(1, 2)
        planes = [p for _, p in ids]
        self.assertEqual(planes, [0, 1, 2, 0, 1, 2])

    def test_rejects_bad_dims(self):
        with self.assertRaises(ValueError):
            tripe_position_ids(0, 2)


class TestRope(unittest.TestCase):
    def test_frequencies_count(self):
        self.assertEqual(len(rope_frequencies(8)), 4)

    def test_odd_dim_rejected(self):
        with self.assertRaises(ValueError):
            rope_frequencies(3)

    def test_pos_zero_is_cos1_sin0(self):
        v = rope_vector(0, 4)
        self.assertEqual(v, [1.0, 0.0, 1.0, 0.0])

    def test_apply_rope_zero_is_identity(self):
        vec = [0.3, -0.7, 1.2, 0.5]
        self.assertEqual(apply_rope(vec, 0), vec)

    def test_rotations_compose_additively(self):
        vec = [1.0, 0.0, 0.5, -0.2]
        once = apply_rope(apply_rope(vec, 2), 3)
        combined = apply_rope(vec, 5)
        for a, b in zip(once, combined):
            self.assertAlmostEqual(a, b, places=9)

    def test_apply_rope_preserves_norm(self):
        vec = [1.0, 2.0, -3.0, 0.5]
        rot = apply_rope(vec, 7)
        n0 = math.sqrt(sum(x * x for x in vec))
        n1 = math.sqrt(sum(x * x for x in rot))
        self.assertAlmostEqual(n0, n1, places=9)


class TestTripeEncoding(unittest.TestCase):
    def test_shape(self):
        enc = tripe_encoding(2, 2, 8)
        self.assertEqual(len(enc), 3 * 2 * 2)
        self.assertTrue(all(len(v) == 8 for v in enc))

    def test_is_additive_fusion(self):
        # First token: pos2d=0, pos1d=0 -> both RoPE vectors are the pos-0 base.
        enc = tripe_encoding(1, 1, 4)
        v0 = rope_vector(0, 4)
        expected = [2 * x for x in v0]
        for a, b in zip(enc[0], expected):
            self.assertAlmostEqual(a, b, places=9)

    def test_planes_at_same_cell_differ(self):
        # Same cell (pos2d equal) but different pos1d must give distinct encodings.
        enc = tripe_encoding(1, 1, 8)
        self.assertNotEqual(enc[0], enc[1])
        self.assertNotEqual(enc[1], enc[2])

    def test_deterministic(self):
        self.assertEqual(tripe_encoding(2, 3, 8), tripe_encoding(2, 3, 8))


if __name__ == "__main__":
    unittest.main()
