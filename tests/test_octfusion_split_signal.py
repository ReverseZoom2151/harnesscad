"""Tests for geometry.octfusion_split_signal."""

import unittest

from harnesscad.domain.geometry.octfusion_octree import Octree
from harnesscad.domain.geometry.octfusion_split_signal import (
    decode_split_signals,
    encode_split_signals,
    full_octree_node_count,
    round_signal,
)


class TestRoundSignal(unittest.TestCase):
    def test_rounds_above_threshold(self):
        self.assertEqual(round_signal([0.9, 0.1, 0.6, 0.4, 0.51, 0.49, 1.0, 0.0]),
                         [1, 0, 1, 0, 1, 0, 1, 0])

    def test_wrong_length(self):
        with self.assertRaises(ValueError):
            round_signal([1.0, 0.0])


class TestEncode(unittest.TestCase):
    def test_single_voxel_signal_chain(self):
        t = Octree.from_voxels([(0, 0, 0)], max_depth=2)
        sig = encode_split_signals(t)
        # root and one child at depth 1 are internal; voxel is child 0 each time
        self.assertIn((0, 0, 0, 0), sig)
        self.assertEqual(sig[(0, 0, 0, 0)], [1, 0, 0, 0, 0, 0, 0, 0])

    def test_leaves_have_no_signal(self):
        t = Octree.from_points([], max_depth=3)
        self.assertEqual(encode_split_signals(t), {})

    def test_two_octants(self):
        t = Octree.from_voxels([(0, 0, 0), (3, 3, 3)], max_depth=2)
        sig = encode_split_signals(t)
        # root splits into child 0 and child 7
        self.assertEqual(sig[(0, 0, 0, 0)], [1, 0, 0, 0, 0, 0, 0, 1])


class TestRoundTrip(unittest.TestCase):
    def test_encode_decode_reproduces_voxels(self):
        occ = {(0, 0, 0), (1, 1, 1), (7, 7, 7), (3, 4, 5), (2, 6, 1)}
        t = Octree.from_voxels(sorted(occ), max_depth=3)
        sig = encode_split_signals(t)
        rebuilt = decode_split_signals(sig, max_depth=3)
        self.assertEqual(rebuilt.to_voxels(depth=3), occ)

    def test_encode_decode_leaf_keys_match(self):
        t = Octree.from_points([(0.1, 0.2, 0.3), (0.8, 0.9, 0.7)], max_depth=4)
        sig = encode_split_signals(t)
        rebuilt = decode_split_signals(sig, max_depth=4)
        orig = {leaf.key() for leaf in t.occupied_leaves()}
        got = {leaf.key() for leaf in rebuilt.occupied_leaves()}
        self.assertEqual(orig, got)


class TestDecode(unittest.TestCase):
    def test_noisy_signal_rounded(self):
        # continuous values near 1 for child 0 only
        signals = {(0, 0, 0, 0): [0.97, 0.02, 0.1, 0.0, 0.3, 0.2, 0.05, 0.4]}
        t = decode_split_signals(signals, max_depth=1)
        self.assertEqual(t.to_voxels(depth=1), {(0, 0, 0)})

    def test_full_depth_forces_full(self):
        # no signals, but full_depth=1 forces a full split at the root
        t = decode_split_signals({}, max_depth=1, full_depth=1)
        # all 8 children created and occupied at max_depth
        self.assertEqual(t.occupied_leaf_count(), 8)

    def test_missing_signal_is_leaf(self):
        t = decode_split_signals({}, max_depth=2, full_depth=0)
        self.assertTrue(t.root.is_leaf)
        self.assertFalse(t.root.occupied)

    def test_invalid_params(self):
        with self.assertRaises(ValueError):
            decode_split_signals({}, max_depth=-1)
        with self.assertRaises(ValueError):
            decode_split_signals({}, max_depth=2, full_depth=3)


class TestFullCount(unittest.TestCase):
    def test_counts(self):
        self.assertEqual(full_octree_node_count(0), 1)
        self.assertEqual(full_octree_node_count(1), 9)
        self.assertEqual(full_octree_node_count(2), 73)

    def test_invalid(self):
        with self.assertRaises(ValueError):
            full_octree_node_count(-1)


if __name__ == "__main__":
    unittest.main()
