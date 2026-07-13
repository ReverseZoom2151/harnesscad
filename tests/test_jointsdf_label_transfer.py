"""Tests for nearest-neighbour label transfer (joint SDF paper)."""

import unittest

from harnesscad.domain.reconstruction.jointsdf_label_transfer import (
    transfer_labels,
    overlap_counts,
    match_labels,
    remap,
    transferred_accuracy,
)


class TransferTest(unittest.TestCase):
    def test_transfer_nearest(self):
        source = [(0.0, 0.0, 0.0), (10.0, 0.0, 0.0)]
        labels = ["a", "b"]
        q = [(1.0, 0.0, 0.0), (9.0, 0.0, 0.0), (4.0, 0.0, 0.0)]
        self.assertEqual(transfer_labels(q, source, labels), ["a", "b", "a"])

    def test_tie_breaks_low_index(self):
        source = [(0.0, 0.0, 0.0), (2.0, 0.0, 0.0)]
        labels = [1, 2]
        # midpoint equidistant -> lower index (label 1) wins
        self.assertEqual(transfer_labels([(1.0, 0.0, 0.0)], source, labels), [1])

    def test_source_length_mismatch(self):
        with self.assertRaises(ValueError):
            transfer_labels([(0.0, 0.0, 0.0)], [(0.0, 0.0, 0.0)], [1, 2])

    def test_empty_source(self):
        with self.assertRaises(ValueError):
            transfer_labels([(0.0, 0.0, 0.0)], [], [])


class MatchTest(unittest.TestCase):
    def test_overlap_counts(self):
        pred = [5, 5, 5, 8]
        gt = [1, 1, 2, 2]
        c = overlap_counts(pred, gt)
        self.assertEqual(c[(5, 1)], 2)
        self.assertEqual(c[(5, 2)], 1)
        self.assertEqual(c[(8, 2)], 1)

    def test_match_majority(self):
        pred = [5, 5, 5, 8, 8]
        gt = [1, 1, 1, 2, 2]
        m = match_labels(pred, gt)
        self.assertEqual(m[5], 1)
        self.assertEqual(m[8], 2)

    def test_oversegmentation_maps_many_to_one(self):
        # two predicted parts both belong to gt part 0
        pred = [10, 10, 20, 20]
        gt = [0, 0, 0, 0]
        m = match_labels(pred, gt)
        self.assertEqual(m[10], 0)
        self.assertEqual(m[20], 0)
        self.assertEqual(remap(pred, m), [0, 0, 0, 0])

    def test_match_deterministic_on_ties(self):
        pred = [7, 7]
        gt = [1, 2]  # tie 1 vs 1
        m1 = match_labels(pred, gt)
        m2 = match_labels(pred, gt)
        self.assertEqual(m1, m2)


class AccuracyTest(unittest.TestCase):
    def test_transferred_accuracy(self):
        source = [(0.0, 0.0, 0.0), (10.0, 0.0, 0.0)]
        labels = [0, 1]
        q = [(1.0, 0.0, 0.0), (9.0, 0.0, 0.0)]
        # predicted labels match transferred GT -> 1.0
        self.assertEqual(transferred_accuracy(q, source, labels, [0, 1]), 1.0)
        self.assertEqual(transferred_accuracy(q, source, labels, [0, 0]), 0.5)


if __name__ == "__main__":
    unittest.main()
