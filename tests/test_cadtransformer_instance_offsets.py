"""Tests for reconstruction.cadtransformer_instance_offsets."""

from __future__ import annotations

import unittest

from harnesscad.domain.reconstruction.recognize.instance_offsets import (
    SENTINEL,
    group_by_shifted_center,
    instance_centroids,
    offset_targets,
    shift_to_centroid,
)


class TestCentroids(unittest.TestCase):
    def test_single_instance(self):
        centers = [(0.0, 0.0), (2.0, 2.0)]
        cents = instance_centroids(centers, [5, 5])
        self.assertAlmostEqual(cents[5][0], 1.0)
        self.assertAlmostEqual(cents[5][1], 1.0)

    def test_background_excluded(self):
        centers = [(0.0, 0.0), (10.0, 10.0)]
        cents = instance_centroids(centers, [-1, 3])
        self.assertNotIn(-1, cents)
        self.assertEqual(cents[3], (10.0, 10.0))

    def test_length_mismatch(self):
        with self.assertRaises(ValueError):
            instance_centroids([(0, 0)], [1, 2])

    def test_multiple_instances(self):
        centers = [(0, 0), (4, 0), (0, 10)]
        cents = instance_centroids(centers, [1, 1, 2])
        self.assertEqual(cents[1], (2.0, 0.0))
        self.assertEqual(cents[2], (0.0, 10.0))


class TestOffsets(unittest.TestCase):
    def test_offset_points_to_centroid(self):
        centers = [(0.0, 0.0), (2.0, 0.0)]
        offs = offset_targets(centers, [7, 7])
        # centroid (1,0); node0 offset (1,0); node1 offset (-1,0)
        self.assertEqual(offs[0], (1.0, 0.0))
        self.assertEqual(offs[1], (-1.0, 0.0))

    def test_background_sentinel(self):
        centers = [(5.0, 5.0)]
        offs = offset_targets(centers, [-1])
        self.assertEqual(offs[0], SENTINEL)

    def test_custom_sentinel(self):
        offs = offset_targets([(1.0, 1.0)], [-1], sentinel=(0.0, 0.0))
        self.assertEqual(offs[0], (0.0, 0.0))


class TestShift(unittest.TestCase):
    def test_shift_recovers_centroid(self):
        centers = [(0.0, 0.0), (2.0, 0.0), (1.0, 3.0)]
        instances = [7, 7, 7]
        offs = offset_targets(centers, instances)
        shifted = shift_to_centroid(centers, offs, instances)
        centroid = (1.0, 1.0)
        for p in shifted:
            self.assertAlmostEqual(p[0], centroid[0])
            self.assertAlmostEqual(p[1], centroid[1])

    def test_background_unchanged(self):
        centers = [(3.0, 4.0)]
        shifted = shift_to_centroid(centers, [SENTINEL], [-1])
        self.assertEqual(shifted[0], (3.0, 4.0))

    def test_length_mismatch(self):
        with self.assertRaises(ValueError):
            shift_to_centroid([(0, 0)], [(0, 0)], [1, 2])


class TestGrouping(unittest.TestCase):
    def test_recovers_instances_from_exact_offsets(self):
        centers = [(0.0, 0.0), (2.0, 0.0), (10.0, 10.0), (12.0, 10.0)]
        instances = [1, 1, 2, 2]
        offs = offset_targets(centers, instances)
        shifted = shift_to_centroid(centers, offs, instances)
        labels = group_by_shifted_center(shifted, instances)
        # two clusters: nodes 0,1 together and 2,3 together
        self.assertEqual(labels[0], labels[1])
        self.assertEqual(labels[2], labels[3])
        self.assertNotEqual(labels[0], labels[2])

    def test_background_labelled_background(self):
        labels = group_by_shifted_center([(0.0, 0.0)], [-1])
        self.assertEqual(labels[0], -1)

    def test_deterministic_labels_start_zero(self):
        labels = group_by_shifted_center([(5.0, 5.0), (5.0, 5.0)], [3, 3])
        self.assertEqual(labels, [0, 0])

    def test_tolerance_separates(self):
        labels = group_by_shifted_center([(0.0, 0.0), (1.0, 0.0)], [1, 1], tol=0.1)
        self.assertNotEqual(labels[0], labels[1])


if __name__ == "__main__":
    unittest.main()
