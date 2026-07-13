"""Tests for the segmentation-consistency metric (joint SDF paper)."""

import random
import unittest

from harnesscad.domain.reconstruction.recognize.segmentation_consistency import (
    per_anchor_consistency,
    segmentation_consistency,
    surface_consistency,
)


class ConsistencyTest(unittest.TestCase):
    def test_single_label_is_perfect(self):
        pts = [(float(i), 0.0, 0.0) for i in range(20)]
        labs = [7] * 20
        self.assertEqual(segmentation_consistency(pts, labs, k=3), 1.0)

    def test_permutation_invariance(self):
        pts = [(float(i), 0.0, 0.0) for i in range(12)]
        labs = [0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3]
        remap = {0: 99, 1: 5, 2: -3, 3: 42}
        relabelled = [remap[x] for x in labs]
        a = segmentation_consistency(pts, labs, k=2)
        b = segmentation_consistency(pts, relabelled, k=2)
        self.assertEqual(a, b)

    def test_clustered_beats_shuffled(self):
        pts = [(float(i), 0.0, 0.0) for i in range(20)]
        clustered = [0] * 10 + [1] * 10
        alternating = [i % 2 for i in range(20)]
        good = segmentation_consistency(pts, clustered, k=2)
        bad = segmentation_consistency(pts, alternating, k=2)
        self.assertGreater(good, bad)

    def test_known_value_alternating(self):
        # On a line, the 2 nearest neighbours of an interior point are its two
        # immediate line-neighbours, which under strict alternation both differ.
        pts = [(float(i), 0.0, 0.0) for i in range(10)]
        labs = [i % 2 for i in range(10)]
        per = per_anchor_consistency(pts, labs, k=2)
        interior = [c for i, c in per if 0 < i < 9]
        for c in interior:
            self.assertEqual(c, 0.0)

    def test_max_anchors_prefix_deterministic(self):
        pts = [(float(i), 0.0, 0.0) for i in range(50)]
        labs = [i % 3 for i in range(50)]
        a = segmentation_consistency(pts, labs, k=4, max_anchors=10)
        b = segmentation_consistency(pts, labs, k=4, max_anchors=10)
        self.assertEqual(a, b)

    def test_rng_sampling_deterministic(self):
        pts = [(float(i), float(i % 4), 0.0) for i in range(40)]
        labs = [i % 2 for i in range(40)]
        a = segmentation_consistency(
            pts, labs, k=3, max_anchors=15, rng=random.Random(0)
        )
        b = segmentation_consistency(
            pts, labs, k=3, max_anchors=15, rng=random.Random(0)
        )
        self.assertEqual(a, b)

    def test_length_mismatch(self):
        with self.assertRaises(ValueError):
            segmentation_consistency([(0.0, 0.0, 0.0)], [1, 2], k=1)

    def test_empty(self):
        self.assertEqual(segmentation_consistency([], [], k=1), 1.0)

    def test_surface_band_filters(self):
        pts = [(float(i), 0.0, 0.0) for i in range(10)]
        labs = [0, 0, 0, 0, 0, 9, 9, 9, 9, 9]
        # Off-surface points carry noisy labels; band keeps only |sdf|<=0.1.
        sdf = [0.0, 0.0, 0.0, 0.0, 0.0, 5.0, 5.0, 5.0, 5.0, 5.0]
        cons, n_surf = surface_consistency(pts, labs, sdf, tau=0.1, k=2)
        self.assertEqual(n_surf, 5)
        self.assertEqual(cons, 1.0)


if __name__ == "__main__":
    unittest.main()
