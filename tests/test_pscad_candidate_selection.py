"""Tests for PS-CAD single-step candidate selection."""

import unittest

from harnesscad.domain.reconstruction.pscad_candidate_selection import (
    StepCandidate,
    axis_aligned_bbox,
    bbox_iou,
    bbox_volume,
    fitness_score,
    select_candidate,
)


def _box_cloud(lo, hi):
    """8 corner points of an axis-aligned box."""
    return tuple((x, y, z) for x in (lo[0], hi[0])
                 for y in (lo[1], hi[1]) for z in (lo[2], hi[2]))


class BboxGeometryTest(unittest.TestCase):
    def test_bbox_and_volume(self):
        cloud = _box_cloud((0, 0, 0), (2, 3, 4))
        box = axis_aligned_bbox(cloud)
        self.assertEqual(box, ((0, 0, 0), (2, 3, 4)))
        self.assertAlmostEqual(bbox_volume(box), 24.0)

    def test_iou_identical(self):
        box = ((0, 0, 0), (1, 1, 1))
        self.assertAlmostEqual(bbox_iou(box, box), 1.0)

    def test_iou_disjoint(self):
        a = ((0, 0, 0), (1, 1, 1))
        b = ((5, 5, 5), (6, 6, 6))
        self.assertEqual(bbox_iou(a, b), 0.0)

    def test_iou_partial(self):
        a = ((0, 0, 0), (2, 2, 2))     # vol 8
        b = ((1, 1, 1), (3, 3, 3))     # vol 8, overlap 1
        # inter=1, union=15
        self.assertAlmostEqual(bbox_iou(a, b), 1.0 / 15.0)

    def test_empty_cloud_rejected(self):
        with self.assertRaises(ValueError):
            axis_aligned_bbox([])


class FitnessScoreTest(unittest.TestCase):
    def test_matching_bool_returns_iou(self):
        box = ((0, 0, 0), (1, 1, 1))
        self.assertAlmostEqual(fitness_score(box, box, 1, 1), 1.0)

    def test_mismatched_bool_zeroed(self):
        box = ((0, 0, 0), (1, 1, 1))
        self.assertEqual(fitness_score(box, box, 1, 0), 0.0)


class SelectCandidateTest(unittest.TestCase):
    def setUp(self):
        # target box (0,0,0)-(2,2,2)
        self.target = _box_cloud((0, 0, 0), (2, 2, 2))
        self.reference_box = ((0, 0, 0), (2, 2, 2))
        self.candidates = [
            StepCandidate(0, _box_cloud((0, 0, 0), (2, 2, 2)), 1),   # perfect
            StepCandidate(1, _box_cloud((0, 0, 0), (5, 5, 5)), 1),   # too big
            StepCandidate(2, _box_cloud((0, 0, 0), (1, 1, 1)), 1),   # too small
        ]

    def test_geo_picks_nearest_to_target(self):
        res = select_candidate(self.candidates, strategy="geo",
                               target_cloud=self.target)
        self.assertEqual(res.winner_index, 0)

    def test_heur_picks_largest_volume(self):
        res = select_candidate(self.candidates, strategy="heur")
        self.assertEqual(res.winner_index, 1)  # the (5,5,5) box

    def test_bbox_iou_picks_best_agreement(self):
        res = select_candidate(self.candidates, strategy="bbox_iou",
                               reference_box=self.reference_box)
        self.assertEqual(res.winner_index, 0)

    def test_bbox_iou_bool_mismatch_demoted(self):
        cands = [
            StepCandidate(0, _box_cloud((0, 0, 0), (2, 2, 2)), 0),  # perfect box, wrong op
            StepCandidate(1, _box_cloud((0, 0, 0), (1.9, 1.9, 1.9)), 1),  # slightly off, right op
        ]
        res = select_candidate(cands, strategy="bbox_iou",
                               reference_box=self.reference_box, reference_bool=1)
        self.assertEqual(res.winner_index, 1)

    def test_rand_is_seed_deterministic(self):
        a = select_candidate(self.candidates, strategy="rand", seed=42)
        b = select_candidate(self.candidates, strategy="rand", seed=42)
        self.assertEqual(a.winner_index, b.winner_index)

    def test_scores_reported_in_candidate_order(self):
        res = select_candidate(self.candidates, strategy="heur")
        self.assertEqual([i for i, _ in res.scores], [0, 1, 2])

    def test_missing_target_rejected(self):
        with self.assertRaises(ValueError):
            select_candidate(self.candidates, strategy="geo")

    def test_missing_reference_rejected(self):
        with self.assertRaises(ValueError):
            select_candidate(self.candidates, strategy="bbox_iou")

    def test_unknown_strategy_rejected(self):
        with self.assertRaises(ValueError):
            select_candidate(self.candidates, strategy="nope")

    def test_empty_candidates_rejected(self):
        with self.assertRaises(ValueError):
            select_candidate([], strategy="heur")


if __name__ == "__main__":
    unittest.main()
