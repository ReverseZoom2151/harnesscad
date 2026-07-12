"""Tests for edit-triplet mining, directional edit alignment, edit locality,
the iterative edit policy, sketch frame tokens, spatial sequence accuracy,
frame coherence, and the spatial challenge set.

Rewritten from bare pytest-style module functions (never collected by
``python -m unittest``) into unittest.TestCase classes.
"""

import math
import unittest

from agent.iterative_edit_policy import IterativeEditPolicy
from bench.spatial_challenge_set import fixtures, stratify
from dataengine.edit_triplets import build
from ingest.sketch_frame_tokens import SketchFrame, quantize
from quality.directional_edit_alignment import alignment, rank
from quality.edit_locality import locality
from quality.frame_coherence import check
from quality.spatial_sequence_accuracy import angle_error, score


class EditTripletsTest(unittest.TestCase):
    def test_changed_pair_yields_a_single_edit(self):
        self.assertEqual(build("x", [1], [2], "a", "b").edits, 1)

    def test_unchanged_pair_yields_no_triplet(self):
        self.assertIsNone(build("x", [1], [1], "a", "a"))


class DirectionalEditAlignmentTest(unittest.TestCase):
    def test_identical_directions_align_perfectly(self):
        self.assertEqual(alignment((0, 0), (1, 0), (0, 0), (1, 0)), 1)

    def test_rank_prefers_the_valid_candidate_over_a_higher_scoring_invalid_one(self):
        ranked = rank(({"id": "b", "valid": True, "score": .2},
                       {"id": "a", "valid": False, "score": 1}))
        self.assertEqual(ranked[0]["id"], "b")


class EditLocalityTest(unittest.TestCase):
    def test_untouched_entities_that_changed_are_reported_as_collateral(self):
        result = locality({"a"}, {"a", "b"}, {"a", "b"})
        self.assertEqual(result["collateral"], ("b",))


class IterativeEditPolicyTest(unittest.TestCase):
    def test_worse_candidate_triggers_a_rollback(self):
        current = {"alignment": .5, "digest": "a", "valid": True}
        worse = {"alignment": .4, "digest": "b", "valid": True}
        self.assertEqual(IterativeEditPolicy().choose(current, worse, ())[1],
                         "rollback")


class SketchFrameTest(unittest.TestCase):
    def setUp(self):
        self.frame = SketchFrame((1, 2, 3), math.pi / 2)

    def test_local_to_world_round_trips_through_world_to_local(self):
        point = (2, 3)
        world = self.frame.local_to_world(point)
        back = self.frame.world_to_local(world)
        for expected, actual in zip(point, back):
            self.assertAlmostEqual(expected, actual, places=9)

    def test_quantize_maps_zero_to_the_bin_centre(self):
        self.assertEqual(quantize((0,), 3)[0], (1,))

    def test_frame_coherence_reports_no_issues_for_a_consistent_frame(self):
        point = (2, 3)
        world = self.frame.local_to_world(point)
        self.assertFalse(check(self.frame, (point,), (world,), (0, 0, 1))["issues"])


class SpatialSequenceAccuracyTest(unittest.TestCase):
    def test_full_turn_has_negligible_angle_error(self):
        self.assertLess(angle_error(0, 2 * math.pi), 1e-7)

    def test_identical_op_sequences_score_perfect_command_accuracy(self):
        self.assertEqual(score([{"op": "x"}], [{"op": "x"}])["command"], 1)


class SpatialChallengeSetTest(unittest.TestCase):
    def test_stratify_buckets_a_fixture_under_its_category(self):
        fixture_set = fixtures()
        self.assertEqual(stratify([(fixture_set[0], 1)])["orientation"], 1)


if __name__ == "__main__":
    unittest.main()
