"""Tests for editing.cadmorph_plan (CADMorph planning / relative-contribution masking)."""
import unittest

from editing.cadmorph_plan import (
    apply_mask, leave_one_out_contribution, plan_mask, relative_scores,
    select_mask_indices,
)
from editing.locate_infill import MASK


class RelativeScoreTests(unittest.TestCase):
    def test_absolute_difference(self):
        # J(i) = |M(i,S') - M(i,S_{r-1})| (paper Eq. 2).
        j = relative_scores([1.0, 2.0, 3.0], [1.0, 5.0, 0.0])
        self.assertEqual(j, (0.0, 3.0, 3.0))

    def test_length_mismatch_raises(self):
        with self.assertRaises(ValueError):
            relative_scores([1.0], [1.0, 2.0])


class SelectMaskTests(unittest.TestCase):
    def test_above_mean(self):
        # mean of (0,3,3) = 2 -> indices 1,2 exceed it.
        self.assertEqual(select_mask_indices([0.0, 3.0, 3.0]), (1, 2))

    def test_max_k_caps(self):
        idx = select_mask_indices([0.0, 3.0, 4.0], max_k=1)
        self.assertEqual(idx, (2,))  # highest score kept

    def test_all_equal_ensures_progress(self):
        # All equal -> none above mean; ensure_progress masks the argmax.
        self.assertEqual(select_mask_indices([2.0, 2.0, 2.0]), (0,))

    def test_all_zero_selects_nothing(self):
        # Nothing to edit -> empty (no positive discrepancy).
        self.assertEqual(select_mask_indices([0.0, 0.0]), ())

    def test_empty(self):
        self.assertEqual(select_mask_indices([]), ())


class ApplyMaskTests(unittest.TestCase):
    def test_replaces_with_mask_token(self):
        out = apply_mask(("a", "b", "c"), [1])
        self.assertEqual(out, ("a", MASK, "c"))

    def test_collapses_consecutive(self):
        out = apply_mask(("a", "b", "c", "d"), [1, 2])
        self.assertEqual(out, ("a", MASK, "d"))

    def test_non_adjacent_stay_separate(self):
        out = apply_mask(("a", "b", "c", "d"), [0, 2])
        self.assertEqual(out, (MASK, "b", MASK, "d"))


class PlanMaskTests(unittest.TestCase):
    def test_end_to_end(self):
        seq = ("SOL", "Line", "Arc")
        plan = plan_mask(seq, contrib_current=[1.0, 2.0, 3.0],
                         contrib_target=[1.0, 5.0, 0.0])
        # J = (0,3,3); mean=2; indices 1,2 masked (consecutive -> collapsed).
        self.assertEqual(plan.masked_indices, (1, 2))
        self.assertEqual(plan.masked_sequence, ("SOL", MASK))
        self.assertAlmostEqual(plan.threshold, 2.0)

    def test_length_mismatch_raises(self):
        with self.assertRaises(ValueError):
            plan_mask(("a", "b"), [1.0], [1.0])


class LeaveOneOutContributionTests(unittest.TestCase):
    """A deterministic geometry stand-in for the P2S cross-attention read-out."""

    def setUp(self):
        # Shape = set of tokens; distance = symmetric-difference size.
        self.render = lambda seq: frozenset(seq)
        self.distance = lambda a, b: len(a ^ b)
        self.contrib = leave_one_out_contribution(self.render, self.distance)

    def test_isolates_mismatched_segment(self):
        seq = ("a", "b", "x")
        target = frozenset({"a", "b", "y"})
        current = self.render(seq)
        j = relative_scores(self.contrib(seq, current),
                            self.contrib(seq, target))
        idx = select_mask_indices(j)
        # Segment 'x' (index 2) is the one that must change to reach the target.
        self.assertEqual(idx, (2,))

    def test_irrelevant_segment_has_low_contribution(self):
        seq = ("a", "b", "x")
        target = frozenset({"a", "b", "y"})
        plan = plan_mask(seq, self.contrib(seq, self.render(seq)),
                         self.contrib(seq, target))
        self.assertEqual(plan.masked_sequence, ("a", "b", MASK))


if __name__ == "__main__":
    unittest.main()
