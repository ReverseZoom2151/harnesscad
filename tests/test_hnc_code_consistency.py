"""Tests for HNC-CAD controllability/consistency metrics (bench.hnc_code_consistency)."""

import unittest

from generation.hnc_code_control import (
    LOOP,
    PROFILE,
    SOLID,
    CodeTree,
    edit_code,
    level_mask,
)
from bench.hnc_code_consistency import (
    changed_levels,
    code_fix_preservation,
    edit_locality,
    instance_agnostic_consistency,
    novelty_rate,
    uniqueness_rate,
)


def sample_tree():
    return CodeTree(solid=7, profiles=((3, (1, 2)), (4, (5, 6))))


class TestCodeFixPreservation(unittest.TestCase):
    def test_preserved_when_only_free_changes(self):
        before = sample_tree()
        mask = level_mask(before, {SOLID, PROFILE})  # loops are free
        after = edit_code(before, LOOP, 0, 99)       # change a free loop
        self.assertEqual(code_fix_preservation(mask, after), 1.0)

    def test_violated_when_fixed_changes(self):
        before = sample_tree()
        mask = level_mask(before, {SOLID})           # solid fixed
        after = edit_code(before, SOLID, 0, 42)      # illegally change solid
        self.assertLess(code_fix_preservation(mask, after), 1.0)

    def test_no_fixed_nodes(self):
        before = sample_tree()
        mask = level_mask(before, set())
        self.assertEqual(code_fix_preservation(mask, before), 1.0)


class TestEditLocality(unittest.TestCase):
    def test_loop_edit_is_local(self):
        before = sample_tree()
        after = edit_code(before, LOOP, 0, 99)
        self.assertEqual(changed_levels(before, after), frozenset({LOOP}))
        self.assertTrue(edit_locality(before, after, LOOP))

    def test_profile_edit_local_to_profile_scope(self):
        before = sample_tree()
        after = edit_code(before, PROFILE, 0, 88)
        self.assertTrue(edit_locality(before, after, PROFILE))
        # a profile change is NOT local if we claim it was a loop edit
        self.assertFalse(edit_locality(before, after, LOOP))

    def test_solid_edit_leaks_upward_for_loop_claim(self):
        before = sample_tree()
        after = edit_code(before, SOLID, 0, 55)
        self.assertTrue(edit_locality(before, after, SOLID))
        self.assertFalse(edit_locality(before, after, LOOP))


class TestInstanceAgnosticConsistency(unittest.TestCase):
    def test_well_separated_positive(self):
        items = [
            ((0.0, 0.0), 0), ((0.1, 0.1), 0),
            ((9.0, 9.0), 1), ((9.1, 9.1), 1),
        ]
        self.assertGreater(instance_agnostic_consistency(items), 0.8)

    def test_mixed_low(self):
        items = [
            ((0.0, 0.0), 0), ((9.0, 9.0), 0),
            ((0.1, 0.1), 1), ((9.1, 9.1), 1),
        ]
        self.assertLess(instance_agnostic_consistency(items), 0.2)

    def test_single_code_returns_zero(self):
        self.assertEqual(instance_agnostic_consistency([((0.0,), 0), ((1.0,), 0)]), 0.0)


class TestDiversitySummaries(unittest.TestCase):
    def test_uniqueness(self):
        a = sample_tree()
        b = CodeTree(solid=0, profiles=((0, (0,)),))
        gen = [a, a, b]  # a appears twice, b once
        self.assertAlmostEqual(uniqueness_rate(gen), 1 / 3)

    def test_novelty(self):
        a = sample_tree()
        b = CodeTree(solid=0, profiles=((0, (0,)),))
        ref = [a]
        self.assertAlmostEqual(novelty_rate([a, b], ref), 0.5)

    def test_empty(self):
        self.assertEqual(uniqueness_rate([]), 0.0)
        self.assertEqual(novelty_rate([], [sample_tree()]), 0.0)


if __name__ == "__main__":
    unittest.main()
