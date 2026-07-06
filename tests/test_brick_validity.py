"""Tests for verifiers.brick_validity (validity check + physics-aware rollback)."""

import unittest

from geometry.brick_structure import Brick, BrickStructure
from verifiers.brick_validity import (
    build_with_validity_and_rollback,
    first_unstable_index,
    is_valid_brick,
    is_valid_placement,
    physics_aware_rollback,
    rejection_sample,
)


def struct(bricks):
    return BrickStructure.from_bricks(bricks)


class TestValidity(unittest.TestCase):
    def test_is_valid_brick(self):
        self.assertTrue(is_valid_brick(Brick(2, 4, 0, 0, 0)))
        self.assertFalse(is_valid_brick(Brick(3, 3, 0, 0, 0)))  # not in library
        self.assertFalse(is_valid_brick(Brick(2, 2, 19, 19, 0)))  # out of bounds

    def test_is_valid_placement(self):
        existing = [Brick(2, 2, 0, 0, 0)]
        self.assertTrue(is_valid_placement(existing, Brick(2, 2, 0, 0, 1)))
        self.assertFalse(is_valid_placement(existing, Brick(1, 1, 1, 1, 0)))  # collide
        self.assertFalse(is_valid_placement(existing, Brick(3, 3, 5, 5, 0)))  # library

    def test_rejection_sample(self):
        existing = [Brick(2, 2, 0, 0, 0)]
        cands = [
            Brick(1, 1, 1, 1, 0),  # collides
            Brick(3, 3, 5, 5, 0),  # not in library
            Brick(2, 2, 0, 0, 1),  # valid -> chosen
            Brick(1, 1, 3, 3, 0),  # also valid but later
        ]
        chosen = rejection_sample(existing, cands)
        self.assertEqual(chosen, Brick(2, 2, 0, 0, 1))
        self.assertIsNone(rejection_sample(existing, [Brick(1, 1, 1, 1, 0)]))


class TestRollback(unittest.TestCase):
    def test_stable_structure_unchanged(self):
        s = struct([Brick(2, 2, 0, 0, 0), Brick(2, 2, 0, 0, 1)])
        r = physics_aware_rollback(s)
        self.assertTrue(r.stable)
        self.assertEqual(r.removed, 0)
        self.assertEqual(len(r.structure.bricks), 2)

    def test_first_unstable_index(self):
        # bricks 0,1 form a solid grounded stack; brick 2 floats (gap at z=2) and
        # is the first (and only) unstable brick.
        s = struct([Brick(2, 2, 0, 0, 0), Brick(2, 2, 0, 0, 1), Brick(2, 2, 0, 0, 3)])
        self.assertEqual(first_unstable_index(s), 2)
        stable = struct([Brick(2, 2, 0, 0, 0), Brick(2, 2, 0, 0, 1)])
        self.assertIsNone(first_unstable_index(stable))

    def test_rollback_removes_unstable_tail(self):
        # a stable 2-brick base, then a floating brick that must be rolled off
        s = struct(
            [Brick(2, 2, 0, 0, 0), Brick(2, 2, 0, 0, 1), Brick(2, 2, 0, 0, 3)]
        )
        r = physics_aware_rollback(s)
        self.assertTrue(r.stable)
        self.assertEqual(len(r.structure.bricks), 2)
        self.assertEqual(r.removed, 1)
        self.assertGreaterEqual(r.rollbacks, 1)

    def test_rollback_extreme_cantilever_returns_stable_prefix(self):
        # A grossly overhanging brick stresses its support too, so rollback may
        # remove more than one brick; the result must still be a stable prefix.
        s = struct([Brick(2, 2, 0, 0, 0), Brick(8, 1, 0, 0, 1)])
        r = physics_aware_rollback(s)
        self.assertTrue(r.stable)
        self.assertLess(len(r.structure.bricks), 2)
        self.assertEqual(
            r.structure.bricks, s.bricks[: len(r.structure.bricks)]
        )

    def test_rollback_of_floating_brick(self):
        s = struct([Brick(2, 2, 0, 0, 0), Brick(2, 2, 0, 0, 2)])
        r = physics_aware_rollback(s)
        self.assertTrue(r.stable)
        self.assertEqual(len(r.structure.bricks), 1)


class TestFullLoop(unittest.TestCase):
    def test_build_skips_invalid_and_stabilises(self):
        candidates = [
            Brick(2, 2, 0, 0, 0),  # valid
            Brick(1, 1, 1, 1, 0),  # collides -> skipped
            Brick(2, 2, 0, 0, 1),  # valid
            Brick(3, 3, 5, 5, 0),  # not in library -> skipped
            Brick(2, 2, 0, 0, 3),  # valid placement but floats -> rolled back
        ]
        r = build_with_validity_and_rollback(candidates)
        self.assertTrue(r.stable)
        # collisions/out-of-library skipped -> 3 placed; floater rolled off -> 2
        self.assertEqual(len(r.structure.bricks), 2)

    def test_build_all_valid_stable(self):
        candidates = [Brick(2, 2, 0, 0, z) for z in range(3)]
        r = build_with_validity_and_rollback(candidates)
        self.assertTrue(r.stable)
        self.assertEqual(len(r.structure.bricks), 3)
        self.assertEqual(r.removed, 0)


if __name__ == "__main__":
    unittest.main()
