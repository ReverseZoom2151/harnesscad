"""Tests for verifiers.brick_buildability (assembly-order analysis)."""

import unittest

from geometry.brick_structure import Brick, BrickStructure
from verifiers.brick_buildability import (
    find_buildable_order,
    is_assembly_stable,
    is_buildable,
    is_supported_order,
    raster_assembly_order,
)
from verifiers.brick_stability import is_stable


def struct(bricks):
    return BrickStructure.from_bricks(bricks)


class TestRasterOrder(unittest.TestCase):
    def test_bottom_to_top(self):
        s = struct(
            [Brick(1, 1, 5, 5, 2), Brick(1, 1, 0, 0, 0), Brick(1, 1, 0, 3, 0)]
        )
        order = raster_assembly_order(s)
        zs = [s.bricks[i].z for i in order]
        self.assertEqual(zs, sorted(zs))
        # ties broken by (y, x): brick at (0,0) before (0,3)
        self.assertEqual(order[0], 1)
        self.assertEqual(order[1], 2)

    def test_raster_is_supported(self):
        s = struct([Brick(2, 2, 0, 0, 0), Brick(2, 2, 0, 0, 1), Brick(2, 2, 0, 0, 2)])
        self.assertTrue(is_supported_order(s, raster_assembly_order(s)))


class TestSupportedOrder(unittest.TestCase):
    def test_top_first_is_unsupported(self):
        s = struct([Brick(2, 2, 0, 0, 0), Brick(2, 2, 0, 0, 1)])
        self.assertFalse(is_supported_order(s, [1, 0]))
        self.assertTrue(is_supported_order(s, [0, 1]))

    def test_requires_permutation(self):
        s = struct([Brick(2, 2, 0, 0, 0), Brick(2, 2, 0, 0, 1)])
        with self.assertRaises(ValueError):
            is_supported_order(s, [0])


class TestBuildable(unittest.TestCase):
    def test_grounded_tower_is_buildable(self):
        s = struct([Brick(2, 2, 0, 0, 0), Brick(2, 2, 0, 0, 1), Brick(2, 2, 0, 0, 2)])
        self.assertTrue(is_buildable(s))
        order = find_buildable_order(s)
        self.assertIsNotNone(order)
        self.assertTrue(is_supported_order(s, order))

    def test_floating_island_not_buildable(self):
        s = struct([Brick(2, 2, 0, 0, 0), Brick(2, 2, 0, 0, 2)])
        self.assertFalse(is_buildable(s))
        self.assertIsNone(find_buildable_order(s))

    def test_bridge_over_two_towers_buildable(self):
        s = struct(
            [
                Brick(2, 2, 0, 0, 0),
                Brick(2, 2, 6, 0, 0),
                Brick(8, 2, 0, 0, 1),
            ]
        )
        self.assertTrue(is_buildable(s))
        order = find_buildable_order(s)
        # the spanning brick must come after both towers it rests on
        self.assertEqual(order.index(2), 2)

    def test_empty(self):
        self.assertTrue(is_buildable(struct([])))
        self.assertEqual(find_buildable_order(struct([])), [])


class TestAssemblyStability(unittest.TestCase):
    def test_stack_is_stable_throughout(self):
        s = struct([Brick(2, 2, 0, 0, 0), Brick(2, 2, 0, 0, 1), Brick(2, 2, 0, 0, 2)])
        self.assertTrue(is_assembly_stable(s, is_stable))

    def test_intermediate_instability_detected(self):
        # A long overhanging top brick makes the final (and its own) step unstable.
        s = struct([Brick(1, 1, 0, 0, 0), Brick(8, 1, 0, 0, 1)])
        self.assertFalse(is_assembly_stable(s, is_stable))


if __name__ == "__main__":
    unittest.main()
