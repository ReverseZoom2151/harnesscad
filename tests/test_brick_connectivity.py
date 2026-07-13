"""Tests for geometry.brick_connectivity (stud-into-tube adjacency graph)."""

import unittest

from harnesscad.domain.geometry.brick_structure import Brick, BrickStructure
from harnesscad.domain.geometry.brick_connectivity import (
    GROUND,
    adjacency,
    connected_components,
    connection_area,
    floating_bricks,
    grounded_bricks,
    is_grounded,
    is_interconnected,
    is_single_component,
    supporting_indices,
    total_connection_area,
)


def struct(bricks):
    return BrickStructure.from_bricks(bricks)


class TestConnectionArea(unittest.TestCase):
    def test_area(self):
        lower = Brick(2, 2, 0, 0, 0)
        upper = Brick(2, 2, 0, 0, 1)
        self.assertEqual(connection_area(lower, upper), 4)
        # partial overlap
        self.assertEqual(connection_area(lower, Brick(2, 2, 1, 0, 1)), 2)
        # not directly above
        self.assertEqual(connection_area(lower, Brick(2, 2, 0, 0, 2)), 0)
        # side by side, same layer
        self.assertEqual(connection_area(lower, Brick(2, 2, 2, 0, 0)), 0)


class TestGraph(unittest.TestCase):
    def test_adjacency_ground(self):
        s = struct([Brick(2, 2, 0, 0, 0), Brick(2, 2, 0, 0, 1)])
        g = adjacency(s)
        self.assertIn(0, g[GROUND])
        self.assertNotIn(1, g[GROUND])
        self.assertIn(1, g[0])
        self.assertIn(0, g[1])

    def test_supporting_indices(self):
        s = struct(
            [Brick(2, 2, 0, 0, 0), Brick(2, 2, 2, 0, 0), Brick(4, 2, 0, 0, 1)]
        )
        self.assertEqual(supporting_indices(s, 2), [0, 1])
        self.assertEqual(supporting_indices(s, 0), [])

    def test_total_connection_area(self):
        s = struct([Brick(2, 2, 0, 0, 0), Brick(2, 2, 0, 0, 1)])
        self.assertEqual(total_connection_area(s), 4)


class TestConnectivity(unittest.TestCase):
    def test_single_component_and_grounded(self):
        s = struct([Brick(2, 2, 0, 0, 0), Brick(2, 2, 0, 0, 1)])
        self.assertTrue(is_single_component(s))
        self.assertTrue(is_grounded(s))
        self.assertTrue(is_interconnected(s))
        self.assertEqual(connected_components(s), [[0, 1]])
        self.assertEqual(floating_bricks(s), [])

    def test_floating_island(self):
        # brick 1 hovers at z=2 with a gap below it -> not grounded
        s = struct([Brick(2, 2, 0, 0, 0), Brick(2, 2, 0, 0, 2)])
        self.assertFalse(is_grounded(s))
        self.assertEqual(floating_bricks(s), [1])
        self.assertEqual(grounded_bricks(s), {0})
        # still two separate components
        self.assertFalse(is_single_component(s))
        self.assertFalse(is_interconnected(s))

    def test_two_grounded_towers_share_baseplate(self):
        # two disjoint towers both on the baseplate -> one component (via GROUND),
        # both grounded.
        s = struct(
            [
                Brick(2, 2, 0, 0, 0),
                Brick(2, 2, 0, 0, 1),
                Brick(2, 2, 10, 10, 0),
                Brick(2, 2, 10, 10, 1),
            ]
        )
        self.assertTrue(is_grounded(s))
        self.assertTrue(is_single_component(s))
        self.assertEqual(connected_components(s), [[0, 1, 2, 3]])

    def test_bridge_connects_components(self):
        s = struct(
            [
                Brick(2, 2, 0, 0, 0),
                Brick(2, 2, 6, 0, 0),
                Brick(8, 2, 0, 0, 1),  # spans both towers
            ]
        )
        self.assertTrue(is_interconnected(s))
        self.assertEqual(connected_components(s), [[0, 1, 2]])

    def test_empty(self):
        s = struct([])
        self.assertTrue(is_single_component(s))
        self.assertTrue(is_grounded(s))


if __name__ == "__main__":
    unittest.main()
