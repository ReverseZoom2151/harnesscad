"""Tests for the Brepler area-descending BFS face linearisation."""

import unittest

from harnesscad.domain.reconstruction.brep import brepler_linearise as bl


class AdjacencyTest(unittest.TestCase):
    def test_symmetric_and_sorted(self):
        adj = bl.adjacency_from_edges([0, 1, 2], [(0, 1), (1, 2)])
        self.assertEqual(adj[1], [0, 2])
        self.assertEqual(adj[0], [1])
        self.assertEqual(adj[2], [1])

    def test_unknown_face_raises(self):
        with self.assertRaises(ValueError):
            bl.adjacency_from_edges([0, 1], [(0, 5)])


class CornerSeedTest(unittest.TestCase):
    def test_closest_to_default_corner(self):
        pts = {0: (0.9, 0.9, 0.9), 1: (-0.9, -0.9, -0.9), 2: (0.0, 0.0, 0.0)}
        self.assertEqual(bl.corner_seed(pts), 1)

    def test_tie_break_by_id(self):
        pts = {2: (0.0, 0.0, 0.0), 1: (0.0, 0.0, 0.0)}
        self.assertEqual(bl.corner_seed(pts), 1)


class BFSTest(unittest.TestCase):
    def test_area_descending_expansion(self):
        # star graph: 0 in the centre, neighbours 1,2,3 with areas 1<2<3
        adj = {0: [1, 2, 3], 1: [0], 2: [0], 3: [0]}
        areas = {0: 10.0, 1: 1.0, 2: 2.0, 3: 3.0}
        order = bl.area_descending_bfs(adj, areas, 0)
        # largest-area neighbour first
        self.assertEqual(order, [0, 3, 2, 1])

    def test_visits_only_component(self):
        adj = {0: [1], 1: [0], 2: [3], 3: [2]}
        areas = {i: 1.0 for i in range(4)}
        self.assertEqual(sorted(bl.area_descending_bfs(adj, areas, 0)), [0, 1])


class LineariseTest(unittest.TestCase):
    def test_full_permutation_connected(self):
        # chain 0-1-2-3, seed by corner at face 0
        adj = bl.adjacency_from_edges([0, 1, 2, 3], [(0, 1), (1, 2), (2, 3)])
        areas = {0: 4.0, 1: 3.0, 2: 2.0, 3: 1.0}
        pts = {0: (-1, -1, -1), 1: (0, 0, 0), 2: (0.5, 0.5, 0.5), 3: (1, 1, 1)}
        order = bl.linearise_faces(adj, areas, pts)
        self.assertEqual(order, [0, 1, 2, 3])
        self.assertEqual(sorted(order), [0, 1, 2, 3])

    def test_explicit_seed_overrides(self):
        adj = bl.adjacency_from_edges([0, 1, 2, 3], [(0, 1), (1, 2), (2, 3)])
        areas = {0: 4.0, 1: 3.0, 2: 2.0, 3: 1.0}
        order = bl.linearise_faces(adj, areas, None, seed=3)
        self.assertEqual(order, [3, 2, 1, 0])

    def test_disconnected_is_total(self):
        adj = {0: [1], 1: [0], 2: [3], 3: [2]}
        areas = {0: 4.0, 1: 3.0, 2: 2.0, 3: 1.0}
        pts = {0: (-1, -1, -1), 1: (0, 0, 0), 2: (0.4, 0.4, 0.4), 3: (1, 1, 1)}
        order = bl.linearise_faces(adj, areas, pts)
        self.assertEqual(sorted(order), [0, 1, 2, 3])
        # first component (seeded at corner face 0) comes before the second
        self.assertEqual(order[:2], [0, 1])

    def test_empty(self):
        self.assertEqual(bl.linearise_faces({}, {}, {}), [])

    def test_matches_reference_area_order_on_branch(self):
        # face 0 has two unvisited neighbours 1 (area 5) and 2 (area 9): 2 first.
        adj = {0: [1, 2], 1: [0], 2: [0]}
        areas = {0: 100.0, 1: 5.0, 2: 9.0}
        pts = {0: (-1, -1, -1), 1: (1, 1, 1), 2: (0.5, 0.5, 0.5)}
        self.assertEqual(bl.linearise_faces(adj, areas, pts), [0, 2, 1])


if __name__ == "__main__":
    unittest.main()
