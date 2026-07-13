"""Tests for geometry.querycad_face_adjacency."""

import unittest

from harnesscad.domain.geometry.topology.querycad_face_adjacency import FaceAdjacencyGraph


class BuildTest(unittest.TestCase):
    def test_from_edges_symmetric(self):
        g = FaceAdjacencyGraph.from_edges([(0, 1), (1, 2)])
        self.assertEqual(sorted(g.faces), [0, 1, 2])
        self.assertEqual(g.neighbors(1), [0, 2])
        self.assertEqual(g.neighbors(0), [1])

    def test_no_self_loop(self):
        g = FaceAdjacencyGraph()
        g.add_adjacency(5, 5)
        self.assertEqual(g.faces, [])

    def test_no_duplicate_neighbor(self):
        g = FaceAdjacencyGraph()
        g.add_adjacency(0, 1)
        g.add_adjacency(0, 1)
        self.assertEqual(g.neighbors(0), [1])


class ConnectedComponentTest(unittest.TestCase):
    def _two_pockets(self):
        # Two disjoint triangles of faces: {0,1,2} and {10,11,12}.
        return FaceAdjacencyGraph.from_edges(
            [(0, 1), (1, 2), (0, 2), (10, 11), (11, 12), (10, 12)]
        )

    def test_component_within_whitelist(self):
        g = self._two_pockets()
        comp = g.connected_component(0, whitelist={0, 1, 2, 10, 11, 12})
        self.assertEqual(sorted(comp), [0, 1, 2])

    def test_whitelist_prunes_across_bridge(self):
        # Bridge 2-10 exists, but whitelist excludes it so components stay split.
        g = self._two_pockets()
        g.add_adjacency(2, 10)
        comp = g.connected_component(0, whitelist={0, 1, 2, 11, 12})
        self.assertEqual(sorted(comp), [0, 1, 2])

    def test_seed_not_in_whitelist(self):
        g = self._two_pockets()
        self.assertEqual(g.connected_component(0, whitelist={1, 2}), [])

    def test_isolated_face(self):
        g = FaceAdjacencyGraph()
        g.add_face(7)
        self.assertEqual(g.connected_component(7), [7])

    def test_first_seen_order(self):
        g = FaceAdjacencyGraph.from_edges([(0, 1), (0, 2), (0, 3)])
        # DFS from 0 pushes neighbours reversed so 1 expands first.
        self.assertEqual(g.connected_component(0)[0], 0)
        self.assertEqual(sorted(g.connected_component(0)), [0, 1, 2, 3])


class PartitionTest(unittest.TestCase):
    def test_partition_into_instances(self):
        g = FaceAdjacencyGraph.from_edges(
            [(0, 1), (1, 2), (10, 11), (11, 12)]
        )
        parts = g.partition([0, 1, 2, 10, 11, 12])
        self.assertEqual(len(parts), 2)
        self.assertEqual(sorted(parts[0]), [0, 1, 2])
        self.assertEqual(sorted(parts[1]), [10, 11, 12])

    def test_partition_respects_subset(self):
        g = FaceAdjacencyGraph.from_edges([(0, 1), (1, 2), (2, 3)])
        # Only faces 0 and 3 tagged; not adjacent within the subset -> 2 parts.
        parts = g.partition([0, 3])
        self.assertEqual(len(parts), 2)
        self.assertEqual([sorted(p) for p in parts], [[0], [3]])

    def test_partition_dedup(self):
        g = FaceAdjacencyGraph.from_edges([(0, 1)])
        parts = g.partition([0, 0, 1])
        self.assertEqual(len(parts), 1)
        self.assertEqual(sorted(parts[0]), [0, 1])

    def test_partition_deterministic(self):
        g = FaceAdjacencyGraph.from_edges([(0, 1), (10, 11)])
        self.assertEqual(g.partition([0, 1, 10, 11]), g.partition([0, 1, 10, 11]))


class DilateTest(unittest.TestCase):
    def test_one_ring(self):
        g = FaceAdjacencyGraph.from_edges([(0, 1), (1, 2), (2, 3)])
        grown = g.dilate([1])
        self.assertEqual(grown[0], 1)
        self.assertEqual(sorted(grown), [0, 1, 2])

    def test_two_rings(self):
        g = FaceAdjacencyGraph.from_edges([(0, 1), (1, 2), (2, 3)])
        grown = g.dilate([1], rings=2)
        self.assertEqual(sorted(grown), [0, 1, 2, 3])

    def test_zero_rings_identity(self):
        g = FaceAdjacencyGraph.from_edges([(0, 1)])
        self.assertEqual(g.dilate([0], rings=0), [0])

    def test_negative_rings_raises(self):
        g = FaceAdjacencyGraph.from_edges([(0, 1)])
        with self.assertRaises(ValueError):
            g.dilate([0], rings=-1)

    def test_dilate_stops_when_saturated(self):
        g = FaceAdjacencyGraph.from_edges([(0, 1)])
        self.assertEqual(sorted(g.dilate([0], rings=99)), [0, 1])

    def test_original_faces_first(self):
        g = FaceAdjacencyGraph.from_edges([(0, 1), (0, 2)])
        grown = g.dilate([0])
        self.assertEqual(grown[0], 0)


class BoundaryTest(unittest.TestCase):
    def test_boundary_faces(self):
        g = FaceAdjacencyGraph.from_edges([(0, 1), (1, 2), (2, 3)])
        # Set {1,2}; face 1 borders 0 (outside), face 2 borders 3 (outside).
        self.assertEqual(sorted(g.boundary_faces([1, 2])), [1, 2])

    def test_interior_face_excluded(self):
        g = FaceAdjacencyGraph.from_edges([(0, 1), (1, 2), (0, 2), (2, 3)])
        # Set {0,1,2}: 0 and 1 only touch each other/2 (inside); 2 touches 3.
        self.assertEqual(g.boundary_faces([0, 1, 2]), [2])


if __name__ == "__main__":
    unittest.main()
