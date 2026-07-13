"""Tests for geometry.octfusion_octree."""

import unittest

from harnesscad.domain.geometry.octfusion_octree import Octree, OctreeNode, _child_index


class TestChildIndex(unittest.TestCase):
    def test_morton_bit_layout(self):
        self.assertEqual(_child_index(0, 0, 0), 0)
        self.assertEqual(_child_index(1, 0, 0), 1)
        self.assertEqual(_child_index(0, 1, 0), 2)
        self.assertEqual(_child_index(0, 0, 1), 4)
        self.assertEqual(_child_index(1, 1, 1), 7)


class TestNode(unittest.TestCase):
    def test_resolution_and_cell_size(self):
        n = OctreeNode(3, 0, 0, 0)
        self.assertEqual(n.resolution(), 8)
        self.assertAlmostEqual(n.cell_size(1.0), 0.125)

    def test_bounds_and_center(self):
        n = OctreeNode(1, 1, 0, 0)
        lo, hi = n.bounds()
        self.assertEqual(lo, (0.5, 0.0, 0.0))
        self.assertEqual(hi, (1.0, 0.5, 0.5))
        self.assertEqual(n.center(), (0.75, 0.25, 0.25))


class TestConstruction(unittest.TestCase):
    def test_empty_stays_root_leaf(self):
        t = Octree.from_points([], max_depth=4)
        self.assertTrue(t.root.is_leaf)
        self.assertFalse(t.root.occupied)
        self.assertEqual(t.leaf_count(), 1)

    def test_single_point_refines_to_max_depth(self):
        t = Octree.from_points([(0.1, 0.1, 0.1)], max_depth=3)
        occ = list(t.occupied_leaves())
        self.assertEqual(len(occ), 1)
        # one occupied leaf at full depth
        self.assertEqual(occ[0].depth, 3)
        self.assertTrue(occ[0].occupied)

    def test_adaptive_partition_covers_volume(self):
        # occupied leaves cover their voxels; all leaves tile without overlap
        t = Octree.from_points([(0.1, 0.1, 0.1), (0.9, 0.9, 0.9)], max_depth=2)
        # sum of leaf cell volumes equals cube volume (partition of unity)
        total = sum(leaf.cell_size() ** 3 for leaf in t.leaves())
        self.assertAlmostEqual(total, 1.0)

    def test_two_far_points_split_root(self):
        t = Octree.from_points([(0.1, 0.1, 0.1), (0.9, 0.9, 0.9)], max_depth=3)
        self.assertFalse(t.root.is_leaf)
        self.assertEqual(t.occupied_leaf_count(), 2)

    def test_max_depth_zero(self):
        t = Octree.from_points([(0.5, 0.5, 0.5)], max_depth=0)
        self.assertTrue(t.root.is_leaf)
        self.assertTrue(t.root.occupied)

    def test_points_outside_ignored(self):
        t = Octree.from_points([(5.0, 5.0, 5.0)], max_depth=3)
        self.assertFalse(t.root.occupied)

    def test_invalid_params(self):
        with self.assertRaises(ValueError):
            Octree.from_points([], max_depth=-1)
        with self.assertRaises(ValueError):
            Octree.from_points([], max_depth=2, size=0.0)


class TestTraversalOrder(unittest.TestCase):
    def test_leaves_deterministic_morton_order(self):
        t = Octree.from_points([(0.1, 0.1, 0.1), (0.9, 0.1, 0.1)], max_depth=1)
        keys = [leaf.key() for leaf in t.leaves()]
        # ascending Morton order of the 8 children of the root
        self.assertEqual(keys, sorted(keys, key=lambda k: (k[0], k[3], k[2], k[1])))
        self.assertEqual(len(keys), 8)

    def test_node_and_leaf_counts(self):
        t = Octree.from_points([(0.1, 0.1, 0.1), (0.9, 0.9, 0.9)], max_depth=2)
        self.assertEqual(t.leaf_count(), sum(1 for _ in t.leaves()))
        self.assertTrue(t.node_count() > t.leaf_count())

    def test_depth_reached(self):
        t = Octree.from_points([(0.1, 0.1, 0.1)], max_depth=5)
        self.assertEqual(t.depth_reached(), 5)


class TestQueries(unittest.TestCase):
    def test_find_leaf_contains_point(self):
        t = Octree.from_points([(0.1, 0.1, 0.1), (0.9, 0.9, 0.9)], max_depth=3)
        leaf = t.find_leaf((0.1, 0.1, 0.1))
        self.assertIsNotNone(leaf)
        lo, hi = leaf.bounds()
        self.assertTrue(lo[0] <= 0.1 <= hi[0])
        self.assertTrue(leaf.occupied)

    def test_find_leaf_outside_returns_none(self):
        t = Octree.from_points([(0.5, 0.5, 0.5)], max_depth=2)
        self.assertIsNone(t.find_leaf((1.5, 0.5, 0.5)))

    def test_face_neighbor(self):
        t = Octree.from_points(
            [(0.1, 0.1, 0.1), (0.9, 0.9, 0.9), (0.1, 0.9, 0.1)], max_depth=2
        )
        leaf = t.find_leaf((0.1, 0.1, 0.1))
        nb = t.face_neighbor(leaf, axis=0, sign=1)
        self.assertIsNotNone(nb)
        # neighbour cell center is one cell over in +x
        self.assertGreater(nb.center()[0], leaf.center()[0])

    def test_face_neighbor_outside(self):
        t = Octree.from_points([(0.1, 0.1, 0.1)], max_depth=2)
        leaf = t.find_leaf((0.1, 0.1, 0.1))
        # -x from a leftmost cell leaves the cube
        self.assertIsNone(t.face_neighbor(leaf, axis=0, sign=-1))

    def test_face_neighbor_invalid_axis(self):
        t = Octree.from_points([(0.1, 0.1, 0.1)], max_depth=2)
        leaf = t.root
        with self.assertRaises(ValueError):
            t.face_neighbor(leaf, axis=3, sign=1)
        with self.assertRaises(ValueError):
            t.face_neighbor(leaf, axis=0, sign=0)


class TestVoxelConversion(unittest.TestCase):
    def test_to_voxels_single(self):
        t = Octree.from_points([(0.1, 0.1, 0.1)], max_depth=3)
        vox = t.to_voxels()
        self.assertEqual(vox, {(0, 0, 0)})

    def test_to_voxels_coarse_leaf_expands(self):
        # occupied root leaf at depth 0 covers whole 2^2 grid
        t = Octree.from_points([(0.5, 0.5, 0.5)], max_depth=0)
        vox = t.to_voxels(depth=2)
        self.assertEqual(len(vox), 8 ** 2 // 8 * 8)  # 4^3 = 64
        self.assertEqual(len(vox), 64)

    def test_roundtrip_voxels_octree_voxels(self):
        occ = {(0, 0, 0), (1, 1, 1), (7, 7, 7), (3, 4, 5)}
        t = Octree.from_voxels(sorted(occ), max_depth=3)
        self.assertEqual(t.to_voxels(depth=3), occ)

    def test_from_voxels_occupancy(self):
        t = Octree.from_voxels([(0, 0, 0)], max_depth=2)
        self.assertTrue(t.root.occupied)
        self.assertEqual(t.occupied_leaf_count(), 1)
        leaf = next(t.occupied_leaves())
        self.assertEqual(leaf.depth, 2)
        self.assertEqual(leaf.key(), (2, 0, 0, 0))

    def test_from_voxels_out_of_range_ignored(self):
        t = Octree.from_voxels([(99, 0, 0)], max_depth=2)
        self.assertFalse(t.root.occupied)


if __name__ == "__main__":
    unittest.main()
