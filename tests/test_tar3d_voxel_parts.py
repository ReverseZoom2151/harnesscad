"""Tests for geometry.tar3d_voxel_parts."""

import unittest

from harnesscad.domain.geometry.assembly.tar3d_voxel_parts import (
    VoxelPart,
    assemble,
    connected_parts,
    covers,
    is_valid_decomposition,
    part_order_key,
    parts_disjoint,
)


class TestVoxelPart(unittest.TestCase):
    def test_bounds_and_corner(self):
        p = VoxelPart([(1, 2, 3), (2, 2, 4)])
        self.assertEqual(p.bounds(), ((1, 2, 3), (2, 2, 4)))
        self.assertEqual(p.min_corner(), (1, 2, 3))

    def test_empty_rejected(self):
        with self.assertRaises(ValueError):
            VoxelPart([])

    def test_equality_and_hash(self):
        a = VoxelPart([(0, 0, 0), (1, 0, 0)])
        b = VoxelPart([(1, 0, 0), (0, 0, 0)])
        self.assertEqual(a, b)
        self.assertEqual(hash(a), hash(b))


class TestConnectedParts(unittest.TestCase):
    def test_two_separate_blobs(self):
        vox = {(0, 0, 0), (1, 0, 0),          # blob A
               (5, 5, 5), (5, 5, 6)}          # blob B
        parts = connected_parts(vox)
        self.assertEqual(len(parts), 2)
        self.assertTrue(is_valid_decomposition(parts, vox))

    def test_single_component(self):
        vox = {(0, 0, 0), (1, 0, 0), (1, 1, 0)}
        parts = connected_parts(vox)
        self.assertEqual(len(parts), 1)
        self.assertEqual(set(parts[0].voxels), vox)

    def test_diagonal_is_not_connected(self):
        # 6-connectivity: face neighbours only, so diagonals are separate parts.
        vox = {(0, 0, 0), (1, 1, 1)}
        parts = connected_parts(vox)
        self.assertEqual(len(parts), 2)

    def test_canonical_order_is_deterministic(self):
        vox = {(9, 9, 9), (0, 0, 0), (0, 1, 0)}
        p1 = [tuple(sorted(p.voxels)) for p in connected_parts(vox)]
        p2 = [tuple(sorted(p.voxels)) for p in connected_parts(set(reversed(list(vox))))]
        self.assertEqual(p1, p2)

    def test_order_key_raster_over_min_corner(self):
        # Part with lower (z,y,x) min corner comes first.
        vox = {(3, 0, 0), (0, 0, 5)}
        parts = connected_parts(vox)
        keys = [part_order_key(p) for p in parts]
        self.assertEqual(keys, sorted(keys))
        # The z=0 corner sorts before the z=5 corner.
        self.assertEqual(parts[0].min_corner(), (3, 0, 0))


class TestAssembly(unittest.TestCase):
    def test_assemble_round_trip(self):
        vox = {(0, 0, 0), (1, 0, 0), (7, 7, 7)}
        parts = connected_parts(vox)
        self.assertEqual(assemble(parts), vox)

    def test_disjoint_true(self):
        parts = [VoxelPart([(0, 0, 0)]), VoxelPart([(1, 0, 0)])]
        self.assertTrue(parts_disjoint(parts))

    def test_disjoint_false_on_overlap(self):
        parts = [VoxelPart([(0, 0, 0)]), VoxelPart([(0, 0, 0), (1, 0, 0)])]
        self.assertFalse(parts_disjoint(parts))
        self.assertFalse(is_valid_decomposition(parts, {(0, 0, 0), (1, 0, 0)}))

    def test_covers_detects_missing(self):
        parts = [VoxelPart([(0, 0, 0)])]
        self.assertFalse(covers(parts, {(0, 0, 0), (1, 0, 0)}))
        self.assertTrue(covers(parts, {(0, 0, 0)}))


if __name__ == "__main__":
    unittest.main()
