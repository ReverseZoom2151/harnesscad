"""Tests for geometry.model_diff (diff-viewer CSG partition)."""

import unittest

from harnesscad.domain.geometry.model_diff import (
    ModelDiff,
    VoxelSolid,
    model_diff,
    voxelize_boxes,
)


class TestVoxelSolid(unittest.TestCase):
    def test_set_ops(self):
        a = VoxelSolid([(0, 0, 0), (1, 0, 0)])
        b = VoxelSolid([(1, 0, 0), (2, 0, 0)])
        self.assertEqual(a.intersection(b), VoxelSolid([(1, 0, 0)]))
        self.assertEqual(a.difference(b), VoxelSolid([(0, 0, 0)]))
        self.assertEqual(len(a.union(b)), 3)


class TestModelDiff(unittest.TestCase):
    def test_partition(self):
        before = VoxelSolid([(0, 0, 0), (1, 0, 0)])
        after = VoxelSolid([(1, 0, 0), (2, 0, 0)])
        d = model_diff(before, after)
        self.assertEqual(d.unchanged, VoxelSolid([(1, 0, 0)]))
        self.assertEqual(d.additions, VoxelSolid([(2, 0, 0)]))
        self.assertEqual(d.deletions, VoxelSolid([(0, 0, 0)]))

    def test_identical(self):
        s = VoxelSolid([(0, 0, 0)])
        d = model_diff(s, s)
        self.assertTrue(d.is_identical)
        self.assertEqual(d.change_ratio, 0.0)

    def test_change_ratio(self):
        before = VoxelSolid([(0, 0, 0), (1, 0, 0)])
        after = VoxelSolid([(1, 0, 0), (2, 0, 0)])
        d = model_diff(before, after)
        # union 3, changed 2 -> 2/3
        self.assertAlmostEqual(d.change_ratio, 2.0 / 3.0)

    def test_disjoint_ratio_one(self):
        d = model_diff(VoxelSolid([(0, 0, 0)]), VoxelSolid([(9, 9, 9)]))
        self.assertEqual(d.change_ratio, 1.0)

    def test_empty_both(self):
        d = model_diff(VoxelSolid(), VoxelSolid())
        self.assertEqual(d.change_ratio, 0.0)
        self.assertTrue(d.is_identical)

    def test_counts(self):
        before = VoxelSolid([(0, 0, 0), (1, 0, 0)])
        after = VoxelSolid([(1, 0, 0), (2, 0, 0), (3, 0, 0)])
        d = model_diff(before, after)
        self.assertEqual(d.unchanged_count, 1)
        self.assertEqual(d.additions_count, 2)
        self.assertEqual(d.deletions_count, 1)
        self.assertEqual(d.union_count, 4)


class TestVoxelizeBoxes(unittest.TestCase):
    def test_unit_box(self):
        # box [0,1]^3 at resolution 1 -> single cell centre (0.5,0.5,0.5)
        s = voxelize_boxes([((0.0, 0.0, 0.0), (1.0, 1.0, 1.0))], 1.0)
        self.assertEqual(s, VoxelSolid([(0, 0, 0)]))

    def test_diff_of_grown_box(self):
        before = voxelize_boxes([((0.0, 0.0, 0.0), (2.0, 1.0, 1.0))], 1.0)
        after = voxelize_boxes([((0.0, 0.0, 0.0), (3.0, 1.0, 1.0))], 1.0)
        d = model_diff(before, after)
        self.assertEqual(d.deletions_count, 0)
        self.assertEqual(d.additions_count, 1)  # one new cell in x
        self.assertGreater(d.unchanged_count, 0)

    def test_bad_resolution(self):
        with self.assertRaises(ValueError):
            voxelize_boxes([], 0.0)

    def test_deterministic(self):
        boxes = [((0.0, 0.0, 0.0), (2.5, 2.5, 2.5))]
        self.assertEqual(voxelize_boxes(boxes, 0.5), voxelize_boxes(boxes, 0.5))


if __name__ == "__main__":
    unittest.main()
