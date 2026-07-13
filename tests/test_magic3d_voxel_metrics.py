"""Tests for bench.magic3d_voxel_metrics (Magic3DSketch voxel-IoU / pose error)."""

import unittest

from harnesscad.eval.bench.magic3d_voxel_metrics import (
    voxel_iou,
    voxelize_points,
    pose_mse,
    pose_mae,
    circular_abs_error,
    azimuth_mae,
    category_mean,
)


class VoxelIoUTest(unittest.TestCase):
    def test_identical(self):
        a = {(0, 0, 0), (1, 0, 0), (0, 1, 0)}
        self.assertAlmostEqual(voxel_iou(a, a), 1.0)

    def test_disjoint(self):
        self.assertAlmostEqual(voxel_iou({(0, 0, 0)}, {(5, 5, 5)}), 0.0)

    def test_partial_overlap(self):
        a = {(0, 0, 0), (1, 0, 0), (2, 0, 0)}
        b = {(1, 0, 0), (2, 0, 0), (3, 0, 0)}
        # intersection 2, union 4
        self.assertAlmostEqual(voxel_iou(a, b), 0.5)

    def test_both_empty_perfect(self):
        self.assertAlmostEqual(voxel_iou(set(), set()), 1.0)

    def test_accepts_lists_with_duplicates(self):
        a = [(0, 0, 0), (0, 0, 0), (1, 1, 1)]
        b = [(1, 1, 1)]
        # A collapses to 2 voxels, intersection 1, union 2
        self.assertAlmostEqual(voxel_iou(a, b), 0.5)

    def test_symmetry(self):
        a = {(0, 0, 0), (1, 0, 0)}
        b = {(1, 0, 0), (2, 0, 0), (3, 0, 0)}
        self.assertAlmostEqual(voxel_iou(a, b), voxel_iou(b, a))


class VoxelizeTest(unittest.TestCase):
    def test_unit_grid(self):
        pts = [(0.2, 0.9, 0.1), (1.5, 0.0, 0.0)]
        self.assertEqual(voxelize_points(pts), {(0, 0, 0), (1, 0, 0)})

    def test_spacing_and_origin(self):
        pts = [(2.0, 2.0, 2.0)]
        v = voxelize_points(pts, origin=(1.0, 1.0, 1.0), spacing=0.5)
        self.assertEqual(v, {(2, 2, 2)})

    def test_negative_coords_floor(self):
        self.assertEqual(voxelize_points([(-0.1, 0.0, 0.0)]), {(-1, 0, 0)})

    def test_bad_spacing(self):
        with self.assertRaises(ValueError):
            voxelize_points([(0, 0, 0)], spacing=0.0)

    def test_iou_pipeline(self):
        a = voxelize_points([(0.1, 0.1, 0.1), (1.1, 0.1, 0.1)])
        b = voxelize_points([(0.9, 0.2, 0.4), (5.0, 5.0, 5.0)])
        # a={(0,0,0),(1,0,0)}, b={(0,0,0),(5,5,5)}, inter=1 union=3
        self.assertAlmostEqual(voxel_iou(a, b), 1.0 / 3.0)


class PoseErrorTest(unittest.TestCase):
    def test_mse_zero(self):
        self.assertAlmostEqual(pose_mse([1.0, 2.0], [1.0, 2.0]), 0.0)

    def test_mse_value(self):
        # diffs 3 and 4 -> (9 + 16)/2 = 12.5
        self.assertAlmostEqual(pose_mse([0.0, 0.0], [3.0, 4.0]), 12.5)

    def test_mae_value(self):
        self.assertAlmostEqual(pose_mae([0.0, 0.0], [3.0, 4.0]), 3.5)

    def test_length_mismatch(self):
        with self.assertRaises(ValueError):
            pose_mse([1.0], [1.0, 2.0])

    def test_empty(self):
        with self.assertRaises(ValueError):
            pose_mae([], [])


class CircularErrorTest(unittest.TestCase):
    def test_wraparound(self):
        self.assertAlmostEqual(circular_abs_error(350.0, 10.0), 20.0)

    def test_no_wrap(self):
        self.assertAlmostEqual(circular_abs_error(10.0, 40.0), 30.0)

    def test_opposite(self):
        self.assertAlmostEqual(circular_abs_error(0.0, 180.0), 180.0)

    def test_bad_period(self):
        with self.assertRaises(ValueError):
            circular_abs_error(1.0, 2.0, period=0.0)

    def test_azimuth_mae(self):
        pred = [350.0, 90.0]
        gt = [10.0, 80.0]
        # errors 20 and 10 -> mean 15
        self.assertAlmostEqual(azimuth_mae(pred, gt), 15.0)

    def test_azimuth_length_mismatch(self):
        with self.assertRaises(ValueError):
            azimuth_mae([1.0], [1.0, 2.0])


class CategoryMeanTest(unittest.TestCase):
    def test_mean(self):
        self.assertAlmostEqual(
            category_mean({"car": 0.7, "sofa": 0.5, "chair": 0.6}), 0.6
        )

    def test_empty_raises(self):
        with self.assertRaises(ValueError):
            category_mean({})


if __name__ == "__main__":
    unittest.main()
