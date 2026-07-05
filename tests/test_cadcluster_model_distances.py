"""Tests for reconstruction.cadcluster_model_distances."""

import unittest

from reconstruction.cadcluster_model_distances import (
    chamfer_distance,
    min_max_normalize,
    pairwise_distance_matrix,
    voxel_jaccard_distance,
    voxelize,
)


class NormalizeTests(unittest.TestCase):
    def test_unit_cube(self):
        pts = [(2.0, -1.0, 5.0), (4.0, 1.0, 5.0), (3.0, 0.0, 5.0)]
        out = min_max_normalize(pts)
        xs = [p[0] for p in out]
        self.assertAlmostEqual(min(xs), 0.0)
        self.assertAlmostEqual(max(xs), 1.0)
        # z has zero span -> all 0.0
        self.assertTrue(all(p[2] == 0.0 for p in out))

    def test_empty_raises(self):
        with self.assertRaises(ValueError):
            min_max_normalize([])


class ChamferTests(unittest.TestCase):
    def test_identical_is_zero(self):
        cloud = [(0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0)]
        self.assertAlmostEqual(chamfer_distance(cloud, cloud), 0.0)

    def test_symmetric(self):
        a = [(0.0, 0.0, 0.0), (1.0, 1.0, 1.0), (2.0, 0.0, 0.0)]
        b = [(0.0, 0.0, 0.0), (0.5, 0.5, 0.5), (2.0, 1.0, 0.0)]
        self.assertAlmostEqual(chamfer_distance(a, b), chamfer_distance(b, a))

    def test_positive_for_different(self):
        a = [(0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0)]
        b = [(0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (1.0, 1.0, 0.0)]
        self.assertGreater(chamfer_distance(a, b), 0.0)

    def test_without_normalize(self):
        a = [(0.0, 0.0, 0.0), (1.0, 0.0, 0.0)]
        b = [(0.0, 0.0, 0.0), (1.0, 0.0, 0.0)]
        self.assertAlmostEqual(chamfer_distance(a, b, normalize=False), 0.0)

    def test_empty_raises(self):
        with self.assertRaises(ValueError):
            chamfer_distance([], [(0.0, 0.0, 0.0)])


class VoxelizeTests(unittest.TestCase):
    def test_occupied_cells(self):
        pts = [(0.0, 0.0, 0.0), (1.0, 1.0, 1.0)]
        cells = voxelize(pts, resolution=4)
        self.assertIn((0, 0, 0), cells)
        self.assertIn((3, 3, 3), cells)

    def test_bad_resolution(self):
        with self.assertRaises(ValueError):
            voxelize([(0.0, 0.0, 0.0)], resolution=0)


class JaccardTests(unittest.TestCase):
    def test_identical_is_zero(self):
        cloud = [(0.0, 0.0, 0.0), (1.0, 1.0, 1.0), (0.5, 0.5, 0.5)]
        self.assertAlmostEqual(voxel_jaccard_distance(cloud, cloud), 0.0)

    def test_range(self):
        a = [(0.0, 0.0, 0.0), (1.0, 1.0, 1.0)]
        b = [(0.0, 0.0, 0.0), (0.5, 0.5, 0.5)]
        d = voxel_jaccard_distance(a, b, resolution=8)
        self.assertGreaterEqual(d, 0.0)
        self.assertLessEqual(d, 1.0)


class MatrixTests(unittest.TestCase):
    def test_chamfer_matrix_symmetric(self):
        clouds = [
            [(0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0)],
            [(0.0, 0.0, 0.0), (2.0, 0.0, 0.0), (0.0, 2.0, 0.0)],
            [(1.0, 1.0, 1.0), (2.0, 1.0, 1.0), (1.0, 2.0, 1.0)],
        ]
        m = pairwise_distance_matrix(clouds, metric="chamfer")
        for i in range(3):
            self.assertAlmostEqual(m[i][i], 0.0)
            for j in range(3):
                self.assertAlmostEqual(m[i][j], m[j][i])

    def test_jaccard_matrix(self):
        clouds = [
            [(0.0, 0.0, 0.0), (1.0, 1.0, 1.0)],
            [(0.0, 0.0, 0.0), (1.0, 1.0, 1.0)],
        ]
        m = pairwise_distance_matrix(clouds, metric="jaccard", resolution=4)
        self.assertAlmostEqual(m[0][1], 0.0)

    def test_bad_metric(self):
        with self.assertRaises(ValueError):
            pairwise_distance_matrix([[(0.0, 0.0, 0.0)]], metric="hausdorff")


if __name__ == "__main__":
    unittest.main()
