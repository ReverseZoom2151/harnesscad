"""Tests for bench.lion_one_nna (LION 1-NNA + voxel-occupancy JSD)."""

import unittest

from harnesscad.eval.bench.generative.lion_one_nna import (
    one_nna,
    one_nna_from_matrix,
    pairwise_distance_matrix,
    voxel_index,
    voxel_occupancy_distribution,
    voxel_jsd,
)


def _abs1(a, b):
    """Scalar |a-b| distance for simple deterministic shapes."""
    return abs(a - b)


class OneNNATest(unittest.TestCase):
    def test_matrix_symmetric_zero_diag(self):
        mat = pairwise_distance_matrix([0.0, 1.0, 3.0], _abs1)
        self.assertEqual(mat[0][0], 0.0)
        self.assertEqual(mat[0][2], 3.0)
        self.assertEqual(mat[2][0], mat[0][2])

    def test_fully_alternating_is_zero(self):
        # Every point's nearest neighbour is the opposite label => 0.0.
        generated = [0.0, 2.0, 4.0]
        reference = [1.0, 3.0, 5.0]
        acc = one_nna(generated, reference, _abs1)
        self.assertAlmostEqual(acc, 0.0, places=6)

    def test_balanced_case_is_half(self):
        # Two points' NN is same-label, two opposite => 0.5 (ideal generator).
        generated = [0.0, 1.0]
        reference = [1.5, 3.0]
        acc = one_nna(generated, reference, _abs1)
        self.assertAlmostEqual(acc, 0.5, places=6)

    def test_well_separated_is_one(self):
        # Two tight, far-apart clusters => nearest neighbour always same label.
        generated = [0.0, 0.1, 0.2]
        reference = [10.0, 10.1, 10.2]
        acc = one_nna(generated, reference, _abs1)
        self.assertAlmostEqual(acc, 1.0, places=6)

    def test_range_bounds(self):
        acc = one_nna([0.0, 5.0], [1.0, 4.0], _abs1)
        self.assertGreaterEqual(acc, 0.0)
        self.assertLessEqual(acc, 1.0)

    def test_from_matrix_matches_direct(self):
        generated = [0.0, 1.0]
        reference = [1.5, 3.0]
        mat = pairwise_distance_matrix(generated + reference, _abs1)
        self.assertAlmostEqual(
            one_nna_from_matrix(mat, len(generated)),
            one_nna(generated, reference, _abs1),
            places=6,
        )

    def test_empty_raises(self):
        with self.assertRaises(ValueError):
            one_nna([], [1.0], _abs1)

    def test_matrix_index_out_of_range_raises(self):
        mat = [[0.0, 1.0], [1.0, 0.0]]
        with self.assertRaises(ValueError):
            one_nna_from_matrix(mat, 2)

    def test_deterministic(self):
        g = [0.0, 2.0, 4.0]
        r = [1.0, 3.0, 5.0]
        self.assertEqual(one_nna(g, r, _abs1), one_nna(g, r, _abs1))


class VoxelOccupancyTest(unittest.TestCase):
    def test_voxel_index_center_and_bounds(self):
        self.assertEqual(voxel_index((0.0, 0.0, 0.0), 2), (1, 1, 1))
        # clamped, not raising
        self.assertEqual(voxel_index((-5.0, 5.0, 0.0), 4), (0, 3, 2))

    def test_index_min_corner(self):
        self.assertEqual(voxel_index((-1.0, -1.0, -1.0), 4), (0, 0, 0))

    def test_occupancy_counts(self):
        cloud = [(-0.9, -0.9, -0.9), (-0.9, -0.9, -0.9), (0.9, 0.9, 0.9)]
        hist = voxel_occupancy_distribution([cloud], grid=2)
        self.assertEqual(hist[(0, 0, 0)], 2)
        self.assertEqual(hist[(1, 1, 1)], 1)
        self.assertEqual(sum(hist.values()), 3)

    def test_jsd_identical_is_zero(self):
        cloud = [(0.1, 0.2, 0.3), (-0.4, 0.5, -0.6)]
        self.assertAlmostEqual(voxel_jsd([cloud], [cloud], grid=8), 0.0, places=9)

    def test_jsd_disjoint_is_one(self):
        a = [[(-0.9, -0.9, -0.9)]]
        b = [[(0.9, 0.9, 0.9)]]
        self.assertAlmostEqual(voxel_jsd(a, b, grid=2), 1.0, places=6)

    def test_jsd_symmetric(self):
        a = [[(-0.9, -0.9, -0.9), (0.1, 0.1, 0.1)]]
        b = [[(0.9, 0.9, 0.9), (0.1, 0.1, 0.1)]]
        self.assertAlmostEqual(
            voxel_jsd(a, b, grid=4), voxel_jsd(b, a, grid=4), places=9
        )

    def test_jsd_bounds(self):
        a = [[(0.0, 0.0, 0.0)], [(-0.5, 0.3, 0.2)]]
        b = [[(0.5, -0.5, 0.5)], [(0.1, 0.1, 0.1)]]
        v = voxel_jsd(a, b, grid=6)
        self.assertGreaterEqual(v, 0.0)
        self.assertLessEqual(v, 1.0 + 1e-9)


if __name__ == "__main__":
    unittest.main()
