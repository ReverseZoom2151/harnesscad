"""Tests for geometry.rag6d_ransac_pose."""

import math
import random
import unittest

from geometry.rag6d_ransac_pose import PoseHypothesis, ransac_rigid_pose


def rot_z(deg):
    c, s = math.cos(math.radians(deg)), math.sin(math.radians(deg))
    return [[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]]


def apply(R, t, p):
    return (
        R[0][0] * p[0] + R[0][1] * p[1] + R[0][2] * p[2] + t[0],
        R[1][0] * p[0] + R[1][1] * p[1] + R[1][2] * p[2] + t[1],
        R[2][0] * p[0] + R[2][1] * p[1] + R[2][2] * p[2] + t[2],
    )


# A spread-out model point set.
SRC = [
    (1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0), (1.0, 1.0, 0.0),
    (1.0, 0.0, 1.0), (0.0, 1.0, 1.0), (2.0, 1.0, 0.5), (-1.0, 2.0, 1.0),
    (3.0, -1.0, 2.0), (0.5, 0.5, 2.5),
]


class TestCleanRecovery(unittest.TestCase):
    def test_identity(self):
        dst = list(SRC)
        h = ransac_rigid_pose(SRC, dst, inlier_thresh=1e-6, iterations=20, seed=1)
        self.assertIsNotNone(h)
        self.assertEqual(h.num_inliers, len(SRC))

    def test_known_transform(self):
        R = rot_z(35.0)
        t = [2.0, -1.0, 0.5]
        dst = [apply(R, t, p) for p in SRC]
        h = ransac_rigid_pose(SRC, dst, inlier_thresh=1e-6, iterations=30, seed=7)
        self.assertIsNotNone(h)
        self.assertEqual(h.num_inliers, len(SRC))
        # recovered R, t reproduce the correspondences
        for p, d in zip(SRC, dst):
            got = h.apply(p)
            self.assertAlmostEqual(got[0], d[0], places=6)
            self.assertAlmostEqual(got[1], d[1], places=6)
            self.assertAlmostEqual(got[2], d[2], places=6)


class TestOutliers(unittest.TestCase):
    def test_rejects_outliers(self):
        R = rot_z(20.0)
        t = [1.0, 0.0, -2.0]
        dst = [apply(R, t, p) for p in SRC]
        # corrupt three correspondences with large errors
        dst[2] = (dst[2][0] + 50.0, dst[2][1], dst[2][2])
        dst[5] = (dst[5][0], dst[5][1] - 40.0, dst[5][2])
        dst[8] = (dst[8][0], dst[8][1], dst[8][2] + 60.0)
        h = ransac_rigid_pose(SRC, dst, inlier_thresh=1e-3, iterations=200, seed=3)
        self.assertIsNotNone(h)
        # the seven clean correspondences are inliers, the three bad ones are not
        self.assertEqual(h.num_inliers, 7)
        self.assertFalse(h.inliers[2])
        self.assertFalse(h.inliers[5])
        self.assertFalse(h.inliers[8])
        # pose recovered from the clean set matches ground truth
        for i, (p, d) in enumerate(zip(SRC, dst)):
            if h.inliers[i]:
                got = h.apply(p)
                self.assertAlmostEqual(got[0], d[0], places=4)


class TestDeterminism(unittest.TestCase):
    def test_same_seed_same_result(self):
        R = rot_z(12.0)
        dst = [apply(R, [0, 0, 0], p) for p in SRC]
        dst[0] = (99.0, 99.0, 99.0)
        h1 = ransac_rigid_pose(SRC, dst, inlier_thresh=1e-3, iterations=50, seed=42)
        h2 = ransac_rigid_pose(SRC, dst, inlier_thresh=1e-3, iterations=50, seed=42)
        self.assertEqual(h1.inliers, h2.inliers)
        self.assertEqual(h1.R, h2.R)
        self.assertEqual(h1.t, h2.t)


class TestEdgeCases(unittest.TestCase):
    def test_too_few_points(self):
        self.assertIsNone(ransac_rigid_pose([(0, 0, 0), (1, 0, 0)],
                                            [(0, 0, 0), (1, 0, 0)],
                                            inlier_thresh=1.0))

    def test_length_mismatch(self):
        with self.assertRaises(ValueError):
            ransac_rigid_pose(SRC, SRC[:-1], inlier_thresh=1.0)

    def test_no_consensus_returns_none(self):
        rng = random.Random(0)
        dst = [(rng.uniform(-100, 100), rng.uniform(-100, 100),
                rng.uniform(-100, 100)) for _ in SRC]
        h = ransac_rigid_pose(SRC, dst, inlier_thresh=1e-9,
                              iterations=20, seed=0, min_inliers=8)
        self.assertIsNone(h)


if __name__ == "__main__":
    unittest.main()
