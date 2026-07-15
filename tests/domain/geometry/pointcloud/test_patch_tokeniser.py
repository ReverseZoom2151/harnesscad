"""Tests for geometry.pointcloud.patch_tokeniser (PointBERT FPS + kNN)."""

import math
import unittest

from harnesscad.domain.geometry.pointcloud.patch_tokeniser import (
    Patches,
    farthest_point_sampling,
    group_patches,
    knn_indices,
    square_distance,
)


class TestFPS(unittest.TestCase):
    def test_picks_spread_corners(self):
        # unit square corners + a centre; FPS from index 0 should grab the
        # far corners before the near centre.
        pts = [
            (0.0, 0.0, 0.0),   # 0
            (1.0, 0.0, 0.0),   # 1
            (0.0, 1.0, 0.0),   # 2
            (1.0, 1.0, 0.0),   # 3
            (0.5, 0.5, 0.0),   # 4 centre
        ]
        idx = farthest_point_sampling(pts, 4, seed=0)
        self.assertEqual(idx[0], 0)
        # first pick is 0, next is the opposite corner 3
        self.assertEqual(idx[1], 3)
        self.assertNotIn(4, idx)  # centre is never the farthest of 4

    def test_deterministic(self):
        pts = [(i * 0.1, (i * 7 % 5) * 0.2, (i % 3) * 0.3) for i in range(20)]
        a = farthest_point_sampling(pts, 6, seed=0)
        b = farthest_point_sampling(pts, 6, seed=0)
        self.assertEqual(a, b)

    def test_seed_shifts_start(self):
        pts = [(0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (2.0, 0.0, 0.0)]
        self.assertEqual(farthest_point_sampling(pts, 1, seed=2)[0], 2)

    def test_empty_raises(self):
        with self.assertRaises(ValueError):
            farthest_point_sampling([], 3)

    def test_zero_number(self):
        self.assertEqual(farthest_point_sampling([(0, 0, 0)], 0), [])


class TestKNN(unittest.TestCase):
    def test_nearest(self):
        pts = [(0, 0, 0), (1, 0, 0), (2, 0, 0), (3, 0, 0)]
        out = knn_indices(pts, [(0, 0, 0)], 2)
        self.assertEqual(out[0], [0, 1])

    def test_tie_break_by_index(self):
        pts = [(1, 0, 0), (-1, 0, 0), (0, 1, 0)]  # all dist 1 from origin
        out = knn_indices(pts, [(0, 0, 0)], 3)
        self.assertEqual(out[0], [0, 1, 2])

    def test_pad_when_more_than_points(self):
        pts = [(0, 0, 0), (5, 0, 0)]
        out = knn_indices(pts, [(0, 0, 0)], 4)
        self.assertEqual(len(out[0]), 4)
        # padded with the nearest (index 0)
        self.assertEqual(out[0][2:], [0, 0])


class TestSquareDistance(unittest.TestCase):
    def test_matrix(self):
        m = square_distance([(0, 0, 0)], [(3, 4, 0), (0, 0, 0)])
        self.assertAlmostEqual(m[0][0], 25.0)
        self.assertAlmostEqual(m[0][1], 0.0)


class TestGroupPatches(unittest.TestCase):
    def _cloud(self):
        return [(i % 4, i // 4, 0.0) for i in range(16)]

    def test_shapes(self):
        p = group_patches(self._cloud(), num_group=4, group_size=3, seed=0)
        self.assertIsInstance(p, Patches)
        self.assertEqual(p.num_group, 4)
        self.assertEqual(p.group_size, 3)
        self.assertEqual(len(p.neighborhoods), 4)
        self.assertEqual(len(p.center_indices), 4)

    def test_centre_is_zero_in_local_frame(self):
        # the centre itself is always its own nearest neighbour, so after
        # subtraction its xyz must be (0,0,0).
        p = group_patches(self._cloud(), num_group=3, group_size=5, seed=0)
        for patch in p.neighborhoods:
            # nearest neighbour is the centre -> local coords (0,0,0)
            self.assertEqual(tuple(patch[0][:3]), (0.0, 0.0, 0.0))

    def test_extra_channels_carried_not_normalised(self):
        # 6-dim points: xyz + rgb.  rgb must survive unchanged.
        pts = [
            (0.0, 0.0, 0.0, 0.9, 0.1, 0.2),
            (1.0, 0.0, 0.0, 0.3, 0.4, 0.5),
            (0.0, 1.0, 0.0, 0.6, 0.7, 0.8),
        ]
        p = group_patches(pts, num_group=1, group_size=1, seed=0)
        # the single neighbour of centre 0 is itself: xyz zeroed, rgb intact
        nb = p.neighborhoods[0][0]
        self.assertEqual(tuple(nb[:3]), (0.0, 0.0, 0.0))
        self.assertEqual(tuple(nb[3:]), (0.9, 0.1, 0.2))

    def test_deterministic(self):
        a = group_patches(self._cloud(), 4, 4, seed=0)
        b = group_patches(self._cloud(), 4, 4, seed=0)
        self.assertEqual(a.center_indices, b.center_indices)
        self.assertEqual(a.neighbor_indices, b.neighbor_indices)

    def test_bad_args(self):
        with self.assertRaises(ValueError):
            group_patches(self._cloud(), 0, 3)
        with self.assertRaises(ValueError):
            group_patches(self._cloud(), 3, 0)


if __name__ == "__main__":
    unittest.main()
