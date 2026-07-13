"""Tests for geometry.dlwheel_spoke_points (paper 112 spoke data processing)."""

import unittest

from harnesscad.domain.geometry.views import dlwheel_spoke_points as sp


class OrderingTests(unittest.TestCase):
    def test_nearest_neighbour_line(self):
        pts = [(0.0, 0.0), (3.0, 0.0), (1.0, 0.0), (2.0, 0.0)]
        ordered = sp.nearest_neighbour_order(pts, start_index=0)
        self.assertEqual(ordered, [(0.0, 0.0), (1.0, 0.0), (2.0, 0.0), (3.0, 0.0)])

    def test_empty(self):
        self.assertEqual(sp.nearest_neighbour_order([]), [])

    def test_bad_start(self):
        with self.assertRaises(ValueError):
            sp.nearest_neighbour_order([(0.0, 0.0)], start_index=5)


class GroupingTests(unittest.TestCase):
    def test_group_split(self):
        ordered = [(0.0, 0.0), (1.0, 0.0), (10.0, 0.0), (11.0, 0.0)]
        groups = sp.group_by_distance(ordered, threshold=5.0)
        self.assertEqual(len(groups), 2)
        self.assertEqual(len(groups[0]), 2)
        self.assertEqual(len(groups[1]), 2)

    def test_single_group(self):
        ordered = [(0.0, 0.0), (1.0, 0.0), (2.0, 0.0)]
        groups = sp.group_by_distance(ordered, threshold=5.0)
        self.assertEqual(len(groups), 1)

    def test_bad_threshold(self):
        with self.assertRaises(ValueError):
            sp.group_by_distance([(0.0, 0.0)], threshold=0.0)


class ReductionTests(unittest.TestCase):
    def test_noise_deleted(self):
        self.assertEqual(sp.reduce_group([(0.0, 0.0), (1.0, 1.0)]), [])

    def test_small_group_kept(self):
        g = [(float(i), 0.0) for i in range(10)]  # 10 <= 20 kept
        self.assertEqual(sp.reduce_group(g), g)

    def test_mid_group_reduced_to_sixth(self):
        g = [(float(i), 0.0) for i in range(60)]  # 20 < 60 < 100 -> keep 60//6=10
        r = sp.reduce_group(g)
        self.assertEqual(len(r), 10)
        # endpoints preserved
        self.assertEqual(r[0], (0.0, 0.0))
        self.assertEqual(r[-1], (59.0, 0.0))

    def test_large_group_reduced_to_twelfth(self):
        g = [(float(i), 0.0) for i in range(120)]  # >=100 -> keep 120//12=10
        r = sp.reduce_group(g)
        self.assertEqual(len(r), 10)

    def test_reduce_groups_drops_noise(self):
        groups = [[(0.0, 0.0)] * 2, [(float(i), 0.0) for i in range(10)]]
        out = sp.reduce_groups(groups)
        self.assertEqual(len(out), 1)


class CenterScaleTests(unittest.TestCase):
    def test_mean_center(self):
        groups = [[(0.0, 0.0), (2.0, 2.0)]]
        centered = sp.mean_center(groups)
        # centroid (1,1) -> points become (-1,-1),(1,1)
        self.assertEqual(centered, [[(-1.0, -1.0), (1.0, 1.0)]])

    def test_scale(self):
        groups = [[(1.0, 2.0)]]
        scaled = sp.scale_groups(groups, scale=0.5)
        self.assertEqual(scaled, [[(0.5, 1.0)]])

    def test_default_scale_constant(self):
        self.assertEqual(sp.WHEEL_SCALE, 0.97)


class PipelineTests(unittest.TestCase):
    def test_full_pipeline(self):
        # Two clusters far apart; each large enough to survive.
        cluster_a = [(float(i) * 0.1, 0.0) for i in range(30)]
        cluster_b = [(100.0 + float(i) * 0.1, 0.0) for i in range(30)]
        pts = cluster_a + cluster_b
        out = sp.process_spoke_points(pts, threshold=5.0, scale=1.0)
        # two groups survive, each reduced from 30 to 5
        self.assertEqual(len(out), 2)
        for g in out:
            self.assertEqual(len(g), 5)
        # mean centered: overall centroid ~ origin
        flat = [p for g in out for p in g]
        cx = sum(p[0] for p in flat) / len(flat)
        self.assertAlmostEqual(cx, 0.0, places=6)


if __name__ == "__main__":
    unittest.main()
