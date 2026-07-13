"""Tests for PS-CAD residual-region (p_ref) computation."""

import unittest

from harnesscad.domain.reconstruction.fitting.pscad_residual_regions import (
    ResidualRegion,
    cluster_residual_regions,
    compute_pref,
    distinct_mask,
    highest_residual_region,
)


class DistinctMaskTest(unittest.TestCase):
    def test_empty_reference_marks_everything_distinct(self):
        src = [(0.0, 0.0, 0.0), (1.0, 0.0, 0.0)]
        self.assertEqual(distinct_mask(src, [], threshold=0.1), (1, 1))

    def test_covered_points_are_not_distinct(self):
        src = [(0.0, 0.0, 0.0), (5.0, 0.0, 0.0)]
        ref = [(0.02, 0.0, 0.0)]  # covers first point only
        self.assertEqual(distinct_mask(src, ref, threshold=0.1), (0, 1))

    def test_negative_threshold_rejected(self):
        with self.assertRaises(ValueError):
            distinct_mask([(0, 0, 0)], [], threshold=-1.0)


class ComputePrefTest(unittest.TestCase):
    def test_first_iteration_all_missing(self):
        p_full = [(0.0, 0.0, 0.0), (1.0, 1.0, 1.0)]
        g = compute_pref(p_full, [], threshold=0.1)
        self.assertEqual(g.full_mask, (1, 1))
        self.assertEqual(g.prev_mask, ())
        self.assertEqual(len(g.missing), 2)
        self.assertEqual(g.auxiliary, ())
        self.assertEqual(g.missing_ratio, 1.0)

    def test_pref_is_concatenation_of_both_regions(self):
        # target = base + armrest; prev = base + a stray auxiliary point.
        p_full = [(0.0, 0.0, 0.0), (0.0, 0.0, 2.0)]     # base + armrest (top)
        p_prev = [(0.01, 0.0, 0.0), (9.0, 9.0, 9.0)]     # base + auxiliary
        g = compute_pref(p_full, p_prev, threshold=0.1)
        # armrest is missing from prev; base is covered.
        self.assertEqual(g.full_mask, (0, 1))
        # auxiliary stray point is distinct in prev; base covered.
        self.assertEqual(g.prev_mask, (0, 1))
        self.assertEqual(g.missing, ((0.0, 0.0, 2.0),))
        self.assertEqual(g.auxiliary, ((9.0, 9.0, 9.0),))
        self.assertEqual(g.p_ref, ((0.0, 0.0, 2.0), (9.0, 9.0, 9.0)))
        self.assertAlmostEqual(g.missing_ratio, 0.5)

    def test_full_coverage_no_residual(self):
        cloud = [(0.0, 0.0, 0.0), (1.0, 0.0, 0.0)]
        g = compute_pref(cloud, cloud, threshold=0.1)
        self.assertEqual(g.p_ref, ())
        self.assertEqual(g.missing_ratio, 0.0)


class ClusterResidualRegionsTest(unittest.TestCase):
    def test_two_separated_clusters(self):
        pts = [(0.0, 0.0, 0.0), (0.1, 0.0, 0.0), (10.0, 0.0, 0.0)]
        regions = cluster_residual_regions(pts, radius=0.5)
        self.assertEqual(len(regions), 2)
        # largest region first
        self.assertEqual(regions[0].size, 2)
        self.assertEqual(regions[1].size, 1)

    def test_deterministic_order(self):
        pts = [(10.0, 0.0, 0.0), (0.0, 0.0, 0.0), (0.05, 0.0, 0.0)]
        a = cluster_residual_regions(pts, radius=0.5)
        b = cluster_residual_regions(list(reversed(pts)), radius=0.5)
        self.assertEqual([r.centroid for r in a], [r.centroid for r in b])

    def test_bounding_box_and_centroid(self):
        region = ResidualRegion(((0.0, 0.0, 0.0), (2.0, 0.0, 0.0)), 2)
        self.assertEqual(region.centroid, (1.0, 0.0, 0.0))
        self.assertEqual(region.bounding_box(), ((0.0, 0.0, 0.0), (2.0, 0.0, 0.0)))

    def test_bad_radius_rejected(self):
        with self.assertRaises(ValueError):
            cluster_residual_regions([(0, 0, 0)], radius=0.0)


class HighestResidualRegionTest(unittest.TestCase):
    def test_picks_largest_missing_cluster(self):
        p_full = [(0.0, 0.0, 0.0), (0.1, 0.0, 0.0), (0.2, 0.0, 0.0), (9.0, 9.0, 9.0)]
        g = compute_pref(p_full, [], threshold=0.05)
        region = highest_residual_region(g, radius=0.5, side="missing")
        self.assertEqual(region.size, 3)

    def test_none_when_no_residual(self):
        cloud = [(0.0, 0.0, 0.0)]
        g = compute_pref(cloud, cloud, threshold=0.1)
        self.assertIsNone(highest_residual_region(g, radius=0.5))

    def test_invalid_side_rejected(self):
        g = compute_pref([(0, 0, 0)], [], threshold=0.1)
        with self.assertRaises(ValueError):
            highest_residual_region(g, radius=0.5, side="nope")


if __name__ == "__main__":
    unittest.main()
