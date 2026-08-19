"""Hand-computed checks for the RLCAD set-level COV / MMD / JSD module.

Point clouds of a single point are used wherever a shape-to-shape distance has
to be predictable: the mean-form symmetric Chamfer between two one-point clouds
is exactly the Euclidean distance between those points, so every expected number
below can be read off by hand.
"""

from __future__ import annotations

import unittest

from harnesscad.eval.bench.generative import brep_set_metrics as brep
from harnesscad.eval.bench.generative import rlcad_set_metrics as rl


def P(x, y=0.0, z=0.0):
    """A one-point cloud at ``(x, y, z)``."""
    return [(float(x), float(y), float(z))]


#: reference {0, 10} against generated {1, 2}: the shape at 10 is not covered.
REF_SPLIT = [P(0.0), P(10.0)]
GEN_CLUMP = [P(1.0), P(2.0)]


class DistanceTest(unittest.TestCase):
    def test_chamfer_of_two_single_point_clouds_is_the_euclidean_distance(self):
        self.assertAlmostEqual(rl.chamfer_distance(P(0.0), P(3.0)), 3.0, places=12)

    def test_emd_of_two_single_point_clouds_is_the_euclidean_distance(self):
        self.assertAlmostEqual(rl.emd_distance(P(0.0), P(3.0)), 3.0, places=12)

    def test_emd_refuses_unequal_cardinality_rather_than_resampling(self):
        with self.assertRaises(ValueError):
            rl.emd_distance([(0.0, 0.0, 0.0)], [(0.0, 0.0, 0.0), (1.0, 0.0, 0.0)])

    def test_chamfer_of_an_empty_cloud_is_refused(self):
        with self.assertRaises(ValueError):
            rl.chamfer_distance([], P(1.0))


class CoverageTest(unittest.TestCase):
    def test_generated_equals_reference_gives_full_coverage(self):
        shapes = [P(0.0), P(1.0), P(2.0)]
        self.assertEqual(rl.coverage(shapes, shapes, rl.chamfer_distance), 1.0)

    def test_coverage_counts_distinct_reference_shapes_matched(self):
        # both generated shapes are nearest to the reference at 0, so only one
        # of the two reference shapes is covered.
        self.assertEqual(
            rl.coverage(GEN_CLUMP, REF_SPLIT, rl.chamfer_distance), 0.5)

    def test_coverage_saturates_at_the_smaller_set(self):
        self.assertEqual(
            rl.coverage([P(0.0)], [P(0.0), P(1.0)], rl.chamfer_distance), 0.5)

    def test_an_empty_set_is_refused(self):
        with self.assertRaises(ValueError):
            rl.coverage([], REF_SPLIT, rl.chamfer_distance)


class MmdTest(unittest.TestCase):
    def test_mmd_is_zero_when_the_sets_coincide(self):
        shapes = [P(0.0), P(1.0), P(2.0)]
        self.assertEqual(rl.mmd(shapes, shapes, rl.chamfer_distance), 0.0)

    def test_mmd_averages_over_the_reference_set(self):
        # ref 0 -> nearest generated is 1 (distance 1); ref 10 -> 2 (distance 8).
        self.assertAlmostEqual(
            rl.mmd(GEN_CLUMP, REF_SPLIT, rl.chamfer_distance), 4.5, places=12)

    def test_mmd_emd_matches_mmd_cd_on_single_point_clouds(self):
        self.assertAlmostEqual(rl.mmd(GEN_CLUMP, REF_SPLIT, rl.emd_distance),
                               4.5, places=12)


class JsdTest(unittest.TestCase):
    def test_identical_sets_have_zero_divergence(self):
        clouds = [[(0.1, 0.2, 0.3), (-0.4, 0.5, -0.6)], [(0.0, 0.0, 0.0)]]
        self.assertAlmostEqual(rl.voxel_jsd(clouds, clouds), 0.0, places=12)

    def test_disjoint_occupancy_saturates_at_one_bit(self):
        a = [[(-0.9, -0.9, -0.9)]]
        b = [[(0.9, 0.9, 0.9)]]
        self.assertAlmostEqual(rl.voxel_jsd(a, b), 1.0, places=12)


class SetReportTest(unittest.TestCase):
    def test_a_perfect_generator_scores_cov_one_mmd_zero_jsd_zero(self):
        shapes = [[(0.1, 0.0, 0.0)], [(-0.2, 0.0, 0.0)], [(0.5, 0.5, 0.5)]]
        report = rl.set_report(shapes, shapes)
        self.assertEqual(report["cov"], 1.0)
        self.assertEqual(report["mmd_cd"], 0.0)
        self.assertAlmostEqual(report["jsd"], 0.0, places=12)
        self.assertNotIn("mmd_emd", report)

    def test_emd_is_opt_in(self):
        shapes = [[(0.1, 0.0, 0.0)], [(-0.2, 0.0, 0.0)]]
        report = rl.set_report(shapes, shapes, with_emd=True)
        self.assertEqual(report["mmd_emd"], 0.0)


class CoverageRivalTest(unittest.TestCase):
    """The two coverage directions are different quantities, not a bug."""

    def test_the_two_directions_disagree_on_the_same_input(self):
        mine = rl.coverage(GEN_CLUMP, REF_SPLIT, rl.chamfer_distance)
        theirs = brep.coverage_mmd(GEN_CLUMP, REF_SPLIT,
                                   rl.chamfer_distance)["coverage"]
        # reference 0 -> generated 1 (index 0); reference 10 -> generated 2
        # (index 1): two distinct generated indices over two references -> 1.0.
        self.assertEqual(mine, 0.5)
        self.assertEqual(theirs, 1.0)

    def test_but_the_two_mmd_terms_are_the_same_quantity(self):
        mine = rl.mmd(GEN_CLUMP, REF_SPLIT, rl.chamfer_distance)
        theirs = brep.coverage_mmd(GEN_CLUMP, REF_SPLIT,
                                   rl.chamfer_distance)["mmd"]
        self.assertAlmostEqual(mine, theirs, places=12)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
