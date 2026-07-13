"""Tests for PS-CAD reconstruction metrics."""

import unittest

from harnesscad.eval.bench.geometry.edge_chamfer_recon import (
    ReconstructionReport,
    chamfer_distance,
    edge_chamfer_distance,
    evaluate_reconstruction,
    hausdorff_distance,
    invalidity_ratio,
    normal_consistency,
    normalize_unit_box,
    sequence_is_valid,
)


class NormalizeTest(unittest.TestCase):
    def test_unit_box_scaling(self):
        cloud = [(0, 0, 0), (10, 0, 0), (0, 5, 0)]
        out = normalize_unit_box(cloud)
        xs = [p[0] for p in out]
        self.assertAlmostEqual(max(xs) - min(xs), 1.0)  # largest extent -> 1

    def test_empty(self):
        self.assertEqual(normalize_unit_box([]), [])

    def test_single_point(self):
        self.assertEqual(normalize_unit_box([(3, 3, 3)]), [(0.0, 0.0, 0.0)])


class ChamferHausdorffTest(unittest.TestCase):
    def test_identical_is_zero(self):
        cloud = [(0, 0, 0), (1, 0, 0), (0, 1, 0)]
        self.assertAlmostEqual(chamfer_distance(cloud, cloud), 0.0)
        self.assertAlmostEqual(hausdorff_distance(cloud, cloud), 0.0)

    def test_hd_ge_cd(self):
        a = [(0, 0, 0), (1, 0, 0), (2, 0, 0)]
        b = [(0, 0, 0), (1, 0, 0), (9, 0, 0)]
        cd = chamfer_distance(a, b)
        hd = hausdorff_distance(a, b)
        self.assertGreaterEqual(hd, cd)

    def test_empty_returns_none(self):
        self.assertIsNone(chamfer_distance([], [(0, 0, 0)]))
        self.assertIsNone(hausdorff_distance([(0, 0, 0)], []))

    def test_without_normalize(self):
        a = [(0, 0, 0)]
        b = [(3, 4, 0)]
        self.assertAlmostEqual(chamfer_distance(a, b, normalize=False), 5.0)


class EdgeChamferTest(unittest.TestCase):
    def test_edge_chamfer_matches_chamfer(self):
        ea = [(0, 0, 0), (1, 1, 1)]
        eb = [(0, 0, 0), (1, 1, 1)]
        self.assertAlmostEqual(edge_chamfer_distance(ea, eb), 0.0)


class NormalConsistencyTest(unittest.TestCase):
    def test_aligned_normals(self):
        pts = [(0, 0, 0), (1, 0, 0)]
        normals = [(0, 0, 1), (0, 0, 1)]
        nc = normal_consistency(pts, normals, pts, normals)
        self.assertAlmostEqual(nc, 1.0)

    def test_flipped_normal_is_absolute(self):
        pts = [(0, 0, 0)]
        nc = normal_consistency(pts, [(0, 0, 1)], pts, [(0, 0, -1)])
        self.assertAlmostEqual(nc, 1.0)

    def test_orthogonal_normals(self):
        pts = [(0, 0, 0)]
        nc = normal_consistency(pts, [(1, 0, 0)], pts, [(0, 1, 0)])
        self.assertAlmostEqual(nc, 0.0)

    def test_length_mismatch_rejected(self):
        with self.assertRaises(ValueError):
            normal_consistency([(0, 0, 0)], [(0, 0, 1), (0, 0, 1)],
                               [(0, 0, 0)], [(0, 0, 1)])


class InvalidityRatioTest(unittest.TestCase):
    def test_sequence_validity(self):
        self.assertTrue(sequence_is_valid([False, True, False]))
        self.assertFalse(sequence_is_valid([False, False]))
        self.assertFalse(sequence_is_valid([]))

    def test_ratio(self):
        seqs = [[True], [False, False], [False, True], [False]]
        # invalid: index 1 and 3 -> 2/4
        self.assertAlmostEqual(invalidity_ratio(seqs), 0.5)

    def test_empty_ratio_zero(self):
        self.assertEqual(invalidity_ratio([]), 0.0)


class EvaluateReconstructionTest(unittest.TestCase):
    def test_full_report(self):
        cloud = [(0, 0, 0), (1, 0, 0), (0, 1, 0)]
        normals = [(0, 0, 1)] * 3
        report = evaluate_reconstruction(
            cloud, cloud,
            target_edges=cloud, prediction_edges=cloud,
            target_normals=normals, prediction_normals=normals,
            sequences=[[True], [False]])
        self.assertIsInstance(report, ReconstructionReport)
        self.assertAlmostEqual(report.cd, 0.0)
        self.assertAlmostEqual(report.hd, 0.0)
        self.assertAlmostEqual(report.ecd, 0.0)
        self.assertAlmostEqual(report.nc, 1.0)
        self.assertAlmostEqual(report.ir, 0.5)

    def test_partial_report_leaves_none(self):
        cloud = [(0, 0, 0), (1, 0, 0)]
        report = evaluate_reconstruction(cloud, cloud)
        self.assertAlmostEqual(report.cd, 0.0)
        self.assertIsNone(report.ecd)
        self.assertIsNone(report.nc)
        self.assertEqual(report.ir, 0.0)


if __name__ == "__main__":
    unittest.main()
