"""Tests for reconstruction.graphbrep_efficiency."""

import unittest

from reconstruction import graphbrep_efficiency as ge
from reconstruction import graphbrep_surface_graph as gsg


class SequenceLengthTest(unittest.TestCase):
    def test_deepcad_training_numbers(self):
        # BrepGen DeepCAD: 30 faces * 20 edges/face = 600 (paper Sec. 4.2.3).
        self.assertEqual(ge.tree_sequence_length(30, 20), 600)
        # GraphBrep DeepCAD edge sequence length = 120.
        self.assertEqual(ge.graph_sequence_length(120), 120)

    def test_deepcad_inference_numbers(self):
        # BrepGen DeepCAD inference: 30 * 30 = 900.
        self.assertEqual(ge.tree_sequence_length(30, 30), 900)

    def test_abc_training_numbers(self):
        # BrepGen ABC training: 50 * 30 = 1500.
        self.assertEqual(ge.tree_sequence_length(50, 30), 1500)

    def test_negative_rejected(self):
        with self.assertRaises(ValueError):
            ge.tree_sequence_length(-1, 20)
        with self.assertRaises(ValueError):
            ge.graph_sequence_length(-1)


class ReductionTest(unittest.TestCase):
    def test_sequence_reduction(self):
        # 600 -> 120 is an 80% reduction.
        self.assertAlmostEqual(ge.sequence_reduction(600, 120), 0.8)

    def test_attention_reduction_is_quadratic(self):
        # attention 600^2 -> 120^2, reduction 1 - (120/600)^2 = 0.96.
        self.assertAlmostEqual(ge.attention_reduction(600, 120), 0.96)

    def test_attention_cost(self):
        self.assertEqual(ge.attention_cost(120), 14400)

    def test_redundancy_ratio(self):
        self.assertAlmostEqual(ge.redundancy_ratio(600, 120), 5.0)

    def test_redundancy_ratio_zero(self):
        with self.assertRaises(ValueError):
            ge.redundancy_ratio(600, 0)

    def test_reduction_zero_baseline(self):
        self.assertEqual(ge.sequence_reduction(0, 0), 0.0)


class CompareTest(unittest.TestCase):
    def test_report_fields(self):
        report = ge.compare(30, 20, 120)
        self.assertEqual(report.tree_length, 600)
        self.assertEqual(report.graph_length, 120)
        self.assertEqual(report.tree_attention, 360000)
        self.assertEqual(report.graph_attention, 14400)
        self.assertAlmostEqual(report.sequence_reduction, 0.8)
        self.assertAlmostEqual(report.attention_reduction, 0.96)
        self.assertAlmostEqual(report.redundancy_ratio, 5.0)

    def test_compare_model(self):
        # A model with 3 surfaces, 3 real edges; padded baseline 3 * 20 = 60.
        A = gsg.build_from_edge_faces([(0, 1), (1, 2), (0, 2)], 3)
        report = ge.compare_model(A, max_edges_per_face=20)
        self.assertEqual(report.tree_length, 60)
        self.assertEqual(report.graph_length, 3)
        self.assertAlmostEqual(report.sequence_reduction, 1.0 - 3 / 60)
        self.assertAlmostEqual(report.redundancy_ratio, 20.0)


if __name__ == "__main__":
    unittest.main()
