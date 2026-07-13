"""Tests for EvoCAD Euler-characteristic topology metrics."""
import unittest

from harnesscad.eval.bench.evocad_topology_metrics import (
    euler_characteristic,
    genus_from_euler,
    topology_error,
    topology_correctness,
    watertight_subset,
    topology_dataset_report,
    TopologyReport,
)


def _tetrahedron():
    # 4 vertices, 4 triangular faces -> chi = 2 (sphere topology, genus 0)
    verts = [(0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, 1)]
    faces = [(0, 1, 2), (0, 1, 3), (0, 2, 3), (1, 2, 3)]
    return verts, faces


class TestEuler(unittest.TestCase):
    def test_tetrahedron_chi_is_two(self):
        v, f = _tetrahedron()
        self.assertEqual(euler_characteristic(v, f), 2)

    def test_genus_from_euler(self):
        self.assertEqual(genus_from_euler(2), 0.0)   # sphere
        self.assertEqual(genus_from_euler(0), 1.0)   # torus (one hole)
        self.assertEqual(genus_from_euler(-2), 2.0)  # two holes


class TestTopologyErrorCorrect(unittest.TestCase):
    def test_topology_error(self):
        self.assertEqual(topology_error(2, -2), 4)
        self.assertEqual(topology_error(0, 0), 0)

    def test_topology_error_invalid(self):
        self.assertIsNone(topology_error(None, 2))
        self.assertIsNone(topology_error(2, None))

    def test_topology_correctness(self):
        self.assertEqual(topology_correctness(-2, -2), 1)
        self.assertEqual(topology_correctness(-2, 2), 0)

    def test_topology_correctness_invalid(self):
        self.assertIsNone(topology_correctness(None, None))


class TestDatasetAggregation(unittest.TestCase):
    def test_watertight_subset_filters_none(self):
        pairs = [(2, 2), (0, None), (None, 0), (-2, -4)]
        self.assertEqual(watertight_subset(pairs), [(2, 2), (-2, -4)])

    def test_dataset_report(self):
        # matches the paper's qualitative table style (chi of gt vs pred).
        pairs = [(-2, -2), (-4, -4), (0, 0), (2, -2), (2, 2), (None, 2)]
        rep = topology_dataset_report(pairs)
        self.assertIsInstance(rep, TopologyReport)
        self.assertEqual(rep.total_samples, 6)
        self.assertEqual(rep.watertight_samples, 5)
        self.assertAlmostEqual(rep.coverage, 5 / 6)
        # 4 of 5 match -> 80%
        self.assertAlmostEqual(rep.topology_correctness_pct, 80.0)
        # T_err: 0,0,0,4,0 -> mean 0.8
        self.assertAlmostEqual(rep.mean_topology_error, 0.8)

    def test_dataset_report_empty_subset(self):
        rep = topology_dataset_report([(None, 1), (2, None)])
        self.assertEqual(rep.total_samples, 2)
        self.assertEqual(rep.watertight_samples, 0)
        self.assertIsNone(rep.topology_correctness_pct)
        self.assertIsNone(rep.mean_topology_error)
        self.assertEqual(rep.coverage, 0.0)

    def test_dataset_report_all_correct(self):
        rep = topology_dataset_report([(2, 2), (0, 0)])
        self.assertEqual(rep.topology_correctness_pct, 100.0)
        self.assertEqual(rep.mean_topology_error, 0.0)


if __name__ == "__main__":
    unittest.main()
