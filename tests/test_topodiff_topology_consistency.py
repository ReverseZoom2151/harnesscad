"""Tests for topology-consistency metrics."""
import math
import unittest

from harnesscad.eval.bench.topodiff_topology_consistency import (
    betti_match,
    betti_match_vectors,
    betti_vector_distance,
    cavity_match,
    collapse_to_diagonal,
    component_match,
    genus_match,
    implies_genus_zero,
    persistence_diagram_distance,
    significant_features,
    topology_consistency,
    topology_consistency_report,
)


def _solid_box(nx, ny, nz):
    return {(x, y, z) for x in range(nx) for y in range(ny) for z in range(nz)}


def _ring():
    return {(x, y, 0) for x in range(3) for y in range(3) if (x, y) != (1, 1)}


class TestBettiConsistency(unittest.TestCase):
    def test_vector_distance(self):
        self.assertEqual(betti_vector_distance((1, 0, 0), (1, 0, 0)), 0)
        self.assertEqual(betti_vector_distance((1, 0, 0), (1, 1, 0)), 1)
        self.assertEqual(betti_vector_distance((2, 1, 0), (1, 0, 1)), 3)

    def test_matching_shapes(self):
        a = _solid_box(2, 2, 2)
        b = _solid_box(3, 3, 3)
        self.assertEqual(betti_match(a, b), 1)  # both solid balls (1,0,0)
        self.assertEqual(genus_match(a, b), 1)
        self.assertEqual(component_match(a, b), 1)
        self.assertEqual(cavity_match(a, b), 1)

    def test_mismatched_topology(self):
        ball = _solid_box(3, 3, 3)
        ring = _ring()
        self.assertEqual(betti_match(ball, ring), 0)
        self.assertEqual(genus_match(ball, ring), 0)  # 0 vs 1

    def test_match_vectors_helper(self):
        self.assertEqual(betti_match_vectors((1, 1, 0), (1, 1, 0)), 1)
        self.assertEqual(betti_match_vectors((1, 1, 0), (1, 0, 0)), 0)

    def test_consistency_result(self):
        r = topology_consistency(_ring(), _solid_box(2, 2, 2))
        self.assertEqual(r.generated, (1, 1, 0))
        self.assertEqual(r.target, (1, 0, 0))
        self.assertEqual(r.betti_match, 0)
        self.assertEqual(r.genus_error, 1)
        self.assertEqual(r.betti_l1, 1)

    def test_perfect_consistency(self):
        r = topology_consistency(_ring(), _ring())
        self.assertEqual(r.betti_match, 1)
        self.assertEqual(r.genus_error, 0)
        self.assertEqual(r.betti_l1, 0)


class TestPersistenceDistance(unittest.TestCase):
    def test_identical_diagrams_zero(self):
        d = [(0.0, 5.0), (1.0, 2.0)]
        self.assertEqual(persistence_diagram_distance(d, d), 0.0)

    def test_empty_diagrams(self):
        self.assertEqual(persistence_diagram_distance([], []), 0.0)

    def test_essential_ignored(self):
        a = [(0.0, math.inf)]
        b = [(0.0, math.inf)]
        self.assertEqual(persistence_diagram_distance(a, b), 0.0)

    def test_extra_point_costs_diagonal(self):
        # b has an extra point (0,4); distance = its diagonal distance = 2.
        a = []
        b = [(0.0, 4.0)]
        self.assertEqual(persistence_diagram_distance(a, b), 2.0)

    def test_symmetric(self):
        a = [(0.0, 5.0)]
        b = [(0.0, 6.0)]
        self.assertEqual(
            persistence_diagram_distance(a, b),
            persistence_diagram_distance(b, a),
        )

    def test_matching_cheaper_than_diagonal(self):
        # Two close points should match (cost 1) not both go to diagonal.
        a = [(0.0, 10.0)]
        b = [(0.0, 11.0)]
        self.assertEqual(persistence_diagram_distance(a, b), 1.0)


class TestDiagonalRule(unittest.TestCase):
    def test_genus_zero_when_on_diagonal(self):
        self.assertTrue(implies_genus_zero([(3.0, 3.0), (5.0, 5.0)]))
        self.assertTrue(implies_genus_zero([]))

    def test_not_genus_zero_with_persistent_loop(self):
        self.assertFalse(implies_genus_zero([(0.0, 4.0)]))

    def test_essential_ignored_in_rule(self):
        self.assertTrue(implies_genus_zero([(2.0, math.inf)]))

    def test_collapse_produces_genus_zero(self):
        diagram = [(0.0, 4.0), (1.0, 7.0)]
        collapsed = collapse_to_diagonal(diagram)
        self.assertTrue(implies_genus_zero(collapsed))

    def test_collapse_keeps_essential(self):
        collapsed = collapse_to_diagonal([(2.0, math.inf)])
        self.assertTrue(math.isinf(collapsed[0][1]))

    def test_significant_features(self):
        diagram = [(0.0, 4.0), (1.0, 1.2), (2.0, math.inf)]
        self.assertEqual(significant_features(diagram, min_persistence=1.0), 1)
        self.assertEqual(significant_features(diagram, min_persistence=0.1), 2)


class TestReport(unittest.TestCase):
    def test_empty(self):
        rep = topology_consistency_report([])
        self.assertEqual(rep.samples, 0)
        self.assertIsNone(rep.betti_match_pct)

    def test_aggregate(self):
        pairs = [
            (_ring(), _ring()),            # match
            (_solid_box(3, 3, 3), _ring()),  # genus error 1, l1 1
        ]
        rep = topology_consistency_report(pairs)
        self.assertEqual(rep.samples, 2)
        self.assertEqual(rep.betti_match_pct, 50.0)
        self.assertEqual(rep.mean_genus_error, 0.5)
        self.assertEqual(rep.mean_betti_l1, 0.5)


if __name__ == "__main__":
    unittest.main()
