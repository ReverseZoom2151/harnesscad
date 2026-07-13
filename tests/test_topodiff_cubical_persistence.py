"""Tests for sublevel-set persistent homology on a scalar/SDF grid."""
import math
import unittest

from harnesscad.domain.numeric.topodiff_cubical_persistence import (
    betti_curve,
    field_from_grid,
    persistence_pairs,
    persistence_points,
    persistence_values,
    top_k_persistent,
)


class TestPersistencePairs(unittest.TestCase):
    def test_empty(self):
        self.assertEqual(persistence_pairs({}), [])

    def test_single_vertex_essential_only(self):
        pairs = persistence_pairs({(0, 0, 0): 2.0})
        self.assertEqual(pairs, [(2.0, math.inf)])

    def test_two_basins_one_line(self):
        # 1D line of values: two minima (0 at ends) separated by a peak (5).
        # coords along x axis.
        field = {
            (0, 0, 0): 0.0,
            (1, 0, 0): 5.0,
            (2, 0, 0): 0.0,
        }
        pairs = persistence_pairs(field)
        finite = [p for p in pairs if not math.isinf(p[1])]
        essential = [p for p in pairs if math.isinf(p[1])]
        # One essential component (global min) + one finite class born at 0,
        # dying when the peak at 5 connects the two basins.
        self.assertEqual(len(essential), 1)
        self.assertEqual(len(finite), 1)
        self.assertEqual(finite[0], (0.0, 5.0))

    def test_no_include_essential(self):
        field = {(0, 0, 0): 1.0, (1, 0, 0): 1.0}
        pairs = persistence_pairs(field, include_essential=False)
        # Same value, single merge at birth -> zero persistence skipped.
        self.assertEqual(pairs, [])

    def test_deterministic(self):
        field = field_from_grid([[[3.0, 1.0], [2.0, 0.0]]])
        a = persistence_pairs(field)
        b = persistence_pairs(field)
        self.assertEqual(a, b)


class TestPersistencePoints(unittest.TestCase):
    def test_birth_persistence_form(self):
        pairs = [(0.0, 5.0), (1.0, 2.0), (3.0, math.inf)]
        pts = persistence_points(pairs)
        self.assertIn((0.0, 5.0), pts)
        self.assertIn((1.0, 1.0), pts)
        # Essential dropped by default.
        self.assertEqual(len(pts), 2)

    def test_keep_essential(self):
        pairs = [(3.0, math.inf)]
        pts = persistence_points(pairs, finite_only=False)
        self.assertEqual(pts[0][0], 3.0)
        self.assertTrue(math.isinf(pts[0][1]))

    def test_persistence_values(self):
        vals = persistence_values([(0.0, 4.0), (1.0, math.inf)])
        self.assertEqual(vals[0], 4.0)
        self.assertTrue(math.isinf(vals[1]))


class TestTopK(unittest.TestCase):
    def test_keeps_longest(self):
        pairs = [(0.0, 1.0), (0.0, 9.0), (0.0, 4.0)]
        top = top_k_persistent(pairs, 2, keep_essential=False)
        self.assertEqual(top, [(0.0, 9.0), (0.0, 4.0)])

    def test_essential_first(self):
        pairs = [(0.0, 1.0), (2.0, math.inf), (0.0, 8.0)]
        top = top_k_persistent(pairs, 2)
        self.assertEqual(top[0], (2.0, math.inf))
        self.assertEqual(top[1], (0.0, 8.0))

    def test_k_zero(self):
        self.assertEqual(top_k_persistent([(0.0, 1.0)], 0), [])

    def test_k_exceeds(self):
        pairs = [(0.0, 1.0)]
        self.assertEqual(len(top_k_persistent(pairs, 5, keep_essential=False)), 1)


class TestBettiCurve(unittest.TestCase):
    def test_curve_merges_to_one(self):
        field = {
            (0, 0, 0): 0.0,
            (1, 0, 0): 5.0,
            (2, 0, 0): 0.0,
        }
        curve = betti_curve(field)
        thresholds = [p.threshold for p in curve]
        betti = [p.beta0 for p in curve]
        self.assertEqual(thresholds, [0.0, 5.0])
        # Two components at threshold 0, merge to one at 5.
        self.assertEqual(betti, [2, 1])

    def test_empty(self):
        self.assertEqual(betti_curve({}), [])


if __name__ == "__main__":
    unittest.main()
