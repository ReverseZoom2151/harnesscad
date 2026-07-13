"""Tests for reconstruction.gaussiancad_emd."""

from __future__ import annotations

import itertools
import unittest

from harnesscad.domain.reconstruction import gaussiancad_emd as emd


def _brute_emd(a, b):
    n = len(a)
    best = None
    for perm in itertools.permutations(range(n)):
        cost = sum(emd._euclidean(a[i], b[perm[i]]) for i in range(n))
        if best is None or cost < best:
            best = cost
    return best


class TestHungarian(unittest.TestCase):
    def test_identity_assignment(self):
        cost = [[0.0, 5.0], [5.0, 0.0]]
        self.assertEqual(emd.hungarian(cost), [0, 1])

    def test_swap_assignment(self):
        cost = [[5.0, 0.0], [0.0, 5.0]]
        self.assertEqual(emd.hungarian(cost), [1, 0])

    def test_empty(self):
        self.assertEqual(emd.hungarian([]), [])

    def test_non_square_raises(self):
        with self.assertRaises(ValueError):
            emd.hungarian([[1.0, 2.0]])

    def test_optimal_3x3(self):
        cost = [[4.0, 1.0, 3.0], [2.0, 0.0, 5.0], [3.0, 2.0, 2.0]]
        assign = emd.hungarian(cost)
        total = sum(cost[i][assign[i]] for i in range(3))
        # brute force minimum
        best = min(sum(cost[i][p[i]] for i in range(3))
                   for p in itertools.permutations(range(3)))
        self.assertAlmostEqual(total, best)
        self.assertEqual(sorted(assign), [0, 1, 2])


class TestEMD(unittest.TestCase):
    def test_identical_clouds_zero(self):
        a = [(0.0, 0.0), (1.0, 1.0), (2.0, 0.0)]
        self.assertAlmostEqual(emd.earth_movers_distance(a, a), 0.0)

    def test_matches_brute_force(self):
        a = [(0.0, 0.0), (1.0, 0.0), (0.0, 1.0), (1.0, 1.0)]
        b = [(0.1, 0.2), (1.2, 0.1), (0.0, 1.1), (0.9, 0.8)]
        self.assertAlmostEqual(emd.earth_movers_distance(a, b), _brute_emd(a, b), places=9)

    def test_translation_cost(self):
        a = [(0.0, 0.0), (0.0, 0.0)]
        b = [(3.0, 4.0), (3.0, 4.0)]
        self.assertAlmostEqual(emd.earth_movers_distance(a, b), 10.0)  # 2 * 5

    def test_mean_emd(self):
        a = [(0.0, 0.0), (0.0, 0.0)]
        b = [(3.0, 4.0), (3.0, 4.0)]
        self.assertAlmostEqual(emd.mean_emd(a, b), 5.0)

    def test_unequal_sizes_raise(self):
        with self.assertRaises(ValueError):
            emd.earth_movers_distance([(0.0, 0.0)], [(0.0, 0.0), (1.0, 1.0)])

    def test_empty_raises(self):
        with self.assertRaises(ValueError):
            emd.earth_movers_distance([], [])

    def test_symmetric(self):
        a = [(0.0, 0.0), (2.0, 0.0), (0.0, 3.0)]
        b = [(1.0, 1.0), (2.0, 2.0), (0.5, 3.5)]
        self.assertAlmostEqual(emd.earth_movers_distance(a, b),
                               emd.earth_movers_distance(b, a), places=9)


if __name__ == "__main__":
    unittest.main()
