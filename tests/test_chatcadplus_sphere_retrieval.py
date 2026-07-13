"""Tests for chatcadplus_sphere_retrieval -- sphere-projected KD-tree cosine k-NN."""

from __future__ import annotations

import math
import random
import unittest

from harnesscad.agents.rag.chatcadplus_sphere_retrieval import SphereKDTree, normalise


def _cos(a, b):
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(x * x for x in b))
    if na == 0 or nb == 0:
        return 0.0
    return sum(x * y for x, y in zip(a, b)) / (na * nb)


class TestSphereRetrieval(unittest.TestCase):
    def test_normalise_unit_length(self):
        v = normalise([3.0, 4.0])
        self.assertTrue(math.isclose(math.sqrt(sum(x * x for x in v)), 1.0))
        self.assertTrue(math.isclose(v[0], 0.6))
        self.assertTrue(math.isclose(v[1], 0.8))

    def test_normalise_zero_vector(self):
        self.assertEqual(normalise([0.0, 0.0]), (0.0, 0.0))

    def test_cosine_invariant_to_magnitude(self):
        tree = SphereKDTree([[1.0, 0.0], [0.0, 1.0]])
        res = tree.query([5.0, 0.0], k=1)
        self.assertEqual(res[0][0], 0)
        self.assertTrue(math.isclose(res[0][1], 1.0))

    def test_top_k_ordering(self):
        vecs = [[1.0, 0.0], [0.9, 0.1], [0.0, 1.0], [-1.0, 0.0]]
        tree = SphereKDTree(vecs)
        res = tree.query([1.0, 0.0], k=3)
        idxs = [i for i, _ in res]
        self.assertEqual(idxs[0], 0)
        self.assertEqual(idxs[1], 1)
        sims = [s for _, s in res]
        self.assertEqual(sims, sorted(sims, reverse=True))

    def test_matches_brute_force_random(self):
        rng = random.Random(1234)
        dim = 6
        vecs = [[rng.gauss(0, 1) for _ in range(dim)] for _ in range(200)]
        tree = SphereKDTree(vecs)
        for _ in range(50):
            q = [rng.gauss(0, 1) for _ in range(dim)]
            kd = tree.query(q, k=5)
            bf = tree.brute_force(q, k=5)
            self.assertEqual([i for i, _ in kd], [i for i, _ in bf])
            for (ik, sk), (ib, sb) in zip(kd, bf):
                self.assertTrue(math.isclose(sk, sb, abs_tol=1e-9))

    def test_cosine_value_matches_direct_computation(self):
        vecs = [[2.0, 1.0], [1.0, 3.0], [-1.0, 2.0]]
        tree = SphereKDTree(vecs)
        q = [1.5, 0.5]
        for idx, sim in tree.query(q, k=3):
            self.assertTrue(math.isclose(sim, _cos(q, vecs[idx]), abs_tol=1e-9))

    def test_stable_tie_break_by_index(self):
        vecs = [[1.0, 0.0], [1.0, 0.0], [0.0, 1.0]]
        tree = SphereKDTree(vecs)
        res = tree.query([1.0, 0.0], k=2)
        self.assertEqual([i for i, _ in res], [0, 1])

    def test_k_larger_than_n(self):
        tree = SphereKDTree([[1.0, 0.0], [0.0, 1.0]])
        self.assertEqual(len(tree.query([1.0, 0.0], k=10)), 2)

    def test_empty_tree_and_k_zero(self):
        self.assertEqual(SphereKDTree([]).query([1.0], k=3), [])
        self.assertEqual(SphereKDTree([[1.0, 0.0]]).query([1.0, 0.0], k=0), [])

    def test_dim_mismatch_raises(self):
        tree = SphereKDTree([[1.0, 0.0], [0.0, 1.0]])
        with self.assertRaises(ValueError):
            tree.query([1.0, 0.0, 0.0], k=1)

    def test_len(self):
        self.assertEqual(len(SphereKDTree([[1.0, 0.0]] * 5)), 5)


if __name__ == "__main__":
    unittest.main()
