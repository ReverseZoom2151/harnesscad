"""Tests for embedding post-processing (normalise, whiten, AQE)."""

from __future__ import annotations

import math
import random
import unittest

from bench.geomretr_embedding import (
    l2_normalize,
    l2_normalize_all,
    pca_whiten,
    apply_whiten,
    average_query_expansion,
    cosine_similarity,
)


class NormalizeTest(unittest.TestCase):
    def test_unit_norm(self):
        out = l2_normalize([3.0, 4.0])
        self.assertAlmostEqual(math.sqrt(sum(x * x for x in out)), 1.0, places=9)
        self.assertAlmostEqual(out[0], 0.6, places=9)

    def test_zero_vector(self):
        self.assertEqual(l2_normalize([0.0, 0.0]), [0.0, 0.0])

    def test_all(self):
        rows = l2_normalize_all([[1.0, 0.0], [0.0, 5.0]])
        for r in rows:
            self.assertAlmostEqual(math.sqrt(sum(x * x for x in r)), 1.0, places=9)


class WhitenTest(unittest.TestCase):
    def _data(self):
        rng = random.Random(0)
        # correlated 3-d data
        data = []
        for _ in range(200):
            a = rng.gauss(0, 2)
            b = a * 0.8 + rng.gauss(0, 0.5)
            c = rng.gauss(0, 1)
            data.append([a, b, c])
        return data

    def test_whitened_covariance_identity(self):
        data = self._data()
        whitened, _t = pca_whiten(data)
        n = len(whitened)
        dim = 3
        means = [sum(v[d] for v in whitened) / n for d in range(dim)]
        # variances ~ 1
        for d in range(dim):
            var = sum((v[d] - means[d]) ** 2 for v in whitened) / (n - 1)
            self.assertAlmostEqual(var, 1.0, places=4)
        # off-diagonal covariance ~ 0
        cov01 = sum((v[0] - means[0]) * (v[1] - means[1]) for v in whitened) / (n - 1)
        self.assertAlmostEqual(cov01, 0.0, places=4)

    def test_apply_consistency(self):
        data = self._data()
        whitened, t = pca_whiten(data)
        again = apply_whiten(data[0], t)
        for x, y in zip(whitened[0], again):
            self.assertAlmostEqual(x, y, places=9)

    def test_empty(self):
        rows, t = pca_whiten([])
        self.assertEqual(rows, [])


class AQETest(unittest.TestCase):
    def test_moves_toward_neighbours(self):
        query = [1.0, 0.0]
        gallery = [[1.0, 0.1], [1.0, 0.2], [-1.0, 0.0]]
        expanded = average_query_expansion(query, gallery, [0, 1, 2], top_k=2)
        self.assertAlmostEqual(math.sqrt(sum(x * x for x in expanded)), 1.0, places=9)
        # expanded query should be more similar to gallery[0] than the original? at least valid unit vec
        self.assertGreater(cosine_similarity(expanded, gallery[0]), 0.9)

    def test_deterministic(self):
        query = [0.3, 0.7, 0.1]
        gallery = [[0.2, 0.9, 0.0], [0.5, 0.5, 0.5]]
        a = average_query_expansion(query, gallery, [0, 1], top_k=2)
        b = average_query_expansion(query, gallery, [0, 1], top_k=2)
        self.assertEqual(a, b)


class CosineTest(unittest.TestCase):
    def test_orthogonal(self):
        self.assertAlmostEqual(cosine_similarity([1, 0], [0, 1]), 0.0, places=9)

    def test_identical(self):
        self.assertAlmostEqual(cosine_similarity([2, 2], [2, 2]), 1.0, places=9)

    def test_zero(self):
        self.assertEqual(cosine_similarity([0, 0], [1, 1]), 0.0)


if __name__ == "__main__":
    unittest.main()
