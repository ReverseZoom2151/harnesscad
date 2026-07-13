"""Tests for bench/contrastcad_contrastive.py — deterministic contrastive maths."""

import math
import unittest

from harnesscad.eval.bench.retrieval.nt_xent_loss import (
    contrastive_loss,
    cosine_similarity,
    dropout_view,
    make_positive_pairs,
    nt_xent_anchor_loss,
    nt_xent_loss,
    positive_index,
    similarity_matrix,
)
import random


class TestCosineSimilarity(unittest.TestCase):
    def test_identical(self):
        self.assertAlmostEqual(cosine_similarity([1, 2, 3], [1, 2, 3]), 1.0)

    def test_orthogonal(self):
        self.assertAlmostEqual(cosine_similarity([1, 0], [0, 1]), 0.0)

    def test_opposite(self):
        self.assertAlmostEqual(cosine_similarity([1, 1], [-1, -1]), -1.0)

    def test_zero_vector_raises(self):
        with self.assertRaises(ValueError):
            cosine_similarity([0, 0], [1, 1])


class TestDropoutView(unittest.TestCase):
    def test_deterministic(self):
        self.assertEqual(dropout_view([1, 2, 3, 4], 0.5, random.Random(0)),
                         dropout_view([1, 2, 3, 4], 0.5, random.Random(0)))

    def test_zero_prob_scales_by_one(self):
        self.assertEqual(dropout_view([1, 2, 3], 0.0, random.Random(0)),
                         [1.0, 2.0, 3.0])

    def test_survivors_scaled(self):
        out = dropout_view([1, 1, 1, 1, 1, 1], 0.5, random.Random(1))
        for v in out:
            self.assertIn(round(v, 6), (0.0, 2.0))

    def test_invalid_prob(self):
        with self.assertRaises(ValueError):
            dropout_view([1], 1.0, random.Random(0))


class TestPositivePairs(unittest.TestCase):
    def setUp(self):
        self.latents = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]

    def test_produces_2m_views(self):
        views = make_positive_pairs(self.latents, 0.2, 42)
        self.assertEqual(len(views), 2 * len(self.latents))

    def test_deterministic(self):
        self.assertEqual(make_positive_pairs(self.latents, 0.2, 42),
                         make_positive_pairs(self.latents, 0.2, 42))

    def test_positive_index_is_involution(self):
        m = 3
        for i in range(2 * m):
            self.assertEqual(positive_index(positive_index(i, m), m), i)


class TestNTXentLoss(unittest.TestCase):
    def setUp(self):
        # Two well-separated clusters -> low loss; identical views.
        self.views = [
            [1.0, 0.0], [0.0, 1.0],   # anchors
            [1.0, 0.0], [0.0, 1.0],   # positives (partner of each anchor)
        ]

    def test_anchor_loss_nonnegative(self):
        for i in range(len(self.views)):
            self.assertGreaterEqual(nt_xent_anchor_loss(self.views, i), 0.0)

    def test_perfect_positives_low_loss(self):
        # anchor 0's positive is view 2 (identical); negatives are orthogonal.
        loss = nt_xent_anchor_loss(self.views, 0, temperature=0.07)
        self.assertLess(loss, 1e-3)

    def test_mean_loss_matches_manual(self):
        manual = sum(nt_xent_anchor_loss(self.views, i)
                     for i in range(len(self.views))) / len(self.views)
        self.assertAlmostEqual(nt_xent_loss(self.views), manual)

    def test_odd_views_raise(self):
        with self.assertRaises(ValueError):
            nt_xent_anchor_loss([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]], 0)

    def test_temperature_must_be_positive(self):
        with self.assertRaises(ValueError):
            nt_xent_anchor_loss(self.views, 0, temperature=0.0)

    def test_closer_positive_lowers_loss(self):
        far = [[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0], [0.0, -1.0]]
        near = [[1.0, 0.0], [0.0, 1.0], [1.0, 0.0], [0.0, 1.0]]
        self.assertLess(nt_xent_loss(near), nt_xent_loss(far))


class TestContrastiveLoss(unittest.TestCase):
    def test_deterministic(self):
        latents = [[1.0, 0.2, 0.0], [0.1, 1.0, 0.3], [0.0, 0.2, 1.0]]
        self.assertAlmostEqual(contrastive_loss(latents, 0.1, 7),
                               contrastive_loss(latents, 0.1, 7))

    def test_zero_dropout_is_reproducible_and_finite(self):
        latents = [[1.0, 0.0], [0.0, 1.0]]
        loss = contrastive_loss(latents, 0.0, 3)
        self.assertTrue(math.isfinite(loss))


class TestSimilarityMatrix(unittest.TestCase):
    def test_symmetric_with_unit_diagonal(self):
        views = [[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]
        m = similarity_matrix(views)
        for i in range(3):
            self.assertAlmostEqual(m[i][i], 1.0)
            for j in range(3):
                self.assertAlmostEqual(m[i][j], m[j][i])


if __name__ == "__main__":
    unittest.main()
