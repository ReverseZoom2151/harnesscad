"""Tests for bench.gencad_retrieval (image-based CAD retrieval accuracy R_B)."""

import random
import unittest

from bench.gencad_retrieval import (
    batch_retrieval_hit,
    retrieval_accuracy,
    random_guess_accuracy,
    retrieval_curve,
)


class HitTest(unittest.TestCase):
    def test_perfect_alignment_hit(self):
        # image latent == its cad latent -> always retrieves itself.
        img = [[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]
        cad = [[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]
        for q in range(3):
            self.assertTrue(batch_retrieval_hit(img, cad, q))

    def test_miss(self):
        # query image aligns with a different cad latent.
        img = [[1.0, 0.0], [0.0, 1.0]]
        cad = [[0.0, 1.0], [1.0, 0.0]]  # swapped
        self.assertFalse(batch_retrieval_hit(img, cad, 0))

    def test_length_mismatch(self):
        with self.assertRaises(ValueError):
            batch_retrieval_hit([[1.0]], [[1.0], [2.0]], 0)


class AccuracyTest(unittest.TestCase):
    def test_perfect_pairs_accuracy_one(self):
        pool = [[float(i), float(-i), 1.0] for i in range(1, 21)]
        mean, std = retrieval_accuracy(pool, pool, batch_size=5, repeats=50, seed=1)
        self.assertEqual(mean, 1.0)
        self.assertEqual(std, 0.0)

    def test_random_pairs_near_floor(self):
        rng = random.Random(0)
        n, dim = 60, 8
        img = [[rng.gauss(0, 1) for _ in range(dim)] for _ in range(n)]
        cad = [[rng.gauss(0, 1) for _ in range(dim)] for _ in range(n)]
        # Unrelated latents: accuracy should sit near the 1/batch floor.
        mean, _ = retrieval_accuracy(img, cad, batch_size=10, repeats=400, seed=2)
        self.assertLess(mean, 0.35)  # well below aligned; near ~0.1 floor

    def test_deterministic_seed(self):
        rng = random.Random(5)
        img = [[rng.gauss(0, 1) for _ in range(4)] for _ in range(40)]
        cad = [[rng.gauss(0, 1) for _ in range(4)] for _ in range(40)]
        r1 = retrieval_accuracy(img, cad, 8, 30, seed=9)
        r2 = retrieval_accuracy(img, cad, 8, 30, seed=9)
        self.assertEqual(r1, r2)

    def test_batch_size_bounds(self):
        pool = [[1.0], [2.0]]
        with self.assertRaises(ValueError):
            retrieval_accuracy(pool, pool, batch_size=1, repeats=5)
        with self.assertRaises(ValueError):
            retrieval_accuracy(pool, pool, batch_size=3, repeats=5)


class GuessTest(unittest.TestCase):
    def test_floor(self):
        self.assertAlmostEqual(random_guess_accuracy(10), 0.1)
        self.assertAlmostEqual(random_guess_accuracy(2048), 1.0 / 2048)

    def test_invalid(self):
        with self.assertRaises(ValueError):
            random_guess_accuracy(0)


class CurveTest(unittest.TestCase):
    def test_curve_shape_and_lift(self):
        pool = [[float(i), float(i * i % 7), 1.0] for i in range(1, 41)]
        rows = retrieval_curve(pool, pool, [4, 8, 16], repeats=20, seed=0)
        self.assertEqual([r["batch_size"] for r in rows], [4, 8, 16])
        for r in rows:
            self.assertEqual(r["mean"], 1.0)  # perfect alignment
            self.assertAlmostEqual(r["random_guess"], 1.0 / r["batch_size"])
            # lift = mean / floor = batch_size for perfect retrieval.
            self.assertAlmostEqual(r["lift"], r["batch_size"], places=6)


if __name__ == "__main__":
    unittest.main()
