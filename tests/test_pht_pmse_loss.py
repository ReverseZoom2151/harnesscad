"""Tests for PHT-CAD P-MSE and combined CE/P-MSE loss."""

import math
import unittest

from reconstruction import pht_pmse_loss as loss


class CrossEntropyTest(unittest.TestCase):
    def test_perfect_onehot_zero(self):
        ce = loss.cross_entropy([1.0, 0.0, 0.0], [1.0, 0.0, 0.0])
        self.assertAlmostEqual(ce, 0.0, places=9)

    def test_uniform(self):
        ce = loss.cross_entropy([1.0, 0.0], [0.5, 0.5])
        self.assertAlmostEqual(ce, -math.log(0.5))

    def test_length_mismatch(self):
        with self.assertRaises(ValueError):
            loss.cross_entropy([1.0], [1.0, 0.0])

    def test_empty(self):
        with self.assertRaises(ValueError):
            loss.cross_entropy([], [])

    def test_negative_target(self):
        with self.assertRaises(ValueError):
            loss.cross_entropy([-0.1, 1.1], [0.5, 0.5])


class PMseTest(unittest.TestCase):
    def test_perfect_zero(self):
        self.assertEqual(loss.p_mse([1.0, 2.0, 3.0], [1.0, 2.0, 3.0]), 0.0)

    def test_mean_squared(self):
        # errors 2 and 0 -> (4 + 0)/2 = 2
        self.assertAlmostEqual(loss.p_mse([3.0, 5.0], [1.0, 5.0]), 2.0)

    def test_mismatch(self):
        with self.assertRaises(ValueError):
            loss.p_mse([1.0], [1.0, 2.0])

    def test_empty(self):
        with self.assertRaises(ValueError):
            loss.p_mse([], [])


class TotalLossTest(unittest.TestCase):
    def test_weighted_sum(self):
        b = loss.total_loss(2.0, 4.0, lambda_ce=0.5, lambda_p_mse=2.0)
        self.assertAlmostEqual(b.total, 0.5 * 2.0 + 2.0 * 4.0)
        self.assertEqual(b.ce, 2.0)
        self.assertEqual(b.p_mse, 4.0)

    def test_default_weights(self):
        b = loss.total_loss(1.0, 3.0)
        self.assertAlmostEqual(b.total, 4.0)

    def test_negative_weight(self):
        with self.assertRaises(ValueError):
            loss.total_loss(1.0, 1.0, lambda_ce=-1.0)


class CombinedLossTest(unittest.TestCase):
    def test_end_to_end(self):
        b = loss.combined_loss([1.0, 0.0], [1.0, 0.0],
                               [1.0, 2.0], [1.0, 4.0],
                               lambda_ce=1.0, lambda_p_mse=1.0)
        self.assertAlmostEqual(b.ce, 0.0, places=9)
        self.assertAlmostEqual(b.p_mse, 2.0)  # (0 + 4)/2
        self.assertAlmostEqual(b.total, 2.0)


if __name__ == "__main__":
    unittest.main()
