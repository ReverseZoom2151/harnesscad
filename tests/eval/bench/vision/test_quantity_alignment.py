"""Tests for eval.bench.vision.quantity_alignment."""

import unittest

from harnesscad.eval.bench.vision.quantity_alignment import (
    counting_errors,
    multi_level_alignment,
    quantity_bin,
)


class ErrorsTest(unittest.TestCase):
    def test_perfect(self):
        e = counting_errors([3, 5, 10], [3, 5, 10])
        self.assertEqual(e["mae"], 0.0)
        self.assertEqual(e["rmse"], 0.0)

    def test_known_values(self):
        e = counting_errors([2, 4], [4, 4])  # errors 2,0
        self.assertAlmostEqual(e["mae"], 1.0)
        self.assertAlmostEqual(e["rmse"], (4 / 2) ** 0.5)

    def test_mismatch(self):
        with self.assertRaises(ValueError):
            counting_errors([1], [1, 2])


class BinTest(unittest.TestCase):
    def test_bins(self):
        self.assertEqual(quantity_bin(1), "few")
        self.assertEqual(quantity_bin(5), "several")
        self.assertEqual(quantity_bin(30), "many")
        self.assertEqual(quantity_bin(1000), "many")

    def test_negative(self):
        with self.assertRaises(ValueError):
            quantity_bin(-1)


class AlignmentTest(unittest.TestCase):
    def test_perfect_alignment(self):
        out = multi_level_alignment([3, 8, 40], [3, 8, 40])
        self.assertAlmostEqual(out["score"], 1.0)
        self.assertAlmostEqual(out["bin_level"], 1.0)

    def test_bin_mismatch_lowers_score(self):
        # predicted 1 (few) vs gt 40 (many): bin miss + large rel error.
        out = multi_level_alignment([1], [40])
        self.assertLess(out["score"], 0.5)

    def test_weighting(self):
        out = multi_level_alignment([3], [3], w_exact=1.0)
        self.assertAlmostEqual(out["score"], out["exact_level"])


if __name__ == "__main__":
    unittest.main()
