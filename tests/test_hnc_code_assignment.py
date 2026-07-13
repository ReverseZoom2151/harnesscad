"""Tests for HNC-CAD nearest-codebook assignment (reconstruction.hnc_code_assignment)."""

import unittest

from harnesscad.domain.reconstruction.tokens.hnc_codebooks import (
    REINIT_THRESHOLD,
    Codebook,
    SPLCodebooks,
    active_code_fraction,
    average_pool,
    codebook_perplexity,
    compression_ratio,
    underutilized_codes,
    utilization,
)


class TestAveragePool(unittest.TestCase):
    def test_mean(self):
        self.assertEqual(average_pool([(1.0, 2.0), (3.0, 4.0)]), (2.0, 3.0))

    def test_empty_raises(self):
        with self.assertRaises(ValueError):
            average_pool([])

    def test_dim_mismatch(self):
        with self.assertRaises(ValueError):
            average_pool([(1.0,), (1.0, 2.0)])


class TestCodebook(unittest.TestCase):
    def setUp(self):
        self.book = Codebook(((0.0, 0.0), (1.0, 1.0), (5.0, 5.0)))

    def test_nearest_assignment(self):
        self.assertEqual(self.book.assign((0.1, 0.1)), 0)
        self.assertEqual(self.book.assign((0.9, 1.1)), 1)
        self.assertEqual(self.book.assign((4.6, 5.2)), 2)

    def test_tie_break_lowest_index(self):
        book = Codebook(((0.0, 0.0), (2.0, 0.0)))
        # (1.0, 0.0) is equidistant -> lowest index
        self.assertEqual(book.assign((1.0, 0.0)), 0)

    def test_quantize_returns_code_vector(self):
        self.assertEqual(self.book.quantize((0.9, 1.1)), (1.0, 1.0))

    def test_batch(self):
        self.assertEqual(self.book.assign_batch([(0.0, 0.0), (5.0, 5.0)]), (0, 2))

    def test_dim_check(self):
        with self.assertRaises(ValueError):
            self.book.assign((1.0,))

    def test_empty_codebook(self):
        with self.assertRaises(ValueError):
            Codebook(())


class TestSPLCodebooks(unittest.TestCase):
    def test_per_level_assign(self):
        books = SPLCodebooks(
            loop=Codebook(((0.0,), (10.0,))),
            profile=Codebook(((0.0,), (10.0,))),
            solid=Codebook(((0.0,), (10.0,))),
        )
        self.assertEqual(books.assign("loop", (9.0,)), 1)
        self.assertEqual(books.assign("solid", (1.0,)), 0)
        with self.assertRaises(ValueError):
            books.assign("bogus", (1.0,))


class TestCodebookHealth(unittest.TestCase):
    def test_utilization(self):
        self.assertEqual(utilization([0, 0, 2], 3), (2, 0, 1))

    def test_utilization_out_of_range(self):
        with self.assertRaises(ValueError):
            utilization([3], 3)

    def test_underutilized(self):
        # code 1 mapped 0 times, code 0 mapped 8 times (> threshold 7)
        assigns = [0] * 8 + [2] * (REINIT_THRESHOLD - 1)
        under = underutilized_codes(assigns, 3)
        self.assertIn(1, under)   # never used
        self.assertIn(2, under)   # below threshold
        self.assertNotIn(0, under)

    def test_active_fraction(self):
        self.assertAlmostEqual(active_code_fraction([0, 0, 1], 4), 0.5)

    def test_perplexity_uniform_is_size(self):
        self.assertAlmostEqual(codebook_perplexity([0, 1, 2, 3], 4), 4.0, places=6)

    def test_perplexity_collapsed_is_one(self):
        self.assertAlmostEqual(codebook_perplexity([2, 2, 2], 4), 1.0, places=6)

    def test_compression_ratio(self):
        self.assertAlmostEqual(compression_ratio(150158, 2500), 60.06, places=2)
        with self.assertRaises(ValueError):
            compression_ratio(10, 0)


if __name__ == "__main__":
    unittest.main()
