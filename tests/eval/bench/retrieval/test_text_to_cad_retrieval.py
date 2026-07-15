import unittest

from harnesscad.eval.bench.retrieval import text_to_cad_retrieval as t2c


class TestRankOf(unittest.TestCase):
    def test_found(self):
        self.assertEqual(t2c.rank_of(["a", "b", "c"], "b"), 2)

    def test_absent(self):
        self.assertEqual(t2c.rank_of(["a", "b"], "z"), 0)


class TestRecall(unittest.TestCase):
    def test_recall_at_k_ranks(self):
        # ranks 1, 3, 6 for three queries
        qs = [1, 3, 6]
        self.assertAlmostEqual(t2c.recall_at_k(qs, 1), 100.0 / 3)
        self.assertAlmostEqual(t2c.recall_at_k(qs, 5), 200.0 / 3)
        self.assertAlmostEqual(t2c.recall_at_k(qs, 10), 100.0)

    def test_recall_with_lists(self):
        qs = [(["x", "y", "gt"], "gt"), (["gt", "a"], "gt")]
        self.assertEqual(t2c.recall_at_k(qs, 1), 50.0)
        self.assertEqual(t2c.recall_at_k(qs, 5), 100.0)

    def test_miss_never_counts(self):
        qs = [0, 2]  # first query missed entirely
        self.assertEqual(t2c.recall_at_k(qs, 20), 50.0)

    def test_bad_k(self):
        with self.assertRaises(ValueError):
            t2c.recall_at_k([1], 0)


class TestMedianRank(unittest.TestCase):
    def test_odd(self):
        self.assertEqual(t2c.median_rank([1, 3, 9]), 3.0)

    def test_even(self):
        self.assertEqual(t2c.median_rank([1, 2, 3, 4]), 2.5)

    def test_miss_penalized(self):
        # observed max is 4; miss becomes 5 -> sorted [2,4,5] median 4
        self.assertEqual(t2c.median_rank([2, 4, 0]), 4.0)


class TestRsumReport(unittest.TestCase):
    def test_rsum_matches_sum(self):
        qs = [1, 1, 1]
        # all at rank 1 -> recall 100 at every K -> rsum = 500
        self.assertEqual(t2c.rsum(qs), 500.0)

    def test_report_structure(self):
        qs = [1, 3, 6, 25]
        rep = t2c.retrieval_report(qs)
        self.assertEqual(rep["num_queries"], 4)
        self.assertEqual(rep["ks"], (1, 2, 5, 10, 20))
        self.assertAlmostEqual(rep["rsum"], sum(rep["recall"].values()))
        self.assertIn(1, rep["recall"])
        self.assertGreater(rep["medr"], 0)


if __name__ == "__main__":
    unittest.main()
