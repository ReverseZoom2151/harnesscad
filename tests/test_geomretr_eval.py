"""Tests for the geometric-object retrieval evaluation protocol."""

from __future__ import annotations

import unittest

from harnesscad.eval.bench.retrieval.geomretr_eval import (
    cosine_distance,
    rank_gallery,
    retrieval_ranking,
    nn_accuracy,
    nn_macro_f1,
    ndcg_at_n,
    average_precision,
    per_category_map,
    evaluate_retrieval,
)


class CosineDistanceTest(unittest.TestCase):
    def test_identical(self):
        self.assertAlmostEqual(cosine_distance([1, 1], [2, 2]), 0.0, places=9)

    def test_orthogonal(self):
        self.assertAlmostEqual(cosine_distance([1, 0], [0, 1]), 1.0, places=9)

    def test_opposite(self):
        self.assertAlmostEqual(cosine_distance([1, 0], [-1, 0]), 2.0, places=9)

    def test_zero_vector(self):
        self.assertEqual(cosine_distance([0, 0], [1, 1]), 1.0)


class RankTest(unittest.TestCase):
    def test_order(self):
        query = [1.0, 0.0]
        gallery = [[-1.0, 0.0], [1.0, 0.0], [0.0, 1.0]]
        order = rank_gallery(query, gallery)
        self.assertEqual(order[0], 1)   # identical direction closest
        self.assertEqual(order[-1], 0)  # opposite farthest

    def test_tie_break_by_index(self):
        query = [1.0, 0.0]
        gallery = [[1.0, 0.0], [1.0, 0.0]]
        self.assertEqual(rank_gallery(query, gallery), [0, 1])


class MetricsTest(unittest.TestCase):
    def setUp(self):
        # two well-separated clusters, query near each
        self.gallery = [[1.0, 0.0], [0.9, 0.1], [0.0, 1.0], [0.1, 0.9]]
        self.gallery_labels = ["A", "A", "B", "B"]
        self.queries = [[0.95, 0.05], [0.05, 0.95]]
        self.query_labels = ["A", "B"]
        self.rankings = retrieval_ranking(self.queries, self.gallery)

    def test_nn_accuracy_perfect(self):
        acc = nn_accuracy(self.rankings, self.query_labels, self.gallery_labels)
        self.assertAlmostEqual(acc, 1.0, places=9)

    def test_nn_f1_perfect(self):
        f1 = nn_macro_f1(self.rankings, self.query_labels, self.gallery_labels)
        self.assertAlmostEqual(f1, 1.0, places=9)

    def test_ndcg_range(self):
        val = ndcg_at_n(self.rankings, self.query_labels, self.gallery_labels, n=4)
        self.assertTrue(0.0 <= val <= 1.0)
        # class A query: same-class items should rank first -> ndcg = 1
        self.assertAlmostEqual(val, 1.0, places=9)

    def test_ndcg_imperfect(self):
        # query whose nearest are wrong class
        gallery = [[1.0, 0.0], [0.0, 1.0]]
        glabels = ["X", "Y"]
        qlabels = ["Y"]
        rankings = retrieval_ranking([[1.0, 0.0]], gallery)
        val = ndcg_at_n(rankings, qlabels, glabels, n=2)
        self.assertLess(val, 1.0)


class APTest(unittest.TestCase):
    def test_all_relevant_first(self):
        order = [0, 1, 2, 3]
        labels = ["A", "A", "B", "B"]
        self.assertAlmostEqual(average_precision(order, "A", labels), 1.0, places=9)

    def test_no_relevant(self):
        self.assertAlmostEqual(average_precision([0, 1], "Z", ["A", "B"]), 0.0)

    def test_known_value(self):
        # relevant at ranks 1 and 3: AP = (1/1 + 2/3)/2
        order = [0, 1, 2]
        labels = ["A", "B", "A"]
        expected = (1.0 + 2.0 / 3.0) / 2.0
        self.assertAlmostEqual(average_precision(order, "A", labels), expected, places=9)


class MapAndReportTest(unittest.TestCase):
    def test_per_category(self):
        gallery = [[1.0, 0.0], [0.9, 0.1], [0.0, 1.0], [0.1, 0.9]]
        glabels = ["A", "A", "B", "B"]
        queries = [[0.95, 0.05], [0.05, 0.95]]
        qlabels = ["A", "B"]
        rankings = retrieval_ranking(queries, gallery)
        mp = per_category_map(rankings, qlabels, glabels)
        self.assertIn("A", mp["per_category"])
        self.assertIn("B", mp["per_category"])
        self.assertAlmostEqual(mp["macro_map"], 1.0, places=9)
        self.assertAlmostEqual(mp["micro_map"], 1.0, places=9)

    def test_report_to_dict(self):
        gallery = [[1.0, 0.0], [0.9, 0.1], [0.0, 1.0], [0.1, 0.9]]
        glabels = ["A", "A", "B", "B"]
        queries = [[0.95, 0.05], [0.05, 0.95]]
        qlabels = ["A", "B"]
        report = evaluate_retrieval(queries, qlabels, gallery, glabels, ndcg_n=4)
        d = report.to_dict()
        self.assertEqual(d["n_queries"], 2)
        self.assertEqual(d["nn_accuracy"], 1.0)
        self.assertEqual(d["nn_f1"], 1.0)
        self.assertIn("A", d["per_category_map"])

    def test_empty(self):
        report = evaluate_retrieval([], [], [], [])
        self.assertEqual(report.n_queries, 0)
        self.assertEqual(report.nn_accuracy, 0.0)


if __name__ == "__main__":
    unittest.main()
