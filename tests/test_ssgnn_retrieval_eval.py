"""Tests for bench.ssgnn_retrieval_eval (graded Recall@k / NDCG@k)."""

from __future__ import annotations

import unittest

from harnesscad.eval.bench.ssgnn_retrieval_eval import (
    GAIN_PARTIAL,
    GAIN_SIMILAR,
    evaluate_retrieval,
    graded_gains,
    ndcg_graded_at_k,
    rank_database,
    recall_at_k,
    retrieval_ranking,
)


class RankDatabaseTests(unittest.TestCase):
    def test_descending_similarity(self):
        query = [1.0, 0.0]
        db = [[1.0, 0.0], [0.0, 1.0], [0.9, 0.1]]
        order = rank_database(query, db)
        self.assertEqual(order[0], 0)   # identical is most similar
        self.assertEqual(order[-1], 1)  # orthogonal is least

    def test_exclude_self(self):
        query = [1.0, 0.0]
        db = [[1.0, 0.0], [0.9, 0.1]]
        order = rank_database(query, db, exclude=0)
        self.assertNotIn(0, order)

    def test_zero_vector_ranked_last(self):
        query = [1.0, 0.0]
        db = [[0.0, 0.0], [1.0, 0.0]]
        order = rank_database(query, db)
        self.assertEqual(order[0], 1)

    def test_tie_broken_by_index(self):
        query = [1.0, 0.0]
        db = [[1.0, 0.0], [1.0, 0.0]]
        self.assertEqual(rank_database(query, db), [0, 1])


class RecallTests(unittest.TestCase):
    def test_full_recall(self):
        ranking = [3, 1, 2, 0]
        self.assertEqual(recall_at_k(ranking, [3, 1], k=2), 1.0)

    def test_partial_recall(self):
        ranking = [3, 0, 1, 2]
        self.assertEqual(recall_at_k(ranking, [3, 1], k=2), 0.5)

    def test_no_relevant(self):
        self.assertEqual(recall_at_k([0, 1], [], k=2), 0.0)

    def test_negative_k(self):
        with self.assertRaises(ValueError):
            recall_at_k([0], [0], k=-1)


class NDCGGradedTests(unittest.TestCase):
    def test_graded_gains_lookup(self):
        gains = graded_gains([2, 0, 1], {2: GAIN_SIMILAR, 1: GAIN_PARTIAL})
        self.assertEqual(gains, [2.0, 0.0, 1.0])

    def test_ideal_ranking_ndcg_one(self):
        # Similar item first, partial second -> already ideal ordering.
        ranking = [0, 1, 2]
        gains = {0: GAIN_SIMILAR, 1: GAIN_PARTIAL}
        self.assertAlmostEqual(ndcg_graded_at_k(ranking, gains, 3), 1.0, places=9)

    def test_reversed_ranking_lower(self):
        ideal = ndcg_graded_at_k([0, 1, 2], {0: 2.0, 1: 1.0}, 3)
        worse = ndcg_graded_at_k([2, 1, 0], {0: 2.0, 1: 1.0}, 3)
        self.assertLess(worse, ideal)


class EvaluateRetrievalTests(unittest.TestCase):
    def _fixture(self):
        # query 0 close to db item 0 (similar) and 2 (partial); db item 1 opposite.
        queries = [[1.0, 0.0]]
        db = [[1.0, 0.0], [0.0, 1.0], [0.8, 0.2]]
        relevant = [[0]]
        gains = [{0: GAIN_SIMILAR, 2: GAIN_PARTIAL}]
        return queries, db, relevant, gains

    def test_report_structure(self):
        queries, db, relevant, gains = self._fixture()
        report = evaluate_retrieval(queries, db, relevant, gains, ks=(1, 2))
        self.assertEqual(report.n_queries, 1)
        self.assertIn(1, report.recall)
        self.assertIn(2, report.ndcg)

    def test_recall_perfect_top1(self):
        queries, db, relevant, gains = self._fixture()
        report = evaluate_retrieval(queries, db, relevant, gains, ks=(1,))
        self.assertEqual(report.recall[1], 1.0)

    def test_to_dict_keys(self):
        queries, db, relevant, gains = self._fixture()
        report = evaluate_retrieval(queries, db, relevant, gains, ks=(5, 10))
        d = report.to_dict()
        self.assertIn("recall@5", d["recall"])
        self.assertIn("ndcg@10", d["ndcg"])

    def test_default_binary_gains(self):
        queries, db, relevant, _ = self._fixture()
        report = evaluate_retrieval(queries, db, relevant, ks=(2,))
        self.assertGreaterEqual(report.ndcg[2], 0.0)

    def test_empty_queries(self):
        report = evaluate_retrieval([], [[1.0]], [], ks=(5,))
        self.assertEqual(report.n_queries, 0)
        self.assertEqual(report.recall[5], 0.0)

    def test_ranking_helper_matches(self):
        queries, db, _, _ = self._fixture()
        rankings = retrieval_ranking(queries, db)
        self.assertEqual(rankings[0][0], 0)


if __name__ == "__main__":
    unittest.main()
