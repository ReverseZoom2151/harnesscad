"""Tests for rag.rag6d_cad_retrieval."""

import unittest

from harnesscad.agents.rag.rag6d_cad_retrieval import (
    CadKnowledgeBase,
    PoseCandidate,
    cosine_similarity,
    hypothesis_score,
    rank_pose_hypotheses,
    select_best_hypothesis,
)


class TestCosine(unittest.TestCase):
    def test_identical(self):
        self.assertAlmostEqual(cosine_similarity([1.0, 2.0, 3.0], [1.0, 2.0, 3.0]), 1.0)

    def test_orthogonal(self):
        self.assertAlmostEqual(cosine_similarity([1.0, 0.0], [0.0, 1.0]), 0.0)

    def test_opposite(self):
        self.assertAlmostEqual(cosine_similarity([1.0, 0.0], [-1.0, 0.0]), -1.0)

    def test_null_vector(self):
        self.assertEqual(cosine_similarity([0.0, 0.0], [1.0, 1.0]), 0.0)

    def test_mismatch_raises(self):
        with self.assertRaises(ValueError):
            cosine_similarity([1.0], [1.0, 2.0])


def make_kb():
    kb = CadKnowledgeBase()
    # model A: two orthogonal views
    kb.add_view("A", "front", [1.0, 0.0, 0.0], pose="A_front")
    kb.add_view("A", "side", [0.0, 1.0, 0.0], pose="A_side")
    # model B: a view along z
    kb.add_view("B", "top", [0.0, 0.0, 1.0], pose="B_top")
    return kb


class TestRetrieval(unittest.TestCase):
    def test_len_and_models(self):
        kb = make_kb()
        self.assertEqual(len(kb), 3)
        self.assertEqual(kb.model_ids(), ("A", "B"))

    def test_nearest_view(self):
        kb = make_kb()
        top = kb.retrieve_views([0.9, 0.1, 0.0], k=1)
        self.assertEqual(len(top), 1)
        score, entry = top[0]
        self.assertEqual(entry.model_id, "A")
        self.assertEqual(entry.view_id, "front")
        self.assertEqual(entry.pose, "A_front")

    def test_topk_order(self):
        kb = make_kb()
        top = kb.retrieve_views([0.0, 0.0, 1.0], k=3)
        self.assertEqual(top[0][1].view_id, "top")   # best match
        # remaining two both score 0 -> tie broken by (model_id, view_id)
        self.assertEqual(top[1][1].model_id, "A")

    def test_best_model(self):
        kb = make_kb()
        result = kb.retrieve_best_model([0.1, 0.0, 0.95])
        self.assertIsNotNone(result)
        model_id, score, entry = result
        self.assertEqual(model_id, "B")
        self.assertEqual(entry.view_id, "top")

    def test_empty_base(self):
        kb = CadKnowledgeBase()
        self.assertEqual(kb.retrieve_views([1.0, 0.0, 0.0], k=1), [])
        self.assertIsNone(kb.retrieve_best_model([1.0, 0.0, 0.0]))

    def test_negative_k_raises(self):
        with self.assertRaises(ValueError):
            make_kb().retrieve_views([1.0, 0.0, 0.0], k=-1)


class TestRanking(unittest.TestCase):
    def test_score(self):
        c = PoseCandidate("p", retrieval_score=0.5, num_inliers=10)
        self.assertAlmostEqual(hypothesis_score(c), 10.5)
        self.assertAlmostEqual(
            hypothesis_score(c, inlier_weight=0.1, retrieval_weight=2.0), 2.0)

    def test_rank_by_inliers(self):
        a = PoseCandidate("a", retrieval_score=0.9, num_inliers=5, label="a")
        b = PoseCandidate("b", retrieval_score=0.1, num_inliers=20, label="b")
        ranked = rank_pose_hypotheses([a, b])
        self.assertEqual(ranked[0].pose, "b")   # far more inliers wins
        self.assertEqual(ranked[1].pose, "a")

    def test_retrieval_breaks_close_inliers(self):
        a = PoseCandidate("a", retrieval_score=0.2, num_inliers=10, label="a")
        b = PoseCandidate("b", retrieval_score=0.8, num_inliers=10, label="b")
        best = select_best_hypothesis([a, b])
        self.assertEqual(best.pose, "b")   # equal inliers, higher retrieval

    def test_deterministic_tie(self):
        a = PoseCandidate("a", retrieval_score=0.5, num_inliers=10, label="z")
        b = PoseCandidate("b", retrieval_score=0.5, num_inliers=10, label="a")
        ranked = rank_pose_hypotheses([a, b])
        self.assertEqual(ranked[0].label, "a")   # tie -> label order

    def test_select_empty(self):
        self.assertIsNone(select_best_hypothesis([]))


if __name__ == "__main__":
    unittest.main()
