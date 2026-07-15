"""Tests for eval.bench.retrieval.two_stage_cascade."""

import unittest

from harnesscad.eval.bench.retrieval.two_stage_cascade import (
    DatabaseModel,
    cosine_similarity,
    retrieve,
    stage1_text_filter,
    stage2_visual_refine,
)


class CosineTest(unittest.TestCase):
    def test_parallel(self):
        self.assertAlmostEqual(cosine_similarity([1, 0], [2, 0]), 1.0)

    def test_orthogonal(self):
        self.assertAlmostEqual(cosine_similarity([1, 0], [0, 1]), 0.0)

    def test_zero_vector_raises(self):
        with self.assertRaises(ValueError):
            cosine_similarity([0, 0], [1, 1])


def _db():
    return [
        DatabaseModel("mustard",
                      caption_embeddings=((1.0, 0.0), (0.9, 0.1)),
                      view_embeddings=((1.0, 0.0),)),
        DatabaseModel("drill",
                      caption_embeddings=((0.0, 1.0),),
                      view_embeddings=((0.0, 1.0),)),
        DatabaseModel("can",
                      caption_embeddings=((0.8, 0.2),),
                      view_embeddings=((0.2, 0.98),)),
    ]


class Stage1Test(unittest.TestCase):
    def test_threshold_filters(self):
        cands = stage1_text_filter((1.0, 0.0), _db(), threshold=0.7)
        idxs = [i for i, _ in cands]
        self.assertIn(0, idxs)   # mustard
        self.assertIn(2, idxs)   # can
        self.assertNotIn(1, idxs)  # drill filtered

    def test_sorted_descending(self):
        cands = stage1_text_filter((1.0, 0.0), _db(), threshold=0.0)
        scores = [s for _, s in cands]
        self.assertEqual(scores, sorted(scores, reverse=True))


class Stage2Test(unittest.TestCase):
    def test_picks_best_visual(self):
        db = _db()
        cands = stage1_text_filter((1.0, 0.0), db, threshold=0.5)
        best = stage2_visual_refine((1.0, 0.0), db, cands)
        self.assertIsNotNone(best)
        self.assertEqual(db[best[0]].name, "mustard")

    def test_no_candidates(self):
        self.assertIsNone(stage2_visual_refine((1.0, 0.0), _db(), []))


class RetrieveTest(unittest.TestCase):
    def test_end_to_end(self):
        model = retrieve((1.0, 0.0), (1.0, 0.0), _db(), threshold=0.7)
        self.assertIsNotNone(model)
        self.assertEqual(model.name, "mustard")

    def test_nothing_passes(self):
        self.assertIsNone(retrieve((1.0, 0.0), (1.0, 0.0), _db(), threshold=1.5))

    def test_deterministic(self):
        a = retrieve((0.5, 0.5), (0.5, 0.5), _db(), threshold=0.0)
        b = retrieve((0.5, 0.5), (0.5, 0.5), _db(), threshold=0.0)
        self.assertEqual(a.name, b.name)


if __name__ == "__main__":
    unittest.main()
