"""Tests for rag.oscar_multimodal_fusion -- OSCAR two-stage text+image retrieval."""

from __future__ import annotations

import unittest

from harnesscad.agents.rag.oscar_multimodal_fusion import (
    OscarModel,
    cosine_similarity,
    text_similarity,
    image_similarity,
    filter_candidates,
    late_fusion_score,
    retrieve,
)


def _model(mid, texts, imgs):
    return OscarModel(mid, tuple(tuple(t) for t in texts), tuple(tuple(v) for v in imgs))


class TestCosine(unittest.TestCase):
    def test_identical(self):
        self.assertAlmostEqual(cosine_similarity([1.0, 0.0], [2.0, 0.0]), 1.0)

    def test_orthogonal(self):
        self.assertAlmostEqual(cosine_similarity([1.0, 0.0], [0.0, 1.0]), 0.0)

    def test_zero_vector(self):
        self.assertAlmostEqual(cosine_similarity([0.0, 0.0], [1.0, 1.0]), 0.0)


class TestModalitySimilarity(unittest.TestCase):
    def test_text_takes_max_over_captions(self):
        m = _model("a", [[0.0, 1.0], [1.0, 0.0]], [[1.0, 0.0]])
        # query aligned with second caption -> sim 1.0
        self.assertAlmostEqual(text_similarity([1.0, 0.0], m), 1.0)

    def test_image_takes_max_over_views(self):
        m = _model("a", [[1.0, 0.0]], [[0.0, 1.0], [1.0, 1.0]])
        # query [1,0]: view0 -> 0, view1 -> cos45 ~ 0.7071
        self.assertAlmostEqual(image_similarity([1.0, 0.0], m), 1.0 / (2 ** 0.5))

    def test_empty_banks_floor(self):
        m = _model("a", [], [])
        self.assertEqual(text_similarity([1.0], m), -1.0)
        self.assertEqual(image_similarity([1.0], m), -1.0)


class TestFilterCandidates(unittest.TestCase):
    def setUp(self):
        # three models; only some align with query [1,0]
        self.models = [
            _model("hit", [[1.0, 0.0]], [[1.0, 0.0]]),
            _model("miss", [[0.0, 1.0]], [[0.0, 1.0]]),
            _model("weak", [[0.9, 0.1]], [[0.9, 0.1]]),
        ]

    def test_threshold_keeps_passing_by_index(self):
        c = filter_candidates([1.0, 0.0], self.models, tau_text=0.5, top_k=2)
        # model0 sim 1.0 passes; model2 sim ~0.994 passes; model1 ~0 fails
        self.assertEqual(c, [0, 2])

    def test_high_threshold_triggers_topk_fallback(self):
        c = filter_candidates([1.0, 0.0], self.models, tau_text=1.5, top_k=2)
        # nobody passes -> top-2 by similarity: model0 (1.0) then model2 (~0.994)
        self.assertEqual(c, [0, 2])

    def test_empty_database(self):
        self.assertEqual(filter_candidates([1.0], [], 0.3, 5), [])

    def test_topk_clamped(self):
        c = filter_candidates([1.0, 0.0], self.models, tau_text=1.5, top_k=99)
        self.assertEqual(len(c), 3)


class TestLateFusion(unittest.TestCase):
    def test_image_only(self):
        self.assertAlmostEqual(late_fusion_score(0.2, 0.9, image_weight=1.0), 0.9)

    def test_text_only(self):
        self.assertAlmostEqual(late_fusion_score(0.2, 0.9, image_weight=0.0), 0.2)

    def test_blend(self):
        self.assertAlmostEqual(late_fusion_score(0.2, 0.8, image_weight=0.5), 0.5)

    def test_invalid_weight(self):
        with self.assertRaises(ValueError):
            late_fusion_score(0.1, 0.2, image_weight=1.5)


class TestRetrieve(unittest.TestCase):
    def setUp(self):
        # Two models share similar text but differ visually. The visually
        # correct model must win via the DINOv2 refinement stage.
        self.models = [
            _model("A", [[1.0, 0.0]], [[1.0, 0.0]]),
            _model("B", [[1.0, 0.0]], [[0.0, 1.0]]),
        ]

    def test_image_refinement_breaks_text_tie(self):
        # Both pass text filter (query text aligned [1,0]); DINO query [0,1]
        # should pick model B whose view matches.
        r = retrieve([1.0, 0.0], [0.0, 1.0], self.models, tau_text=0.5)
        self.assertEqual(r.model_id, "B")
        self.assertEqual(r.model_index, 1)
        self.assertFalse(r.used_fallback)
        self.assertEqual(set(r.candidates), {0, 1})

    def test_fallback_flag_set(self):
        r = retrieve([1.0, 0.0], [1.0, 0.0], self.models, tau_text=1.5, top_k=1)
        self.assertTrue(r.used_fallback)
        # top-1 text candidate is model A (index 0), so only it is considered
        self.assertEqual(r.candidates, (0,))
        self.assertEqual(r.model_id, "A")

    def test_empty_database(self):
        r = retrieve([1.0], [1.0], [], tau_text=0.3)
        self.assertEqual(r.model_index, -1)
        self.assertIsNone(r.model_id)

    def test_deterministic_tie_lowest_index(self):
        # identical models -> lowest index wins
        models = [_model("X", [[1.0, 0.0]], [[1.0, 0.0]]),
                  _model("Y", [[1.0, 0.0]], [[1.0, 0.0]])]
        r = retrieve([1.0, 0.0], [1.0, 0.0], models, tau_text=0.5)
        self.assertEqual(r.model_index, 0)


if __name__ == "__main__":
    unittest.main()
