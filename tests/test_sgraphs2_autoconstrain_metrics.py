"""Tests for bench.sgraphs2_autoconstrain_metrics."""

import unittest

from harnesscad.eval.bench.sketch.sgraphs2_autoconstrain_metrics import (
    corpus_scores,
    edge_key,
    edge_key_set,
    f1,
    micro_scores,
    per_type_scores,
    sketch_scores,
)
from harnesscad.domain.reconstruction.sketch.sgraphs2_dof_mask import EdgeOp


class TestEdgeKey(unittest.TestCase):
    def test_key_is_label_first_last(self):
        self.assertEqual(edge_key(EdgeOp("coincident", (1, 2))), ("coincident", 1, 2))

    def test_hyperedge_keeps_only_extremities(self):
        self.assertEqual(edge_key(EdgeOp("mirror", (1, 2, 3, 4))), ("mirror", 1, 4))

    def test_self_loop(self):
        self.assertEqual(edge_key(EdgeOp("horizontal", (3,))), ("horizontal", 3, 3))

    def test_empty_references_rejected(self):
        with self.assertRaises(ValueError):
            edge_key(EdgeOp("coincident", ()))

    def test_subnode_edges_dropped(self):
        ops = [EdgeOp("subnode", (1, 2)), EdgeOp("coincident", (1, 2))]
        self.assertEqual(edge_key_set(ops), {("coincident", 1, 2)})

    def test_duplicates_collapse(self):
        ops = [EdgeOp("parallel", (1, 2)), EdgeOp("parallel", (1, 2))]
        self.assertEqual(len(edge_key_set(ops)), 1)


class TestF1(unittest.TestCase):
    def test_harmonic_mean(self):
        self.assertAlmostEqual(f1(1.0, 0.5), 2 / 3)

    def test_zero_zero(self):
        self.assertEqual(f1(0.0, 0.0), 0.0)


class TestSketchScores(unittest.TestCase):
    def test_perfect(self):
        ops = [EdgeOp("coincident", (1, 2)), EdgeOp("parallel", (2, 3))]
        score = sketch_scores(ops, list(ops))
        self.assertEqual(score.precision, 1.0)
        self.assertEqual(score.recall, 1.0)
        self.assertEqual(score.f1, 1.0)
        self.assertEqual(score.num_correct, 2)

    def test_partial(self):
        gt = [EdgeOp("coincident", (1, 2)), EdgeOp("parallel", (2, 3))]
        pred = [EdgeOp("coincident", (1, 2)), EdgeOp("tangent", (3, 4))]
        score = sketch_scores(gt, pred)
        self.assertEqual(score.precision, 0.5)
        self.assertEqual(score.recall, 0.5)
        self.assertEqual(score.num_correct, 1)

    def test_order_insensitive(self):
        gt = [EdgeOp("coincident", (1, 2)), EdgeOp("parallel", (2, 3))]
        score = sketch_scores(gt, list(reversed(gt)))
        self.assertEqual(score.precision, 1.0)

    def test_duplicate_predictions_do_not_inflate(self):
        gt = [EdgeOp("coincident", (1, 2))]
        pred = [EdgeOp("coincident", (1, 2)), EdgeOp("coincident", (1, 2))]
        score = sketch_scores(gt, pred)
        self.assertEqual(score.precision, 1.0)
        self.assertEqual(score.num_predicted, 1)

    def test_no_predictions_gives_zero_precision(self):
        score = sketch_scores([EdgeOp("coincident", (1, 2))], [])
        self.assertEqual(score.precision, 0.0)
        self.assertEqual(score.recall, 0.0)

    def test_no_ground_truth_gives_recall_one(self):
        score = sketch_scores([], [EdgeOp("coincident", (1, 2))])
        self.assertEqual(score.recall, 1.0)
        self.assertEqual(score.precision, 0.0)

    def test_both_empty(self):
        # Abstaining on an unconstrained sketch: recall 1 (nothing missed) but
        # precision 0 (no credit for abstaining).
        score = sketch_scores([], [])
        self.assertEqual(score.recall, 1.0)
        self.assertEqual(score.precision, 0.0)

    def test_subnode_excluded_from_both_sides(self):
        gt = [EdgeOp("subnode", (1, 2)), EdgeOp("coincident", (1, 2))]
        pred = [EdgeOp("subnode", (1, 2)), EdgeOp("coincident", (1, 2))]
        score = sketch_scores(gt, pred)
        self.assertEqual(score.num_ground_truth, 1)
        self.assertEqual(score.num_predicted, 1)
        self.assertEqual(score.precision, 1.0)

    def test_direction_matters(self):
        # (1, 2) and (2, 1) are distinct keys -- the reference does not canonicalise
        # reference order.
        score = sketch_scores([EdgeOp("coincident", (1, 2))], [EdgeOp("coincident", (2, 1))])
        self.assertEqual(score.num_correct, 0)


class TestCorpusScores(unittest.TestCase):
    def setUp(self):
        # Sketch A: 1 of 1 correct. Sketch B: 1 of 2 predicted, 1 of 4 in GT.
        self.pairs = [
            ([EdgeOp("coincident", (1, 2))], [EdgeOp("coincident", (1, 2))]),
            (
                [EdgeOp("coincident", (1, 2)), EdgeOp("parallel", (2, 3)),
                 EdgeOp("tangent", (3, 4)), EdgeOp("vertical", (4,))],
                [EdgeOp("coincident", (1, 2)), EdgeOp("radius", (5, 5))],
            ),
        ]

    def test_macro_average_weights_sketches_equally(self):
        score = corpus_scores(self.pairs)
        # A: p=1, r=1.  B: p=0.5, r=0.25.
        self.assertAlmostEqual(score.precision, 0.75)
        self.assertAlmostEqual(score.recall, 0.625)
        self.assertEqual(score.num_sketches, 2)

    def test_micro_average_weights_edges(self):
        score = micro_scores(self.pairs)
        # correct=2, predicted=3, ground truth=5.
        self.assertAlmostEqual(score.precision, 2 / 3)
        self.assertAlmostEqual(score.recall, 2 / 5)

    def test_macro_and_micro_differ_on_skewed_corpus(self):
        self.assertNotAlmostEqual(
            corpus_scores(self.pairs).recall, micro_scores(self.pairs).recall
        )

    def test_abstaining_model_does_not_score_perfectly(self):
        # Many unconstrained sketches, one constrained: abstaining gets recall
        # credit on the empty ones but zero precision throughout.
        pairs = [([], [])] * 9 + [([EdgeOp("coincident", (1, 2))], [])]
        score = corpus_scores(pairs)
        self.assertEqual(score.precision, 0.0)
        self.assertAlmostEqual(score.recall, 0.9)
        self.assertEqual(score.f1, 0.0)

    def test_empty_corpus(self):
        score = corpus_scores([])
        self.assertEqual((score.precision, score.recall, score.num_sketches), (0.0, 0.0, 0))

    def test_micro_empty_corpus(self):
        score = micro_scores([])
        self.assertEqual(score.precision, 0.0)
        self.assertEqual(score.recall, 1.0)

    def test_corpus_f1(self):
        score = corpus_scores(self.pairs)
        self.assertAlmostEqual(score.f1, f1(0.75, 0.625))


class TestPerTypeScores(unittest.TestCase):
    def test_breakdown_exposes_per_type_failure(self):
        pairs = [
            (
                [EdgeOp("coincident", (1, 2)), EdgeOp("tangent", (3, 4))],
                [EdgeOp("coincident", (1, 2)), EdgeOp("tangent", (9, 9))],
            ),
            (
                [EdgeOp("coincident", (5, 6)), EdgeOp("tangent", (7, 8))],
                [EdgeOp("coincident", (5, 6))],
            ),
        ]
        by_type = per_type_scores(pairs)
        self.assertEqual(sorted(by_type), ["coincident", "tangent"])
        self.assertEqual(by_type["coincident"].precision, 1.0)
        self.assertEqual(by_type["coincident"].recall, 1.0)
        self.assertEqual(by_type["tangent"].precision, 0.0)
        self.assertEqual(by_type["tangent"].recall, 0.0)
        self.assertEqual(by_type["tangent"].num_ground_truth, 2)
        self.assertEqual(by_type["tangent"].num_predicted, 1)

    def test_hallucinated_type_has_zero_precision_and_recall_one(self):
        pairs = [([EdgeOp("coincident", (1, 2))], [EdgeOp("radius", (3, 3))])]
        by_type = per_type_scores(pairs)
        self.assertEqual(by_type["radius"].precision, 0.0)
        self.assertEqual(by_type["radius"].recall, 1.0)  # nothing to find
        self.assertEqual(by_type["radius"].num_ground_truth, 0)

    def test_subnode_absent_from_breakdown(self):
        pairs = [([EdgeOp("subnode", (1, 2))], [EdgeOp("subnode", (1, 2))])]
        self.assertEqual(per_type_scores(pairs), {})

    def test_empty_corpus(self):
        self.assertEqual(per_type_scores([]), {})

    def test_deterministic(self):
        pairs = [([EdgeOp("coincident", (1, 2))], [EdgeOp("coincident", (1, 2))])]
        self.assertEqual(per_type_scores(pairs), per_type_scores(pairs))


if __name__ == "__main__":
    unittest.main()
