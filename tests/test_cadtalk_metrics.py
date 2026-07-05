import unittest

from cadtalk_metrics import (
    apply_synonyms,
    block_accuracy,
    semantic_iou,
    evaluate,
    aggregate,
)


class TestBlockAccuracy(unittest.TestCase):
    def test_all_correct(self):
        pred = {0: "body", 1: "wing", 2: "tail"}
        gt = {0: "body", 1: "wing", 2: "tail"}
        self.assertEqual(block_accuracy(pred, gt), 1.0)

    def test_half(self):
        pred = {0: "body", 1: "tail"}
        gt = {0: "body", 1: "wing"}
        self.assertEqual(block_accuracy(pred, gt), 0.5)

    def test_multilabel_gt(self):
        # block 0 gt accepts either 'wing' or 'engine'
        pred = {0: "engine"}
        gt = {0: {"wing", "engine"}}
        self.assertEqual(block_accuracy(pred, gt), 1.0)

    def test_missing_prediction_is_wrong(self):
        pred = {0: "body"}
        gt = {0: "body", 1: "wing"}
        self.assertEqual(block_accuracy(pred, gt), 0.5)

    def test_empty_gt(self):
        self.assertEqual(block_accuracy({}, {}), 0.0)


class TestSemanticIoU(unittest.TestCase):
    def test_perfect(self):
        pred = {0: "a", 1: "b"}
        gt = {0: "a", 1: "b"}
        self.assertEqual(semantic_iou(pred, gt), 1.0)

    def test_partial(self):
        # label a: pred {0}, gt {0} -> IoU 1. label b: pred {1}, gt {2} -> 0.
        # label c handled: pred none? build a concrete case.
        pred = {0: "a", 1: "b"}
        gt = {0: "a", 1: "c"}
        # labels: a (1.0), b (pred{1},gt{}) ->0, c (pred{},gt{1})->0
        mean, per = semantic_iou(pred, gt, per_label=True)
        self.assertEqual(per["a"], 1.0)
        self.assertEqual(per["b"], 0.0)
        self.assertEqual(per["c"], 0.0)
        self.assertAlmostEqual(mean, 1.0 / 3.0)

    def test_multilabel_gt_iou(self):
        pred = {0: "wing"}
        gt = {0: {"wing", "engine"}}
        mean, per = semantic_iou(pred, gt, per_label=True)
        # wing: pred{0} gt{0} -> 1.0 ; engine: pred{} gt{0} -> 0.0
        self.assertEqual(per["wing"], 1.0)
        self.assertEqual(per["engine"], 0.0)
        self.assertAlmostEqual(mean, 0.5)

    def test_empty(self):
        self.assertEqual(semantic_iou({}, {}), 0.0)


class TestSynonyms(unittest.TestCase):
    def test_apply(self):
        pred = {0: "seat", 1: "backrest"}
        mapping = {"backrest": "back"}
        mapped = apply_synonyms(pred, mapping)
        self.assertEqual(mapped, {0: "seat", 1: "back"})

    def test_evaluate_with_synonyms(self):
        pred = {0: "backrest"}
        gt = {0: "back"}
        rep = evaluate(pred, gt, synonyms={"backrest": "back"})
        self.assertEqual(rep["block_accuracy"], 1.0)
        self.assertEqual(rep["semantic_iou"], 1.0)


class TestEvaluateAggregate(unittest.TestCase):
    def test_report_fields(self):
        rep = evaluate({0: "a", 1: "b"}, {0: "a", 1: "c"})
        self.assertEqual(rep["n_blocks"], 2)
        self.assertEqual(rep["n_correct"], 1)
        self.assertIn("per_label_iou", rep)

    def test_aggregate(self):
        r1 = evaluate({0: "a"}, {0: "a"})
        r2 = evaluate({0: "a"}, {0: "b"})
        agg = aggregate([r1, r2])
        self.assertEqual(agg["n_programs"], 2)
        self.assertAlmostEqual(agg["block_accuracy"], 0.5)

    def test_aggregate_empty(self):
        self.assertEqual(aggregate([])["n_programs"], 0)


if __name__ == "__main__":
    unittest.main()
