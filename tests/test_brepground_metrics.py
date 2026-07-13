"""Tests for bench.brepground_metrics."""

import unittest

from harnesscad.eval.bench.retrieval.brepground_metrics import (
    GroundingCase,
    average_precision,
    evaluate,
    f1,
    mean_average_precision,
    mean_f1,
    precision_recall_f1,
    recall_at_k,
)


class TestRecallAtK(unittest.TestCase):
    def test_all_in_topk(self):
        self.assertEqual(recall_at_k([1, 2, 3], {1, 2}, 3), 1.0)

    def test_partial(self):
        self.assertAlmostEqual(recall_at_k([1, 9, 8], {1, 2}, 3), 0.5)

    def test_beyond_k_not_counted(self):
        self.assertEqual(recall_at_k([9, 8, 1], {1}, 2), 0.0)

    def test_empty_truth_is_one(self):
        self.assertEqual(recall_at_k([1, 2], set(), 3), 1.0)

    def test_k_zero(self):
        self.assertEqual(recall_at_k([1], {1}, 0), 0.0)


class TestAveragePrecision(unittest.TestCase):
    def test_perfect_ranking(self):
        # both relevant items at the top -> AP = 1.
        self.assertAlmostEqual(average_precision([1, 2, 3, 4], {1, 2}), 1.0)

    def test_interleaved(self):
        # ranks of relevant items: 1 and 3 -> (1/1 + 2/3)/2.
        ap = average_precision([1, 9, 2, 8], {1, 2})
        self.assertAlmostEqual(ap, (1.0 + 2.0 / 3.0) / 2.0)

    def test_missing_relevant_penalised(self):
        # only one of two relevant retrieved at rank 1 -> (1/1)/2 = 0.5.
        self.assertAlmostEqual(average_precision([1, 9], {1, 2}), 0.5)

    def test_empty_truth(self):
        self.assertEqual(average_precision([1], set()), 1.0)


class TestPrecisionRecallF1(unittest.TestCase):
    def test_perfect(self):
        self.assertEqual(precision_recall_f1({1, 2}, {1, 2}), (1.0, 1.0, 1.0))

    def test_half_precision(self):
        p, r, fscore = precision_recall_f1({1, 9}, {1})
        self.assertAlmostEqual(p, 0.5)
        self.assertAlmostEqual(r, 1.0)
        self.assertAlmostEqual(fscore, 2 * 0.5 * 1.0 / 1.5)

    def test_no_overlap(self):
        self.assertEqual(precision_recall_f1({9}, {1}), (0.0, 0.0, 0.0))

    def test_both_empty(self):
        self.assertEqual(precision_recall_f1(set(), set()), (1.0, 1.0, 1.0))


class TestF1Case(unittest.TestCase):
    def test_falls_back_to_ranked(self):
        c = GroundingCase(ranked=[1, 2], truth={1, 2})
        self.assertAlmostEqual(f1(c), 1.0)

    def test_uses_selected_when_given(self):
        c = GroundingCase(ranked=[1, 2, 3], truth={1}, selected=[1])
        self.assertAlmostEqual(f1(c), 1.0)

    def test_duplicate_ranked_rejected(self):
        with self.assertRaises(ValueError):
            GroundingCase(ranked=[1, 1], truth={1})


class TestAggregate(unittest.TestCase):
    def setUp(self):
        self.cases = [
            GroundingCase(ranked=[1, 2, 3], truth={1, 2}),
            GroundingCase(ranked=[9, 1, 2], truth={1}),
        ]

    def test_map(self):
        expected = (average_precision([1, 2, 3], {1, 2})
                    + average_precision([9, 1, 2], {1})) / 2
        self.assertAlmostEqual(mean_average_precision(self.cases), expected)

    def test_mean_f1(self):
        self.assertTrue(0.0 <= mean_f1(self.cases) <= 1.0)

    def test_evaluate_keys(self):
        rep = evaluate(self.cases, ks=(3, 5))
        self.assertIn("recall@3", rep)
        self.assertIn("recall@5", rep)
        self.assertIn("mAP", rep)
        self.assertIn("F1", rep)

    def test_evaluate_range(self):
        rep = evaluate(self.cases)
        for v in rep.values():
            self.assertTrue(0.0 <= v <= 1.0)

    def test_empty_cases(self):
        self.assertEqual(mean_average_precision([]), 0.0)
        self.assertEqual(mean_f1([]), 0.0)


class TestGrounderIntegration(unittest.TestCase):
    def test_from_grounder_output(self):
        from harnesscad.domain.reconstruction.translate.brepground_grounding import BRepPrimitive, ground
        prims = [
            BRepPrimitive(0, "face", "cylindrical", (0.0, 0.0, 0.0), 8.0,
                          is_hole=True),
            BRepPrimitive(1, "face", "cylindrical", (0.0, 0.0, 0.0), 20.0,
                          is_hole=True),
            BRepPrimitive(2, "face", "planar", (0.0, 0.0, 5.0), 100.0),
        ]
        ranked = [p.index for p in ground("all holes", prims)]
        case = GroundingCase(ranked=ranked, truth={0, 1})
        self.assertAlmostEqual(f1(case), 1.0)
        self.assertAlmostEqual(recall_at_k(ranked, {0, 1}, 3), 1.0)


if __name__ == "__main__":
    unittest.main()
