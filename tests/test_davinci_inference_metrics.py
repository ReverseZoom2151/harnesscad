import unittest

from harnesscad.io.ingest.davinci_primitive_tokens import encode_primitive
from harnesscad.eval.bench.davinci_inference_metrics import (
    chamfer_distance, constraint_f1, evaluate, hungarian, match_primitives,
    primitive_f1, primitive_true_positives, sample_primitive, token_accuracy,
    token_cost,
)


class TestHungarian(unittest.TestCase):
    def test_simple_optimal(self):
        cost = [[4, 1, 3], [2, 0, 5], [3, 2, 2]]
        a = hungarian(cost)
        total = sum(cost[i][a[i]] for i in range(3))
        self.assertEqual(total, 5)   # 4+0+... optimal is 1+2+2=5? verify below

    def test_optimal_value_bruteforce(self):
        import itertools
        cost = [[7, 2, 9], [6, 4, 3], [5, 8, 1]]
        a = hungarian(cost)
        got = sum(cost[i][a[i]] for i in range(3))
        best = min(sum(cost[i][p[i]] for i in range(3))
                   for p in itertools.permutations(range(3)))
        self.assertEqual(got, best)

    def test_rectangular(self):
        cost = [[1, 2, 3], [4, 1, 5]]
        a = hungarian(cost)
        self.assertEqual(len(a), 2)

    def test_empty(self):
        self.assertEqual(hungarian([]), [])


class TestMatching(unittest.TestCase):
    def test_identity_match(self):
        prims = [encode_primitive("line", (0.1, 0.2, 0.3, 0.4)),
                 encode_primitive("circle", (0.5, 0.5, 0.2))]
        mapping = match_primitives(prims, prims)
        self.assertEqual(mapping, {0: 0, 1: 1})

    def test_permuted_match(self):
        a = encode_primitive("line", (0.1, 0.2, 0.3, 0.4))
        b = encode_primitive("point", (0.7, 0.7))
        mapping = match_primitives([b, a], [a, b])
        self.assertEqual(mapping[0], 1)
        self.assertEqual(mapping[1], 0)

    def test_token_cost(self):
        a = encode_primitive("line", (0.1, 0.2, 0.3, 0.4))
        self.assertEqual(token_cost(a, a), 0)


class TestPrimitiveMetrics(unittest.TestCase):
    def test_perfect_accuracy_and_f1(self):
        prims = [encode_primitive("line", (0.1, 0.2, 0.3, 0.4)),
                 encode_primitive("arc", (0.1, 0.2, 0.3, 0.4, 0.5, 0.6))]
        self.assertEqual(token_accuracy(prims, prims), 1.0)
        pf1 = primitive_f1(prims, prims)
        self.assertEqual(pf1["f1"], 1.0)

    def test_tolerance_5_units(self):
        gt = [encode_primitive("point", (0.5, 0.5))]
        # shift by ~4 quant units -> still TP; by ~10 -> not
        near = [encode_primitive("point", (0.5 + 4 / 64, 0.5))]
        far = [encode_primitive("point", (0.5 + 12 / 64, 0.5))]
        self.assertEqual(primitive_f1(near, gt)["tp"], 1)
        self.assertEqual(primitive_f1(far, gt)["tp"], 0)

    def test_type_mismatch_not_tp(self):
        gt = [encode_primitive("line", (0.1, 0.2, 0.3, 0.4))]
        pred = [encode_primitive("point", (0.1, 0.2))]
        self.assertEqual(primitive_f1(pred, gt)["tp"], 0)

    def test_missing_prediction_recall(self):
        gt = [encode_primitive("point", (0.1, 0.1)),
              encode_primitive("point", (0.9, 0.9))]
        pred = [encode_primitive("point", (0.1, 0.1))]
        pf1 = primitive_f1(pred, gt)
        self.assertEqual(pf1["tp"], 1)
        self.assertAlmostEqual(pf1["recall"], 0.5)
        self.assertAlmostEqual(pf1["precision"], 1.0)


class TestConstraintF1(unittest.TestCase):
    def test_constraint_tp_requires_primitive_tp(self):
        gt = [encode_primitive("line", (0.1, 0.2, 0.3, 0.4)),
              encode_primitive("line", (0.5, 0.6, 0.7, 0.8))]
        pred = list(gt)
        mapping = match_primitives(pred, gt)
        tps = primitive_true_positives(pred, gt, mapping)
        gt_cons = [("parallel", 0, 4, 1, 4)]
        pred_cons = [("parallel", 0, 4, 1, 4)]
        cf1 = constraint_f1(pred_cons, gt_cons, tps)
        self.assertEqual(cf1["f1"], 1.0)

    def test_constraint_fails_when_primitive_wrong(self):
        gt = [encode_primitive("line", (0.1, 0.2, 0.3, 0.4)),
              encode_primitive("line", (0.5, 0.6, 0.7, 0.8))]
        # second predicted primitive badly wrong -> not a primitive TP
        pred = [encode_primitive("line", (0.1, 0.2, 0.3, 0.4)),
                encode_primitive("point", (0.9, 0.9))]
        mapping = match_primitives(pred, gt)
        tps = primitive_true_positives(pred, gt, mapping)
        cf1 = constraint_f1([("parallel", 0, 4, 1, 4)],
                            [("parallel", 0, 4, 1, 4)], tps)
        self.assertEqual(cf1["tp"], 0)

    def test_undirected_canonicalisation(self):
        gt = [encode_primitive("line", (0.1, 0.2, 0.3, 0.4)),
              encode_primitive("line", (0.5, 0.6, 0.7, 0.8))]
        pred = list(gt)
        tps = primitive_true_positives(pred, gt)
        # predicted with endpoints swapped must still match
        cf1 = constraint_f1([("parallel", 1, 4, 0, 4)],
                            [("parallel", 0, 4, 1, 4)], tps)
        self.assertEqual(cf1["tp"], 1)


class TestChamfer(unittest.TestCase):
    def test_zero_for_identical(self):
        prims = [encode_primitive("circle", (0.5, 0.5, 0.2))]
        self.assertAlmostEqual(chamfer_distance(prims, prims), 0.0, places=6)

    def test_positive_for_shifted(self):
        a = [encode_primitive("point", (0.1, 0.1))]
        b = [encode_primitive("point", (0.9, 0.9))]
        self.assertGreater(chamfer_distance(a, b), 0.0)

    def test_sample_counts(self):
        self.assertEqual(len(sample_primitive(encode_primitive("point", (0.5, 0.5)))), 1)
        self.assertEqual(len(sample_primitive(encode_primitive("none", ()))), 0)
        self.assertEqual(len(sample_primitive(encode_primitive("circle", (0.5, 0.5, 0.2)), samples=6)), 6)


class TestEvaluate(unittest.TestCase):
    def test_bundle_perfect(self):
        gt = [encode_primitive("line", (0.1, 0.2, 0.3, 0.4)),
              encode_primitive("line", (0.5, 0.6, 0.7, 0.8))]
        cons = [("parallel", 0, 4, 1, 4)]
        res = evaluate(gt, gt, cons, cons)
        self.assertEqual(res["accuracy"], 1.0)
        self.assertEqual(res["primitive"]["f1"], 1.0)
        self.assertEqual(res["constraint"]["f1"], 1.0)
        self.assertAlmostEqual(res["chamfer"], 0.0, places=6)


if __name__ == "__main__":
    unittest.main()
