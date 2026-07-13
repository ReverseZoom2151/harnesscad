import unittest

from harnesscad.domain.reconstruction.translate.cad2program_shape_program import (
    Bbox, ShapeProgram, make_instance,
)
from harnesscad.domain.reconstruction.evaluate.cad2program_metrics import (
    box_iou_3d, hungarian, match_primitives, reconstruction_prf,
    model_retrieval_accuracy, parameter_estimation_accuracy, evaluate,
)


class BoxIouTest(unittest.TestCase):
    def test_identical(self):
        b = Bbox(0, 0, 0, 2, 2, 2, 0)
        self.assertAlmostEqual(box_iou_3d(b, b), 1.0)

    def test_disjoint(self):
        a = Bbox(0, 0, 0, 2, 2, 2, 0)
        b = Bbox(10, 0, 0, 2, 2, 2, 0)
        self.assertEqual(box_iou_3d(a, b), 0.0)

    def test_half_overlap(self):
        # Two unit cubes overlapping half along x.
        a = Bbox(0, 0, 0, 2, 2, 2, 0)
        b = Bbox(1, 0, 0, 2, 2, 2, 0)
        # intersection = 1*2*2 = 4, union = 8+8-4 = 12 -> 1/3
        self.assertAlmostEqual(box_iou_3d(a, b), 1.0 / 3.0)

    def test_90_degree_fold(self):
        a = Bbox(0, 0, 0, 4, 2, 2, 0)
        b = Bbox(0, 0, 0, 2, 4, 2, 90)   # 90-deg rotation swaps x/y extents
        self.assertAlmostEqual(box_iou_3d(a, b), 1.0)

    def test_non_axis_angle_zero(self):
        a = Bbox(0, 0, 0, 2, 2, 2, 0)
        b = Bbox(0, 0, 0, 2, 2, 2, 45)
        self.assertEqual(box_iou_3d(a, b), 0.0)


class HungarianTest(unittest.TestCase):
    def test_identity(self):
        cost = [[0, 9, 9], [9, 0, 9], [9, 9, 0]]
        self.assertEqual(hungarian(cost), [(0, 0), (1, 1), (2, 2)])

    def test_permutation(self):
        cost = [[9, 0, 9], [9, 9, 0], [0, 9, 9]]
        pairs = dict(hungarian(cost))
        self.assertEqual(pairs, {0: 1, 1: 2, 2: 0})

    def test_rectangular(self):
        cost = [[1, 2, 3], [3, 1, 2]]   # 2 rows, 3 cols
        pairs = hungarian(cost)
        self.assertEqual(len(pairs), 2)
        total = sum(cost[r][c] for r, c in pairs)
        self.assertEqual(total, 2)   # (0,0)=1 + (1,1)=1

    def test_empty(self):
        self.assertEqual(hungarian([]), [])


def _prog(*specs):
    return ShapeProgram([make_instance(mid, box, params)
                         for mid, box, params in specs])


class MatchTest(unittest.TestCase):
    def test_perfect_match(self):
        gt = _prog(("m1", (0, 0, 0, 2, 2, 2, 0), None),
                   ("m2", (10, 0, 0, 2, 2, 2, 0), None))
        res = match_primitives(gt, gt)
        self.assertEqual(res.true_positives, 2)
        self.assertAlmostEqual(res.precision, 1.0)
        self.assertAlmostEqual(res.recall, 1.0)
        self.assertAlmostEqual(res.f1, 1.0)

    def test_below_threshold_not_tp(self):
        gt = _prog(("m1", (0, 0, 0, 2, 2, 2, 0), None))
        pred = _prog(("m1", (1.5, 0, 0, 2, 2, 2, 0), None))  # small overlap
        res = match_primitives(pred, gt)
        self.assertEqual(res.true_positives, 0)
        self.assertEqual(len(res.matches), 1)  # still matched, just not TP

    def test_extra_prediction_hurts_precision(self):
        gt = _prog(("m1", (0, 0, 0, 2, 2, 2, 0), None))
        pred = _prog(("m1", (0, 0, 0, 2, 2, 2, 0), None),
                     ("m2", (10, 0, 0, 2, 2, 2, 0), None))
        r = reconstruction_prf(pred, gt)
        self.assertAlmostEqual(r["recall"], 1.0)
        self.assertAlmostEqual(r["precision"], 0.5)

    def test_empty_both(self):
        r = reconstruction_prf(ShapeProgram(), ShapeProgram())
        self.assertAlmostEqual(r["precision"], 1.0)
        self.assertAlmostEqual(r["recall"], 1.0)


class RetrievalParamTest(unittest.TestCase):
    def test_retrieval_accuracy(self):
        gt = _prog(("m1", (0, 0, 0, 2, 2, 2, 0), None),
                   ("m2", (10, 0, 0, 2, 2, 2, 0), None))
        pred = _prog(("m1", (0, 0, 0, 2, 2, 2, 0), None),
                     ("mX", (10, 0, 0, 2, 2, 2, 0), None))  # wrong id
        r = model_retrieval_accuracy(pred, gt)
        self.assertEqual(r["considered"], 2)
        self.assertEqual(r["correct"], 1)
        self.assertAlmostEqual(r["accuracy"], 0.5)

    def test_param_all_or_nothing(self):
        gt = _prog(("m1", (0, 0, 0, 2, 2, 2, 0), {"N": 1, "BT": 18}))
        pred_good = _prog(("m1", (0, 0, 0, 2, 2, 2, 0), {"N": 1, "BT": 18}))
        pred_bad = _prog(("m1", (0, 0, 0, 2, 2, 2, 0), {"N": 1, "BT": 9}))
        self.assertAlmostEqual(
            parameter_estimation_accuracy(pred_good, gt)["accuracy"], 1.0)
        self.assertAlmostEqual(
            parameter_estimation_accuracy(pred_bad, gt)["accuracy"], 0.0)

    def test_param_only_on_retrieved(self):
        # Wrong model id -> not counted in param denominator.
        gt = _prog(("m1", (0, 0, 0, 2, 2, 2, 0), {"N": 1}))
        pred = _prog(("mX", (0, 0, 0, 2, 2, 2, 0), {"N": 1}))
        r = parameter_estimation_accuracy(pred, gt)
        self.assertEqual(r["considered"], 0)

    def test_evaluate_bundle(self):
        gt = _prog(("m1", (0, 0, 0, 2, 2, 2, 0), {"N": 1}))
        out = evaluate(gt, gt)
        self.assertIn("reconstruction", out)
        self.assertIn("retrieval", out)
        self.assertIn("parameter", out)
        self.assertAlmostEqual(out["reconstruction"]["f1"], 1.0)


if __name__ == "__main__":
    unittest.main()
