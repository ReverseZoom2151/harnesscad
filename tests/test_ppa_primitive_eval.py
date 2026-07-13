"""Tests for the PPA primitive-prediction evaluation protocol."""

import math
import unittest

from harnesscad.domain.reconstruction import ppa_primitive as pp
from harnesscad.domain.reconstruction import ppa_quantization as pq
from harnesscad.eval.bench import ppa_primitive_eval as ev


def _norm(sketch):
    return pq.normalize_sketch(sketch)[0]


class TestMatching(unittest.TestCase):
    def test_identity_match_permuted(self):
        a = pp.line((0, 0), (1, 0))
        b = pp.circle((5, 5), 2)
        c = pp.point((3, 3))
        gt = pp.Sketch([a, b, c])
        pred = pp.Sketch([c, a, b])  # same primitives, shuffled
        sigma = ev.match_primitives(gt, pred)
        self.assertEqual(len(sigma), 3)
        for i, j in sigma.items():
            self.assertEqual(list(gt)[i], list(pred)[j])

    def test_more_predictions_than_gt(self):
        gt = pp.Sketch([pp.line((0, 0), (1, 0))])
        pred = pp.Sketch([pp.circle((9, 9), 1), pp.line((0, 0), (1, 0))])
        sigma = ev.match_primitives(gt, pred)
        self.assertEqual(len(sigma), 1)
        self.assertEqual(list(pred)[sigma[0]].ptype, pp.LINE)

    def test_empty_gt(self):
        self.assertEqual(ev.match_primitives(pp.Sketch([]), pp.Sketch([pp.point((0, 0))])), {})

    def test_cost_ordering(self):
        gt = pp.line((0, 0), (1, 0))
        close = pp.line((0, 0), (1, 0.01))
        far = pp.circle((5, 5), 3)
        self.assertLess(ev.match_cost(gt, close), ev.match_cost(gt, far))


class TestAccuracies(unittest.TestCase):
    def setUp(self):
        self.gt = pp.Sketch([
            pp.line((0, 0), (10, 0), flag=True),
            pp.circle((5, 5), 3, flag=False),
            pp.point((8, 8), flag=True),
        ])

    def test_perfect_prediction(self):
        r = ev.evaluate(_norm(self.gt), _norm(self.gt))
        self.assertEqual(r["ACC_ptype"], 1.0)
        self.assertEqual(r["ACC_flag"], 1.0)
        self.assertEqual(r["ACC_ppar"], 1.0)
        self.assertLess(r["chamfer"], 1e-9)

    def test_type_error_lowers_ptype(self):
        pred = pp.Sketch([
            pp.circle((0, 0), 1, flag=True),   # wrong type vs GT line
            pp.circle((5, 5), 3, flag=False),
            pp.point((8, 8), flag=True),
        ])
        acc = ev.primitive_type_accuracy(_norm(self.gt), _norm(pred))
        self.assertAlmostEqual(acc, 2 / 3)

    def test_flag_error_lowers_flag_acc(self):
        pred = pp.Sketch([
            pp.line((0, 0), (10, 0), flag=False),  # flipped flag
            pp.circle((5, 5), 3, flag=False),
            pp.point((8, 8), flag=True),
        ])
        acc = ev.boolean_flag_accuracy(_norm(self.gt), _norm(pred))
        self.assertAlmostEqual(acc, 2 / 3)

    def test_param_tolerance(self):
        # perturb the line endpoint by a hair -> within eta=1 level, still correct
        gt_n = _norm(self.gt)
        # rebuild a slightly perturbed predicted sketch from GT normalised coords
        perturbed = []
        for prim in gt_n:
            if prim.ptype == pp.LINE:
                p = list(prim.params)
                p[2] += 0.3 / 63  # under one level
                perturbed.append(pp.Primitive(pp.LINE, prim.flag, tuple(p)))
            else:
                perturbed.append(prim)
        acc = ev.parameter_accuracy(gt_n, pp.Sketch(perturbed), eta=1)
        self.assertAlmostEqual(acc, 1.0)

    def test_param_large_error_fails(self):
        gt_n = _norm(self.gt)
        perturbed = []
        for prim in gt_n:
            if prim.ptype == pp.LINE:
                p = list(prim.params)
                p[2] -= 0.5  # ~32 levels off -> fails tolerance
                perturbed.append(pp.Primitive(pp.LINE, prim.flag, tuple(p)))
            else:
                perturbed.append(prim)
        acc = ev.parameter_accuracy(gt_n, pp.Sketch(perturbed), eta=1)
        self.assertAlmostEqual(acc, 2 / 3)

    def test_empty_gt_accuracy_one(self):
        self.assertEqual(ev.primitive_type_accuracy(pp.Sketch([]), pp.Sketch([])), 1.0)


class TestChamfer(unittest.TestCase):
    def test_shifted_sketch_positive_cd(self):
        gt = pp.Sketch([pp.line((0, 0), (10, 0))])
        pred = pp.Sketch([pp.line((0, 5), (10, 5))])
        cd = ev.sketch_chamfer(gt, pred)
        self.assertAlmostEqual(cd, 10.0, places=6)  # 5 up + 5 back

    def test_empty_raises(self):
        with self.assertRaises(ValueError):
            ev.sketch_chamfer(pp.Sketch([]), pp.Sketch([pp.point((0, 0))]))

    def test_evaluate_reports_all_fields(self):
        gt = _norm(pp.Sketch([pp.line((0, 0), (1, 1))]))
        r = ev.evaluate(gt, gt)
        for key in ("matched", "num_gt", "num_pred", "ACC_ptype", "ACC_flag",
                    "ACC_ppar", "chamfer"):
            self.assertIn(key, r)


if __name__ == "__main__":
    unittest.main()
