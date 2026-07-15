import math
import unittest

from harnesscad.eval.judge.best_of_n import (
    Candidate,
    SampleResult,
    aggregate_run,
    normalize_to_unit_box,
    select_best,
)


class BestOfNTests(unittest.TestCase):
    def test_select_best_min_cd_max_iou(self):
        cands = [
            Candidate(True, cd=0.5, iou=0.7),
            Candidate(True, cd=0.2, iou=0.9),
            Candidate(False),
        ]
        r = select_best(cands)
        self.assertEqual(r.n_valid, 2)
        self.assertEqual(r.n_candidates, 3)
        self.assertAlmostEqual(r.best_cd, 0.2)
        self.assertAlmostEqual(r.best_iou, 0.9)

    def test_select_best_all_invalid(self):
        r = select_best([Candidate(False), Candidate(False)])
        self.assertEqual(r.n_valid, 0)
        self.assertIsNone(r.best_cd)
        self.assertIsNone(r.best_iou)
        self.assertFalse(r.any_valid)

    def test_normalize_unit_box_centers_and_scales(self):
        pts = [(0, 0, 0), (2, 0, 0), (0, 2, 0), (0, 0, 2)]
        out = normalize_to_unit_box(pts)
        # Largest extent is 2 -> scale 0.5; centre (1,1,1) maps to (0.5,0.5,0.5).
        for p in out:
            for c in p:
                self.assertGreaterEqual(c, 0.0 - 1e-9)
                self.assertLessEqual(c, 1.0 + 1e-9)
        # The bbox centre maps exactly to 0.5 in every axis.
        centre_pt = normalize_to_unit_box([(1, 1, 1), (1, 1, 1)])
        self.assertEqual(centre_pt[0], (0.5, 0.5, 0.5))

    def test_normalize_degenerate_extent(self):
        out = normalize_to_unit_box([(3, 3, 3)])
        self.assertEqual(out[0], (0.5, 0.5, 0.5))

    def test_normalize_empty_raises(self):
        with self.assertRaises(ValueError):
            normalize_to_unit_box([])

    def test_aggregate_invalidity_ratio(self):
        samples = [
            SampleResult(3, 2, 0.1, 0.9),
            SampleResult(3, 0, None, None),  # invalid sample
            SampleResult(3, 1, 0.3, 0.5),
        ]
        rep = aggregate_run(samples)
        self.assertEqual(rep.n_samples, 3)
        self.assertAlmostEqual(rep.invalidity_ratio, 1 / 3)
        self.assertAlmostEqual(rep.mean_cd, (0.1 + 0.3) / 2)
        self.assertAlmostEqual(rep.median_cd, 0.2)
        self.assertAlmostEqual(rep.mean_iou, (0.9 + 0.5) / 2)

    def test_aggregate_skip_curve(self):
        samples = [
            SampleResult(1, 1, 0.1, None),
            SampleResult(1, 1, 0.9, None),
            SampleResult(1, 0, None, None),
        ]
        rep = aggregate_run(samples, skip_max=1)
        k0, ir0, mean0 = rep.skip_curve[0]
        self.assertEqual(k0, 0)
        self.assertAlmostEqual(ir0, 1 / 3)
        self.assertAlmostEqual(mean0, 0.5)
        k1, ir1, mean1 = rep.skip_curve[1]
        # drop worst CD (0.9): ir rises by 1/3, mean over remaining {0.1}.
        self.assertAlmostEqual(ir1, 2 / 3)
        self.assertAlmostEqual(mean1, 0.1)

    def test_aggregate_empty(self):
        rep = aggregate_run([])
        self.assertEqual(rep.n_samples, 0)
        self.assertTrue(math.isnan(rep.mean_cd))

    def test_determinism(self):
        samples = [SampleResult(2, 1, 0.4, 0.6)]
        self.assertEqual(aggregate_run(samples), aggregate_run(samples))


if __name__ == "__main__":
    unittest.main()
