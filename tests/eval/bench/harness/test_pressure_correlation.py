"""~200 intrinsic bench metrics, and until now zero correlation with `solved`."""

from __future__ import annotations

import os
import unittest

from harnesscad.eval.bench.harness import pressure_correlation as pc

RESULTS = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "..", "..", "..", "..", "assets", "pressure", "results.json")


class TestFeatures(unittest.TestCase):
    def test_an_ungraded_attempt_yields_nothing(self):
        self.assertIsNone(pc.features({"grade": None}, {}))

    def test_features_are_intrinsic_and_complete(self):
        rec = {
            "grade": {
                "applied": 3, "apply_ok": True, "built": True,
                "measure": {"bbox": [50.0, 30.0, 6.0], "volume": 4500.0,
                            "validity": {"is_valid": True, "manifold": True,
                                         "watertight": True, "genus": 0,
                                         "euler_characteristic": 2, "issues": 0,
                                         "solid_present": True}},
                "diagnostics": [{"severity": "error", "code": "x"}],
                "fleet_actionable": [], "fleet_caught": False, "solved": True,
            },
            "ops": [{"op": "new_sketch"}, {"op": "extrude"}, {"op": "hole"}],
            "raw": "[]", "parse_ok": True, "attempt": 1, "seconds": 0.5,
        }
        f = pc.features(rec, {})
        self.assertEqual(f["n_ops"], 3.0)
        self.assertEqual(f["n_op_hole"], 1.0)
        self.assertEqual(f["bbox_volume"], 9000.0)
        self.assertEqual(f["fill_ratio"], 0.5)
        self.assertEqual(f["aspect_ratio"], 50.0 / 6.0)
        self.assertEqual(f["n_error_diags"], 1.0)

    def test_a_zero_volume_bbox_does_not_divide_by_zero(self):
        f = pc.features({"grade": {"measure": {"bbox": [0, 0, 0], "volume": 0}},
                         "ops": []}, {})
        self.assertEqual(f["fill_ratio"], 0.0)
        self.assertEqual(f["aspect_ratio"], 0.0)


class TestCorrelate(unittest.TestCase):
    def test_a_perfect_predictor_scores_one(self):
        xs = [{"m": float(i)} for i in range(6)]
        ys = [0.0, 0.0, 0.0, 1.0, 1.0, 1.0]
        (name, r, ar, distinct), = pc.correlate(xs, ys)
        self.assertEqual(name, "m")
        self.assertGreater(r, 0.8)

    def test_a_constant_metric_is_noise_by_construction(self):
        xs = [{"m": 1.0} for _ in range(6)]
        ys = [0.0, 1.0, 0.0, 1.0, 0.0, 1.0]
        (_, r, _, distinct), = pc.correlate(xs, ys)
        self.assertEqual(r, 0.0)
        self.assertEqual(distinct, 1)

    def test_results_are_sorted_by_absolute_r(self):
        xs = [{"weak": float(i % 2), "strong": float(i)} for i in range(8)]
        ys = [float(i >= 4) for i in range(8)]
        names = [t[0] for t in pc.correlate(xs, ys)]
        self.assertEqual(names[0], "strong")


class TestTheRealCorpus(unittest.TestCase):
    def test_the_pressure_corpus_yields_graded_attempts(self):
        if not os.path.exists(RESULTS):
            self.skipTest("assets/pressure/results.json not present")
        import json
        with open(RESULTS, "r", encoding="utf-8") as fh:
            data = json.load(fh)
        xs, ys = pc.rows(data)
        self.assertGreater(len(xs), 100)
        self.assertEqual(len(xs), len(ys))
        scored = pc.correlate(xs, ys)
        by_name = {t[0]: t for t in scored}
        # THE FINDING, pinned: the kernel-validity family NEVER VARIES across 208
        # graded attempts. Every built part is valid, manifold and watertight, and
        # solved and unsolved parts are indistinguishable by any of them. A metric
        # that cannot vary cannot predict, and `brep-validity` is a MODEL-FACING
        # verifier reporting exactly this family.
        for dead in ("is_valid", "manifold", "watertight", "solid_present"):
            self.assertEqual(by_name[dead][3], 1,
                             "%s varied; the finding has changed and the report "
                             "must be re-run" % dead)


if __name__ == "__main__":
    unittest.main()
