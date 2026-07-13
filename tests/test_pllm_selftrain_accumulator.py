import unittest

from harnesscad.data.dataengine.selftrain.pllm_selftrain_accumulator import (
    PseudoLabelPool, detect_drift, label_efficiency, run_selftraining,
)


def _rec(sid, cd, prog=None, length=5):
    return {"shape_id": sid, "program": prog or ("p_" + sid),
            "chamfer": cd, "length": length}


class TestPool(unittest.TestCase):
    def test_add_new(self):
        p = PseudoLabelPool()
        out = p.add_round(0, [_rec("a", 0.5), _rec("b", 0.3)])
        self.assertEqual(out["added"], 2)
        self.assertEqual(len(p), 2)

    def test_improve_replaces(self):
        p = PseudoLabelPool()
        p.add_round(0, [_rec("a", 0.5)])
        out = p.add_round(1, [_rec("a", 0.2)])
        self.assertEqual(out["improved"], 1)
        self.assertEqual(out["added"], 0)
        self.assertAlmostEqual(p.get("a")["chamfer"], 0.2)
        self.assertEqual(p.get("a")["round"], 1)

    def test_worse_kept_unchanged(self):
        p = PseudoLabelPool()
        p.add_round(0, [_rec("a", 0.2)])
        out = p.add_round(1, [_rec("a", 0.9)])
        self.assertEqual(out["unchanged"], 1)
        self.assertAlmostEqual(p.get("a")["chamfer"], 0.2)

    def test_dedup_single_per_shape(self):
        p = PseudoLabelPool()
        p.add_round(0, [_rec("a", 0.5)])
        p.add_round(1, [_rec("a", 0.4)])
        p.add_round(2, [_rec("a", 0.3)])
        self.assertEqual(len(p), 1)

    def test_mean_chamfer(self):
        p = PseudoLabelPool()
        self.assertIsNone(p.mean_chamfer())
        p.add_round(0, [_rec("a", 0.2), _rec("b", 0.4)])
        self.assertAlmostEqual(p.mean_chamfer(), 0.3)


class TestDrift(unittest.TestCase):
    def test_no_drift_when_improving(self):
        d = detect_drift([0.5, 0.4, 0.3, 0.2])
        self.assertFalse(d["drift"])

    def test_drift_when_rising(self):
        d = detect_drift([0.2, 0.3, 0.4, 0.5])
        self.assertTrue(d["drift"])
        self.assertAlmostEqual(d["increase"], 0.2)

    def test_not_enough_rounds(self):
        d = detect_drift([0.2, 0.3])
        self.assertFalse(d["drift"])

    def test_flat_not_drift(self):
        # equal values are not a strict increase beyond tol -> no drift
        d = detect_drift([0.3, 0.3, 0.3])
        self.assertFalse(d["drift"])

    def test_bad_args(self):
        with self.assertRaises(ValueError):
            detect_drift([0.1, 0.2], min_rounds=1)
        with self.assertRaises(ValueError):
            detect_drift([0.1, 0.2], tol=-1)


class TestEfficiency(unittest.TestCase):
    def test_coverage(self):
        e = label_efficiency(30, 100)
        self.assertAlmostEqual(e["coverage"], 0.3)

    def test_zero_unlabeled(self):
        e = label_efficiency(0, 0)
        self.assertEqual(e["coverage"], 0.0)

    def test_fidelity_gain(self):
        e = label_efficiency(10, 100, [0.6, 0.4, 0.2])
        self.assertAlmostEqual(e["fidelity_gain"], 0.4)
        self.assertAlmostEqual(e["gain_per_round"], 0.2)

    def test_negative_total_raises(self):
        with self.assertRaises(ValueError):
            label_efficiency(1, -1)


class TestRunSelftraining(unittest.TestCase):
    def test_improving_run_no_drift(self):
        rounds = [
            [_rec("a", 0.6), _rec("b", 0.5)],
            [_rec("a", 0.4), _rec("b", 0.3)],
            [_rec("a", 0.2), _rec("b", 0.2)],
        ]
        out = run_selftraining(rounds, total_unlabeled=10)
        self.assertEqual(out["pool_size"], 2)
        self.assertFalse(out["drift"]["drift"])
        self.assertIsNone(out["stopped_round"])
        self.assertGreater(out["efficiency"]["fidelity_gain"], 0)
        # mean chamfer decreased across rounds
        mh = out["mean_chamfer_history"]
        self.assertTrue(mh[0] > mh[-1])

    def test_drift_triggers_early_stop(self):
        # pool fidelity worsens every round -> drift detected, stop at round 2
        rounds = [
            [_rec("a", 0.1)],
            [_rec("b", 0.4)],
            [_rec("c", 0.9)],
            [_rec("d", 1.2)],
        ]
        out = run_selftraining(rounds, total_unlabeled=10, drift_min_rounds=3)
        self.assertTrue(out["drift"]["drift"])
        self.assertEqual(out["stopped_round"], 2)
        # stopped before consuming round 3 -> only a,b,c in pool
        self.assertEqual(out["pool_size"], 3)

    def test_no_stop_when_disabled(self):
        rounds = [[_rec("a", 0.1)], [_rec("b", 0.4)], [_rec("c", 0.9)]]
        out = run_selftraining(rounds, total_unlabeled=10, stop_on_drift=False)
        self.assertIsNone(out["stopped_round"])
        self.assertEqual(out["pool_size"], 3)


if __name__ == "__main__":
    unittest.main()
