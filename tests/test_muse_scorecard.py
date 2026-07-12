"""Tests for bench.muse_scorecard."""

import unittest

from bench.muse_scorecard import (
    cascade_dropoff,
    evaluate_design,
    muse_scorecard,
)


def _perfect_record():
    return {
        "sandbox_success": True,
        "watertight": 1, "manifold": 1, "self_intersection_free": 1,
        "overlap_free": 1,
        "functional": 1, "robust": 1, "well_toleranced": 1,
        "manufacturable": 1, "assembly_ready": 1, "connectable": 1,
    }


class EvaluateDesignTests(unittest.TestCase):
    def test_perfect_design(self):
        r = evaluate_design(_perfect_record())
        self.assertEqual(r["geom_valid"], 1.0)
        self.assertEqual(r["final_score"], 1.0)
        self.assertEqual(r["functionality"], 1.0)

    def test_code_failure_zeros_everything(self):
        rec = _perfect_record()
        rec["sandbox_success"] = False
        r = evaluate_design(rec)
        self.assertEqual(r["sandbox_success"], 0.0)
        self.assertEqual(r["geom_valid"], 0.0)
        self.assertEqual(r["watertight"], 0.0)
        self.assertEqual(r["final_score"], 0.0)

    def test_geometry_failure_zeros_alignment(self):
        rec = _perfect_record()
        rec["overlap_free"] = 0  # one geometry check fails
        r = evaluate_design(rec)
        self.assertEqual(r["sandbox_success"], 1.0)
        self.assertEqual(r["geom_valid"], 0.0)
        # upstream geometry checks that passed are still credited
        self.assertEqual(r["watertight"], 1.0)
        # downstream alignment gated to zero
        self.assertEqual(r["functionality"], 0.0)
        self.assertEqual(r["final_score"], 0.0)

    def test_pillar_averaging(self):
        rec = _perfect_record()
        rec["robust"] = 0
        rec["manufacturable"] = 0
        r = evaluate_design(rec)
        self.assertEqual(r["functionality"], 0.5)
        self.assertEqual(r["manufacturability"], 0.5)
        self.assertEqual(r["assemblability"], 1.0)
        self.assertAlmostEqual(r["final_score"], (0.5 + 0.5 + 1.0) / 3.0)

    def test_out_of_range_raises(self):
        rec = _perfect_record()
        rec["functional"] = 2
        with self.assertRaises(ValueError):
            evaluate_design(rec)

    def test_missing_fields_default_zero(self):
        r = evaluate_design({"sandbox_success": True})
        self.assertEqual(r["geom_valid"], 0.0)
        self.assertEqual(r["final_score"], 0.0)


class ScorecardTests(unittest.TestCase):
    def test_aggregate_percent(self):
        good = _perfect_record()
        bad = _perfect_record()
        bad["sandbox_success"] = False
        sc = muse_scorecard([good, bad])
        self.assertEqual(sc["n"], 2)
        self.assertEqual(sc["sandbox_success"], 50.0)
        self.assertEqual(sc["final_score"], 50.0)

    def test_aggregate_fraction(self):
        sc = muse_scorecard([_perfect_record()], as_percent=False)
        self.assertEqual(sc["final_score"], 1.0)

    def test_empty_raises(self):
        with self.assertRaises(ValueError):
            muse_scorecard([])

    def test_funnel_monotonic_cascade(self):
        # code >= geom >= alignment must hold as in the paper's funnel.
        recs = []
        recs.append(_perfect_record())
        r2 = _perfect_record(); r2["overlap_free"] = 0; recs.append(r2)
        r3 = _perfect_record(); r3["sandbox_success"] = False; recs.append(r3)
        r4 = _perfect_record(); r4["connectable"] = 0; recs.append(r4)
        sc = muse_scorecard(recs)
        self.assertGreaterEqual(sc["sandbox_success"], sc["geom_valid"])
        self.assertGreaterEqual(sc["geom_valid"], sc["final_score"])


class CascadeTests(unittest.TestCase):
    def test_dropoff(self):
        good = _perfect_record()
        bad = _perfect_record(); bad["overlap_free"] = 0
        sc = muse_scorecard([good, bad])
        drop = cascade_dropoff(sc)
        # code 100, geom 50, final 50
        self.assertEqual(drop["code_to_geometry"], 50.0)
        self.assertEqual(drop["geometry_to_alignment"], 0.0)
        self.assertEqual(drop["code_to_alignment"], 50.0)


if __name__ == "__main__":
    unittest.main()
