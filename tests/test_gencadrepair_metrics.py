"""Tests for GenCAD-Self-Repairing feasibility / repair-success metrics."""

from __future__ import annotations

import unittest

from reconstruction.deepcad_command_spec import Command, command
from reliability.gencadrepair_metrics import (
    RepairBenchmark,
    benchmark_repair,
    evaluate_sequences,
    feasibility_rate,
    feasibility_report,
    repair_success_rate,
)


def _ext(**kw):
    params = dict(theta=0.0, phi=0.0, gamma=0.0, px=0.0, py=0.0, pz=0.0,
                  s=1.0, e1=0.5, e2=0.0, b=0.0, u=0.0)
    params.update(kw)
    return command("Ext", **params)


class TestFeasibilityRate(unittest.TestCase):
    def test_basic_rate(self):
        self.assertAlmostEqual(feasibility_rate([True, True, False, True]), 0.75)

    def test_empty_is_zero(self):
        self.assertEqual(feasibility_rate([]), 0.0)

    def test_report_counts(self):
        r = feasibility_report([True, False, False, True, True])
        self.assertEqual(r.valid, 3)
        self.assertEqual(r.invalid, 2)
        self.assertEqual(r.total, 5)
        self.assertAlmostEqual(r.rate, 0.6)


class TestRepairSuccess(unittest.TestCase):
    def test_paper_figure_65_84_percent(self):
        # Paper: fixed 532 of 808 baseline-infeasible samples.
        before = [False] * 808
        after = [True] * 532 + [False] * 276
        self.assertAlmostEqual(repair_success_rate(before, after), 532 / 808, 4)

    def test_no_infeasible_is_zero(self):
        self.assertEqual(repair_success_rate([True, True], [True, True]), 0.0)

    def test_length_mismatch_raises(self):
        with self.assertRaises(ValueError):
            repair_success_rate([True], [True, False])


class TestBenchmark(unittest.TestCase):
    def test_paper_style_benchmark(self):
        # 8515 samples, baseline 7707 valid -> 808 infeasible; repaired 8239 valid.
        before = [True] * 7707 + [False] * 808
        after = [True] * 7707 + [True] * 532 + [False] * 276
        bench = benchmark_repair(before, after)
        self.assertIsInstance(bench, RepairBenchmark)
        self.assertEqual(bench.total, 8515)
        self.assertEqual(bench.infeasible_before, 808)
        self.assertEqual(bench.fixed, 532)
        self.assertEqual(bench.regressions, 0)
        self.assertAlmostEqual(bench.baseline_rate, 7707 / 8515, 4)
        self.assertAlmostEqual(bench.repaired_rate, 8239 / 8515, 4)
        self.assertAlmostEqual(bench.repair_success_rate, 532 / 808, 4)
        self.assertGreater(bench.rate_improvement, 0.0)

    def test_regressions_counted(self):
        before = [True, False, True]
        after = [False, True, True]
        bench = benchmark_repair(before, after)
        self.assertEqual(bench.fixed, 1)
        self.assertEqual(bench.regressions, 1)

    def test_to_dict_stable(self):
        bench = benchmark_repair([False, True], [True, True])
        d = bench.to_dict()
        self.assertEqual(d["fixed"], 1)
        self.assertEqual(d["repair_success_rate"], 1.0)


class TestEvaluateSequences(unittest.TestCase):
    def test_default_predicate_uses_taxonomy(self):
        good = [Command("SOL"), command("Circle", x=0.0, y=0.0, r=0.5),
                _ext(), Command("EOS")]
        bad = [_ext(), Command("EOS")]
        report = evaluate_sequences([good, bad, good])
        self.assertEqual(report.valid, 2)
        self.assertEqual(report.invalid, 1)

    def test_custom_predicate(self):
        report = evaluate_sequences([1, 2, 3], is_feasible=lambda s: s % 2 == 1)
        self.assertEqual(report.valid, 2)


if __name__ == "__main__":
    unittest.main()
