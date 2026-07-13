import unittest

from harnesscad.eval.bench.harness.llm3dmodel_run_metrics import (
    FIRST_ATTEMPT, CONVERGED, DID_NOT_CONVERGE, RunRecord, refinements,
    total_time, outcome, summarize)


def _rec(name, iters, converged, cap=50, times=None, err=None):
    return RunRecord(name=name, iterations=iters, converged=converged,
                     max_retries=cap, iter_seconds=times or [], error_type=err)


class RunRecordTests(unittest.TestCase):
    def test_rejects_zero_iterations(self):
        with self.assertRaises(ValueError):
            _rec("x", 0, True)

    def test_rejects_negative_cap(self):
        with self.assertRaises(ValueError):
            RunRecord("x", 1, True, max_retries=-1)


class DerivedMetricTests(unittest.TestCase):
    def test_refinements(self):
        self.assertEqual(refinements(_rec("a", 1, True)), 0)
        self.assertEqual(refinements(_rec("a", 3, True)), 2)

    def test_total_time(self):
        self.assertAlmostEqual(total_time(_rec("a", 2, True, times=[19.0, 23.0])),
                               42.0)

    def test_outcome_first_attempt(self):
        self.assertEqual(outcome(_rec("a", 1, True)), FIRST_ATTEMPT)

    def test_outcome_converged(self):
        self.assertEqual(outcome(_rec("a", 3, True)), CONVERGED)

    def test_outcome_did_not_converge(self):
        self.assertEqual(outcome(_rec("a", 50, False)), DID_NOT_CONVERGE)


class SummarizeTests(unittest.TestCase):
    def setUp(self):
        # Mirrors Table 1 shape: some first-attempt, some converged, some failed.
        self.records = [
            _rec("cube", 1, True, times=[19.06]),
            _rec("cylinder", 1, True, times=[20.29]),
            _rec("fillet", 2, True, times=[42.0], err="syntax"),
            _rec("hinge", 3, True, times=[53.53], err="geometric"),
            _rec("gear", 50, False, times=[836.46], err="execution"),
            _rec("frame", 50, False, times=[909.11], err="geometric"),
        ]

    def test_rates(self):
        s = summarize(self.records)
        self.assertEqual(s["n"], 6)
        self.assertAlmostEqual(s["first_attempt_rate"], 2 / 6)
        self.assertAlmostEqual(s["convergence_rate"], 4 / 6)
        self.assertEqual(s["failure_count"], 2)

    def test_refinement_stats(self):
        s = summarize(self.records)
        # converged runs have refinements 0,0,1,2 -> mean 0.75, max 2
        self.assertAlmostEqual(s["mean_refinements"], 0.75)
        self.assertEqual(s["max_refinements"], 2)

    def test_error_distribution(self):
        s = summarize(self.records)
        self.assertEqual(s["error_distribution"]["geometric"], 2)
        self.assertEqual(s["error_distribution"]["execution"], 1)

    def test_outcomes_table(self):
        s = summarize(self.records)
        self.assertEqual(s["outcomes"]["cube"], FIRST_ATTEMPT)
        self.assertEqual(s["outcomes"]["fillet"], CONVERGED)
        self.assertEqual(s["outcomes"]["gear"], DID_NOT_CONVERGE)

    def test_time_totals(self):
        s = summarize(self.records)
        self.assertAlmostEqual(s["total_time"],
                               19.06 + 20.29 + 42.0 + 53.53 + 836.46 + 909.11)

    def test_empty_rejected(self):
        with self.assertRaises(ValueError):
            summarize([])


if __name__ == "__main__":
    unittest.main()
