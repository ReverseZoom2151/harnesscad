"""Tests for run-level aggregation and leaderboard reporting."""
import unittest

from harnesscad.eval.bench.harness.run_summary import (
    SampleResult,
    build_run_summary,
    leaderboard_row,
    parse_result,
    rank_leaderboard,
)


def _results():
    return [
        SampleResult("101", "valid", 0.8, "generation"),
        SampleResult("102", "invalid", 0.9, "generation"),   # reported score ignored
        SampleResult("103", "missing", 0.0, "generation"),
        SampleResult("201", "valid", 0.6, "editing"),
    ]


class TestSampleResult(unittest.TestCase):
    def test_invalid_sample_scores_zero(self):
        self.assertEqual(SampleResult("x", "invalid", 0.99).effective_score, 0.0)

    def test_missing_sample_scores_zero(self):
        self.assertEqual(SampleResult("x", "missing", 0.99).effective_score, 0.0)

    def test_unknown_status_raises(self):
        with self.assertRaises(ValueError):
            SampleResult("x", "crashed", 0.0)

    def test_parse_missing_payload(self):
        r = parse_result("101", None)
        self.assertEqual(r.status, "missing")
        self.assertEqual(r.cad_score, 0.0)
        self.assertEqual(r.task_type, "generation")

    def test_parse_null_score(self):
        r = parse_result("101", {"status": "valid", "cad_score": None})
        self.assertEqual(r.effective_score, 0.0)

    def test_parse_task_type_override(self):
        r = parse_result("201", {"status": "valid", "cad_score": 0.5}, task_type="editing")
        self.assertEqual(r.task_type, "editing")


class TestRunSummary(unittest.TestCase):
    def test_aggregate_includes_zeros(self):
        summary = build_run_summary(_results())
        # (0.8 + 0 + 0 + 0.6) / 4
        self.assertAlmostEqual(summary["aggregate_score"], 0.35, places=6)

    def test_counts(self):
        summary = build_run_summary(_results())
        self.assertEqual(summary["n_samples"], 4)
        self.assertEqual(summary["n_valid"], 2)
        self.assertEqual(summary["n_invalid"], 1)
        self.assertEqual(summary["n_missing"], 1)
        self.assertAlmostEqual(summary["validity_rate"], 0.5)

    def test_per_task_buckets(self):
        summary = build_run_summary(_results())
        gen = summary["per_task_scores"]["generation"]
        self.assertEqual(gen["n_samples"], 3)
        self.assertAlmostEqual(gen["score"], round(0.8 / 3, 4))
        self.assertAlmostEqual(gen["validity_rate"], round(1 / 3, 4))
        self.assertAlmostEqual(summary["per_task_scores"]["editing"]["score"], 0.6)

    def test_unknown_task_type_gets_its_own_bucket_after_known_ones(self):
        results = _results() + [SampleResult("301", "valid", 1.0, "assembly")]
        summary = build_run_summary(results)
        self.assertEqual(
            list(summary["score_by_task_type"]), ["generation", "editing", "assembly"]
        )
        self.assertAlmostEqual(summary["per_task_scores"]["assembly"]["score"], 1.0)

    def test_per_sample_scores_sorted_and_zeroed(self):
        summary = build_run_summary(_results())
        self.assertEqual(list(summary["per_sample_scores"]), ["101", "102", "103", "201"])
        self.assertEqual(summary["per_sample_scores"]["102"]["cad_score"], 0.0)
        self.assertEqual(summary["per_sample_scores"]["102"]["status"], "invalid")

    def test_empty_run(self):
        summary = build_run_summary([])
        self.assertEqual(summary["aggregate_score"], 0.0)
        self.assertEqual(summary["n_samples"], 0)
        self.assertEqual(summary["score_by_task_type"], {})

    def test_deterministic(self):
        self.assertEqual(build_run_summary(_results()), build_run_summary(_results()))


class TestLeaderboard(unittest.TestCase):
    def test_row_publishes_unvalidated(self):
        row = leaderboard_row("agent v1", build_run_summary(_results()))
        self.assertFalse(row.validated)
        self.assertEqual(row.n_samples, 4)
        self.assertIn("aggregate_score", row.to_dict())

    def test_ranking_order_and_tie_break(self):
        summary = build_run_summary(_results())
        rows = [
            leaderboard_row("b", summary),
            leaderboard_row("a", summary),
            leaderboard_row("c", build_run_summary([SampleResult("1", "valid", 1.0)])),
        ]
        ranked = rank_leaderboard(rows)
        self.assertEqual([r.submission for r in ranked], ["c", "a", "b"])


if __name__ == "__main__":
    unittest.main()
