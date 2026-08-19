"""Hand-computed checks for the SUC / Pass@1 / AVG Re linked report.

The fixture below is small enough to score by hand:

    t0  [ok]                       -> succeeds at index 0 (0 retries)
    t1  [fail, fail, ok]           -> succeeds at index 2 (2 retries)
    t2  [fail, ok]                 -> succeeds at index 1 (1 retry)
    t3  [fail, fail, fail]         -> never succeeds

so over the full trace SUC = 3/4 = 0.75, Pass@1 = 1/4 = 0.25 and
AVG Re = (0 + 2 + 1) / 3 = 1.0 (successful tasks only).
"""

from __future__ import annotations

import unittest

from harnesscad.eval.bench.harness import correction_budget as cb


def _ok():
    return {"executed": True, "valid": True}


def _ran_but_broken():
    return {"executed": True, "valid": False}


def _crashed():
    return {"executed": False, "valid": False}


TRACES = (
    {"task_id": "t0", "attempts": [_ok()]},
    {"task_id": "t1", "attempts": [_crashed(), _ran_but_broken(), _ok()]},
    {"task_id": "t2", "attempts": [_ran_but_broken(), _ok()]},
    {"task_id": "t3", "attempts": [_crashed(), _crashed(), _ran_but_broken()]},
)


class AttemptTest(unittest.TestCase):
    def test_executing_without_being_valid_is_not_a_success(self):
        self.assertFalse(cb.attempt_succeeded(_ran_but_broken()))
        self.assertFalse(cb.attempt_succeeded(_crashed()))
        self.assertTrue(cb.attempt_succeeded(_ok()))

    def test_an_unchecked_attempt_is_refused_not_guessed(self):
        with self.assertRaises(ValueError):
            cb.attempt_succeeded({"executed": True})

    def test_first_success_index_is_the_retry_count(self):
        self.assertEqual(cb.first_success_index(TRACES[0]["attempts"]), 0)
        self.assertEqual(cb.first_success_index(TRACES[1]["attempts"]), 2)
        self.assertEqual(cb.first_success_index(TRACES[2]["attempts"]), 1)
        self.assertIsNone(cb.first_success_index(TRACES[3]["attempts"]))

    def test_a_success_outside_the_budget_does_not_count(self):
        # budget=1 allows attempts 0 and 1 only; t1 only succeeds at index 2.
        self.assertIsNone(cb.first_success_index(TRACES[1]["attempts"], budget=1))
        self.assertEqual(cb.first_success_index(TRACES[2]["attempts"], budget=1), 1)


class StabilityReportTest(unittest.TestCase):
    def test_the_triple_over_the_full_trace(self):
        r = cb.stability_report(TRACES)
        self.assertEqual(r.n_tasks, 4)
        self.assertEqual(r.n_success, 3)
        self.assertEqual(r.suc, 0.75)
        self.assertEqual(r.pass_at_1, 0.25)
        self.assertEqual(r.avg_retries, 1.0)

    def test_a_tighter_budget_lowers_suc_and_lowers_avg_retries(self):
        # budget=1: t0 (idx 0) and t2 (idx 1) succeed, t1 and t3 do not.
        r = cb.stability_report(TRACES, budget=1)
        self.assertEqual(r.suc, 0.5)
        self.assertEqual(r.pass_at_1, 0.25)
        self.assertEqual(r.avg_retries, 0.5)

    def test_avg_retries_averages_successes_only(self):
        # Dropping the never-solved task must not move AVG Re at all.
        full = cb.stability_report(TRACES)
        solved_only = cb.stability_report(TRACES[:3])
        self.assertEqual(full.avg_retries, solved_only.avg_retries)
        self.assertEqual(solved_only.suc, 1.0)

    def test_a_task_with_no_attempts_still_counts_in_the_denominator(self):
        r = cb.stability_report([{"task_id": "x", "attempts": []},
                                 {"task_id": "y", "attempts": [_ok()]}])
        self.assertEqual(r.n_tasks, 2)
        self.assertEqual(r.suc, 0.5)

    def test_pass_at_one_can_never_exceed_suc(self):
        with self.assertRaises(ValueError):
            cb.StabilityReport(n_tasks=2, n_success=1, n_first_attempt=2,
                               suc=0.5, pass_at_1=1.0, avg_retries=0.0)

    def test_an_empty_task_set_is_undefined(self):
        with self.assertRaises(ValueError):
            cb.stability_report([])

    def test_summary_prints_all_three_together(self):
        text = cb.stability_report(TRACES).summary()
        self.assertIn("SUC 0.7500", text)
        self.assertIn("Pass@1 0.2500", text)
        self.assertIn("AVG Re 1.0000", text)

    def test_no_public_helper_returns_avg_retries_alone(self):
        """AVG Re is reachable only through the linked report, by design."""
        self.assertNotIn("avg_retries", cb.__all__)
        self.assertFalse([n for n in cb.__all__ if "retr" in n.lower()])


class ComparisonTest(unittest.TestCase):
    """The paper's caveat: more retries alongside more success is not a loss."""

    def test_rising_suc_with_rising_retries_is_not_called_a_regression(self):
        before = cb.stability_report(TRACES, budget=1)   # SUC 0.50, AVG Re 0.50
        after = cb.stability_report(TRACES)              # SUC 0.75, AVG Re 1.00
        verdict = cb.compare(before, after)
        self.assertEqual(verdict["suc_delta"], 0.25)
        self.assertEqual(verdict["avg_retries_delta"], 0.5)
        self.assertEqual(verdict["verdict"], "more_success_more_retries")
        self.assertIn(verdict["verdict"], cb.VERDICTS)

    def test_falling_suc_is_the_headline_whatever_the_retries_did(self):
        before = cb.stability_report(TRACES)
        after = cb.stability_report(TRACES, budget=1)
        self.assertEqual(cb.compare(before, after)["verdict"], "less_success")

    def test_tied_suc_makes_retries_comparable_again(self):
        cheap = cb.stability_report(
            [{"task_id": "a", "attempts": [_ok()]},
             {"task_id": "b", "attempts": [_crashed(), _ok()]}])
        dear = cb.stability_report(
            [{"task_id": "a", "attempts": [_ok()]},
             {"task_id": "b", "attempts": [_crashed(), _crashed(), _ok()]}])
        self.assertEqual(cheap.suc, dear.suc)
        self.assertEqual(cheap.avg_retries, 0.5)
        self.assertEqual(dear.avg_retries, 1.0)
        self.assertEqual(cb.compare(cheap, dear)["verdict"],
                         "same_success_more_retries")
        self.assertEqual(cb.compare(dear, cheap)["verdict"],
                         "same_success_fewer_retries")
        self.assertEqual(cb.compare(cheap, cheap)["verdict"],
                         "same_success_same_retries")

    def test_paper_table_one_reproduces_the_reported_verdict(self):
        """arXiv:2605.19748 Table 1: both-memory beats wo-memory on SUC while
        spending MORE retries (0.3467 vs 0.3018)."""
        wo = cb.StabilityReport(n_tasks=10000, n_success=9494, n_first_attempt=7528,
                                suc=0.9494, pass_at_1=0.7528, avg_retries=0.3018)
        both = cb.StabilityReport(n_tasks=10000, n_success=9950, n_first_attempt=8300,
                                  suc=0.9950, pass_at_1=0.8300, avg_retries=0.3467)
        verdict = cb.compare(wo, both)
        self.assertEqual(verdict["verdict"], "more_success_more_retries")
        self.assertGreater(verdict["suc_delta"], 0.0)
        self.assertGreater(verdict["avg_retries_delta"], 0.0)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
