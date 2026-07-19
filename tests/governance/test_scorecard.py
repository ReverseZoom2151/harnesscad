"""The tri-state 'no silent green' scorecard: unknown is contagious to green
but distinct from fail, and every unknown carries its reason.

Guards the aggregator invariant ported from anvilate (scorecard.py) and its
composition with credibility_tier.
"""

import unittest

from harnesscad.governance.scorecard import (
    CheckResult,
    Scorecard,
    TriState,
    tri_state_rollup,
)
from harnesscad.governance.credibility_tier import classify_credibility


class TestReasonInvariant(unittest.TestCase):
    def test_unknown_without_reason_is_rejected(self):
        with self.assertRaises(ValueError):
            CheckResult.unknown("x", "")

    def test_fail_without_reason_is_rejected(self):
        with self.assertRaises(ValueError):
            CheckResult.failing("x", "")

    def test_pass_needs_no_reason(self):
        r = CheckResult.passing("x")
        self.assertTrue(r.passed)

    def test_unknown_is_not_passed(self):
        self.assertFalse(CheckResult.unknown("x", "not measured").passed)
        self.assertFalse(CheckResult.unknown("x", "not measured").ran)


class TestRollup(unittest.TestCase):
    def test_all_pass_is_green(self):
        card = Scorecard.of([CheckResult.passing("a"), CheckResult.passing("b")])
        self.assertIs(card.verdict, TriState.PASS)
        self.assertTrue(card.is_green)

    def test_one_unknown_among_pass_is_not_green(self):
        card = Scorecard.of([
            CheckResult.passing("a"),
            CheckResult.passing("b"),
            CheckResult.unknown("c", "could not run: backend missing"),
        ])
        self.assertFalse(card.is_green)
        self.assertIs(card.verdict, TriState.UNKNOWN)

    def test_unknown_is_distinct_from_fail(self):
        card = Scorecard.of([
            CheckResult.passing("a"),
            CheckResult.unknown("c", "not measured"),
        ])
        self.assertIsNot(card.verdict, TriState.FAIL)

    def test_fail_dominates_pass(self):
        card = Scorecard.of([
            CheckResult.passing("a"),
            CheckResult.failing("b", "below allowable"),
        ])
        self.assertIs(card.verdict, TriState.FAIL)

    def test_fail_dominates_unknown(self):
        card = Scorecard.of([
            CheckResult.failing("a", "below allowable"),
            CheckResult.unknown("b", "not measured"),
        ])
        self.assertIs(card.verdict, TriState.FAIL)

    def test_empty_is_unknown_not_green(self):
        card = Scorecard.of([])
        self.assertIs(card.verdict, TriState.UNKNOWN)
        self.assertFalse(card.is_green)

    def test_free_function_matches_scorecard(self):
        results = [CheckResult.passing("a"), CheckResult.unknown("b", "x")]
        self.assertIs(tri_state_rollup(results), Scorecard.of(results).verdict)


class TestReasonsPreserved(unittest.TestCase):
    def test_every_unknown_reason_is_surfaced(self):
        card = Scorecard.of([
            CheckResult.passing("a"),
            CheckResult.unknown("b", "mesh too large to check"),
            CheckResult.unknown("c", "S-N curve unavailable"),
        ])
        reasons = card.unknown_reasons()
        self.assertEqual(len(reasons), 2)
        for r in reasons:
            self.assertIn(": ", r)
            self.assertTrue(r.split(": ", 1)[1])

    def test_serialisation_is_deterministic(self):
        card = Scorecard.of([
            CheckResult.passing("a"),
            CheckResult.unknown("b", "x"),
        ])
        self.assertEqual(card.to_dict(), card.to_dict())


class TestCredibilityComposition(unittest.TestCase):
    def test_green_scorecard_still_bounded_by_weakest_tier(self):
        solver = classify_credibility("solver", solver_executed=True)
        critique = classify_credibility("critique")
        card = Scorecard.of([
            CheckResult.passing("stress", credibility=solver),
            CheckResult.passing("dfm", credibility=critique),
        ])
        # Green on the run-state axis, weak on the evidence-strength axis.
        self.assertTrue(card.is_green)
        self.assertEqual(card.credibility(), "critique_finding")

    def test_no_stamps_is_unverified(self):
        card = Scorecard.of([CheckResult.passing("a")])
        self.assertEqual(card.credibility(), "unverified")


class TestSelfcheck(unittest.TestCase):
    def test_selfcheck_exits_zero(self):
        from harnesscad.governance import scorecard
        self.assertEqual(scorecard.main(["--selfcheck"]), 0)


if __name__ == "__main__":
    unittest.main()
