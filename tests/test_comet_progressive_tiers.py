"""Tests for token-budgeted progressive-tier read planning."""

import unittest

from context.comet_progressive_tiers import (
    DETAIL,
    RAW,
    SUMMARY,
    TierProfile,
    estimate_tokens,
    make_profile,
    plan_reads,
)


class TestEstimateTokens(unittest.TestCase):
    def test_ceil_division(self):
        self.assertEqual(estimate_tokens("abcd", 4), 1)
        self.assertEqual(estimate_tokens("abcde", 4), 2)

    def test_empty_is_zero(self):
        self.assertEqual(estimate_tokens("", 4), 0)

    def test_bad_ratio_raises(self):
        with self.assertRaises(ValueError):
            estimate_tokens("x", 0)


class TestMakeProfile(unittest.TestCase):
    def test_missing_tiers_fall_back_upward(self):
        p = make_profile("n", "a" * 8, chars_per_token=4)
        # detail and raw absent -> both fall back to summary cost (2)
        self.assertEqual(p.costs, (2, 2, 2))

    def test_costs_monotonic_non_decreasing(self):
        p = make_profile("n", "a" * 4, "b" * 40, "c" * 400, chars_per_token=4)
        self.assertEqual(p.costs, (1, 10, 100))


class TestProfileValidation(unittest.TestCase):
    def test_bad_costs_length(self):
        with self.assertRaises(ValueError):
            TierProfile("n", (1, 2))

    def test_negative_cost(self):
        with self.assertRaises(ValueError):
            TierProfile("n", (1, -2, 3))


class TestPlanReads(unittest.TestCase):
    def test_all_summaries_fit_then_deepen(self):
        profs = [
            TierProfile("a", (2, 5, 20), priority=0),
            TierProfile("b", (2, 5, 20), priority=1),
        ]
        plan = plan_reads(profs, budget=100)
        # Both admitted, budget ample -> both reach raw.
        self.assertEqual(plan.tier_of("a"), RAW)
        self.assertEqual(plan.tier_of("b"), RAW)
        self.assertEqual(plan.dropped, [])
        self.assertLessEqual(plan.total_tokens, plan.budget)

    def test_tight_budget_admits_only_summaries(self):
        profs = [
            TierProfile("a", (2, 50, 90), priority=0),
            TierProfile("b", (2, 50, 90), priority=1),
        ]
        plan = plan_reads(profs, budget=4)
        self.assertEqual(plan.tier_of("a"), SUMMARY)
        self.assertEqual(plan.tier_of("b"), SUMMARY)
        self.assertEqual(plan.total_tokens, 4)

    def test_priority_order_admission_and_drop(self):
        profs = [
            TierProfile("hi", (3, 9, 9), priority=0),
            TierProfile("lo", (3, 9, 9), priority=5),
        ]
        # Budget fits only one summary.
        plan = plan_reads(profs, budget=3)
        self.assertEqual(plan.tier_of("hi"), SUMMARY)
        self.assertIsNone(plan.tier_of("lo"))
        self.assertEqual(plan.dropped, ["lo"])

    def test_risk_node_deepened_first(self):
        # Two nodes, only enough spare budget to deepen one to raw.
        profs = [
            TierProfile("plain", (2, 6, 6), priority=0, risk=False),
            TierProfile("risky", (2, 6, 6), priority=1, risk=True),
        ]
        # summaries cost 4; only 4 spare tokens -> one node can reach detail
        # (delta 4), and raw is free from there. Risk node wins the escalation.
        plan = plan_reads(profs, budget=4 + 4)
        self.assertEqual(plan.tier_of("risky"), RAW)
        self.assertEqual(plan.tier_of("plain"), SUMMARY)

    def test_pinned_claims_budget_first(self):
        profs = [
            TierProfile("hot", (3, 8, 20), priority=0, pinned=False),
            TierProfile("pin", (3, 8, 20), priority=9, pinned=True),
        ]
        # Only 3 tokens: pinned wins even though its priority is worse.
        plan = plan_reads(profs, budget=3)
        self.assertEqual(plan.tier_of("pin"), SUMMARY)
        self.assertIsNone(plan.tier_of("hot"))

    def test_pinned_overflow_drops_lowest_priority_pin(self):
        profs = [
            TierProfile("p1", (3, 8, 20), priority=0, pinned=True),
            TierProfile("p2", (3, 8, 20), priority=1, pinned=True),
        ]
        plan = plan_reads(profs, budget=3)
        self.assertEqual(plan.tier_of("p1"), SUMMARY)
        self.assertIsNone(plan.tier_of("p2"))
        self.assertEqual(plan.dropped, ["p2"])

    def test_max_tier_cap(self):
        profs = [TierProfile("a", (2, 5, 500), priority=0)]
        plan = plan_reads(profs, budget=1000, max_tier=DETAIL)
        self.assertEqual(plan.tier_of("a"), DETAIL)

    def test_never_exceeds_budget(self):
        profs = [
            TierProfile("a", (5, 40, 200), priority=0),
            TierProfile("b", (5, 40, 200), priority=1),
            TierProfile("c", (5, 40, 200), priority=2),
        ]
        plan = plan_reads(profs, budget=97)
        self.assertLessEqual(plan.total_tokens, 97)

    def test_deterministic_repeat(self):
        profs = [
            TierProfile("a", (5, 40, 200), priority=0, risk=True),
            TierProfile("b", (5, 40, 200), priority=1),
            TierProfile("c", (5, 40, 200), priority=1),
        ]
        p1 = plan_reads(profs, budget=120).to_dict()
        p2 = plan_reads(list(reversed(profs)), budget=120).to_dict()
        self.assertEqual(p1, p2)

    def test_duplicate_node_id_raises(self):
        with self.assertRaises(ValueError):
            plan_reads([TierProfile("a", (1, 2, 3)), TierProfile("a", (1, 2, 3))], 10)

    def test_negative_budget_raises(self):
        with self.assertRaises(ValueError):
            plan_reads([], -1)


if __name__ == "__main__":
    unittest.main()
