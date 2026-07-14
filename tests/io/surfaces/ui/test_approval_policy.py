"""The MECHANICAL approval gate (ui/approval.py: ApprovalPolicy).

The three tiers, the risk indicator and the dry-run preview were all built and
every write surface ignored them. These tests pin the enforcing half:

  * ``require`` RAISES on a denial (a caller cannot forget to check a bool);
  * a headless process REFUSES tier-3 by default;
  * an unattended auto-approve must be asked for BY NAME and carry a reason;
  * every decision -- approved or denied, tier-1 through tier-3 -- is recorded.

Deterministic, stdlib-only, no kernel.
"""

import unittest

from harnesscad.core.cisp.ops import Extrude, NewSketch
from harnesscad.io.surfaces.ui.approval import (
    ApprovalDenied,
    ApprovalPolicy,
    ApprovalTier,
    HeadlessPolicy,
    RiskLevel,
)


class HeadlessRefuseTest(unittest.TestCase):
    def test_tier3_refused_when_no_human_is_attached(self):
        policy = ApprovalPolicy(surface="test")
        self.assertTrue(policy.headless_context)
        record = policy.decide("export")
        self.assertIs(record.tier, ApprovalTier.REQUIRE)
        self.assertIs(record.risk, RiskLevel.HIGH)
        self.assertFalse(record.approved)
        self.assertEqual(record.decided_by, "policy:headless-refuse")

    def test_require_raises_so_a_caller_cannot_proceed_by_accident(self):
        policy = ApprovalPolicy()
        with self.assertRaises(ApprovalDenied) as ctx:
            policy.require("delete")
        self.assertFalse(ctx.exception.record.approved)
        self.assertEqual(len(policy.audit), 1)

    def test_tier1_and_tier2_proceed_but_are_still_recorded(self):
        policy = ApprovalPolicy()
        auto = policy.require("measure")
        notify = policy.require(NewSketch())
        self.assertIs(auto.tier, ApprovalTier.AUTO)
        self.assertIs(notify.tier, ApprovalTier.NOTIFY)
        self.assertTrue(auto.approved and notify.approved)
        self.assertEqual([r["op"] for r in policy.audit_dicts()],
                         ["measure", "new_sketch"])


class HumanApproverTest(unittest.TestCase):
    def test_human_decides_tier3_and_is_named_in_the_record(self):
        asked = []
        policy = ApprovalPolicy(lambda op: asked.append(op) or False,
                                principal="alice", surface="cli")
        with self.assertRaises(ApprovalDenied):
            policy.require("export")
        self.assertEqual(asked, ["export"])
        record = policy.audit[-1]
        self.assertEqual(record.decided_by, "human:alice")
        self.assertEqual(record.surface, "cli")

    def test_human_is_not_consulted_for_tier2(self):
        asked = []
        policy = ApprovalPolicy(lambda op: asked.append(op) or False)
        record = policy.require(Extrude(sketch="sk1", distance=5.0))
        self.assertTrue(record.approved)
        self.assertEqual(asked, [])  # tier-2 never bothers the human


class ExplicitHeadlessAutoApproveTest(unittest.TestCase):
    def test_auto_approve_without_a_reason_is_rejected_at_construction(self):
        with self.assertRaises(ValueError):
            ApprovalPolicy(None, headless=HeadlessPolicy.AUTO_APPROVE)

    def test_auto_approve_records_the_stated_policy(self):
        policy = ApprovalPolicy.headless_auto_approve(
            "nightly regression run: exports go to a scratch dir",
            principal="ci", surface="ci")
        record = policy.require("export")
        self.assertTrue(record.approved)
        self.assertEqual(record.decided_by, "policy:headless-auto-approve")
        self.assertIn("nightly regression run", record.reason)

    def test_tier_override_is_honoured(self):
        # The MCP surface passes the tier its annotations already carry, so the
        # two classifiers cannot drift.
        policy = ApprovalPolicy()
        with self.assertRaises(ApprovalDenied):
            policy.require("reset", tier=ApprovalTier.REQUIRE)


if __name__ == "__main__":
    unittest.main()
