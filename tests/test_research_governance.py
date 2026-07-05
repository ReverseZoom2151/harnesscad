import unittest

from research.governance import Claim, Evidence, ResearchGovernance, Review


def ready_governance():
    governance = ResearchGovernance()
    governance.add_evidence(Evidence(
        "e1", "tests/run-1.json", "sha256:abc", metadata={
            "expected_result": "pass", "observed_result": "pass",
        }
    ))
    governance.add_claim(Claim("c1", "The generated part satisfies its contract.", ("e1",)))
    return governance


class ResearchGovernanceTests(unittest.TestCase):
    def test_advances_with_consistent_evidence_and_review_ensemble(self):
        governance = ready_governance()
        decision = governance.evaluate_gate([
            Review("geometry", {"validity": .9, "coverage": .8}, "advance"),
            Review("manufacturing", {"validity": .8}, "advance"),
        ])
        self.assertEqual("advance", decision.outcome)
        self.assertEqual("validation", governance.stage)
        self.assertTrue(decision.checks.ok)
        self.assertEqual(.825, decision.ensemble_score)

    def test_missing_or_irreproducible_evidence_forces_refine(self):
        governance = ResearchGovernance()
        governance.add_evidence(Evidence("e1", "run.json", "hash", reproducible=False))
        governance.add_claim(Claim("c1", "claim", ("e1", "missing")))
        decision = governance.evaluate_gate([Review("r", {"quality": 1}, "advance")])
        self.assertEqual("refine", decision.outcome)
        self.assertIn("claim c1 missing evidence: missing", decision.checks.errors)
        self.assertIn("evidence e1 is not reproducible", decision.checks.errors)

    def test_result_consistency_check(self):
        governance = ResearchGovernance()
        governance.add_evidence(Evidence(
            "e", "run", "hash", metadata={"expected_result": 10, "observed_result": 11}
        ))
        governance.add_claim(Claim("c", "dimension is ten", ("e",)))
        self.assertIn("evidence e result is inconsistent", governance.check().errors)

    def test_majority_reject_wins_even_with_high_scores(self):
        governance = ready_governance()
        decision = governance.evaluate_gate([
            Review("a", {"quality": .95}, "reject"),
            Review("b", {"quality": .95}, "reject"),
            Review("c", {"quality": .95}, "advance"),
        ])
        self.assertEqual("reject", decision.outcome)
        self.assertEqual("discovery", governance.stage)

    def test_checkpoint_rollback_restores_complete_state(self):
        governance = ready_governance()
        digest = governance.checkpoint("baseline")
        governance.add_evidence(Evidence("extra", "other", "hash"))
        governance.evaluate_gate([Review("r", {"quality": .9}, "advance")])
        self.assertNotEqual(digest, governance.state_digest())
        governance.rollback("baseline")
        self.assertEqual(digest, governance.state_digest())
        self.assertEqual("discovery", governance.stage)
        self.assertNotIn("extra", governance.evidence)
        self.assertEqual([], governance.decisions)

    def test_digest_is_insertion_order_independent(self):
        first = ResearchGovernance()
        second = ResearchGovernance()
        evidence = [
            Evidence("a", "a.json", "a"), Evidence("b", "b.json", "b")
        ]
        claims = [Claim("a", "A", ("a",)), Claim("b", "B", ("b",))]
        for item in evidence:
            first.add_evidence(item)
        for item in claims:
            first.add_claim(item)
        for item in reversed(evidence):
            second.add_evidence(item)
        for item in reversed(claims):
            second.add_claim(item)
        self.assertEqual(first.state_digest(), second.state_digest())


if __name__ == "__main__":
    unittest.main()
