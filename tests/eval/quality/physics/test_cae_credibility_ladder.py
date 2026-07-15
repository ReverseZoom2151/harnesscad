"""Tests for the cad-cae-copilot CAE credibility ladder."""

import unittest

from harnesscad.eval.quality.physics import cae_credibility_ladder as cl


class LadderOrderTest(unittest.TestCase):
    def test_ladder_is_ordered_low_to_high(self):
        ranks = [cl.ladder_rank(x) for x in cl.LADDER]
        self.assertEqual(ranks, sorted(ranks))
        self.assertEqual(cl.ladder_rank("no_result_artifact"), 0)

    def test_unknown_level_is_negative(self):
        self.assertEqual(cl.ladder_rank("nope"), -1)


class EvidenceLevelTest(unittest.TestCase):
    def test_no_artifact(self):
        a = cl.assess_cae_credibility(cl.CaeEvidence())
        self.assertEqual(a.level, "no_result_artifact")
        self.assertFalse(a.certified)

    def test_full_chain_reaches_human_review(self):
        ev = cl.CaeEvidence(
            artifact_present=True,
            solver_completed=True,
            metrics_parsed=True,
            plausibility_checked=True,
            design_target_compared=True,
            benchmark_passed=True,
            human_review_supported=True,
            mesh_status=cl.MeshStatus.CONVERGED,
        )
        a = cl.assess_cae_credibility(ev)
        self.assertEqual(a.level, "human_review_supported")
        self.assertFalse(a.certified)

    def test_cumulative_gap_stops_at_break(self):
        ev = cl.CaeEvidence(artifact_present=True, solver_completed=True, metrics_parsed=False)
        a = cl.assess_cae_credibility(ev)
        self.assertEqual(a.level, "solver_completed")


class MeshDisciplineTest(unittest.TestCase):
    def test_failed_mesh_caps_below_benchmark(self):
        ev = cl.CaeEvidence(
            artifact_present=True,
            solver_completed=True,
            metrics_parsed=True,
            plausibility_checked=True,
            design_target_compared=True,
            benchmark_passed=True,
            mesh_status=cl.MeshStatus.FAILED,
        )
        a = cl.assess_cae_credibility(ev)
        self.assertLess(cl.ladder_rank(a.level), cl.ladder_rank("benchmark_calibrated"))
        self.assertEqual(a.uncapped_level, "benchmark_calibrated")
        self.assertTrue(a.limitations)

    def test_unknown_mesh_records_limitation(self):
        ev = cl.CaeEvidence(
            artifact_present=True, solver_completed=True, metrics_parsed=True,
            mesh_status=cl.MeshStatus.UNKNOWN,
        )
        a = cl.assess_cae_credibility(ev)
        self.assertTrue(any("unknown" in x for x in a.limitations))


class Vnv40Test(unittest.TestCase):
    def test_tiers_ordered(self):
        ranks = [cl.vnv40_rank(t) for t in cl.VNV40_TIERS]
        self.assertEqual(ranks, sorted(ranks))

    def test_solver_claim_without_execution_downgraded(self):
        tier, reason = cl.assess_vnv40_tier("executed_solver_result", solver_executed=False)
        self.assertEqual(tier, "unverified")
        self.assertIsNotNone(reason)

    def test_solver_claim_with_execution_kept(self):
        tier, reason = cl.assess_vnv40_tier("executed_solver_result", solver_executed=True)
        self.assertEqual(tier, "executed_solver_result")
        self.assertIsNone(reason)

    def test_unknown_tier_is_unverified(self):
        tier, reason = cl.assess_vnv40_tier("magic", solver_executed=True)
        self.assertEqual(tier, "unverified")


if __name__ == "__main__":
    unittest.main()
