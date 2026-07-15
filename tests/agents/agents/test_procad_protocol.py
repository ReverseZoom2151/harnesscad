"""Tests for the Pro-CAD two-agent protocol scaffold."""

import unittest

from harnesscad.agents.agents.procad_protocol import (
    Critique,
    Message,
    Proposal,
    Role,
    TwoAgentProtocol,
    TwoAgentResult,
    Verdict,
    default_designer,
    default_reviewer,
)


class TestDefaultReviewer(unittest.TestCase):
    def test_approves_clean_proposal(self):
        review = default_reviewer()
        c = review(Proposal(brief="a box", ops=("sketch rect", "extrude 10")))
        self.assertTrue(c.approved)
        self.assertEqual(c.verdict, Verdict.APPROVE)

    def test_flags_empty_proposal(self):
        c = default_reviewer()(Proposal(brief="x", ops=()))
        self.assertEqual(c.verdict, Verdict.REVISE)
        self.assertTrue(c.issues)
        self.assertTrue(c.requests)

    def test_flags_ambiguity_marker(self):
        c = default_reviewer()(Proposal(brief="x", ops=("extrude about 10mm",)))
        self.assertEqual(c.verdict, Verdict.REVISE)
        self.assertTrue(any("about" in i for i in c.issues))

    def test_flags_op_budget(self):
        ops = tuple(f"op{i}" for i in range(40))
        c = default_reviewer(max_ops=32)(Proposal(brief="x", ops=ops))
        self.assertEqual(c.verdict, Verdict.REVISE)
        self.assertTrue(any("budget" in i for i in c.issues))


class TestDefaultDesigner(unittest.TestCase):
    def test_passthrough_without_critique(self):
        p = Proposal(brief="x", ops=("a", "b"))
        self.assertEqual(default_designer()(p, None), p)

    def test_strips_markers_on_revision(self):
        design = default_designer()
        p = Proposal(brief="x", ops=("extrude about 10mm",))
        crit = Critique(verdict=Verdict.REVISE, requests=("resolve 'about'",))
        revised = design(p, crit)
        self.assertEqual(revised.revision, 1)
        self.assertNotIn("about", revised.ops[0])


class TestProtocol(unittest.TestCase):
    def test_clean_proposal_approved_first_round(self):
        proto = TwoAgentProtocol()
        res = proto.run(Proposal(brief="box", ops=("sketch rect", "extrude 10")))
        self.assertIsInstance(res, TwoAgentResult)
        self.assertTrue(res.approved)
        self.assertEqual(res.rounds, 1)
        # transcript: designer + reviewer for the single round.
        self.assertEqual(len(res.transcript), 2)
        self.assertEqual(res.transcript[0].role, Role.DESIGNER)
        self.assertEqual(res.transcript[1].role, Role.REVIEWER)

    def test_ambiguous_proposal_gets_revised_then_approved(self):
        proto = TwoAgentProtocol()
        res = proto.run(Proposal(brief="box", ops=("extrude about 10mm",)))
        # round 1 flags 'about', designer strips it, round 2 approves.
        self.assertTrue(res.approved)
        self.assertEqual(res.rounds, 2)
        self.assertNotIn("about", res.final_ops[0])
        self.assertEqual(res.final_proposal.revision, 1)

    def test_alternating_roles_in_transcript(self):
        proto = TwoAgentProtocol()
        res = proto.run(Proposal(brief="box", ops=("extrude about 10mm",)))
        roles = [m.role for m in res.transcript]
        for i, r in enumerate(roles):
            expected = Role.DESIGNER if i % 2 == 0 else Role.REVIEWER
            self.assertEqual(r, expected)

    def test_budget_exhausted_without_approval(self):
        # a Reviewer that never approves; a Designer that never changes anything.
        proto = TwoAgentProtocol(
            designer=lambda p, c: p,
            reviewer=lambda p: Critique(verdict=Verdict.REVISE, issues=("nope",)),
            max_rounds=3,
        )
        res = proto.run(Proposal(brief="x", ops=("a",)))
        self.assertFalse(res.approved)
        self.assertEqual(res.rounds, 3)
        self.assertEqual(res.final_critique.verdict, Verdict.REVISE)

    def test_reject_stops_immediately(self):
        proto = TwoAgentProtocol(
            reviewer=lambda p: Critique(verdict=Verdict.REJECT, issues=("fatal",)),
            max_rounds=5,
        )
        res = proto.run(Proposal(brief="x", ops=("a",)))
        self.assertFalse(res.approved)
        self.assertEqual(res.rounds, 1)
        self.assertEqual(res.final_critique.verdict, Verdict.REJECT)

    def test_deterministic(self):
        p = Proposal(brief="box", ops=("extrude about 10mm",))
        a = TwoAgentProtocol().run(p)
        b = TwoAgentProtocol().run(p)
        self.assertEqual(a.approved, b.approved)
        self.assertEqual(a.rounds, b.rounds)
        self.assertEqual(a.final_ops, b.final_ops)


if __name__ == "__main__":
    unittest.main()
