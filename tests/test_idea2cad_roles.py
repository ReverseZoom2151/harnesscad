"""Tests for agents/idea2cad_roles.py (paper 86: From Idea to CAD)."""

import unittest

from harnesscad.agents.agents.idea2cad_blackboard import VPhase
from harnesscad.agents.agents.idea2cad_roles import (
    RequirementsEngineer,
    CadEngineer,
    QualityAssuranceEngineer,
    User,
    HandoffDAG,
    HANDOFFS,
    ROLE_RE,
    ROLE_CAD,
    ROLE_QA,
    ROLE_USER,
    ROLE_PHASE,
    _ast_ok,
)


class TestRolePhaseMap(unittest.TestCase):
    def test_role_owns_phase(self):
        self.assertEqual(ROLE_PHASE[ROLE_RE], VPhase.REQUIREMENTS)
        self.assertEqual(ROLE_PHASE[ROLE_CAD], VPhase.DESIGN)
        self.assertEqual(ROLE_PHASE[ROLE_QA], VPhase.VERIFICATION)
        self.assertEqual(ROLE_PHASE[ROLE_USER], VPhase.VALIDATION)


class TestHandoffDAG(unittest.TestCase):
    def setUp(self):
        self.dag = HandoffDAG()

    def test_roles_present(self):
        roles = self.dag.roles()
        for r in (ROLE_RE, ROLE_CAD, ROLE_QA, ROLE_USER):
            self.assertIn(r, roles)

    def test_forward_edges(self):
        fwd = self.dag.forward_edges()
        # RE->CAD, CAD->QA, CAD->USER are the three forward edges
        pairs = {(h.src, h.dst) for h in fwd}
        self.assertIn((ROLE_RE, ROLE_CAD), pairs)
        self.assertIn((ROLE_CAD, ROLE_QA), pairs)
        self.assertIn((ROLE_CAD, ROLE_USER), pairs)

    def test_feedback_back_edges(self):
        fb = self.dag.feedback_edges()
        pairs = {(h.src, h.dst) for h in fb}
        # QA->CAD and USER->CAD are the corrective back-edges
        self.assertEqual(pairs, {(ROLE_QA, ROLE_CAD), (ROLE_USER, ROLE_CAD)})

    def test_re_hands_to_cad(self):
        outs = self.dag.out_edges(ROLE_RE)
        self.assertEqual(len(outs), 1)
        self.assertEqual(outs[0].dst, ROLE_CAD)
        self.assertEqual(outs[0].artifact, "specification")

    def test_cad_receives_two_feedback_edges(self):
        ins = self.dag.in_edges(ROLE_CAD, kind="feedback")
        self.assertEqual(len(ins), 2)

    def test_forward_order_re_first(self):
        order = self.dag.forward_order()
        self.assertEqual(order[0], ROLE_RE)
        self.assertLess(order.index(ROLE_RE), order.index(ROLE_CAD))
        self.assertLess(order.index(ROLE_CAD), order.index(ROLE_QA))


class TestRequirementsEngineer(unittest.TestCase):
    def test_clarify_default_heuristic(self):
        re = RequirementsEngineer()
        self.assertTrue(re.clarify(None, "a block"))          # no dims -> ambiguous
        self.assertEqual(re.clarify(None, "length=10 width=5cm"), [])

    def test_summarise_extracts_addendum(self):
        re = RequirementsEngineer(
            summarise_fn=lambda s, t: f"chatter <SUMMARY>{t}</SUMMARY>")
        self.assertEqual(re.summarise(None, "len=10"), "len=10")

    def test_summarise_none_when_no_block(self):
        re = RequirementsEngineer(summarise_fn=lambda s, t: "still a question?")
        self.assertIsNone(re.summarise(None, "x"))

    def test_default_summary_wraps(self):
        re = RequirementsEngineer()
        self.assertEqual(re.summarise(None, "len=10"), "len=10")


class TestCadEngineer(unittest.TestCase):
    def test_plan_generate_check_exec(self):
        cad = CadEngineer()
        plan = cad.plan("a box")
        self.assertIn("build model", plan)
        code = cad.generate("a box")
        self.assertTrue(cad.check(code))
        self.assertIsNotNone(cad.execute(code))

    def test_check_rejects_bad_code(self):
        cad = CadEngineer()
        self.assertFalse(cad.check("def (:"))
        self.assertTrue(cad.check("x = 1"))

    def test_hints_from_docs(self):
        cad = CadEngineer()
        hints = cad.hints_from_docs("code", "docs", ["fix A", "fix B"])
        self.assertIn("fix A", hints)

    def test_exec_failure_returns_none(self):
        cad = CadEngineer(exec_fn=lambda c: None)
        self.assertIsNone(cad.execute("x = 1"))


class TestQAEngineer(unittest.TestCase):
    def test_default_acceptable(self):
        qa = QualityAssuranceEngineer()
        rep = qa.review("spec", object())
        self.assertTrue(rep.acceptable)
        self.assertEqual(rep.issues, [])
        self.assertEqual(len(rep.views), 7)

    def test_bounds_to_two_issues(self):
        qa = QualityAssuranceEngineer(qa_fn=lambda r, imgs: ["i1", "i2", "i3"])
        rep = qa.review("spec", object())
        self.assertEqual(len(rep.issues), 2)
        self.assertFalse(rep.acceptable)

    def test_render_uses_seven_views(self):
        seen = {}

        def render(m, views):
            seen["views"] = views
            return {v: None for v in views}

        qa = QualityAssuranceEngineer(render_fn=render)
        qa.review("spec", object())
        self.assertEqual(len(seen["views"]), 7)


class TestUser(unittest.TestCase):
    def test_default_accepts(self):
        self.assertEqual(User().validate("spec", object()), [])

    def test_feedback_proxy(self):
        u = User(feedback_fn=lambda r, m: ["make wheels parallel to XZ"])
        self.assertEqual(u.validate("spec", object()), ["make wheels parallel to XZ"])


class TestAstGate(unittest.TestCase):
    def test_ast_ok(self):
        self.assertTrue(_ast_ok("import cadquery as cq\nresult = cq"))
        self.assertFalse(_ast_ok("this is (not python"))
        self.assertTrue(_ast_ok(""))


if __name__ == "__main__":
    unittest.main()
