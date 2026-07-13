import unittest

from harnesscad.agents.generation.escalation import (
    EscalationPolicy, Strategy, fingerprint_code,
)


class TestFingerprint(unittest.TestCase):
    def test_whitespace_insensitive(self):
        self.assertEqual(fingerprint_code("a  b\nc"), fingerprint_code("a b c"))

    def test_distinct_code_distinct(self):
        self.assertNotEqual(fingerprint_code("box(1)"), fingerprint_code("box(2)"))


class TestPolicy(unittest.TestCase):
    def test_normal_adjust(self):
        p = EscalationPolicy()
        p.record("bbox_z", "cyl(10)")
        d = p.directive(iteration=1)
        self.assertIs(d.strategy, Strategy.ADJUST)
        self.assertFalse(d.escalated)

    def test_escalate_at_threshold_iteration(self):
        p = EscalationPolicy(escalate_iteration=3)
        p.record("bbox_z", "a")
        d = p.directive(iteration=3)
        self.assertIs(d.strategy, Strategy.ESCALATE)

    def test_persisting_issue_escalates(self):
        p = EscalationPolicy(persist_threshold=2)
        p.record("bbox_z", "a")
        p.record("bbox_z", "b")
        d = p.directive(iteration=1)
        self.assertIs(d.strategy, Strategy.ESCALATE)
        self.assertEqual(d.persisting_issue, "bbox_z")

    def test_changing_issue_no_persist(self):
        p = EscalationPolicy(persist_threshold=2)
        p.record("bbox_z", "a")
        p.record("hole_count", "b")
        d = p.directive(iteration=1)
        self.assertIs(d.strategy, Strategy.ADJUST)

    def test_oscillation_detected(self):
        p = EscalationPolicy()
        p.record("issueA", "codeA")
        p.record("issueB", "codeB")
        p.record("issueA", "codeA")   # back to A -> oscillation
        d = p.directive(iteration=1)
        self.assertIs(d.strategy, Strategy.BREAK_CYCLE)

    def test_forbidden_lists_prior_attempts(self):
        p = EscalationPolicy()
        p.record("x", "codeA")
        d = p.directive(iteration=0)
        self.assertIn(fingerprint_code("codeA"), d.forbidden_fingerprints)

    def test_is_forbidden(self):
        p = EscalationPolicy()
        p.record("x", "codeA")
        self.assertTrue(p.is_forbidden("codeA"))
        self.assertFalse(p.is_forbidden("codeB"))


if __name__ == "__main__":
    unittest.main()
