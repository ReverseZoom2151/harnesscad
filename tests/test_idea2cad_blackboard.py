"""Tests for agents/idea2cad_blackboard.py (paper 86: From Idea to CAD)."""

import unittest

from agents.idea2cad_blackboard import DesignBlackboard, VPhase, PHASE_ORDER, Revision


class TestPhases(unittest.TestCase):
    def test_phase_order(self):
        self.assertEqual(PHASE_ORDER,
                         (VPhase.REQUIREMENTS, VPhase.DESIGN,
                          VPhase.VERIFICATION, VPhase.VALIDATION))

    def test_str_value(self):
        self.assertEqual(VPhase.DESIGN.value, "design")
        self.assertEqual(VPhase.DESIGN, "design")


class TestBlackboardSpec(unittest.TestCase):
    def test_post_input_and_spec(self):
        bb = DesignBlackboard()
        bb.post_input("sketch-handle", "length=10")
        self.assertEqual(bb.sketch, "sketch-handle")
        self.assertEqual(bb.specification, "length=10")

    def test_append_text(self):
        bb = DesignBlackboard()
        bb.post_input(None, "base")
        bb.append_text("more")
        self.assertEqual(bb.text, "base\nmore")

    def test_addendum_included_in_spec(self):
        bb = DesignBlackboard()
        bb.post_input(None, "T-part")
        bb.post_addendum("addendum-part")
        self.assertEqual(bb.specification, "T-part\naddendum-part")


class TestFeedbackComposition(unittest.TestCase):
    def test_combined_feedback_order(self):
        bb = DesignBlackboard()
        bb.post_verification_feedback(["ver1", "ver2"])
        bb.post_validation_feedback(["val1"])
        # Algorithm 3: design(R, Fval + Fver) -- validation first
        self.assertEqual(bb.combined_feedback, ["val1", "ver1", "ver2"])

    def test_has_feedback(self):
        bb = DesignBlackboard()
        self.assertFalse(bb.has_feedback)
        bb.post_verification_feedback(["x"])
        self.assertTrue(bb.has_feedback)

    def test_empty_feedback(self):
        bb = DesignBlackboard()
        bb.post_verification_feedback([])
        bb.post_validation_feedback([])
        self.assertEqual(bb.combined_feedback, [])
        self.assertFalse(bb.has_feedback)


class TestRevisionLog(unittest.TestCase):
    def test_monotonic_revisions(self):
        bb = DesignBlackboard()
        bb.post_input(None, "a")
        bb.post_plan("p")
        bb.post_code("c")
        revs = [r.rev for r in bb.log]
        self.assertEqual(revs, list(range(len(revs))))
        self.assertEqual(bb.revision, len(bb.log))

    def test_enter_phase_recorded(self):
        bb = DesignBlackboard()
        bb.enter_phase(VPhase.VERIFICATION)
        self.assertEqual(bb.phase, VPhase.VERIFICATION)
        self.assertEqual(bb.log[-1].field, "phase")
        self.assertIsInstance(bb.log[-1], Revision)

    def test_model_flag_in_snapshot(self):
        bb = DesignBlackboard()
        self.assertFalse(bb.snapshot()["has_model"])
        bb.post_model(object())
        self.assertTrue(bb.snapshot()["has_model"])

    def test_snapshot_json_shape(self):
        bb = DesignBlackboard()
        bb.post_input("s", "len=10")
        snap = bb.snapshot()
        self.assertEqual(snap["phase"], "requirements")
        self.assertIn("specification", snap)
        self.assertEqual(len(bb.log_dicts()), len(bb.log))


if __name__ == "__main__":
    unittest.main()
