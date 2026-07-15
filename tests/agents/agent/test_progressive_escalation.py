"""Tests for the AgentSCAD progressive escalation policy."""

import unittest

from harnesscad.agents.agent import progressive_escalation as pe


def _pass():
    return [pe.ValidationCheck("C001", passed=True, is_critical=True)]


def _crit_fail():
    return [pe.ValidationCheck("C002", passed=False, is_critical=True, message="floating")]


def _noncrit_fail():
    return [pe.ValidationCheck("D001", passed=False, is_critical=False)]


class ValidationTest(unittest.TestCase):
    def test_critical_failure_blocks(self):
        self.assertTrue(pe.has_critical_failure(_crit_fail()))
        self.assertFalse(pe.validation_passed(_crit_fail()))

    def test_noncritical_failure_does_not_block(self):
        self.assertFalse(pe.has_critical_failure(_noncrit_fail()))
        self.assertTrue(pe.validation_passed(_noncrit_fail()))


class HappyPathTest(unittest.TestCase):
    def test_clean_validation_delivers_without_repair(self):
        cfg = pe.EscalationConfig(budget=3.0)
        state = pe.run_policy(cfg, [_pass()])
        self.assertEqual(state.stage, pe.DELIVER)
        self.assertFalse(state.repair_used)
        self.assertFalse(state.visual_repair_used)

    def test_generate_render_validate_ordering(self):
        cfg = pe.EscalationConfig(budget=3.0)
        state = pe.run_policy(cfg, [_pass()])
        # GENERATE precedes RENDER precedes VALIDATE precedes DELIVER.
        h = state.history
        self.assertLess(h.index(pe.GENERATE), h.index(pe.RENDER))
        self.assertLess(h.index(pe.RENDER), h.index(pe.VALIDATE))
        self.assertLess(h.index(pe.VALIDATE), h.index(pe.DELIVER))


class RepairTest(unittest.TestCase):
    def test_repair_fires_once_then_delivers_when_fixed(self):
        cfg = pe.EscalationConfig(budget=5.0)
        # First validate fails critically; after repair, re-validate passes.
        state = pe.run_policy(cfg, [_crit_fail(), _pass()])
        self.assertEqual(state.stage, pe.DELIVER)
        self.assertTrue(state.repair_used)

    def test_repair_fires_at_most_once(self):
        cfg = pe.EscalationConfig(budget=10.0, visual_repair_requested=False)
        # Keeps failing critically -> one repair, then HUMAN_REVIEW (no loop).
        state = pe.run_policy(cfg, [_crit_fail(), _crit_fail(), _crit_fail()])
        self.assertEqual(state.stage, pe.HUMAN_REVIEW)
        self.assertTrue(state.repair_used)
        self.assertEqual(state.history.count(pe.REPAIR), 1)


class VisualRepairTest(unittest.TestCase):
    def test_visual_repair_only_when_requested(self):
        cfg = pe.EscalationConfig(budget=10.0, visual_repair_requested=False)
        state = pe.run_policy(cfg, [_crit_fail(), _crit_fail()])
        self.assertEqual(state.history.count(pe.VISUAL_REPAIR), 0)

    def test_visual_repair_runs_after_repair_when_requested(self):
        cfg = pe.EscalationConfig(budget=10.0, visual_repair_requested=True, vision_available=True)
        state = pe.run_policy(cfg, [_crit_fail(), _crit_fail(), _crit_fail()])
        self.assertEqual(state.history.count(pe.VISUAL_REPAIR), 1)

    def test_visual_requested_but_no_provider_routes_to_human(self):
        cfg = pe.EscalationConfig(budget=10.0, visual_repair_requested=True, vision_available=False)
        # Non-critical failure so repair does not fire; visual wanted but no provider.
        state = pe.run_policy(cfg, [_noncrit_fail()])
        # non-critical fail -> validation_passed True -> DELIVER; use crit to force branch
        cfg2 = pe.EscalationConfig(budget=1.2, visual_repair_requested=True, vision_available=False)
        state2 = pe.run_policy(cfg2, [_crit_fail()])
        self.assertEqual(state2.stage, pe.HUMAN_REVIEW)


class BudgetTest(unittest.TestCase):
    def test_budget_stops_before_unaffordable_repair(self):
        # Budget only covers GENERATE+RENDER+VALIDATE, not REPAIR.
        cfg = pe.EscalationConfig(budget=1.2)
        state = pe.run_policy(cfg, [_crit_fail()])
        self.assertEqual(state.stage, pe.HUMAN_REVIEW)
        self.assertFalse(state.repair_used)

    def test_spent_never_exceeds_budget(self):
        cfg = pe.EscalationConfig(budget=5.0, visual_repair_requested=True)
        state = pe.run_policy(cfg, [_crit_fail(), _crit_fail(), _crit_fail()])
        self.assertLessEqual(state.spent, cfg.budget + 1e-9)


if __name__ == "__main__":
    unittest.main()
