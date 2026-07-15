"""Tests for TuriX two-screenshot step-eval, SOP skill-files, and Brain rules."""

import unittest

from harnesscad.agents.cua.turix import (
    DEFAULT_GRACE, SOP, SOPStep, SkillLibrary, StepEval, evaluate_step,
    is_settled, require_explicit, settle_index,
)


class TestEvaluateStep(unittest.TestCase):
    def test_no_change_is_a_failed_step(self):
        state = {"marks": [{"id": 1, "label": "Pad"}]}
        ev = evaluate_step(state, state)
        self.assertFalse(ev.changed)
        self.assertFalse(ev.ok)

    def test_change_without_expectation_is_ok(self):
        ev = evaluate_step({"a": 1}, {"a": 2})
        self.assertTrue(ev.changed)
        self.assertIsNone(ev.expectation_met)
        self.assertTrue(ev.ok)

    def test_expectation_met_when_tokens_appear(self):
        before = {"dialog": "none"}
        after = {"dialog": "Pad dialog open", "field": "Length"}
        ev = evaluate_step(before, after, expectation="the Pad dialog is open")
        self.assertTrue(ev.changed)
        self.assertTrue(ev.expectation_met)
        self.assertTrue(ev.ok)

    def test_expectation_not_met_fails_even_if_changed(self):
        ev = evaluate_step({"x": 0}, {"y": 999},
                           expectation="the Sketch panel appears")
        self.assertTrue(ev.changed)
        self.assertFalse(ev.expectation_met)
        self.assertFalse(ev.ok)

    def test_key_order_does_not_count_as_change(self):
        ev = evaluate_step({"a": 1, "b": 2}, {"b": 2, "a": 1})
        self.assertFalse(ev.changed)


class TestSOP(unittest.TestCase):
    def _pad_sop(self):
        return SOP(name="pad_rect", trigger="pad",
                   steps=(SOPStep("open", "Part Design"),
                          SOPStep("click", "Pad"),
                          SOPStep("type", "Length", "10")))

    def test_matches_trigger(self):
        s = self._pad_sop()
        self.assertTrue(s.matches("please pad the sketch"))
        self.assertFalse(s.matches("fillet the edge"))

    def test_dict_roundtrip(self):
        s = self._pad_sop()
        self.assertEqual(SOP.from_dict(s.to_dict()), s)


class TestSkillLibrary(unittest.TestCase):
    def setUp(self):
        self.lib = SkillLibrary([
            SOP(name="pad", trigger="pad", steps=()),
            SOP(name="pad_rect", trigger="pad a rectangle", steps=()),
            SOP(name="fillet", trigger="fillet", steps=()),
        ])

    def test_recall_longest_trigger_wins(self):
        sop = self.lib.recall("pad a rectangle at the origin")
        self.assertEqual(sop.name, "pad_rect")

    def test_recall_none_when_no_trigger(self):
        self.assertIsNone(self.lib.recall("draw a circle"))

    def test_get_by_name_and_len(self):
        self.assertEqual(self.lib.get("fillet").trigger, "fillet")
        self.assertEqual(len(self.lib), 3)

    def test_dict_roundtrip(self):
        back = SkillLibrary.from_dict(self.lib.to_dict())
        self.assertEqual(back.to_dict(), self.lib.to_dict())


class TestSettle(unittest.TestCase):
    def test_settles_when_stable_for_grace(self):
        # loading... loading... A A A -> settles at the first stable A.
        obs = [{"s": "load"}, {"s": "load2"}, {"s": "done"}, {"s": "done"},
               {"s": "done"}]
        self.assertEqual(settle_index(obs, grace=2), 2)
        self.assertTrue(is_settled(obs, grace=2))

    def test_never_settles_returns_none(self):
        obs = [{"s": i} for i in range(4)]  # always changing
        self.assertIsNone(settle_index(obs, grace=2))
        self.assertFalse(is_settled(obs, grace=2))

    def test_grace_zero_settles_immediately(self):
        self.assertEqual(settle_index([{"s": 1}], grace=0), 0)

    def test_default_grace_constant(self):
        self.assertEqual(DEFAULT_GRACE, 2)

    def test_empty_stream(self):
        self.assertIsNone(settle_index([]))


class TestRequireExplicit(unittest.TestCase):
    def test_explicit_dimensioned_instruction_ok(self):
        ok, reasons = require_explicit(
            "make a block 30 mm long, 20 mm wide and 10 mm tall")
        self.assertTrue(ok)
        self.assertEqual(reasons, [])

    def test_vague_dimension_talk_refused(self):
        ok, reasons = require_explicit("make it long and tall and thick")
        self.assertFalse(ok)
        self.assertTrue(reasons)

    def test_no_dimension_talk_is_left_alone(self):
        # Not this rule's job to refuse; the planner handles non-measured briefs.
        ok, reasons = require_explicit("open the part design workbench")
        self.assertTrue(ok)

    def test_min_numbers_threshold(self):
        ok, _ = require_explicit("a bar 5 mm long", min_numbers=2)
        self.assertFalse(ok)


if __name__ == "__main__":
    unittest.main()
