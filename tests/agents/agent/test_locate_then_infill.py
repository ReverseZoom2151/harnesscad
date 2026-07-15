import unittest

from harnesscad.agents.agent.locate_then_infill import (
    MASK_TOKEN,
    apply_infill,
    build_infill_plan,
    locate_span,
)

TOKENS = ["a", "b", "c", "d", "e"]


class LocateSpanTests(unittest.TestCase):
    def test_explicit_span(self):
        self.assertEqual(locate_span(TOKENS, span=(1, 3)), (1, 3))

    def test_anchor(self):
        self.assertEqual(locate_span(TOKENS, anchor=["c", "d"]), (2, 4))

    def test_requires_exactly_one_locator(self):
        with self.assertRaises(ValueError):
            locate_span(TOKENS)
        with self.assertRaises(ValueError):
            locate_span(TOKENS, span=(0, 1), anchor=["a"])

    def test_span_out_of_bounds(self):
        with self.assertRaises(ValueError):
            locate_span(TOKENS, span=(0, 99))

    def test_anchor_absent(self):
        with self.assertRaises(ValueError):
            locate_span(TOKENS, anchor=["z"])


class InfillPlanTests(unittest.TestCase):
    def test_build_plan_masks_span(self):
        plan = build_infill_plan(TOKENS, span=(1, 3))
        self.assertEqual(plan.prefix, ("a",))
        self.assertEqual(plan.suffix, ("d", "e"))
        self.assertEqual(plan.span, (1, 3))
        self.assertEqual(plan.masked, ("a", MASK_TOKEN, "d", "e"))

    def test_build_plan_by_anchor(self):
        plan = build_infill_plan(TOKENS, anchor=["c"])
        self.assertEqual(plan.span, (2, 3))
        self.assertEqual(plan.masked, ("a", "b", MASK_TOKEN, "d", "e"))

    def test_apply_infill_splices(self):
        plan = build_infill_plan(TOKENS, span=(1, 3))
        self.assertEqual(apply_infill(plan, ["X", "Y"]), ("a", "X", "Y", "d", "e"))

    def test_apply_empty_infill_is_deletion(self):
        plan = build_infill_plan(TOKENS, span=(1, 3))
        self.assertEqual(apply_infill(plan, []), ("a", "d", "e"))

    def test_context_preserved_verbatim(self):
        plan = build_infill_plan(TOKENS, span=(2, 2))  # zero-width insertion point
        self.assertEqual(apply_infill(plan, ["NEW"]), ("a", "b", "NEW", "c", "d", "e"))


if __name__ == "__main__":
    unittest.main()
