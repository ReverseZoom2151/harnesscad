"""Tests for eval.bench.geometry.constraint_editability."""

import unittest

from harnesscad.eval.bench.geometry.constraint_editability import (
    CONSTRAINT_TYPES,
    EditInstance,
    conditional_pcsr,
    edit_reachability,
    evaluate,
    overall_editable_success,
)


class ConstraintTypesTest(unittest.TestCase):
    def test_nineteen_types(self):
        self.assertEqual(len(CONSTRAINT_TYPES), 19)

    def test_unknown_constraint_rejected(self):
        with self.assertRaises(ValueError):
            EditInstance(True, {"nonexistent": True})


class MetricsTest(unittest.TestCase):
    def setUp(self):
        # 4 instances: 3 reachable, 1 not.
        self.insts = [
            EditInstance(True, {"perpendicular": True, "parallel": True}),   # 1.0
            EditInstance(True, {"perpendicular": True, "parallel": False}),  # 0.5
            EditInstance(True, {"equal": True}),                             # 1.0
            EditInstance(False, {"length": True}),                          # ignored
        ]

    def test_er(self):
        self.assertAlmostEqual(edit_reachability(self.insts), 0.75)

    def test_cpcsr_over_reachable_only(self):
        # (1.0 + 0.5 + 1.0) / 3 = 0.8333...
        self.assertAlmostEqual(conditional_pcsr(self.insts), 2.5 / 3.0)

    def test_oes_is_product(self):
        self.assertAlmostEqual(
            overall_editable_success(self.insts), 0.75 * (2.5 / 3.0)
        )

    def test_evaluate_keys(self):
        out = evaluate(self.insts)
        self.assertEqual(set(out), {"ER", "cPCSR", "OES"})
        self.assertAlmostEqual(out["OES"], out["ER"] * out["cPCSR"])

    def test_empty_preservation_is_full(self):
        self.assertAlmostEqual(conditional_pcsr([EditInstance(True, {})]), 1.0)

    def test_no_reachable_cpcsr_zero(self):
        self.assertEqual(conditional_pcsr([EditInstance(False)]), 0.0)

    def test_empty_raises(self):
        with self.assertRaises(ValueError):
            edit_reachability([])


if __name__ == "__main__":
    unittest.main()
