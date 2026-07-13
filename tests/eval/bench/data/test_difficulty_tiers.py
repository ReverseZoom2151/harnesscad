"""Tests for bench.t2cadbench_taxonomy."""

import unittest

from harnesscad.eval.bench.data.difficulty_tiers import (
    LEVEL_COUNTS,
    classify_example,
    classify_operations,
    l4_domain_deviation,
    normalize_operation,
    operation_tier,
    validate_split,
)


class NormalizeTests(unittest.TestCase):
    def test_strips_dot_and_args(self):
        self.assertEqual(normalize_operation(".chamfer(2)"), "chamfer")
        self.assertEqual(normalize_operation("Box"), "box")
        self.assertEqual(normalize_operation("  .loft(ruled=False) "), "loft")

    def test_operation_tier_lookup(self):
        self.assertEqual(operation_tier("box"), 1)
        self.assertEqual(operation_tier(".cut"), 2)
        self.assertEqual(operation_tier("sweep()"), 3)
        self.assertIsNone(operation_tier("frobnicate"))


class ClassifyOperationsTests(unittest.TestCase):
    def test_l1_primitive_with_finishing(self):
        # Appendix F L1 example: polygon/extrude/circle/cutThruAll/chamfer.
        r = classify_operations(
            ["polygon", "extrude", "faces", "circle", "cutThruAll",
             "edges", "chamfer"])
        self.assertEqual(r["tier"], 1)
        self.assertEqual(r["label"], "L1")
        self.assertIn("chamfer", r["advanced_features"])

    def test_l2_boolean_drives_tier(self):
        # Appendix F L2: sphere + box cut -> boolean -> L2.
        r = classify_operations(["sphere", "cut", "box"])
        self.assertEqual(r["tier"], 2)
        self.assertEqual(r["label"], "L2")
        self.assertEqual(r["driver"], "cut")

    def test_l3_loft_and_shell(self):
        r = classify_operations(["workplane", "add", "loft", "shell"])
        self.assertEqual(r["tier"], 3)
        self.assertEqual(r["label"], "L3")
        # "add" is unknown, recorded.
        self.assertIn("add", r["unknown"])
        self.assertEqual(set(r["advanced_features"]), {"loft", "shell"})

    def test_empty_defaults_to_l1(self):
        r = classify_operations([])
        self.assertEqual(r["tier"], 1)
        self.assertIsNone(r["driver"])
        self.assertEqual(r["op_count"], 0)

    def test_op_count(self):
        r = classify_operations(["box", "hole", "chamfer"])
        self.assertEqual(r["op_count"], 3)


class ClassifyExampleTests(unittest.TestCase):
    def test_domain_forces_l4_but_keeps_geometry(self):
        r = classify_example(["box", "hole"], application_domain="Medical")
        self.assertEqual(r["label"], "L4")
        self.assertEqual(r["tier"], 4)
        self.assertEqual(r["geometric_label"], "L1")
        self.assertEqual(r["application_domain"], "medical")

    def test_no_domain_stays_geometric(self):
        r = classify_example(["sphere", "cut"])
        self.assertEqual(r["label"], "L2")
        self.assertIsNone(r["application_domain"])

    def test_blank_domain_ignored(self):
        r = classify_example(["box"], application_domain="   ")
        self.assertEqual(r["label"], "L1")


class SplitValidationTests(unittest.TestCase):
    def test_exact_paper_split_matches(self):
        r = validate_split(dict(LEVEL_COUNTS))
        self.assertTrue(r["matches"])
        self.assertEqual(r["total"], 600)
        self.assertEqual(r["deltas"]["L1"], 0)

    def test_mismatch_reports_deltas(self):
        r = validate_split({"L1": 190, "L2": 200, "L3": 100, "L4": 100})
        self.assertFalse(r["matches"])
        self.assertEqual(r["deltas"]["L1"], -10)
        self.assertEqual(r["total"], 590)


class L4DomainTests(unittest.TestCase):
    def test_on_target_mix_zero_deviation(self):
        r = l4_domain_deviation({
            "industrial": 40, "consumer": 25, "medical": 15,
            "architectural": 10, "educational": 10})
        self.assertAlmostEqual(r["max_deviation"], 0.0)
        self.assertEqual(r["total"], 100)

    def test_skewed_mix_flags_deviation(self):
        r = l4_domain_deviation({"industrial": 100})
        self.assertAlmostEqual(r["per_domain"]["industrial"]["fraction"], 1.0)
        self.assertGreater(r["max_deviation"], 0.5)

    def test_empty_raises(self):
        with self.assertRaises(ValueError):
            l4_domain_deviation({})


if __name__ == "__main__":
    unittest.main()
