"""Tests for procedural.proccad_key_params."""

import unittest

from procedural.proccad_key_params import (
    KeyParameterTemplate,
    classify,
    two_stage_opamp_template,
)


class ClassifyTest(unittest.TestCase):
    def test_name(self):
        self.assertEqual(
            classify(2, "differential", "push_pull"),
            "2stage_differential_in_push_pull_out",
        )

    def test_case_insensitive(self):
        self.assertEqual(classify(1, "Differential", "SINGLE_SIDED"),
                         "1stage_differential_in_single_sided_out")

    def test_bad_stage_count(self):
        with self.assertRaises(ValueError):
            classify(0, "differential", "differential")

    def test_bad_type(self):
        with self.assertRaises(ValueError):
            classify(2, "weird", "differential")


class TemplateTest(unittest.TestCase):
    def test_dimensionality(self):
        t = two_stage_opamp_template()
        self.assertEqual(t.dimensionality(), 5)
        self.assertEqual(t.full_dimensionality(), 9)

    def test_reduction_ratio(self):
        t = two_stage_opamp_template()
        self.assertAlmostEqual(t.reduction_ratio(), 9 / 5)

    def test_realize_computes_derived(self):
        t = two_stage_opamp_template()
        keys = {
            "bias_current": 4.0,
            "output_swing": 2.0,
            "load_cap": 10.0,
            "gain_target": 5.0,
            "supply_v": 3.3,
        }
        full = t.realize(keys)
        self.assertEqual(full["tail_current"], 8.0)  # 2*bias
        self.assertAlmostEqual(full["input_gm"], 2.0)  # sqrt(4)
        self.assertAlmostEqual(full["comp_cap"], 10.0 * (1 + 0.5))  # load*(1+0.1*gain)
        self.assertAlmostEqual(full["slew_rate"], 8.0 / 15.0)

    def test_realize_deterministic(self):
        t = two_stage_opamp_template()
        keys = {
            "bias_current": 1.0,
            "output_swing": 1.0,
            "load_cap": 1.0,
            "gain_target": 1.0,
            "supply_v": 1.0,
        }
        self.assertEqual(t.realize(keys), t.realize(keys))

    def test_missing_key_rejected(self):
        t = two_stage_opamp_template()
        with self.assertRaises(ValueError):
            t.realize({"bias_current": 1.0})

    def test_extra_key_rejected(self):
        t = two_stage_opamp_template()
        keys = {
            "bias_current": 1.0,
            "output_swing": 1.0,
            "load_cap": 1.0,
            "gain_target": 1.0,
            "supply_v": 1.0,
            "junk": 0.0,
        }
        with self.assertRaises(ValueError):
            t.realize(keys)

    def test_duplicate_key_rejected(self):
        with self.assertRaises(ValueError):
            KeyParameterTemplate("c", ("a", "a"))

    def test_derived_name_collision_rejected(self):
        with self.assertRaises(ValueError):
            KeyParameterTemplate("c", ("a",), derivations=[("a", lambda v: 0.0)])

    def test_ordered_derivation_dependency(self):
        # second derivation depends on the first
        t = KeyParameterTemplate(
            "c",
            ("x",),
            derivations=[("y", lambda v: v["x"] + 1), ("z", lambda v: v["y"] * 2)],
        )
        full = t.realize({"x": 3.0})
        self.assertEqual(full["y"], 4.0)
        self.assertEqual(full["z"], 8.0)


if __name__ == "__main__":
    unittest.main()
