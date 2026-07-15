"""Tests for agents.agent.triz_matrix."""

import unittest

from harnesscad.agents.agent.triz_matrix import (
    ENGINEERING_PARAMETERS,
    INVENTIVE_PRINCIPLES,
    enhancement_context,
    parameter_name,
    principle_name,
    recommend_named,
    recommend_principles,
)


class EnumerationTest(unittest.TestCase):
    def test_forty_principles(self):
        self.assertEqual(len(INVENTIVE_PRINCIPLES), 40)
        self.assertEqual(set(INVENTIVE_PRINCIPLES), set(range(1, 41)))

    def test_thirtynine_parameters(self):
        self.assertEqual(len(ENGINEERING_PARAMETERS), 39)
        self.assertEqual(set(ENGINEERING_PARAMETERS), set(range(1, 40)))

    def test_named_lookups(self):
        self.assertEqual(principle_name(1), "Segmentation")
        self.assertEqual(principle_name(40), "Composite materials")
        self.assertEqual(parameter_name(14), "Strength")
        self.assertEqual(parameter_name(1), "Weight of moving object")

    def test_unknown_ids_raise(self):
        with self.assertRaises(ValueError):
            principle_name(41)
        with self.assertRaises(ValueError):
            parameter_name(40)


class MatrixTest(unittest.TestCase):
    def test_documented_cell(self):
        # Section 4.2: Strength (#14) vs Weight of moving object (#1).
        self.assertEqual(recommend_principles(14, 1), (1, 8, 15, 40))

    def test_named_recommendation(self):
        named = recommend_named(14, 1)
        self.assertEqual(named[0], (1, "Segmentation"))
        self.assertEqual(named[-1], (40, "Composite materials"))

    def test_unseeded_cell_is_empty(self):
        self.assertEqual(recommend_principles(9, 21), ())

    def test_self_contradiction_raises(self):
        with self.assertRaises(ValueError):
            recommend_principles(14, 14)


class ContextTest(unittest.TestCase):
    def test_scaffold_structure(self):
        ctx = enhancement_context("baseline_chair.py", 14, 1)
        self.assertEqual(set(ctx), {"role", "task", "requirements", "context"})
        c = ctx["context"]
        self.assertEqual(c["baseline_reference"], "baseline_chair.py")
        self.assertEqual(c["improving_feature"], (14, "Strength"))
        self.assertEqual(c["worsening_feature"], (1, "Weight of moving object"))
        self.assertEqual(c["recommended_principles"][0], (1, "Segmentation"))


if __name__ == "__main__":
    unittest.main()
