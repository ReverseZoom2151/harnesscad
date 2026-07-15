"""Tests for domain.spec.representation_completeness."""

import unittest

from harnesscad.domain.spec.representation_completeness import (
    SEMANTIC_LAYERS,
    compare_representations,
    score_representation,
)


class ScoreTest(unittest.TestCase):
    def test_deepcad_is_solid_modeling(self):
        s = score_representation("deepcad", ["geometry", "sketch"])
        self.assertEqual(s["level"], "solid_modeling")
        self.assertIn("topological_naming", s["missing"])

    def test_whucad_is_industrial(self):
        s = score_representation("whucad", list(SEMANTIC_LAYERS))
        self.assertEqual(s["level"], "industrial_parametric")
        self.assertEqual(s["missing"], [])
        self.assertAlmostEqual(s["coverage"], 1.0)

    def test_pointcloud_no_level(self):
        s = score_representation("pointcloud", ["geometry"])
        self.assertEqual(s["level"], "none")

    def test_unknown_layer_rejected(self):
        with self.assertRaises(ValueError):
            score_representation("x", ["geometry", "bogus"])


class CompareTest(unittest.TestCase):
    def test_whucad_ranks_above_deepcad(self):
        ranked = compare_representations(
            {
                "deepcad": ["geometry", "sketch"],
                "whucad": list(SEMANTIC_LAYERS),
                "mesh": ["geometry"],
            }
        )
        self.assertEqual(ranked[0]["name"], "whucad")
        self.assertEqual(ranked[-1]["name"], "mesh")

    def test_deterministic_order(self):
        reps = {"a": ["geometry", "sketch"], "b": ["geometry", "sketch"]}
        r1 = [s["name"] for s in compare_representations(reps)]
        r2 = [s["name"] for s in compare_representations(reps)]
        self.assertEqual(r1, r2)


if __name__ == "__main__":
    unittest.main()
