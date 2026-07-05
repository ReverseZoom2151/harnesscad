import unittest

from bench.engdesign_cad_rubric import (
    score_part_description, score_dimension_extraction, score_cad_features,
    iteration_trajectory,
)


class PartDescriptionTest(unittest.TestCase):
    def test_blind_hole_scores_one(self):
        self.assertEqual(
            score_part_description("A rectangular block with a blind hole."), 1)

    def test_through_hole_scores_zero(self):
        self.assertEqual(
            score_part_description("A block with a cylindrical through-hole."), 0)

    def test_generic_block_with_hole(self):
        self.assertEqual(score_part_description("a block with a hole"), 1)


class DimensionExtractionTest(unittest.TestCase):
    def setUp(self):
        self.expected = [
            {"value": 8.0, "labels": ("length", "width", "height")},
            {"value": 5.0, "labels": ("length", "width", "height")},
            {"value": 12.0, "labels": ("length", "width", "height")},
            {"value": 5.0, "labels": ("hole diameter",)},
            {"value": 4.0, "labels": ("hole depth",)},
        ]

    def test_perfect_ten(self):
        extracted = [
            {"value": 8.0, "label": "height"},
            {"value": 5.0, "label": "width"},
            {"value": 12.0, "label": "length"},
            {"value": 5.0, "label": "hole diameter"},
            {"value": 4.0, "label": "hole depth"},
        ]
        res = score_dimension_extraction(extracted, self.expected)
        self.assertEqual(res["score"], 10)
        self.assertEqual(res["max"], 10)

    def test_extra_dimension_penalised(self):
        extracted = [
            {"value": 8.0, "label": "height"},
            {"value": 5.0, "label": "width"},
            {"value": 12.0, "label": "length"},
            {"value": 5.0, "label": "hole diameter"},
            {"value": 4.0, "label": "hole depth"},
            {"value": 2.0, "label": "made up"},
        ]
        res = score_dimension_extraction(extracted, self.expected)
        self.assertEqual(res["extra_penalty"], 1)
        self.assertEqual(res["score"], 9)


class CadFeatureTest(unittest.TestCase):
    def test_perfect_six(self):
        feats = {"runs_no_errors": True, "correct_dimensions": True,
                 "hole_on_largest_face": True, "hole_centered": True,
                 "correct_depth": True, "correct_diameter": True}
        self.assertEqual(score_cad_features(feats)["score"], 6)

    def test_extra_incorrect_feature(self):
        feats = {"runs_no_errors": True, "correct_dimensions": True,
                 "hole_on_largest_face": False, "hole_centered": False,
                 "correct_depth": False, "correct_diameter": False}
        res = score_cad_features(feats, extra_incorrect=1)
        self.assertEqual(res["earned"], 2)
        self.assertEqual(res["score"], 1)


class TrajectoryTest(unittest.TestCase):
    def test_degrading_iterations(self):
        # Paper: CAD Gen 5 (P7) worse than CAD Gen 1 (P3).
        res = iteration_trajectory([4, 4, 2, 1, 0])
        self.assertTrue(res["final_worse_than_first"])
        self.assertFalse(res["improved"])
        self.assertEqual(res["best"], 4)
        self.assertTrue(res["peaked_at_first"])

    def test_empty_raises(self):
        with self.assertRaises(ValueError):
            iteration_trajectory([])


if __name__ == "__main__":
    unittest.main()
