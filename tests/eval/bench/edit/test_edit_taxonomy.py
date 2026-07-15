"""Tests for the neuralCAD-Edit edit-operation taxonomy and benchmark structure."""

import unittest

from harnesscad.eval.bench.edit import edit_taxonomy as et


class ClassifyTest(unittest.TestCase):
    def test_families(self):
        cases = {
            "Add a hole through the top face": et.EditOperation.ADD,
            "Remove the small boss on the side": et.EditOperation.REMOVE,
            "Increase the width of the base by 10mm": et.EditOperation.MODIFY,
            "Fillet the four vertical edges": et.EditOperation.FILLET,
            "Chamfer the top edge at 45 degrees": et.EditOperation.CHAMFER,
            "Mirror the bracket across the YZ plane": et.EditOperation.PATTERN,
            "Cut a cylindrical pocket from the block": et.EditOperation.BOOLEAN,
            "Shell the body to 2mm wall thickness": et.EditOperation.SHELL,
            "Move the flange 5mm along X": et.EditOperation.TRANSFORM,
        }
        for text, expected in cases.items():
            self.assertEqual(et.classify_instruction(text), expected, text)

    def test_default_is_modify(self):
        self.assertEqual(et.classify_instruction("please refine this somehow"), et.EditOperation.MODIFY)


class TaskTest(unittest.TestCase):
    def test_task_autotags_operation(self):
        t = et.EditTask("t1", "Fillet all edges", "easy", "brepA", "brepB")
        self.assertEqual(t.operation, et.EditOperation.FILLET)

    def test_bad_difficulty_raises(self):
        with self.assertRaises(ValueError):
            et.EditTask("t", "x", "trivial", "a", "b")


class BenchmarkTest(unittest.TestCase):
    def setUp(self):
        self.bench = et.EditBenchmark(tasks=(
            et.EditTask("1", "Add a rib", "easy", "a", "b"),
            et.EditTask("2", "Fillet edges", "medium", "a", "b"),
            et.EditTask("3", "Fillet more edges", "medium", "a", "b"),
            et.EditTask("4", "Shell the part", "hard", "a", "b"),
        ))

    def test_difficulty_counts(self):
        self.assertEqual(self.bench.difficulty_counts(), {"easy": 1, "medium": 2, "hard": 1})

    def test_operation_counts(self):
        oc = self.bench.operation_counts()
        self.assertEqual(oc[et.EditOperation.FILLET], 2)
        self.assertEqual(oc[et.EditOperation.SHELL], 1)


class MetricTest(unittest.TestCase):
    def test_chamfer_similarity_monotone(self):
        near = et.chamfer_similarity(0.01)
        far = et.chamfer_similarity(1.0)
        self.assertGreater(near, far)

    def test_chamfer_negative_raises(self):
        with self.assertRaises(ValueError):
            et.chamfer_similarity(-0.1)

    def test_voxel_iou(self):
        self.assertEqual(et.voxel_iou_similarity(5, 10), 0.5)
        self.assertEqual(et.voxel_iou_similarity(0, 0), 0.0)

    def test_aggregate_rating(self):
        r = et.aggregate_rating([
            {"instruction_understanding": 6, "quality": 5},
            {"instruction_understanding": 4, "quality": 3},
        ])
        self.assertAlmostEqual(r["instruction_understanding"], 5.0)
        self.assertAlmostEqual(r["quality"], 4.0)

    def test_effectiveness_and_score_bundle(self):
        s = et.score_edit(gt_similarity=0.8, start_similarity=0.3,
                          ratings=[{"instruction_understanding": 6, "quality": 6}])
        self.assertAlmostEqual(s["effectiveness"], 0.5)
        self.assertAlmostEqual(s["rating_quality"], 6.0)

    def test_rating_scales_present(self):
        self.assertEqual(set(et.RATING_SCALES), {"instruction_understanding", "quality"})
        self.assertEqual(len(et.RATING_SCALES["quality"]), 7)


if __name__ == "__main__":
    unittest.main()
