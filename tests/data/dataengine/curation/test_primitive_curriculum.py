import unittest

from harnesscad.data.dataengine.curation.primitive_curriculum import (
    PRIMITIVE_ORDER,
    curriculum_indices,
    difficulty_key,
    order_curriculum,
    primitive_rank,
    stage_batches,
)


class TestPrimitiveRank(unittest.TestCase):
    def test_order(self):
        ranks = [primitive_rank(p) for p in PRIMITIVE_ORDER]
        self.assertEqual(ranks, [0, 1, 2, 3, 4])

    def test_loop_before_mse(self):
        self.assertLess(primitive_rank("L"), primitive_rank("MSE"))

    def test_unknown_raises(self):
        with self.assertRaises(ValueError):
            primitive_rank("X")


class TestDifficultyKey(unittest.TestCase):
    def test_key(self):
        self.assertEqual(
            difficulty_key({"primitive": "S", "curves": 4}), (2, 4))

    def test_negative_curves_raises(self):
        with self.assertRaises(ValueError):
            difficulty_key({"primitive": "L", "curves": -1})


class TestOrderCurriculum(unittest.TestCase):
    def setUp(self):
        self.samples = [
            {"id": "a", "primitive": "SE", "curves": 2},
            {"id": "b", "primitive": "L", "curves": 3},
            {"id": "c", "primitive": "L", "curves": 1},
            {"id": "d", "primitive": "MSE", "curves": 1},
            {"id": "e", "primitive": "F", "curves": 5},
        ]

    def test_level_then_curves(self):
        ids = [s["id"] for s in order_curriculum(self.samples)]
        # L(1), L(3), F(5), SE(2), MSE(1)
        self.assertEqual(ids, ["c", "b", "e", "a", "d"])

    def test_stable_within_key(self):
        samples = [
            {"id": "x", "primitive": "L", "curves": 2},
            {"id": "y", "primitive": "L", "curves": 2},
        ]
        ids = [s["id"] for s in order_curriculum(samples)]
        self.assertEqual(ids, ["x", "y"])


class TestStageBatches(unittest.TestCase):
    def test_stages(self):
        samples = [
            {"id": "a", "primitive": "L", "curves": 2},
            {"id": "b", "primitive": "L", "curves": 1},
            {"id": "c", "primitive": "SE", "curves": 1},
        ]
        stages = stage_batches(samples)
        self.assertEqual([s[0] for s in stages], ["L", "SE"])
        self.assertEqual([x["id"] for x in stages[0][1]], ["b", "a"])
        self.assertEqual([x["id"] for x in stages[1][1]], ["c"])

    def test_empty_levels_omitted(self):
        stages = stage_batches([{"id": "z", "primitive": "MSE", "curves": 0}])
        self.assertEqual(len(stages), 1)
        self.assertEqual(stages[0][0], "MSE")


class TestCurriculumIndices(unittest.TestCase):
    def test_indices(self):
        samples = [
            {"primitive": "MSE", "curves": 1},
            {"primitive": "L", "curves": 1},
        ]
        self.assertEqual(curriculum_indices(samples), [1, 0])


if __name__ == "__main__":
    unittest.main()
