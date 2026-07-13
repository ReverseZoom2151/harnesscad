import unittest

from harnesscad.eval.bench.data.stratify_mesh_complexity import (
    GEOMETRIC_COMPLEXITY_LEVELS, mesh_complexity_score, split_mesh_complexity,
    compilation_difficulty, classify_geometric_complexity, object_statistics,
    dataset_statistics,
)


class TestMeshComplexity(unittest.TestCase):
    def test_score(self):
        self.assertEqual(mesh_complexity_score(6, 8), 14)

    def test_median_split(self):
        objs = [
            {"vertices": 6, "faces": 8},     # 14 -> simple
            {"vertices": 25, "faces": 47},   # 72 (median) -> simple (<=)
            {"vertices": 100, "faces": 200},  # 300 -> complex
        ]
        labeled = split_mesh_complexity(objs)
        self.assertEqual([o["mesh_complexity"] for o in labeled],
                         ["simple", "simple", "complex"])

    def test_empty(self):
        self.assertEqual(split_mesh_complexity([]), [])


class TestCompilationDifficulty(unittest.TestCase):
    def test_easy(self):
        label, n = compilation_difficulty([True, True, True, True, False, False])
        self.assertEqual(label, "easy")
        self.assertEqual(n, 4)

    def test_hard(self):
        label, n = compilation_difficulty([True, True, True, False, False, False])
        self.assertEqual(label, "hard")
        self.assertEqual(n, 3)

    def test_wrong_count(self):
        with self.assertRaises(ValueError):
            compilation_difficulty([True, False])


class TestGeometricComplexity(unittest.TestCase):
    def test_levels(self):
        self.assertEqual(len(GEOMETRIC_COMPLEXITY_LEVELS), 4)

    def test_normalise(self):
        self.assertEqual(classify_geometric_complexity("Very Complex"), "very_complex")

    def test_unknown(self):
        with self.assertRaises(ValueError):
            classify_geometric_complexity("nonsense")


class TestStatistics(unittest.TestCase):
    def test_object_statistics(self):
        obj = {
            "vertices": 10, "faces": 20,
            "description": "Draw a desk. It has four legs.",
            "code": "import cadquery as cq\n\nresult = cq.Workplane()\n",
        }
        s = object_statistics(obj)
        self.assertEqual(s["words"], 7)
        self.assertEqual(s["sentences"], 2)
        self.assertEqual(s["code_lines"], 2)
        self.assertEqual(s["code_tokens"], 7)

    def test_single_sentence_no_punct(self):
        obj = {"vertices": 1, "faces": 1, "description": "a cube", "code": "x=1"}
        self.assertEqual(object_statistics(obj)["sentences"], 1)

    def test_dataset_statistics(self):
        objs = [
            {"vertices": 6, "faces": 8, "description": "one two", "code": "a\nb"},
            {"vertices": 10, "faces": 12, "description": "one two three", "code": "a"},
        ]
        ds = dataset_statistics(objs)
        self.assertEqual(ds["datapoints"], 2)
        self.assertEqual(ds["vertices"]["min"], 6)
        self.assertEqual(ds["vertices"]["max"], 10)
        self.assertEqual(ds["vertices"]["avg"], 8.0)

    def test_empty_dataset(self):
        with self.assertRaises(ValueError):
            dataset_statistics([])


if __name__ == "__main__":
    unittest.main()
