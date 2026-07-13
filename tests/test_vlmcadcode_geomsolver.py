import unittest

from harnesscad.eval.bench.judges.vlmcadcode_geomsolver import (
    CATEGORIES, geometric_properties, solver_feedback, verbalize,
)


def _unit_cube_mesh(scale=1.0):
    v = [(0, 0, 0), (scale, 0, 0), (scale, scale, 0), (0, scale, 0),
         (0, 0, scale), (scale, 0, scale), (scale, scale, scale), (0, scale, scale)]
    f = [
        (0, 2, 1), (0, 3, 2),        # bottom (z=0), normals down
        (4, 5, 6), (4, 6, 7),        # top (z=scale)
        (0, 1, 5), (0, 5, 4),        # y=0
        (2, 3, 7), (2, 7, 6),        # y=scale
        (1, 2, 6), (1, 6, 5),        # x=scale
        (3, 0, 4), (3, 4, 7),        # x=0
    ]
    return v, f


class TestProperties(unittest.TestCase):
    def test_thirteen_categories(self):
        self.assertEqual(len(CATEGORIES), 13)
        props = geometric_properties(*_unit_cube_mesh())
        self.assertEqual(set(props), set(CATEGORIES))

    def test_cube_geometry(self):
        props = geometric_properties(*_unit_cube_mesh(scale=2.0))
        self.assertAlmostEqual(props["width"], 2.0)
        self.assertAlmostEqual(props["height"], 2.0)
        self.assertAlmostEqual(props["depth"], 2.0)
        self.assertEqual(props["num_vertices"], 8.0)
        self.assertEqual(props["num_faces"], 12.0)
        self.assertEqual(props["num_edges"], 18.0)  # 12 cube edges + 6 face diagonals
        self.assertAlmostEqual(props["volume"], 8.0)
        self.assertAlmostEqual(props["surface_area"], 24.0)
        self.assertAlmostEqual(props["bbox_volume"], 8.0)
        self.assertAlmostEqual(props["centroid_x"], 1.0)

    def test_non_triangle_raises(self):
        with self.assertRaises(ValueError):
            geometric_properties([(0, 0, 0)], [(0, 0, 0, 0)])


class TestFeedback(unittest.TestCase):
    def test_paired_feedback_records(self):
        fb = solver_feedback(_unit_cube_mesh(scale=1.0), _unit_cube_mesh(scale=2.0))
        self.assertEqual(len(fb), 13)
        width = next(r for r in fb if r["category"] == "width")
        self.assertAlmostEqual(width["generated"], 1.0)
        self.assertAlmostEqual(width["ground_truth"], 2.0)
        self.assertAlmostEqual(width["abs_diff"], 1.0)
        self.assertAlmostEqual(width["rel_diff"], -0.5)

    def test_verbalize_reports_differences(self):
        fb = solver_feedback(_unit_cube_mesh(scale=1.0), _unit_cube_mesh(scale=2.0))
        text = verbalize(fb)
        self.assertIn("smaller", text)
        self.assertIn("width", text)

    def test_verbalize_match(self):
        fb = solver_feedback(_unit_cube_mesh(scale=2.0), _unit_cube_mesh(scale=2.0))
        self.assertIn("matches the ground truth", verbalize(fb))


if __name__ == "__main__":
    unittest.main()
