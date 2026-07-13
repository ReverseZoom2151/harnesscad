import unittest

from harnesscad.domain.programs.validate.blenderllm_bpy_script import (
    BpyCall,
    check_syntax,
    extract_calls,
    extract_primitives,
    is_recognized_vocabulary,
    vocabulary_coverage,
)

GOOD = (
    "import bpy\n"
    "bpy.ops.mesh.primitive_cube_add(size=2, location=(0, 0, 0))\n"
    "bpy.ops.transform.resize(value=(1, 2, 3))\n"
    "bpy.ops.mesh.primitive_uv_sphere_add(radius=1)\n"
)

BAD = "bpy.ops.mesh.primitive_cube_add(size=2\n"  # unbalanced paren


class SyntaxTests(unittest.TestCase):
    def test_valid_script(self):
        result = check_syntax(GOOD)
        self.assertTrue(result.ok)
        self.assertIsNone(result.error)

    def test_invalid_script_reports_location(self):
        result = check_syntax(BAD)
        self.assertFalse(result.ok)
        self.assertIsNotNone(result.error)
        self.assertIsNotNone(result.lineno)


class ExtractTests(unittest.TestCase):
    def test_extracts_ordered_calls(self):
        calls = extract_calls(GOOD)
        self.assertEqual(
            [c.dotted for c in calls],
            [
                "bpy.ops.mesh.primitive_cube_add",
                "bpy.ops.transform.resize",
                "bpy.ops.mesh.primitive_uv_sphere_add",
            ],
        )

    def test_keyword_and_arg_capture(self):
        cube = extract_calls(GOOD)[0]
        self.assertEqual(cube.group, "mesh")
        self.assertEqual(cube.op, "primitive_cube_add")
        self.assertEqual(cube.num_args, 0)
        self.assertEqual(cube.keywords, ("size", "location"))

    def test_ignores_non_bpy_ops_calls(self):
        script = "import bpy\nprint('hi')\nbpy.context.scene.frame_set(1)\n"
        self.assertEqual(extract_calls(script), [])

    def test_extract_primitives_only(self):
        prims = extract_primitives(GOOD)
        self.assertEqual([c.op for c in prims],
                         ["primitive_cube_add", "primitive_uv_sphere_add"])

    def test_syntax_error_raises_on_extract(self):
        with self.assertRaises(SyntaxError):
            extract_calls(BAD)


class VocabularyTests(unittest.TestCase):
    def test_primitive_and_transform_recognized(self):
        cube = BpyCall("mesh", "primitive_cube_add", 0, ())
        resize = BpyCall("transform", "resize", 0, ("value",))
        self.assertTrue(cube.is_primitive)
        self.assertTrue(resize.is_transform)
        self.assertTrue(is_recognized_vocabulary(cube))
        self.assertTrue(is_recognized_vocabulary(resize))

    def test_unknown_group_not_recognized(self):
        weird = BpyCall("render", "render", 0, ())
        self.assertFalse(is_recognized_vocabulary(weird))

    def test_coverage_full_for_known_script(self):
        self.assertEqual(vocabulary_coverage(GOOD), 1.0)

    def test_coverage_empty_script_is_one(self):
        self.assertEqual(vocabulary_coverage("import bpy\nx = 1\n"), 1.0)

    def test_coverage_partial(self):
        script = (
            "import bpy\n"
            "bpy.ops.mesh.primitive_cube_add()\n"
            "bpy.ops.render.render()\n"
        )
        self.assertEqual(vocabulary_coverage(script), 0.5)


if __name__ == "__main__":
    unittest.main()
