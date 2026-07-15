import unittest

from harnesscad.domain.spec import zoo_catalog as zc


class TestStdlib(unittest.TestCase):
    def test_modules_sorted_and_present(self):
        mods = zc.std_modules()
        self.assertEqual(mods, sorted(mods))
        for m in ("sketch", "solid", "csg", "math", "constraints"):
            self.assertIn(m, mods)

    def test_function_names_qualified(self):
        names = zc.std_function_names("csg")
        self.assertIn("csg::union", names)
        self.assertIn("csg::subtract", names)
        self.assertTrue(all(n.startswith("csg::") for n in names))

    def test_all_function_names_deduped_sorted(self):
        names = zc.std_function_names()
        self.assertEqual(names, sorted(set(names)))


class TestEngineOps(unittest.TestCase):
    def test_known_ops(self):
        for op in ("extrude", "revolve", "loft", "sweep", "boolean_union",
                   "import_files", "export", "take_snapshot"):
            self.assertTrue(zc.is_engine_op(op), op)

    def test_unknown_op(self):
        self.assertFalse(zc.is_engine_op("frobnicate"))

    def test_op_set_unique(self):
        self.assertEqual(len(zc.ENGINE_OPS), len(set(zc.ENGINE_OPS)))


class TestFormatMatrix(unittest.TestCase):
    def test_can_convert(self):
        self.assertTrue(zc.can_convert("step", "stl"))
        self.assertTrue(zc.can_convert("sldprt", "obj"))
        self.assertFalse(zc.can_convert("stl", "stl"))  # same fmt
        self.assertFalse(zc.can_convert("dxf", "stl"))  # dxf not importable
        self.assertFalse(zc.can_convert("step", "dxf"))  # dxf is 2d only

    def test_conversions_from_step(self):
        outs = zc.conversions_from("step")
        self.assertIn("stl", outs)
        self.assertIn("glb", outs)
        self.assertNotIn("step", outs)

    def test_conversions_to_glb(self):
        srcs = zc.conversions_to("glb")
        self.assertIn("step", srcs)
        self.assertIn("sldprt", srcs)

    def test_extension_resolution(self):
        self.assertEqual(zc.import_format_for_extension("part.stp"), "step")
        self.assertEqual(zc.import_format_for_extension("model.STL"), "stl")
        self.assertEqual(zc.import_format_for_extension("x.dae"), "")  # disabled
        self.assertEqual(zc.import_format_for_extension("noext"), "")

    def test_conversion_matrix_wellformed(self):
        for src, dst in zc.CONVERSION_MATRIX:
            self.assertNotEqual(src, dst)
            self.assertIn(src, zc.IMPORT_FORMATS)
            self.assertIn(dst, zc.EXPORT_FORMATS_3D)
        self.assertEqual(len(zc.CONVERSION_MATRIX), len(set(zc.CONVERSION_MATRIX)))


if __name__ == "__main__":
    unittest.main()
