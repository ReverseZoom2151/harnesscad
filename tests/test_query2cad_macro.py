import unittest

from generation.query2cad_macro import (
    Primitive, BooleanOp, FreeCADMacro, estimate_difficulty,
    PRIMITIVE_PARAMS, BOOLEAN_FUNCS,
)


class TestPrimitive(unittest.TestCase):
    def test_box_lines(self):
        p = Primitive("Box", "box", {"length": 10, "width": 10, "height": 5})
        lines = p.to_lines()
        self.assertEqual(lines[0], "Box = Part.Box()")
        self.assertIn("Box.Length = 10", lines)
        self.assertIn("Box.Height = 5", lines)

    def test_placement_emitted_when_offset(self):
        p = Primitive("S", "sphere", {"radius": 8}, position=(0, 0, 10))
        lines = p.to_lines()
        self.assertTrue(any("Placement" in l for l in lines))

    def test_no_placement_at_origin(self):
        p = Primitive("S", "sphere", {"radius": 8})
        self.assertFalse(any("Placement" in l for l in p.to_lines()))

    def test_cone_underscore_param_camelcased(self):
        p = Primitive("C", "cone", {"radius1": 5, "radius2": 2, "height": 9})
        lines = p.to_lines()
        self.assertIn("C.Radius1 = 5", lines)
        self.assertIn("C.Radius2 = 2", lines)

    def test_float_formatting(self):
        p = Primitive("S", "sphere", {"radius": 2.5})
        self.assertIn("S.Radius = 2.5", p.to_lines())

    def test_unknown_kind_rejected(self):
        with self.assertRaises(ValueError):
            Primitive("X", "pyramid", {"a": 1})

    def test_wrong_params_rejected(self):
        with self.assertRaises(ValueError):
            Primitive("B", "box", {"length": 1})

    def test_negative_param_rejected(self):
        with self.assertRaises(ValueError):
            Primitive("S", "sphere", {"radius": -1})

    def test_bool_param_rejected(self):
        with self.assertRaises(ValueError):
            Primitive("S", "sphere", {"radius": True})

    def test_all_kinds_have_params(self):
        for kind, params in PRIMITIVE_PARAMS.items():
            self.assertTrue(len(params) >= 1)


class TestBooleanOp(unittest.TestCase):
    def test_fuse_line(self):
        b = BooleanOp("U", "fuse", "A", "B")
        self.assertEqual(b.to_lines(), ["U = A.fuse(B)"])

    def test_all_ops_map(self):
        for op in ("fuse", "cut", "common"):
            self.assertIn(BOOLEAN_FUNCS[op], BooleanOp("R", op, "A", "B").to_lines()[0])

    def test_unknown_op_rejected(self):
        with self.assertRaises(ValueError):
            BooleanOp("R", "xor", "A", "B")


class TestMacro(unittest.TestCase):
    def _cube_on_sphere(self):
        return FreeCADMacro(
            primitives=[
                Primitive("Sph", "sphere", {"radius": 8}),
                Primitive("Box", "box", {"length": 10, "width": 10, "height": 10},
                          position=(0, 0, 8)),
            ],
            booleans=[BooleanOp("Model", "fuse", "Box", "Sph")],
        )

    def test_source_scaffolding(self):
        src = self._cube_on_sphere().to_source()
        self.assertIn("import FreeCAD", src)
        self.assertIn("import Part", src)
        self.assertIn("Part.show(Model)", src)
        self.assertIn("doc.recompute()", src)
        self.assertTrue(src.endswith("\n"))

    def test_deterministic_source(self):
        self.assertEqual(self._cube_on_sphere().to_source(),
                         self._cube_on_sphere().to_source())

    def test_final_shape_defaults_to_last(self):
        m = FreeCADMacro(primitives=[Primitive("S", "sphere", {"radius": 3})])
        self.assertEqual(m.final_shape(), "S")

    def test_explicit_result(self):
        m = self._cube_on_sphere()
        m.result = "Sph"
        self.assertEqual(m.final_shape(), "Sph")

    def test_duplicate_name_rejected(self):
        m = FreeCADMacro(primitives=[
            Primitive("A", "sphere", {"radius": 1}),
            Primitive("A", "sphere", {"radius": 2}),
        ])
        with self.assertRaises(ValueError):
            m.validate()

    def test_boolean_undefined_operand_rejected(self):
        m = FreeCADMacro(
            primitives=[Primitive("A", "sphere", {"radius": 1})],
            booleans=[BooleanOp("R", "cut", "A", "Ghost")])
        with self.assertRaises(ValueError):
            m.validate()

    def test_bad_result_rejected(self):
        m = FreeCADMacro(primitives=[Primitive("A", "sphere", {"radius": 1})],
                         result="Nope")
        with self.assertRaises(ValueError):
            m.validate()

    def test_empty_macro_rejected(self):
        with self.assertRaises(ValueError):
            FreeCADMacro().validate()

    def test_operation_summary(self):
        s = self._cube_on_sphere().operation_summary()
        self.assertEqual(s["num_primitives"], 2)
        self.assertEqual(s["num_booleans"], 1)
        self.assertEqual(s["primitive_kinds"], {"sphere": 1, "box": 1})
        self.assertEqual(s["boolean_ops"], {"fuse": 1})


class TestDifficulty(unittest.TestCase):
    def test_easy_single_primitive(self):
        m = FreeCADMacro(primitives=[Primitive("C", "box",
                         {"length": 10, "width": 10, "height": 10})])
        self.assertEqual(estimate_difficulty(m), "easy")

    def test_medium_placement(self):
        m = FreeCADMacro(primitives=[
            Primitive("A", "sphere", {"radius": 8}),
            Primitive("B", "box", {"length": 10, "width": 10, "height": 10},
                      position=(0, 0, 8)),
        ], booleans=[BooleanOp("R", "fuse", "A", "B")])
        self.assertEqual(estimate_difficulty(m), "medium")

    def test_hard_many_primitives(self):
        prims = [Primitive("P%d" % i, "sphere", {"radius": i + 1})
                 for i in range(4)]
        m = FreeCADMacro(primitives=prims)
        self.assertEqual(estimate_difficulty(m), "hard")

    def test_hard_many_booleans(self):
        m = FreeCADMacro(
            primitives=[Primitive("A", "sphere", {"radius": 1}),
                        Primitive("B", "sphere", {"radius": 2}),
                        Primitive("C", "sphere", {"radius": 3})],
            booleans=[BooleanOp("R1", "fuse", "A", "B"),
                      BooleanOp("R2", "cut", "R1", "C")])
        self.assertEqual(estimate_difficulty(m), "hard")


if __name__ == "__main__":
    unittest.main()
