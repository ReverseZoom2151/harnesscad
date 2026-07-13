"""Tests for reconstruction.t2cq_translate (DeepCAD -> CadQuery code translation)."""

import unittest

from harnesscad.domain.reconstruction.tokens.deepcad_command_spec import command
from harnesscad.domain.programs.ast.t2cq_ast import VarRef, parse_program, validate, serialize
from harnesscad.domain.reconstruction.translate.t2cq_translate import (
    translate_to_code,
    translate_to_program,
    _arc_midpoint,
)


def _circle_extrude():
    return [
        command("SOL"),
        command("Circle", x=0.375, y=0.375, r=0.375),
        command("Ext", theta=0.0, phi=0.0, gamma=0.0, px=0.0, py=0.0, pz=0.0,
                s=0.75, e1=0.1046, e2=0.0, b=0, u=0),
        command("EOS"),
    ]


def _rect_extrude():
    return [
        command("SOL"),
        command("Line", x=0.75, y=0.0),
        command("Line", x=0.75, y=0.5),
        command("Line", x=0.0, y=0.5),
        command("Line", x=0.0, y=0.0),
        command("Ext", theta=0.0, phi=0.0, gamma=0.0, px=0.0, py=0.0, pz=0.0,
                s=1.0, e1=0.5, e2=0.0, b=0, u=0),
        command("EOS"),
    ]


class TestCircle(unittest.TestCase):
    def test_circle_program(self):
        prog = translate_to_program(_circle_extrude())
        self.assertEqual(len(prog.statements), 1)
        methods = [c.method for c in prog.statements[0].chain.calls]
        self.assertEqual(methods, ["moveTo", "circle", "extrude"])

    def test_circle_code_parses_and_valid(self):
        code = translate_to_code(_circle_extrude())
        self.assertIn(".circle(0.375)", code)
        self.assertIn(".extrude(0.1046)", code)
        self.assertEqual(validate(parse_program(code)), [])

    def test_origin_circle_omits_moveto(self):
        cmds = [
            command("SOL"),
            command("Circle", x=0.0, y=0.0, r=0.2),
            command("Ext", e1=0.5, b=0, u=0),
            command("EOS"),
        ]
        methods = [c.method for c in translate_to_program(cmds).statements[0].chain.calls]
        self.assertEqual(methods, ["circle", "extrude"])


class TestPolyline(unittest.TestCase):
    def test_rect_program(self):
        prog = translate_to_program(_rect_extrude())
        methods = [c.method for c in prog.statements[0].chain.calls]
        self.assertEqual(methods[0], "moveTo")
        self.assertEqual(methods[-2], "close")
        self.assertEqual(methods[-1], "extrude")
        self.assertEqual(methods.count("lineTo"), 4)

    def test_rect_code_valid(self):
        code = translate_to_code(_rect_extrude())
        self.assertEqual(validate(parse_program(code)), [])
        self.assertTrue(code.startswith("import cadquery as cq"))


class TestArc(unittest.TestCase):
    def test_arc_produces_threepointarc(self):
        cmds = [
            command("SOL"),
            command("Line", x=1.0, y=0.0),
            command("Arc", x=0.0, y=1.0, alpha=1.5708, f=1),
            command("Line", x=0.0, y=0.0),
            command("Ext", e1=0.3, b=0, u=0),
            command("EOS"),
        ]
        prog = translate_to_program(cmds)
        methods = [c.method for c in prog.statements[0].chain.calls]
        self.assertIn("threePointArc", methods)
        self.assertEqual(validate(prog), [])

    def test_arc_midpoint_semicircle_bulges_outward(self):
        # A 180-degree ccw arc from (1,0) to (-1,0) bulges to +y.
        mid = _arc_midpoint((1.0, 0.0), (-1.0, 0.0), 3.14159, True)
        self.assertAlmostEqual(mid[0], 0.0, places=3)
        self.assertGreater(mid[1], 0.5)

    def test_arc_midpoint_flag_flips_side(self):
        up = _arc_midpoint((1.0, 0.0), (-1.0, 0.0), 3.14159, True)
        down = _arc_midpoint((1.0, 0.0), (-1.0, 0.0), 3.14159, False)
        self.assertGreater(up[1], 0.0)
        self.assertLess(down[1], 0.0)


class TestBooleanAndPose(unittest.TestCase):
    def test_two_parts_union(self):
        cmds = [
            command("SOL"), command("Circle", x=0.0, y=0.0, r=0.3),
            command("Ext", e1=0.5, b=0, u=0),
            command("SOL"), command("Circle", x=0.0, y=0.0, r=0.2),
            command("Ext", e1=0.5, b=1, u=0),
            command("EOS"),
        ]
        prog = translate_to_program(cmds)
        vars_ = [s.var for s in prog.statements]
        self.assertIn("part_1", vars_)
        self.assertIn("part_2", vars_)
        # A union statement references both parts.
        union_stmt = prog.statements[-1]
        self.assertEqual(union_stmt.chain.calls[0].method, "union")
        self.assertEqual(union_stmt.chain.calls[0].args, (VarRef("part_2"),))
        self.assertEqual(prog.result_var, union_stmt.var)
        self.assertEqual(validate(prog), [])

    def test_cut_boolean(self):
        cmds = [
            command("SOL"), command("Circle", x=0.0, y=0.0, r=0.3),
            command("Ext", e1=0.5, b=0, u=0),
            command("SOL"), command("Circle", x=0.0, y=0.0, r=0.1),
            command("Ext", e1=0.5, b=2, u=0),
            command("EOS"),
        ]
        prog = translate_to_program(cmds)
        self.assertEqual(prog.statements[-1].chain.calls[0].method, "cut")

    def test_pose_emits_rotate_translate(self):
        cmds = [
            command("SOL"), command("Circle", x=0.0, y=0.0, r=0.3),
            command("Ext", theta=0.0, phi=0.0, gamma=1.5708,
                    px=0.0, py=0.5625, pz=0.0, e1=0.5, b=0, u=0),
            command("EOS"),
        ]
        prog = translate_to_program(cmds)
        methods = [c.method for c in prog.statements[0].chain.calls]
        self.assertIn("rotate", methods)
        self.assertIn("translate", methods)
        code = serialize(prog)
        self.assertIn("90.0", code)  # gamma 1.5708 rad ~= 90 deg


class TestErrors(unittest.TestCase):
    def test_no_extrude_raises(self):
        with self.assertRaises(ValueError):
            translate_to_program([command("SOL"), command("Circle", x=0, y=0, r=1),
                                  command("EOS")])

    def test_depth_sums_both_sides(self):
        cmds = [
            command("SOL"), command("Circle", x=0.0, y=0.0, r=0.3),
            command("Ext", e1=0.4, e2=0.1, b=0, u=0),
            command("EOS"),
        ]
        code = translate_to_code(cmds)
        self.assertIn(".extrude(0.5)", code)


if __name__ == "__main__":
    unittest.main()
