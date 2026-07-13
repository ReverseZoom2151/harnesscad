"""Tests for programs.t2cq_ast (CadQuery-subset AST: build/serialize/parse/validate)."""

import ast as _ast
import unittest

from harnesscad.domain.programs.ast.t2cq_ast import (
    Assign,
    Call,
    Chain,
    CqProgram,
    VarRef,
    Workplane,
    format_arg,
    is_valid,
    parse_program,
    serialize,
    serialize_chain,
    validate,
)


def _cylinder_program() -> CqProgram:
    chain = Chain(Workplane("XY"), (Call("circle", (0.28125,)), Call("extrude", (0.1046,))))
    return CqProgram((Assign("part_1", chain),), "part_1")


class TestSerialize(unittest.TestCase):
    def test_serialize_cylinder(self):
        code = serialize(_cylinder_program())
        self.assertIn("import cadquery as cq", code)
        self.assertIn('part_1 = cq.Workplane("XY").circle(0.28125).extrude(0.1046)', code)

    def test_serialize_is_valid_python(self):
        code = serialize(_cylinder_program())
        _ast.parse(code)  # must not raise

    def test_format_arg_tuple_and_varref(self):
        self.assertEqual(format_arg((0.375, 0.2969)), "(0.375, 0.2969)")
        self.assertEqual(format_arg(VarRef("part_2")), "part_2")
        self.assertEqual(format_arg("XY"), "'XY'")
        self.assertEqual(format_arg((1.0,)), "(1.0,)")

    def test_serialize_chain_varref_root(self):
        chain = Chain(VarRef("part_1"), (Call("union", (VarRef("part_2"),)),))
        self.assertEqual(serialize_chain(chain), "part_1.union(part_2)")

    def test_result_line_emitted_when_last_var_differs(self):
        prog = CqProgram(
            (Assign("part_1", Chain(Workplane("XY"), (Call("box", (1.0, 2.0, 3.0)),))),
             Assign("part_2", Chain(Workplane("XY"), (Call("box", (1.0, 1.0, 1.0)),)))),
            "part_1",
        )
        code = serialize(prog)
        self.assertTrue(code.rstrip().endswith("result = part_1"))

    def test_serialize_bad_result_var_raises(self):
        prog = CqProgram((Assign("a", Chain(Workplane("XY"), (Call("close"),))),), "zzz")
        with self.assertRaises(ValueError):
            serialize(prog)


class TestParse(unittest.TestCase):
    def test_roundtrip_cylinder(self):
        prog = _cylinder_program()
        code = serialize(prog)
        parsed = parse_program(code)
        self.assertEqual(serialize(parsed), code)

    def test_parse_lines_and_arc(self):
        code = (
            "import cadquery as cq\n"
            'part_1 = cq.Workplane("XY").moveTo(0.0, 0.0).lineTo(0.75, 0.0)'
            ".threePointArc((0.375, 0.2969), (0.2188, 0.4531)).close().extrude(0.5625)\n"
        )
        prog = parse_program(code)
        self.assertEqual(len(prog.statements), 1)
        methods = [c.method for c in prog.statements[0].chain.calls]
        self.assertEqual(methods, ["moveTo", "lineTo", "threePointArc", "close", "extrude"])
        arc = prog.statements[0].chain.calls[2]
        self.assertEqual(arc.args, ((0.375, 0.2969), (0.2188, 0.4531)))

    def test_parse_varref_root_and_arg(self):
        code = "a = cq.Workplane(\"XY\").box(1, 1, 1)\nb = a.union(a)\n"
        prog = parse_program(code)
        self.assertIsInstance(prog.statements[1].chain.root, VarRef)
        self.assertEqual(prog.statements[1].chain.calls[0].args, (VarRef("a"),))
        self.assertEqual(prog.result_var, "b")

    def test_parse_rejects_non_assignment(self):
        with self.assertRaises(ValueError):
            parse_program("cq.Workplane('XY').box(1, 1, 1)\n")

    def test_parse_rejects_keyword_args(self):
        with self.assertRaises(ValueError):
            parse_program("a = cq.Workplane('XY').extrude(1, both=True)\n")

    def test_parse_syntax_error(self):
        with self.assertRaises(SyntaxError):
            parse_program("a = cq.Workplane('XY'.box(\n")


class TestValidate(unittest.TestCase):
    def test_valid_program(self):
        self.assertTrue(is_valid(_cylinder_program()))
        self.assertEqual(validate(_cylinder_program()), [])

    def test_unknown_method(self):
        prog = CqProgram(
            (Assign("a", Chain(Workplane("XY"), (Call("frobnicate", (1.0,)),))),), "a")
        errs = validate(prog)
        self.assertTrue(any("frobnicate" in e for e in errs))

    def test_bad_arity(self):
        prog = CqProgram(
            (Assign("a", Chain(Workplane("XY"), (Call("circle", (1.0, 2.0)),))),), "a")
        errs = validate(prog)
        self.assertTrue(any("circle" in e and "args" in e for e in errs))

    def test_unknown_workplane(self):
        prog = CqProgram((Assign("a", Chain(Workplane("WW"), (Call("close"),))),), "a")
        self.assertTrue(any("workplane" in e for e in validate(prog)))

    def test_use_before_definition(self):
        prog = CqProgram(
            (Assign("a", Chain(VarRef("b"), (Call("close"),))),
             Assign("b", Chain(Workplane("XY"), (Call("box", (1.0, 1.0, 1.0)),)))),
            "a",
        )
        errs = validate(prog)
        self.assertTrue(any("used before assignment" in e for e in errs))

    def test_varref_argument_undefined(self):
        prog = CqProgram(
            (Assign("a", Chain(Workplane("XY"), (Call("union", (VarRef("ghost"),)),))),), "a")
        self.assertTrue(any("ghost" in e for e in validate(prog)))


class MoveToAritySignatureTest(unittest.TestCase):
    """moveTo(x=0, y=0): nought, one and two positional args are all legal.

    A (2, 2) arity here made validate() reject real CadQuery programs; the
    corpus in resources/cadbible/cadquery-contrib calls moveTo with one arg.
    """

    def _validate(self, call):
        code = 'import cadquery as cq\npart_1 = cq.Workplane("XY").%s.close().extrude(0.5)\n' % call
        return validate(parse_program(code))

    def test_two_positional_args_valid(self):
        self.assertEqual(self._validate("moveTo(1.0, 2.0).lineTo(3.0, 4.0)"), [])

    def test_one_positional_arg_valid(self):
        self.assertEqual(self._validate("moveTo(1.0).lineTo(3.0, 4.0)"), [])

    def test_no_positional_args_valid(self):
        self.assertEqual(self._validate("moveTo().lineTo(3.0, 4.0)"), [])

    def test_three_positional_args_still_rejected(self):
        self.assertTrue(self._validate("moveTo(1.0, 2.0, 3.0).lineTo(3.0, 4.0)"))


if __name__ == "__main__":
    unittest.main()
