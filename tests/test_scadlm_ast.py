"""Tests for programs.scadlm_ast (OpenSCAD lexer / parser / AST / unparser)."""

from __future__ import annotations

import unittest

from harnesscad.domain.programs.ast.scadlm_ast import (
    Argument,
    Assign,
    Binary,
    Block,
    Call,
    Comprehension,
    ForStmt,
    FunctionDef,
    IfStmt,
    Include,
    Index,
    LetExpr,
    ModuleCall,
    ModuleDef,
    Name,
    Num,
    Range,
    ScadSyntaxError,
    Str,
    Ternary,
    Unary,
    Vector,
    parse,
    parse_expression,
    tokenize,
    unparse,
    walk,
)


class TestTokenizer(unittest.TestCase):
    def test_strips_comments(self):
        toks = tokenize("a = 1; // hi\n/* block\n comment */ b = 2;")
        kinds = [(t.kind, t.value) for t in toks if t.kind != "EOF"]
        self.assertEqual(
            kinds,
            [("NAME", "a"), ("OP", "="), ("NUMBER", "1"), ("OP", ";"),
             ("NAME", "b"), ("OP", "="), ("NUMBER", "2"), ("OP", ";")],
        )

    def test_special_variables_and_numbers(self):
        toks = tokenize("$fn=1.5e2;")
        self.assertEqual(toks[0].value, "$fn")
        self.assertEqual(toks[2].kind, "NUMBER")
        self.assertEqual(float(toks[2].value), 150.0)

    def test_strings_with_escapes(self):
        toks = tokenize('t = "a\\"b";')
        self.assertEqual(toks[2].kind, "STRING")
        self.assertEqual(toks[2].value, 'a"b')

    def test_line_and_column_tracking(self):
        toks = tokenize("a=1;\nb=2;")
        b = [t for t in toks if t.value == "b"][0]
        self.assertEqual((b.line, b.column), (2, 1))

    def test_unterminated_string_raises(self):
        with self.assertRaises(ScadSyntaxError):
            tokenize('a = "oops;')

    def test_unterminated_block_comment_raises(self):
        with self.assertRaises(ScadSyntaxError):
            tokenize("/* nope")


class TestParseStatements(unittest.TestCase):
    def test_assignment(self):
        (stmt,) = parse("width = 10;")
        self.assertIsInstance(stmt, Assign)
        self.assertEqual(stmt.name, "width")
        self.assertEqual(stmt.value, Num(10.0))

    def test_module_call_with_children(self):
        (stmt,) = parse("difference() { cube(10); sphere(r=4); }")
        self.assertIsInstance(stmt, ModuleCall)
        self.assertEqual(stmt.name, "difference")
        self.assertEqual(len(stmt.children), 2)
        self.assertEqual(stmt.children[1].args[0].name, "r")

    def test_single_child_without_braces(self):
        (stmt,) = parse("translate([1,2,3]) cube(2);")
        self.assertEqual(stmt.name, "translate")
        self.assertEqual(len(stmt.children), 1)
        self.assertEqual(stmt.children[0].name, "cube")

    def test_module_definition(self):
        (stmt,) = parse("module foo(a, b=2) { cube(a); }")
        self.assertIsInstance(stmt, ModuleDef)
        self.assertEqual([p.name for p in stmt.params], ["a", "b"])
        self.assertEqual(stmt.params[1].default, Num(2.0))
        self.assertIsInstance(stmt.body, Block)

    def test_function_definition(self):
        (stmt,) = parse("function sq(x) = x * x;")
        self.assertIsInstance(stmt, FunctionDef)
        self.assertIsInstance(stmt.body, Binary)

    def test_if_else(self):
        (stmt,) = parse("if (a > 1) cube(1); else sphere(1);")
        self.assertIsInstance(stmt, IfStmt)
        self.assertIsInstance(stmt.orelse, ModuleCall)

    def test_for_and_intersection_for(self):
        a, b = parse("for (i = [0:3]) cube(i); intersection_for(j=[1,2]) sphere(j);")
        self.assertIsInstance(a, ForStmt)
        self.assertFalse(a.intersect)
        self.assertTrue(b.intersect)
        self.assertEqual(a.bindings[0][0], "i")
        self.assertIsInstance(a.bindings[0][1], Range)

    def test_modifier_characters(self):
        stmts = parse("*cube(1); #sphere(1); %cylinder(1); !cube(2);")
        self.assertEqual([s.modifier for s in stmts], ["*", "#", "%", "!"])

    def test_include_and_use(self):
        a, b = parse("include <lib/foo.scad>\nuse <bar.scad>\n")
        self.assertIsInstance(a, Include)
        self.assertEqual(a.kind, "include")
        self.assertEqual(a.path, "lib/foo.scad")
        self.assertEqual(b.kind, "use")

    def test_stray_semicolon_is_noop(self):
        stmts = parse(";;cube(1);")
        self.assertEqual(len(stmts), 3)

    def test_syntax_error_has_position(self):
        with self.assertRaises(ScadSyntaxError) as ctx:
            parse("cube(10)\nsphere(")
        self.assertEqual(ctx.exception.line, 2)

    def test_missing_semicolon_raises(self):
        with self.assertRaises(ScadSyntaxError):
            parse("a = 1")

    def test_unbalanced_brace_raises(self):
        with self.assertRaises(ScadSyntaxError):
            parse("union() { cube(1);")


class TestParseExpressions(unittest.TestCase):
    def test_precedence(self):
        e = parse_expression("1 + 2 * 3")
        self.assertEqual(e.op, "+")
        self.assertEqual(e.right.op, "*")

    def test_power_right_associative(self):
        e = parse_expression("2 ^ 3 ^ 2")
        self.assertEqual(e.op, "^")
        self.assertEqual(e.right.op, "^")

    def test_unary_and_not(self):
        e = parse_expression("-x")
        self.assertIsInstance(e, Unary)
        self.assertEqual(parse_expression("!b").op, "!")

    def test_ternary(self):
        e = parse_expression("a ? 1 : 2")
        self.assertIsInstance(e, Ternary)

    def test_vector_index_member(self):
        e = parse_expression("v[1]")
        self.assertIsInstance(e, Index)
        self.assertEqual(parse_expression("p.z").name, "z")

    def test_range_forms(self):
        r1 = parse_expression("[0:5]")
        r2 = parse_expression("[0:2:10]")
        self.assertIsNone(r1.step)
        self.assertEqual(r2.step, Num(2.0))
        self.assertEqual(r2.end, Num(10.0))

    def test_vector_literal(self):
        v = parse_expression("[1, 2, 3]")
        self.assertIsInstance(v, Vector)
        self.assertEqual(len(v.items), 3)

    def test_empty_vector(self):
        self.assertEqual(parse_expression("[]"), Vector([]))

    def test_function_call(self):
        e = parse_expression("max(1, 2)")
        self.assertIsInstance(e, Call)
        self.assertEqual(e.name, Name("max"))

    def test_let_expression(self):
        e = parse_expression("let(a = 2) a + 1")
        self.assertIsInstance(e, LetExpr)
        self.assertEqual(e.bindings, [("a", Num(2.0))])

    def test_list_comprehension_for_if(self):
        e = parse_expression("[for (i = [0:4]) if (i % 2 == 0) i]")
        self.assertIsInstance(e, Vector)
        comp = e.items[0]
        self.assertIsInstance(comp, Comprehension)
        self.assertEqual(comp.kind, "for")
        self.assertEqual(comp.body.kind, "if")

    def test_list_comprehension_each(self):
        e = parse_expression("[each [1, 2], 3]")
        self.assertEqual(e.items[0].kind, "each")
        self.assertEqual(e.items[1], Num(3.0))

    def test_string_literal(self):
        self.assertEqual(parse_expression('"hi"'), Str("hi"))


class TestUnparse(unittest.TestCase):
    def _roundtrip(self, src: str) -> str:
        first = unparse(parse(src))
        second = unparse(parse(first))
        self.assertEqual(first, second)
        return first

    def test_roundtrip_is_idempotent(self):
        src = """
        // a bracket
        width = 20;
        module bracket(w = 10, h = 5) {
            difference() {
                cube([w, w, h], center = true);
                translate([0, 0, -1]) cylinder(h = h + 2, r = 2, $fn = 32);
            }
        }
        for (i = [0 : 2 : 6]) translate([i * 5, 0, 0]) bracket(w = width);
        """
        out = self._roundtrip(src)
        self.assertIn("module bracket(w = 10, h = 5) {", out)
        self.assertIn("cube([w, w, h], center = true);", out)
        self.assertIn("for (i = [0 : 2 : 6])", out)

    def test_parenthesisation_preserved(self):
        self.assertEqual(unparse(parse_expression("(1 + 2) * 3")), "(1 + 2) * 3")
        self.assertEqual(unparse(parse_expression("1 + 2 * 3")), "1 + 2 * 3")

    def test_modifier_roundtrip(self):
        self.assertEqual(self._roundtrip("#cube(1);"), "#cube(1);")

    def test_numbers_render_cleanly(self):
        self.assertEqual(unparse(parse_expression("10.0")), "10")
        self.assertEqual(unparse(parse_expression("2.5")), "2.5")

    def test_expression_unparse(self):
        self.assertEqual(unparse(parse_expression("a ? b : c")), "a ? b : c")


class TestWalk(unittest.TestCase):
    def test_walk_visits_nested_nodes(self):
        tree = parse("translate([1,2,3]) cube(size = 4);")
        names = [n.name for n in walk(tree) if isinstance(n, ModuleCall)]
        self.assertEqual(names, ["translate", "cube"])
        nums = [n.value for n in walk(tree) if isinstance(n, Num)]
        self.assertEqual(sorted(nums), [1.0, 2.0, 3.0, 4.0])

    def test_walk_is_deterministic(self):
        tree = parse("union() { cube(1); sphere(2); }")
        self.assertEqual([type(n).__name__ for n in walk(tree)],
                         [type(n).__name__ for n in walk(tree)])

    def test_walk_covers_arguments(self):
        tree = parse("cylinder(h = 4, r = 2);")
        args = [n for n in walk(tree) if isinstance(n, Argument)]
        # Arguments are unwrapped: their values are visited directly.
        self.assertEqual(args, [])
        self.assertEqual(len([n for n in walk(tree) if isinstance(n, Num)]), 2)


if __name__ == "__main__":
    unittest.main()
