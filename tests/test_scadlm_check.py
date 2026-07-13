"""Tests for programs.scadlm_check (OpenSCAD static validity gate)."""

from __future__ import annotations

import unittest

from harnesscad.domain.programs.validate.openscad_check import (
    BUILTIN_MODULES,
    Issue,
    check,
    format_report,
    is_valid,
)


def codes(source: str):
    return [i.code for i in check(source)]


def errors(source: str):
    return [i for i in check(source) if i.severity == "error"]


class TestSyntax(unittest.TestCase):
    def test_syntax_error_is_reported_with_line(self):
        issues = check("cube(10);\nsphere(")
        self.assertEqual(len(issues), 1)
        self.assertEqual(issues[0].code, "syntax")
        self.assertEqual(issues[0].line, 2)
        self.assertFalse(is_valid("cube(10);\nsphere("))

    def test_valid_program_passes(self):
        src = """
        module bracket(w = 10, h = 5) {
            difference() {
                cube([w, w, h], center = true);
                translate([0, 0, -1]) cylinder(h = h + 2, r = 2, $fn = 32);
            }
        }
        bracket();
        """
        self.assertTrue(is_valid(src))
        self.assertEqual(errors(src), [])


class TestUnknownNames(unittest.TestCase):
    def test_unknown_module(self):
        self.assertIn("unknown-module", codes("frobnicate(3);"))
        self.assertFalse(is_valid("frobnicate(3);"))

    def test_user_module_is_known(self):
        self.assertTrue(is_valid("module m() { cube(1); } m();"))

    def test_module_defined_after_use(self):
        self.assertTrue(is_valid("m(); module m() { cube(1); }"))

    def test_unknown_function(self):
        self.assertIn("unknown-function", codes("cube(bogus(2));"))

    def test_builtin_functions_are_known(self):
        self.assertTrue(is_valid("cube([sqrt(4), max(1,2), sin(30)]);"))

    def test_user_function_is_known(self):
        self.assertTrue(is_valid("function f(x) = x + 1; cube(f(2));"))


class TestVariables(unittest.TestCase):
    def test_undefined_variable(self):
        self.assertIn("undefined-variable", codes("cube(width);"))

    def test_assignment_is_hoisted_in_scope(self):
        self.assertTrue(is_valid("cube(w); w = 4;"))

    def test_module_parameters_are_in_scope(self):
        self.assertTrue(is_valid("module m(a) { cube(a); } m(2);"))

    def test_for_binding_is_in_scope(self):
        self.assertTrue(is_valid("for (i = [0:3]) translate([i, 0, 0]) cube(1);"))

    def test_for_binding_does_not_leak(self):
        self.assertIn("undefined-variable",
                      codes("for (i = [0:3]) cube(1);\ncube(i);"))

    def test_let_binding_is_in_scope(self):
        self.assertTrue(is_valid("let (s = 2) cube(s);"))

    def test_comprehension_binding_is_in_scope(self):
        self.assertTrue(is_valid("v = [for (i = [1:3]) i * 2]; cube(v[0]);"))

    def test_special_variables_are_defined(self):
        self.assertTrue(is_valid("sphere(r = 2, $fn = $preview ? 12 : 64);"))

    def test_bad_member_access(self):
        self.assertIn("bad-member", codes("v = [1,2,3]; cube(v.w);"))


class TestArguments(unittest.TestCase):
    def test_unknown_named_argument_on_builtin(self):
        self.assertIn("unknown-argument", codes("cube(size = 2, radius = 4);"))

    def test_known_named_arguments_pass(self):
        self.assertTrue(is_valid("cylinder(h = 4, r1 = 2, r2 = 1, center = true);"))

    def test_too_many_positional_arguments(self):
        self.assertIn("too-many-arguments", codes("sphere(1, 2, 3);"))

    def test_unknown_argument_on_user_module(self):
        self.assertIn("unknown-argument",
                      codes("module m(a) { cube(a); } m(b = 1);"))

    def test_too_many_positional_on_user_module(self):
        self.assertIn("too-many-arguments",
                      codes("module m(a) { cube(a); } m(1, 2);"))

    def test_positional_after_named(self):
        self.assertIn("positional-after-named",
                      codes("cube(size = 2, true);"))

    def test_duplicate_parameter(self):
        self.assertIn("duplicate-parameter",
                      codes("module m(a, a) { cube(1); } m(1);"))

    def test_function_arity(self):
        self.assertIn("too-many-arguments",
                      codes("function f(x) = x; cube(f(1, 2));"))

    def test_echo_accepts_any_arity(self):
        self.assertTrue(is_valid("cube(1); echo(1, 2, 3, 4);"))


class TestSemanticWarnings(unittest.TestCase):
    def test_degenerate_difference(self):
        self.assertIn("degenerate-boolean", codes("difference() { cube(1); }"))
        self.assertTrue(is_valid("difference() { cube(1); }"))  # warning only

    def test_healthy_difference_has_no_warning(self):
        self.assertNotIn("degenerate-boolean",
                         codes("difference() { cube(4); sphere(1); }"))

    def test_empty_boolean(self):
        self.assertIn("empty-boolean", codes("union();"))

    def test_children_of_primitive_ignored(self):
        self.assertIn("ignored-children", codes("cube(2) sphere(1);"))

    def test_nonpositive_dimension(self):
        self.assertIn("nonpositive-dimension", codes("cube([2, 0, 5]);"))
        self.assertIn("nonpositive-dimension", codes("cylinder(h = -1, r = 2);"))
        self.assertNotIn("nonpositive-dimension", codes("cube([2, 3, 5]);"))

    def test_no_geometry(self):
        self.assertIn("no-geometry", codes("w = 4;"))
        self.assertNotIn("no-geometry", codes("cube(1);"))

    def test_unused_module(self):
        self.assertIn("unused-module", codes("module m() { cube(1); } cube(2);"))
        self.assertNotIn("unused-module", codes("module m() { cube(1); } m();"))

    def test_children_outside_module(self):
        self.assertIn("children-outside-module", codes("children();"))
        self.assertTrue(is_valid(
            "module wrap() { translate([1,0,0]) children(); } wrap() cube(1);"))


class TestReport(unittest.TestCase):
    def test_determinism(self):
        src = "cube(bad); frobnicate(); difference() { cube(1); }"
        self.assertEqual([i.render() for i in check(src)],
                         [i.render() for i in check(src)])

    def test_format_report_lists_all(self):
        text = format_report(check("frobnicate(1);"))
        self.assertIn("unknown-module", text)
        self.assertIn("[error]", text)

    def test_format_report_when_clean(self):
        self.assertEqual(format_report(check("cube(1);")), "No issues found.")

    def test_issue_render_includes_line_when_known(self):
        self.assertIn("(line 1)", Issue("error", "syntax", "boom", 1, 2).render())

    def test_builtin_table_covers_cheatsheet_core(self):
        for name in ("cube", "sphere", "cylinder", "translate", "rotate",
                     "difference", "union", "intersection", "hull", "minkowski",
                     "linear_extrude", "children"):
            self.assertIn(name, BUILTIN_MODULES)


if __name__ == "__main__":
    unittest.main()
