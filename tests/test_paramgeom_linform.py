"""Tests for programs.paramgeom_linform (linear-form algebra + expression parser)."""

import unittest
from fractions import Fraction

from harnesscad.domain.programs.expressions.linear_form import (
    LinearForm,
    NonLinearError,
    Num,
    Var,
    BinOp,
    Ternary,
    Call,
    parse_expression,
    to_linear_form,
)


class LinearFormAlgebraTest(unittest.TestCase):
    def test_const_and_var_constructors(self):
        c = LinearForm.const(3)
        self.assertTrue(c.is_constant)
        self.assertEqual(c.constant, Fraction(3))
        v = LinearForm.var("x")
        self.assertEqual(v.coefficient("x"), Fraction(1))
        self.assertEqual(v.constant, Fraction(0))

    def test_zero_coefficient_pruned(self):
        v = LinearForm.var("x", 0)
        self.assertTrue(v.is_zero)
        self.assertEqual(v.variables, ())

    def test_addition_collects_like_terms(self):
        f = LinearForm.var("x", 2) + LinearForm.var("x", 3) + LinearForm.const(1)
        self.assertEqual(f.coefficient("x"), Fraction(5))
        self.assertEqual(f.constant, Fraction(1))

    def test_subtraction_cancels(self):
        f = LinearForm.var("x") - LinearForm.var("x")
        self.assertTrue(f.is_zero)

    def test_scaling(self):
        f = (LinearForm.var("x", 2) + LinearForm.const(3)).scaled(Fraction(1, 2))
        self.assertEqual(f.coefficient("x"), Fraction(1))
        self.assertEqual(f.constant, Fraction(3, 2))

    def test_scale_by_zero_is_zero(self):
        f = (LinearForm.var("x") + LinearForm.const(9)).scaled(0)
        self.assertTrue(f.is_zero)

    def test_evaluate(self):
        f = LinearForm.var("x", 2) + LinearForm.var("y", -1) + LinearForm.const(5)
        self.assertEqual(f.evaluate({"x": 10, "y": 3}), Fraction(22))

    def test_evaluate_missing_var_raises(self):
        with self.assertRaises(KeyError):
            LinearForm.var("x").evaluate({})

    def test_equality_and_hash(self):
        a = LinearForm.var("x", 2) + LinearForm.const(1)
        b = LinearForm.const(1) + LinearForm.var("x", 2)
        self.assertEqual(a, b)
        self.assertEqual(hash(a), hash(b))


class LinearFormRenderTest(unittest.TestCase):
    def test_render_linear_combination(self):
        # 3 + 2*var1 - var2  (paper's canonical C3 example)
        f = (
            LinearForm.const(3)
            + LinearForm.var("var1", 2)
            + LinearForm.var("var2", -1)
        )
        self.assertEqual(f.to_code(var_order=["var1", "var2"]), "2*var1 - var2 + 3")

    def test_render_zero(self):
        self.assertEqual(LinearForm.const(0).to_code(), "0")

    def test_render_single_variable(self):
        self.assertEqual(LinearForm.var("h").to_code(), "h")

    def test_render_fraction_coeff(self):
        f = LinearForm.var("size", Fraction(1, 2))
        self.assertEqual(f.to_code(), "1/2*size")

    def test_render_leading_negative(self):
        f = LinearForm.var("x", -2)
        self.assertEqual(f.to_code(), "-2*x")


class ParserTest(unittest.TestCase):
    def test_number(self):
        self.assertEqual(parse_expression("4.0"), Num(Fraction(4)))

    def test_variable(self):
        self.assertEqual(parse_expression("var1"), Var("var1"))

    def test_precedence(self):
        e = parse_expression("3 + 2*var1 - var2")
        # top-level should be subtraction
        self.assertIsInstance(e, BinOp)
        self.assertEqual(e.op, "-")

    def test_parens(self):
        e = parse_expression("(a + b) * 2")
        self.assertIsInstance(e, BinOp)
        self.assertEqual(e.op, "*")

    def test_ternary(self):
        e = parse_expression("(var1 > 3) ? 1 : 2")
        self.assertIsInstance(e, Ternary)

    def test_function_call(self):
        e = parse_expression("sin(x)")
        self.assertIsInstance(e, Call)
        self.assertEqual(e.name, "sin")

    def test_trailing_tokens_raise(self):
        with self.assertRaises(SyntaxError):
            parse_expression("1 2")

    def test_empty_raises(self):
        with self.assertRaises(SyntaxError):
            parse_expression("   ")


class ToLinearFormTest(unittest.TestCase):
    def test_affine_reduction(self):
        f = to_linear_form("3 + 2*var1 - var2")
        self.assertEqual(f.coefficient("var1"), Fraction(2))
        self.assertEqual(f.coefficient("var2"), Fraction(-1))
        self.assertEqual(f.constant, Fraction(3))

    def test_constant_times_var_both_orders(self):
        self.assertEqual(to_linear_form("2*x"), to_linear_form("x*2"))

    def test_division_by_constant(self):
        f = to_linear_form("x/2 + 1")
        self.assertEqual(f.coefficient("x"), Fraction(1, 2))

    def test_nested_linear(self):
        # matches paper Listing 2: size_cube_a/2 + size_cube_b/2
        f = to_linear_form("size_cube_a/2 + size_cube_b/2")
        self.assertEqual(f.coefficient("size_cube_a"), Fraction(1, 2))
        self.assertEqual(f.coefficient("size_cube_b"), Fraction(1, 2))

    def test_var_times_var_raises(self):
        with self.assertRaises(NonLinearError):
            to_linear_form("var1*var2")

    def test_div_by_var_raises(self):
        with self.assertRaises(NonLinearError):
            to_linear_form("1/x")

    def test_ternary_raises(self):
        with self.assertRaises(NonLinearError):
            to_linear_form("(x>3)?1:2")

    def test_division_by_zero_raises(self):
        with self.assertRaises(NonLinearError):
            to_linear_form("x/0")


if __name__ == "__main__":
    unittest.main()
