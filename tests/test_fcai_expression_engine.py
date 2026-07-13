"""Tests for the FreeCAD expression-language parser and evaluator."""

import math
import unittest

from harnesscad.domain.programs.expressions.freecad_expressions import (
    Expression,
    ExpressionError,
    Reference,
    parse,
    tokenize,
)


class TokenizeTest(unittest.TestCase):
    def test_simple(self):
        toks = tokenize("1 + 2")
        self.assertEqual([t.value for t in toks], ["1", "+", "2"])

    def test_reference_with_subscript(self):
        toks = tokenize("Sketch.Constraints[0]")
        self.assertEqual([t.value for t in toks],
                         ["Sketch", ".", "Constraints", "[", "0", "]"])

    def test_float_and_exponent(self):
        toks = tokenize("1.5e3")
        self.assertEqual(len(toks), 1)
        self.assertEqual(toks[0].value, "1.5e3")

    def test_bad_char(self):
        with self.assertRaises(ExpressionError):
            tokenize("1 & 2")


class ArithmeticTest(unittest.TestCase):
    def ev(self, s, env=None):
        return parse(s).evaluate(env or {})

    def test_precedence(self):
        self.assertEqual(self.ev("2 + 3 * 4"), 14)

    def test_parentheses(self):
        self.assertEqual(self.ev("(2 + 3) * 4"), 20)

    def test_unary_minus(self):
        self.assertEqual(self.ev("-5 + 2"), -3)
        self.assertEqual(self.ev("3 * -2"), -6)

    def test_power_right_assoc(self):
        # 2 ^ 3 ^ 2 = 2 ^ 9 = 512
        self.assertEqual(self.ev("2 ^ 3 ^ 2"), 512)

    def test_modulo(self):
        self.assertEqual(self.ev("10 % 3"), 1)

    def test_division(self):
        self.assertEqual(self.ev("7 / 2"), 3.5)

    def test_division_by_zero(self):
        with self.assertRaises(ExpressionError):
            self.ev("1 / 0")

    def test_modulo_by_zero(self):
        with self.assertRaises(ExpressionError):
            self.ev("1 % 0")


class UnitsTest(unittest.TestCase):
    def test_mm_is_base(self):
        self.assertEqual(parse("10 mm").evaluate(), 10.0)

    def test_cm_converts(self):
        self.assertEqual(parse("2 cm").evaluate(), 20.0)

    def test_inch_converts(self):
        self.assertAlmostEqual(parse("1 in").evaluate(), 25.4)

    def test_mixed_units_add(self):
        self.assertEqual(parse("10 mm + 1 cm").evaluate(), 20.0)


class FunctionTest(unittest.TestCase):
    def test_trig_degrees(self):
        self.assertAlmostEqual(parse("sin(30)").evaluate(), 0.5)
        self.assertAlmostEqual(parse("cos(60)").evaluate(), 0.5)

    def test_sqrt(self):
        self.assertEqual(parse("sqrt(16)").evaluate(), 4.0)

    def test_min_max_multi_arg(self):
        self.assertEqual(parse("max(3, 7, 5)").evaluate(), 7)
        self.assertEqual(parse("min(3, 7, 5)").evaluate(), 3)

    def test_nested_calls(self):
        self.assertAlmostEqual(parse("sqrt(pow(3, 2) + pow(4, 2))").evaluate(),
                               5.0)

    def test_unknown_function(self):
        with self.assertRaises(ExpressionError):
            parse("frobnicate(1)").evaluate()

    def test_bad_arity(self):
        with self.assertRaises(ExpressionError):
            parse("sqrt(1, 2)").evaluate()


class ReferenceTest(unittest.TestCase):
    def test_variable_reference(self):
        e = parse("Variables.height")
        self.assertEqual(e.reference_keys(), ["Variables.height"])
        self.assertEqual(e.evaluate({"Variables.height": 42}), 42)

    def test_subscript_reference_key(self):
        e = parse("Sketch.Constraints[3]")
        self.assertEqual(e.reference_keys(), ["Sketch.Constraints[3]"])

    def test_nested_placement_reference(self):
        e = parse("Box.Placement.Base.x")
        self.assertEqual(e.reference_keys(), ["Box.Placement.Base.x"])

    def test_formula_references_dedup(self):
        e = parse("Variables.w * 2 + Variables.w")
        self.assertEqual(e.reference_keys(), ["Variables.w"])

    def test_multiple_references_ordered(self):
        e = parse("Variables.a + Variables.b * Variables.c")
        self.assertEqual(e.reference_keys(),
                         ["Variables.a", "Variables.b", "Variables.c"])

    def test_unresolved_reference_raises(self):
        with self.assertRaises(ExpressionError):
            parse("Variables.missing").evaluate({})

    def test_reference_key_roundtrip(self):
        r = Reference(("Sketch", "Constraints"), ((1, 8),))
        self.assertEqual(r.key(), "Sketch.Constraints[8]")


class ParametricFormulaTest(unittest.TestCase):
    def test_pad_length_expression(self):
        # set_expression('Pad', 'Length', 'Variables.height')
        e = parse("Variables.height")
        self.assertEqual(e.evaluate({"Variables.height": 25}), 25)

    def test_wall_times_two(self):
        e = parse("Variables.wall * 2")
        self.assertEqual(e.evaluate({"Variables.wall": 3}), 6)

    def test_derived_dimension(self):
        e = parse("(Variables.length - 2 * Variables.wall)")
        self.assertEqual(
            e.evaluate({"Variables.length": 50, "Variables.wall": 2}), 46)


class MalformedTest(unittest.TestCase):
    def test_empty(self):
        with self.assertRaises(ExpressionError):
            parse("")

    def test_trailing_tokens(self):
        with self.assertRaises(ExpressionError):
            parse("1 2")

    def test_unbalanced_paren(self):
        with self.assertRaises(ExpressionError):
            parse("(1 + 2")

    def test_dangling_operator(self):
        with self.assertRaises(ExpressionError):
            parse("1 +")

    def test_is_expression_instance(self):
        self.assertIsInstance(parse("1"), Expression)


if __name__ == "__main__":
    unittest.main()
