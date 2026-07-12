import math
import unittest

from numeric.codetocad_length_expression import (
    ANGLE,
    LENGTH,
    PERCENT,
    ExpressionError,
    Quantity,
    convert_angle,
    convert_length,
    format_length,
    parse_angle,
    parse_length,
    parse_quantity,
    tokenize,
)


class TestLengthParsing(unittest.TestCase):
    def test_simple_units(self):
        self.assertAlmostEqual(parse_length("5mm"), 0.005)
        self.assertAlmostEqual(parse_length("2cm"), 0.02)
        self.assertAlmostEqual(parse_length("5in"), 0.127)
        self.assertAlmostEqual(parse_length("2ft"), 0.6096)
        self.assertAlmostEqual(parse_length("1m"), 1.0)

    def test_arithmetic(self):
        self.assertAlmostEqual(parse_length("2mm + 1m"), 1.002)
        self.assertAlmostEqual(parse_length("6in + 2ft"), 0.1524 + 0.6096)
        self.assertAlmostEqual(parse_length("6m / 2"), 3.0)
        self.assertAlmostEqual(parse_length("2 * 3mm"), 0.006)
        self.assertAlmostEqual(parse_length("(1m - 250mm) / 2"), 0.375)

    def test_unary_minus(self):
        self.assertAlmostEqual(parse_length("-5mm"), -0.005)
        self.assertAlmostEqual(parse_length("10mm + -2mm"), 0.008)

    def test_bare_number_is_metres(self):
        self.assertAlmostEqual(parse_length(3), 3.0)
        self.assertAlmostEqual(parse_length("3"), 3.0)
        self.assertAlmostEqual(parse_length(0.5), 0.5)

    def test_imperial_fraction_and_mixed(self):
        self.assertAlmostEqual(parse_length("1/2in"), 0.0127)
        self.assertAlmostEqual(parse_length("3/4in"), 0.75 * 0.0254)
        self.assertAlmostEqual(parse_length("1-1/2in"), 1.5 * 0.0254)
        self.assertAlmostEqual(parse_length('1/2"'), 0.0127)

    def test_ratio_is_dimensionless(self):
        q = parse_quantity("1m / 250mm")
        self.assertEqual(q.kind, "scalar")
        self.assertAlmostEqual(q.value, 4.0)


class TestPercent(unittest.TestCase):
    def test_percent_needs_base(self):
        with self.assertRaises(ExpressionError):
            parse_length("50%")

    def test_percent_resolves_against_base(self):
        self.assertAlmostEqual(parse_length("50%", base=0.08), 0.04)
        self.assertAlmostEqual(parse_length("150%", base=2.0), 3.0)

    def test_percent_multiplier(self):
        self.assertAlmostEqual(parse_length("50% * 2m"), 1.0)

    def test_percent_token(self):
        q = parse_quantity("25%", base=4.0)
        self.assertEqual(q.kind, LENGTH)
        self.assertAlmostEqual(q.value, 1.0)


class TestAngleParsing(unittest.TestCase):
    def test_degrees_and_radians(self):
        self.assertAlmostEqual(parse_angle("90deg"), math.pi / 2)
        self.assertAlmostEqual(parse_angle("0.5rad"), 0.5)
        self.assertAlmostEqual(parse_angle("90deg + 0.5rad"), 2.0707963, places=6)
        self.assertAlmostEqual(parse_angle("1turn"), 2 * math.pi)

    def test_bare_number_is_radians(self):
        self.assertAlmostEqual(parse_angle(1.5), 1.5)

    def test_angle_scaling(self):
        self.assertAlmostEqual(parse_angle("180deg / 2"), math.pi / 2)


class TestDimensionalErrors(unittest.TestCase):
    def test_length_plus_angle(self):
        with self.assertRaises(ExpressionError):
            parse_quantity("1mm + 1deg")

    def test_length_times_length(self):
        with self.assertRaises(ExpressionError):
            parse_quantity("1mm * 1in")

    def test_length_plus_scalar(self):
        with self.assertRaises(ExpressionError):
            parse_quantity("1mm + 2")

    def test_angle_in_length_context(self):
        with self.assertRaises(ExpressionError):
            parse_length("90deg")

    def test_division_by_zero(self):
        with self.assertRaises(ExpressionError):
            parse_length("1mm / 0")

    def test_bad_character(self):
        with self.assertRaises(ExpressionError):
            parse_length("5mm $ 2")

    def test_unbalanced_parens(self):
        with self.assertRaises(ExpressionError):
            parse_length("(5mm + 1mm")

    def test_empty(self):
        with self.assertRaises(ExpressionError):
            parse_length("   ")

    def test_no_code_execution(self):
        with self.assertRaises(ExpressionError):
            parse_length("__import__('os').getcwd()")


class TestConversionHelpers(unittest.TestCase):
    def test_convert(self):
        self.assertAlmostEqual(convert_length(0.005, "mm"), 5.0)
        self.assertAlmostEqual(convert_length(0.0254, "in"), 1.0)
        self.assertAlmostEqual(convert_angle(math.pi, "deg"), 180.0)

    def test_convert_unknown_unit(self):
        with self.assertRaises(ExpressionError):
            convert_length(1.0, "furlong")

    def test_format(self):
        self.assertEqual(format_length(0.005, "mm"), "5mm")
        self.assertEqual(format_length(0.0127, "in"), "0.5in")
        self.assertEqual(format_length(0.0, "mm"), "0mm")

    def test_roundtrip(self):
        for text in ("5mm", "1-1/2in", "2ft + 3in"):
            metres = parse_length(text)
            self.assertAlmostEqual(parse_length(format_length(metres, "mm")), metres)


class TestQuantity(unittest.TestCase):
    def test_kinds(self):
        self.assertEqual(parse_quantity("5mm").kind, LENGTH)
        self.assertEqual(parse_quantity("5deg").kind, ANGLE)
        self.assertEqual(parse_quantity("5%").kind, PERCENT)

    def test_bad_kind(self):
        with self.assertRaises(ExpressionError):
            Quantity(1.0, "mass")

    def test_tokenize(self):
        tokens = tokenize("2mm + 1m")
        self.assertEqual([t[0] for t in tokens], ["q", "op", "q"])

    def test_quantity_passthrough(self):
        q = Quantity(0.01, LENGTH)
        self.assertIs(parse_quantity(q), q)


if __name__ == "__main__":
    unittest.main()
