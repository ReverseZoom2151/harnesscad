"""Tests for RapCAD's numeric/literal semantics (domain.programs.rapcad_language).

Every input here is written from scratch for this test; no RapCAD (GPL-3) file
content is reproduced or vendored.
"""

import contextlib
import io
import unittest
from fractions import Fraction

from harnesscad.domain.programs.rapcad_language import (
    FACTS,
    UNIT_FACTORS,
    UNIT_SUFFIXES,
    Interval,
    Quaternion,
    ang,
    is_exact_rational_dialect,
    main,
    parse_number,
    split_unit,
)


I = Quaternion.of(0, 1, 0, 0)
J = Quaternion.of(0, 0, 1, 0)
K = Quaternion.of(0, 0, 0, 1)


class UnitsTest(unittest.TestCase):
    def test_unit_factors_are_exact_and_millimetre_based(self):
        cases = [
            ("5", Fraction(5)),
            ("5mm", Fraction(5)),
            ("1um", Fraction(1, 1000)),
            ("1cm", Fraction(10)),
            ("1m", Fraction(1000)),
            ("1th", Fraction(254, 10000)),
            ("1in", Fraction(254, 10)),
            ("1ft", Fraction(3048, 10)),
        ]
        for literal, expected in cases:
            with self.subTest(literal=literal):
                self.assertEqual(parse_number(literal), expected)

    def test_imperial_units_are_exactly_related(self):
        self.assertEqual(parse_number("1in"), parse_number("1000th"))
        self.assertEqual(parse_number("1ft"), parse_number("12in"))
        self.assertEqual(parse_number("1in"), Fraction(254, 10))  # 25.4 mm exactly

    def test_metric_units_are_exactly_related(self):
        self.assertEqual(parse_number("1m"), parse_number("100cm"))
        self.assertEqual(parse_number("1cm"), parse_number("10mm"))
        self.assertEqual(parse_number("1mm"), parse_number("1000um"))

    def test_suffix_matching_is_longest_first(self):
        # If the bare "m" were tried first, "5cm" would lex as "5c" metres.
        self.assertLess(UNIT_SUFFIXES.index("cm"), UNIT_SUFFIXES.index("m"))
        self.assertLess(UNIT_SUFFIXES.index("mm"), UNIT_SUFFIXES.index("m"))
        self.assertLess(UNIT_SUFFIXES.index("um"), UNIT_SUFFIXES.index("m"))
        self.assertEqual(parse_number("5cm"), Fraction(50))
        self.assertEqual(split_unit("5cm"), ("5", Fraction(10)))

    def test_unit_suffix_set_matches_factor_table(self):
        self.assertEqual(set(UNIT_SUFFIXES), set(UNIT_FACTORS))
        self.assertEqual(UNIT_FACTORS["mm"], 1)  # millimetre is the base


class RationalsWithUnitsTest(unittest.TestCase):
    def test_rational_composes_with_unit_as_a_single_literal(self):
        self.assertEqual(parse_number("1/2m"), Fraction(500))
        self.assertEqual(parse_number("3/4in"), Fraction(3, 4) * Fraction(254, 10))

    def test_repeated_slash_is_left_associative(self):
        self.assertEqual(parse_number("1/2/4mm"), Fraction(1, 8))
        self.assertNotEqual(parse_number("1/2/4mm"), Fraction(2))  # not 1/(2/4)

    def test_mixed_number_form(self):
        self.assertEqual(parse_number("1 1/2in"), Fraction(381, 10))
        self.assertEqual(parse_number("2 3/4mm"), Fraction(11, 4))

    def test_zero_denominator_is_undef_not_an_exception(self):
        self.assertIsNone(parse_number("1/0m"))
        self.assertIsNone(parse_number("5/0in"))

    def test_non_literals_are_undef(self):
        self.assertIsNone(parse_number(""))
        self.assertIsNone(parse_number("cube"))
        self.assertIsNone(parse_number("1.2.3"))


class ExactnessTest(unittest.TestCase):
    def test_decimals_are_exact_rationals_not_doubles(self):
        self.assertEqual(parse_number("1.5"), Fraction(15, 10))
        self.assertEqual(parse_number("0.1") + parse_number("0.2"), parse_number("0.3"))
        # ... which is famously false in binary floating point:
        self.assertNotEqual(0.1 + 0.2, 0.3)

    def test_the_one_line_dialect_discriminator(self):
        self.assertNotEqual(parse_number("1.0000000000000001"), parse_number("1.0"))
        self.assertEqual(float("1.0000000000000001"), float("1.0"))

    def test_is_exact_rational_dialect_separates_the_two_towers(self):
        exact = is_exact_rational_dialect(
            lambda a, b: parse_number(a) != parse_number(b))
        doubles = is_exact_rational_dialect(lambda a, b: float(a) != float(b))
        self.assertIs(exact, True)
        self.assertIs(doubles, False)

    def test_exponent_form(self):
        self.assertEqual(parse_number("2e-3mm"), Fraction(1, 500))
        self.assertEqual(parse_number("1e3"), Fraction(1000))
        self.assertEqual(parse_number("1e+3"), Fraction(1000))


class IntervalsTest(unittest.TestCase):
    def test_asymmetric_interval_deviations_are_upper_then_lower(self):
        # N[a,b] == [N-b, N+a]. Getting this backwards gives [9,12].
        self.assertEqual(Interval.from_literal(10, 1, 2),
                         Interval(Fraction(8), Fraction(11)))
        self.assertNotEqual(Interval.from_literal(10, 1, 2),
                            Interval(Fraction(9), Fraction(12)))

    def test_single_deviation_and_plus_minus_are_symmetric(self):
        self.assertEqual(Interval.from_literal(10, 1),
                         Interval(Fraction(9), Fraction(11)))
        self.assertEqual(Interval.from_literal(10, 1),
                         Interval.from_literal(10, 1, 1))

    def test_interval_addition_and_subtraction(self):
        a = Interval(Fraction(8), Fraction(11))
        b = Interval(Fraction(9, 2), Fraction(11, 2))
        self.assertEqual(a + b, Interval(Fraction(25, 2), Fraction(33, 2)))
        self.assertEqual(a - b, Interval(Fraction(5, 2), Fraction(13, 2)))
        # x - x is not zero: uncertainty accumulates.
        self.assertEqual(a - a, Interval(Fraction(-3), Fraction(3)))

    def test_interval_multiplication_takes_corner_extremes(self):
        neg = Interval(Fraction(-2), Fraction(3))
        self.assertEqual(neg * neg, Interval(Fraction(-6), Fraction(9)))
        a = Interval(Fraction(8), Fraction(11))
        b = Interval(Fraction(9, 2), Fraction(11, 2))
        self.assertEqual(a * b, Interval(Fraction(36), Fraction(121, 2)))

    def test_interval_division_and_zero_divisor(self):
        a = Interval(Fraction(8), Fraction(12))
        b = Interval(Fraction(2), Fraction(4))
        self.assertEqual(a / b, Interval(Fraction(2), Fraction(6)))
        with self.assertRaises(ZeroDivisionError):
            a / Interval(Fraction(0), Fraction(2))

    def test_interval_negation_swaps_ends(self):
        self.assertEqual(-Interval(Fraction(8), Fraction(11)),
                         Interval(Fraction(-11), Fraction(-8)))

    def test_interval_midpoint_tolerance_and_containment(self):
        a = Interval.from_literal(10, 1, 2)
        self.assertEqual(a.midpoint, Fraction(19, 2))
        self.assertEqual(a.tolerance, Fraction(3, 2))
        self.assertTrue(a.contains(8) and a.contains(11) and a.contains(10))
        self.assertFalse(a.contains(Fraction(159, 20)))  # 7.95

    def test_intervals_accept_exact_rational_bounds(self):
        a = Interval.from_literal(5, "1/2")
        self.assertEqual(a, Interval(Fraction(9, 2), Fraction(11, 2)))


class QuaternionsTest(unittest.TestCase):
    def test_hamilton_product_basis_identities(self):
        self.assertEqual(I * J, K)
        self.assertEqual(J * K, I)
        self.assertEqual(K * I, J)
        self.assertEqual(I * I, Quaternion.of(-1, 0, 0, 0))
        self.assertEqual(J * J, Quaternion.of(-1, 0, 0, 0))
        self.assertEqual(K * K, Quaternion.of(-1, 0, 0, 0))
        self.assertEqual(I * J * K, Quaternion.of(-1, 0, 0, 0))

    def test_hamilton_product_is_not_componentwise_and_not_commutative(self):
        self.assertEqual(J * I, Quaternion.of(0, 0, 0, -1))
        self.assertNotEqual(I * J, J * I)
        # componentwise multiplication would give the zero quaternion here
        self.assertNotEqual(I * J, Quaternion.of(0, 0, 0, 0))

    def test_hamilton_product_is_associative_with_identity(self):
        self.assertEqual((I * J) * K, I * (J * K))
        self.assertEqual(Quaternion.identity() * I, I)
        self.assertEqual(I * Quaternion.identity(), I)

    def test_conjugate_and_norm(self):
        q = Quaternion.of(1, 2, 3, 4)
        self.assertEqual(q.norm_squared(), Fraction(30))
        self.assertEqual(q * q.conjugate(), Quaternion.of(30, 0, 0, 0))

    def test_ang_identity_and_right_angles_are_exact(self):
        self.assertEqual(ang(0, (0, 0, 1)), Quaternion.identity())
        self.assertEqual(ang(0, (1, 2, 3)), Quaternion.identity())
        self.assertEqual(ang(180, (0, 0, 1)), K)
        self.assertEqual(ang(180, (1, 0, 0)), I)

    def test_ang_composition_matches_angle_addition(self):
        composed = ang(90, (0, 0, 1)) * ang(90, (0, 0, 1))
        target = ang(180, (0, 0, 1))
        for component in ("w", "x", "y", "z"):
            self.assertLess(
                abs(float(getattr(composed, component)
                          - getattr(target, component))), 1e-9)


class FactsAndCliTest(unittest.TestCase):
    def test_every_fact_carries_a_source_citation(self):
        self.assertTrue(FACTS)
        for fact in FACTS:
            self.assertTrue(fact.topic and fact.claim)
            self.assertIn(":", fact.citation)  # a path and a line number
            self.assertTrue(fact.citation.startswith(("src/", "doc/")))
            self.assertTrue(fact.render())

    def test_facts_are_ascii_only(self):
        for fact in FACTS:
            fact.render().encode("ascii")

    def test_selfcheck_exits_zero(self):
        self.assertEqual(main(["--selfcheck"]), 0)

    def test_cli_facts_and_number(self):
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            self.assertEqual(main(["--facts"]), 0)
            self.assertEqual(main(["--number", "1/2m"]), 0)
        out = buf.getvalue()
        self.assertIn("500", out)
        self.assertEqual(main(["--number", "1/0m"]), 1)


if __name__ == "__main__":
    unittest.main()
