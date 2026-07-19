"""Tests for RapCAD's numeric/literal semantics (domain.programs.rapcad_language).

Every input here is written from scratch for this test; no RapCAD (GPL-3) file
content is reproduced or vendored.
"""

from fractions import Fraction

import pytest

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


# --------------------------------------------------------------------------- #
# units                                                                       #
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("literal,expected", [
    ("5", Fraction(5)),
    ("5mm", Fraction(5)),
    ("1um", Fraction(1, 1000)),
    ("1cm", Fraction(10)),
    ("1m", Fraction(1000)),
    ("1th", Fraction(254, 10000)),
    ("1in", Fraction(254, 10)),
    ("1ft", Fraction(3048, 10)),
])
def test_unit_factors_are_exact_and_millimetre_based(literal, expected):
    assert parse_number(literal) == expected


def test_imperial_units_are_exactly_related():
    assert parse_number("1in") == parse_number("1000th")
    assert parse_number("1ft") == parse_number("12in")
    assert parse_number("1in") == Fraction(254, 10)  # 25.4 mm exactly


def test_metric_units_are_exactly_related():
    assert parse_number("1m") == parse_number("100cm")
    assert parse_number("1cm") == parse_number("10mm")
    assert parse_number("1mm") == parse_number("1000um")


def test_suffix_matching_is_longest_first():
    # If the bare "m" were tried first, "5cm" would lex as "5c" metres.
    assert UNIT_SUFFIXES.index("cm") < UNIT_SUFFIXES.index("m")
    assert UNIT_SUFFIXES.index("mm") < UNIT_SUFFIXES.index("m")
    assert UNIT_SUFFIXES.index("um") < UNIT_SUFFIXES.index("m")
    assert parse_number("5cm") == Fraction(50)
    assert split_unit("5cm") == ("5", Fraction(10))


def test_unit_suffix_set_matches_factor_table():
    assert set(UNIT_SUFFIXES) == set(UNIT_FACTORS)
    assert UNIT_FACTORS["mm"] == 1  # millimetre is the base


# --------------------------------------------------------------------------- #
# rationals composed with units -- one literal, not a division                 #
# --------------------------------------------------------------------------- #
def test_rational_composes_with_unit_as_a_single_literal():
    assert parse_number("1/2m") == Fraction(500)
    assert parse_number("3/4in") == Fraction(3, 4) * Fraction(254, 10)


def test_repeated_slash_is_left_associative():
    assert parse_number("1/2/4mm") == Fraction(1, 8)
    assert parse_number("1/2/4mm") != Fraction(2)  # not 1/(2/4)


def test_mixed_number_form():
    assert parse_number("1 1/2in") == Fraction(381, 10)
    assert parse_number("2 3/4mm") == Fraction(11, 4)


def test_zero_denominator_is_undef_not_an_exception():
    assert parse_number("1/0m") is None
    assert parse_number("5/0in") is None


def test_non_literals_are_undef():
    assert parse_number("") is None
    assert parse_number("cube") is None
    assert parse_number("1.2.3") is None


# --------------------------------------------------------------------------- #
# exactness                                                                   #
# --------------------------------------------------------------------------- #
def test_decimals_are_exact_rationals_not_doubles():
    assert parse_number("1.5") == Fraction(15, 10)
    assert parse_number("0.1") + parse_number("0.2") == parse_number("0.3")
    # ... which is famously false in binary floating point:
    assert 0.1 + 0.2 != 0.3


def test_the_one_line_dialect_discriminator():
    assert parse_number("1.0000000000000001") != parse_number("1.0")
    assert float("1.0000000000000001") == float("1.0")


def test_is_exact_rational_dialect_separates_the_two_towers():
    exact = is_exact_rational_dialect(
        lambda a, b: parse_number(a) != parse_number(b))
    doubles = is_exact_rational_dialect(lambda a, b: float(a) != float(b))
    assert exact is True
    assert doubles is False


def test_exponent_form():
    assert parse_number("2e-3mm") == Fraction(1, 500)
    assert parse_number("1e3") == Fraction(1000)
    assert parse_number("1e+3") == Fraction(1000)


# --------------------------------------------------------------------------- #
# intervals                                                                   #
# --------------------------------------------------------------------------- #
def test_asymmetric_interval_deviations_are_upper_then_lower():
    # N[a,b] == [N-b, N+a]. Getting this backwards gives [9,12].
    assert Interval.from_literal(10, 1, 2) == Interval(Fraction(8),
                                                       Fraction(11))
    assert Interval.from_literal(10, 1, 2) != Interval(Fraction(9),
                                                       Fraction(12))


def test_single_deviation_and_plus_minus_are_symmetric():
    assert Interval.from_literal(10, 1) == Interval(Fraction(9), Fraction(11))
    assert Interval.from_literal(10, 1) == Interval.from_literal(10, 1, 1)


def test_interval_addition_and_subtraction():
    a = Interval(Fraction(8), Fraction(11))
    b = Interval(Fraction(9, 2), Fraction(11, 2))
    assert a + b == Interval(Fraction(25, 2), Fraction(33, 2))
    assert a - b == Interval(Fraction(5, 2), Fraction(13, 2))
    # x - x is not zero: uncertainty accumulates.
    assert a - a == Interval(Fraction(-3), Fraction(3))


def test_interval_multiplication_takes_corner_extremes():
    neg = Interval(Fraction(-2), Fraction(3))
    assert neg * neg == Interval(Fraction(-6), Fraction(9))
    a = Interval(Fraction(8), Fraction(11))
    b = Interval(Fraction(9, 2), Fraction(11, 2))
    assert a * b == Interval(Fraction(36), Fraction(121, 2))


def test_interval_division_and_zero_divisor():
    a = Interval(Fraction(8), Fraction(12))
    b = Interval(Fraction(2), Fraction(4))
    assert a / b == Interval(Fraction(2), Fraction(6))
    with pytest.raises(ZeroDivisionError):
        a / Interval(Fraction(0), Fraction(2))


def test_interval_negation_swaps_ends():
    assert -Interval(Fraction(8), Fraction(11)) == Interval(Fraction(-11),
                                                            Fraction(-8))


def test_interval_midpoint_tolerance_and_containment():
    a = Interval.from_literal(10, 1, 2)
    assert a.midpoint == Fraction(19, 2)
    assert a.tolerance == Fraction(3, 2)
    assert a.contains(8) and a.contains(11) and a.contains(10)
    assert not a.contains(Fraction(159, 20))  # 7.95


def test_intervals_accept_exact_rational_bounds():
    a = Interval.from_literal(5, "1/2")
    assert a == Interval(Fraction(9, 2), Fraction(11, 2))


# --------------------------------------------------------------------------- #
# quaternions                                                                 #
# --------------------------------------------------------------------------- #
I = Quaternion.of(0, 1, 0, 0)
J = Quaternion.of(0, 0, 1, 0)
K = Quaternion.of(0, 0, 0, 1)


def test_hamilton_product_basis_identities():
    assert I * J == K
    assert J * K == I
    assert K * I == J
    assert I * I == Quaternion.of(-1, 0, 0, 0)
    assert J * J == Quaternion.of(-1, 0, 0, 0)
    assert K * K == Quaternion.of(-1, 0, 0, 0)
    assert I * J * K == Quaternion.of(-1, 0, 0, 0)


def test_hamilton_product_is_not_componentwise_and_not_commutative():
    assert J * I == Quaternion.of(0, 0, 0, -1)
    assert I * J != J * I
    # componentwise multiplication would give the zero quaternion here
    assert I * J != Quaternion.of(0, 0, 0, 0)


def test_hamilton_product_is_associative_with_identity():
    assert (I * J) * K == I * (J * K)
    assert Quaternion.identity() * I == I
    assert I * Quaternion.identity() == I


def test_conjugate_and_norm():
    q = Quaternion.of(1, 2, 3, 4)
    assert q.norm_squared() == Fraction(30)
    assert q * q.conjugate() == Quaternion.of(30, 0, 0, 0)


def test_ang_identity_and_right_angles_are_exact():
    assert ang(0, (0, 0, 1)) == Quaternion.identity()
    assert ang(0, (1, 2, 3)) == Quaternion.identity()
    assert ang(180, (0, 0, 1)) == K
    assert ang(180, (1, 0, 0)) == I


def test_ang_composition_matches_angle_addition():
    composed = ang(90, (0, 0, 1)) * ang(90, (0, 0, 1))
    target = ang(180, (0, 0, 1))
    for component in ("w", "x", "y", "z"):
        assert abs(float(getattr(composed, component)
                         - getattr(target, component))) < 1e-9


# --------------------------------------------------------------------------- #
# facts + CLI                                                                 #
# --------------------------------------------------------------------------- #
def test_every_fact_carries_a_source_citation():
    assert FACTS
    for fact in FACTS:
        assert fact.topic and fact.claim
        assert ":" in fact.citation  # a path and a line number
        assert fact.citation.startswith(("src/", "doc/"))
        assert fact.render()


def test_facts_are_ascii_only():
    for fact in FACTS:
        fact.render().encode("ascii")


def test_selfcheck_exits_zero():
    assert main(["--selfcheck"]) == 0


def test_cli_facts_and_number(capsys):
    assert main(["--facts"]) == 0
    assert main(["--number", "1/2m"]) == 0
    out = capsys.readouterr().out
    assert "500" in out
    assert main(["--number", "1/0m"]) == 1
