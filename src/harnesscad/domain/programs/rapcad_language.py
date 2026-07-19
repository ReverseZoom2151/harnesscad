"""Exact numeric and literal semantics of a rational-numeric OpenSCAD dialect.

A rational-numeric OpenSCAD-family dialect -- the one whose files carry the
``.rcad`` extension -- is usually filed away as "OpenSCAD with a nicer IDE".
That is wrong in a way that matters to any tool that reads or generates those
files: its *numeric tower and literal grammar* are a different language from
OpenSCAD's, and the differences are silent -- a program that means one thing in
OpenSCAD means another in this dialect without any syntax error to warn you.

This module models the four novel pieces in stdlib
:class:`fractions.Fraction`, so the harness can reason about those numbers
instead of guessing:

1.  **Unit-suffixed literals over a millimetre base, with exact rational
    factors.** ``5mm``, ``2in``, ``10th`` are single numeric *literals*, not
    expressions; the suffix multiplies by an exact rational and the result is a
    plain number in millimetres. See :data:`UNIT_FACTORS`, :func:`parse_number`.

2.  **An exact-rational numeric tower, not IEEE-754 doubles.** ``"1.5"`` becomes
    the exact rational ``15/10``, not the nearest double. The one-line
    discriminator between a true exact-rational dialect and any double-based
    OpenSCAD is ``1.0000000000000001 != 1.0``: **true** under exact rationals,
    **false** everywhere else, because in doubles both literals are the same
    bit pattern. See :func:`is_exact_rational_dialect`.

3.  **Engineering tolerance intervals** ``N[a,b]``, ``N[a]`` and ``N PM e``,
    with real interval arithmetic. The asymmetric form is easy to get backwards:
    ``N[a,b]`` is the closed range ``[N-b, N+a]`` -- ``a`` is the *upper*
    deviation and ``b`` the *lower*. See :class:`Interval`.

4.  **Quaternion literals** ``<w,x,y,z>`` whose ``*`` is the Hamilton product,
    plus ``ang(angle, axis)`` as an axis-angle constructor with identity
    ``<1,0,0,0>``. See :class:`Quaternion` and :func:`ang`.

Method
------
Everything below is a *fact about the dialect*, stated as behaviour. No
external source text is reproduced here, and no ``.rcad`` fixture is vendored
-- the test inputs in :data:`FACTS` and in the selfcheck are written from
scratch for this module. The behaviours modelled are:

* a decimal string is rewritten into an exact rational (digits over a power of
  ten), so parsing is exact;
* exponent, mixed-number and repeated-``/`` literal forms;
* seven unit suffixes with exact factors, tested longest-first;
* the character classes that define what a numeric literal *is*, and the four
  number token rules -- note both rational rules require a unit suffix;
* a literal's value is mantissa * unit; the zero-denominator rule yields the
  undefined value rather than raising or producing an infinity, and runtime
  division/modulus by zero is likewise undefined;
* three interval literal productions, evaluated as ``lower = n - less`` and
  ``upper = n + more``, with ``less`` defaulting to ``more``;
* interval add/subtract/multiply/divide and the equality rules;
* the quaternion literal production, ``ang(angle, axis)``, and exact rounding
  of sine/cosine at right angles, which is why ``ang(0, axis)`` is exactly
  ``<1,0,0,0>``;
* the Hamilton product for ``*``.

Not a duplicate of existing harness modules -- checked before writing:
:mod:`harnesscad.domain.numeric.interval_arithmetic` is f-rep *spatial*
bounding (a different purpose: pruning an implicit-surface octree), and
:mod:`harnesscad.domain.numeric.unit_expressions` is a float, SI-metre-base
expression evaluator with no ``th`` suffix and no notion of a unit-suffixed
*literal*. Neither can represent ``1/2m`` as one exact number.

Pure stdlib, deterministic, ASCII, nothing executed.
"""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from fractions import Fraction
from typing import Dict, List, Optional, Sequence, Tuple

__all__ = [
    "Fact",
    "FACTS",
    "UNIT_FACTORS",
    "UNIT_SUFFIXES",
    "split_unit",
    "parse_number",
    "is_exact_rational_dialect",
    "Interval",
    "Quaternion",
    "ang",
    "main",
]


# --------------------------------------------------------------------------- #
# 0. the fact record                                                          #
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Fact:
    """One verified claim about the dialect, with its recorded reference."""

    topic: str
    claim: str
    citation: str

    def render(self) -> str:
        return "[%s] %s  (%s)" % (self.topic, self.claim, self.citation)


# --------------------------------------------------------------------------- #
# 1. unit-suffixed literals -- exact rationals over a millimetre base          #
# --------------------------------------------------------------------------- #

#: Suffix -> exact multiplier, in millimetres. The base unit of the dialect is
#: the millimetre: a bare number and a ``mm``-suffixed number are the same value.
UNIT_FACTORS: Dict[str, Fraction] = {
    "um": Fraction(1, 1000),      # micrometre
    "mm": Fraction(1),            # millimetre -- the base
    "cm": Fraction(10),
    "m": Fraction(1000),
    "th": Fraction(254, 10000),   # thou / mil = 1/1000 inch, exactly
    "in": Fraction(254, 10),      # inch = 25.4 mm, exactly
    "ft": Fraction(3048, 10),     # foot = 304.8 mm, exactly
}

#: Suffixes in the order the dialect tests them. The order is load-bearing: ``um``,
#: ``mm`` and ``cm`` must be tried before the bare ``m``, or ``5cm`` would lex
#: as the number ``5c`` times metres. (``th``/``in``/``ft`` do not end in ``m``,
#: so their position after ``m`` is harmless.)
UNIT_SUFFIXES: Tuple[str, ...] = ("um", "mm", "cm", "m", "th", "in", "ft")


def split_unit(text: str) -> Tuple[str, Fraction]:
    """Strip a trailing unit suffix, returning ``(mantissa_text, factor)``.

    An unsuffixed number keeps a factor of 1 -- millimetres are the base unit.
    Matching is longest-first, mirroring the dialect's ordered suffix-test
    chain.
    """
    for suffix in UNIT_SUFFIXES:
        if text.endswith(suffix):
            return text[: -len(suffix)], UNIT_FACTORS[suffix]
    return text, Fraction(1)


def _parse_plain(text: str) -> Optional[Fraction]:
    """Exact value of a decimal digit string: ``"1.5"`` -> ``Fraction(15, 10)``.

    The point is removed and the digits divided by the corresponding power of
    ten -- so the result is exact, never the nearest double.
    """
    if not text:
        return None
    body = text
    point = body.find(".")
    if point >= 0:
        if "." in body[point + 1:]:
            return None
        digits = body[:point] + body[point + 1:]
        scale = len(body) - point - 1
    else:
        digits = body
        scale = 0
    if not digits or not digits.isdigit():
        return None
    return Fraction(int(digits), 10 ** scale)


def _parse_exp(text: str) -> Optional[Fraction]:
    """Exponent form: mantissa ``e`` power.

    The exponent may itself be fractional in the dialect's grammar; a non-integer
    exponent cannot stay exact, so it is refused here rather than silently
    returning a float.
    """
    lowered = text.lower()
    at = lowered.find("e")
    if at < 0:
        return _parse_plain(text)
    mantissa = _parse_plain(text[:at])
    exponent_text = text[at + 1:].replace("+", "", 1)
    negative = exponent_text.startswith("-")
    if negative:
        exponent_text = exponent_text[1:]
    exponent = _parse_plain(exponent_text)
    if mantissa is None or exponent is None or exponent.denominator != 1:
        return None
    power = Fraction(10) ** int(exponent)
    return mantissa / power if negative else mantissa * power


def _parse_rational(text: str) -> Optional[Fraction]:
    """Rational form: repeated ``/``.

    Splits on the *last* slash and recurses left, so ``1/2/4`` is ``(1/2)/4``
    -- left-associative. A zero divisor makes the whole literal undefined
    (see :func:`parse_number`).
    """
    at = text.rfind("/")
    if at < 0:
        return _parse_exp(text)
    left = _parse_rational(text[:at])
    right = _parse_exp(text[at + 1:])
    if left is None or right is None or right == 0:
        return None
    return left / right


def _parse_mixed(text: str) -> Optional[Fraction]:
    """Mixed-number form: ``1 1/2`` = 3/2.

    Splits on the *first* space; the whole part is parsed plainly and the
    remainder as a rational.
    """
    at = text.find(" ")
    if at < 0:
        return _parse_rational(text)
    whole = _parse_plain(text[:at])
    part = _parse_rational(text[at + 1:])
    if whole is None or part is None:
        return None
    return whole + part


def parse_number(literal: str) -> Optional[Fraction]:
    """Exact millimetre value of one numeric literal of the dialect, or ``None``.

    ``None`` means the literal is undefined -- which is what a zero-denominator
    literal such as ``1/0m`` produces: the lexer has a dedicated rule for a zero
    denominator that yields the undefined token, and at runtime division by zero
    is likewise undefined rather than an infinity or an exception. It also means
    "not a literal".

    The whole string is *one* literal, suffix included. ``1/2m`` is a single
    token worth exactly 500 mm; there is no division operator involved.
    """
    text = literal.strip()
    if not text:
        return None
    mantissa_text, factor = split_unit(text)
    value = _parse_mixed(mantissa_text)
    if value is None:
        return None
    return value * factor


def is_exact_rational_dialect(evaluate_ne) -> bool:
    """Does ``evaluate_ne`` behave like an exact rational tower or like doubles?

    ``evaluate_ne`` is any callable taking two literal *strings* and returning
    whether the dialect considers them unequal. The probe is the sharpest
    one-liner available: ``1.0000000000000001 != 1.0``. Under IEEE-754 doubles
    both literals round to the same bit pattern, so the answer is False; under
    an exact rational tower they are distinct rationals, so it is True.
    """
    return bool(evaluate_ne("1.0000000000000001", "1.0"))


# --------------------------------------------------------------------------- #
# 2. tolerance intervals                                                      #
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Interval:
    """A tolerance interval: a closed range ``[lower, upper]``.

    Build it from a literal with :meth:`from_literal`, which applies the
    dialect's (easily-inverted) convention: for ``N[a, b]`` the first bracketed
    value is the deviation *upward* and the second the deviation *downward*, so
    the range is ``[N - b, N + a]``.

    Arithmetic is ordinary interval arithmetic over the four corner products.
    """

    lower: Fraction
    upper: Fraction

    @classmethod
    def from_literal(cls, number, more, less=None) -> "Interval":
        """``N[more, less]``; ``N[more]`` and ``N PM more`` omit ``less``.

        When ``less`` is omitted it defaults to ``more``, giving the symmetric
        ``N +/- more``.
        """
        n = Fraction(number)
        up = Fraction(more)
        down = up if less is None else Fraction(less)
        return cls(n - down, n + up)

    # -- presentation ------------------------------------------------------ #
    @property
    def midpoint(self) -> Fraction:
        """The centre the dialect prints: ``upper - (upper - lower)/2``."""
        return self.upper - (self.upper - self.lower) / 2

    @property
    def tolerance(self) -> Fraction:
        """Half-width -- the ``t`` in the ``n +/- t`` display form."""
        return (self.upper - self.lower) / 2

    def __str__(self) -> str:
        return "%s+/-%s" % (self.midpoint, self.tolerance)

    # -- arithmetic -------------------------------------------------------- #
    def __neg__(self) -> "Interval":
        """Unary minus swaps and negates the ends."""
        return Interval(-self.upper, -self.lower)

    def __pos__(self) -> "Interval":
        return self

    def __add__(self, other: "Interval") -> "Interval":
        return Interval(self.lower + other.lower, self.upper + other.upper)

    def __sub__(self, other: "Interval") -> "Interval":
        return Interval(self.lower - other.upper, self.upper - other.lower)

    def _corners(self, other: "Interval", divide: bool) -> List[Fraction]:
        vals = []
        for a in (self.lower, self.upper):
            for b in (other.lower, other.upper):
                if divide:
                    if b == 0:
                        raise ZeroDivisionError(
                            "interval divisor spans zero -- undef in RapCAD")
                    vals.append(a / b)
                else:
                    vals.append(a * b)
        return vals

    def __mul__(self, other: "Interval") -> "Interval":
        vals = self._corners(other, divide=False)
        return Interval(min(vals), max(vals))

    def __truediv__(self, other: "Interval") -> "Interval":
        vals = self._corners(other, divide=True)
        return Interval(min(vals), max(vals))

    def contains(self, value) -> bool:
        """Is ``value`` inside the closed range? (Not a dialect operator.)"""
        return self.lower <= Fraction(value) <= self.upper


# --------------------------------------------------------------------------- #
# 3. quaternions                                                              #
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Quaternion:
    """A ``<w, x, y, z>`` literal; ``*`` is the Hamilton product.

    The dialect spells a quaternion with angle brackets and internally calls it
    a "complex" value. Its ``*`` is not componentwise: it is the Hamilton
    product, so it is associative but **not** commutative -- the property the
    selfcheck proves.
    """

    w: Fraction
    x: Fraction
    y: Fraction
    z: Fraction

    @classmethod
    def of(cls, w, x, y, z) -> "Quaternion":
        return cls(Fraction(w), Fraction(x), Fraction(y), Fraction(z))

    @classmethod
    def identity(cls) -> "Quaternion":
        return cls.of(1, 0, 0, 0)

    def __mul__(self, other: "Quaternion") -> "Quaternion":
        w1, x1, y1, z1 = self.w, self.x, self.y, self.z
        w2, x2, y2, z2 = other.w, other.x, other.y, other.z
        return Quaternion(
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        )

    def norm_squared(self) -> Fraction:
        return self.w ** 2 + self.x ** 2 + self.y ** 2 + self.z ** 2

    def conjugate(self) -> "Quaternion":
        return Quaternion(self.w, -self.x, -self.y, -self.z)

    def __str__(self) -> str:
        return "<%s,%s,%s,%s>" % (self.w, self.x, self.y, self.z)


def ang(angle_degrees, axis: Sequence) -> Quaternion:
    """The ``ang(angle, axis)`` axis-angle constructor.

    ``w = cos(angle/2)``, ``(x,y,z) = axis * sin(angle/2)``, with the angle in
    **degrees**. The dialect evaluates the half-angle with right-angle-exact
    sine and cosine, which round exactly at multiples of 90 degrees; that is
    reproduced here so that right-angle rotations -- and the identity
    ``ang(0, axis)`` -> ``<1,0,0,0>`` -- come out exact rather than 1e-17 off.
    """
    half = Fraction(angle_degrees) / 2
    if half.denominator == 1 and int(half) % 90 == 0:
        cos_h = Fraction(round(math.cos(math.radians(float(half)))))
        sin_h = Fraction(round(math.sin(math.radians(float(half)))))
    else:
        radians = math.radians(float(half))
        cos_h = Fraction(math.cos(radians)).limit_denominator(10 ** 12)
        sin_h = Fraction(math.sin(radians)).limit_denominator(10 ** 12)
    ax, ay, az = (Fraction(c) for c in axis)
    return Quaternion(cos_h, ax * sin_h, ay * sin_h, az * sin_h)


# --------------------------------------------------------------------------- #
# 4. the recorded facts                                                       #
# --------------------------------------------------------------------------- #
FACTS: Tuple[Fact, ...] = (
    Fact("units",
         "RapCAD's base length unit is the millimetre: a bare number and an "
         "mm-suffixed number are the same value, and every other suffix "
         "multiplies by an exact rational (um=1/1000, cm=10, m=1000, "
         "th=254/10000, in=254/10, ft=3048/10).",
         "src/decimal.cpp:142-173"),
    Fact("units",
         "A unit suffix is part of the numeric literal, not an operator: the "
         "lexer's number rules carry an optional (for rationals, mandatory) "
         "unit class, so 1/2m is ONE token worth exactly 500.",
         "src/lexer.l:68,113-118; src/tokenbuilder.cpp:332-358"),
    Fact("units",
         "Suffix matching is ordered longest-first, testing um/mm/cm before "
         "the bare m; reordering the chain would make 5cm lex as 5c metres.",
         "src/decimal.cpp:142-160"),
    Fact("numbers",
         "Numbers are exact rationals (GMP mpq), not IEEE-754 doubles: a "
         "decimal string is rewritten as digits over a power of ten, so 1.5 is "
         "exactly 15/10.",
         "src/decimal.cpp:28-56"),
    Fact("numbers",
         "One-line dialect discriminator: 1.0000000000000001 != 1.0 is TRUE in "
         "RapCAD and FALSE in any double-based OpenSCAD, where both literals "
         "round to the same bit pattern.",
         "src/decimal.cpp:28-56"),
    Fact("numbers",
         "A zero denominator yields undef, not an exception and not an "
         "infinity: the lexer has a dedicated zero-denominator rule returning "
         "the UNDEF token, and runtime division/modulus by zero is undefined "
         "too, so 1/0m == undef.",
         "src/lexer.l:116; src/tokenbuilder.cpp:348-351; "
         "src/numbervalue.cpp:85-88"),
    Fact("numbers",
         "Repeated slashes are left-associative (the C++ splits on the LAST "
         "slash and recurses left), so 1/2/4mm is (1/2)/4 = 1/8.",
         "src/decimal.cpp:132-140"),
    Fact("numbers",
         "Mixed numbers exist: a space splits a whole part from a fraction, so "
         "1 1/2in is 3/2 inches = 381/10 mm exactly.",
         "src/decimal.cpp:124-130; src/lexer.l:58"),
    Fact("intervals",
         "N[a,b] is the CLOSED range [N-b, N+a] -- the first bracketed value "
         "is the upward deviation and the second the downward one, which is "
         "the reverse of the reading order. N[a] and N PM e omit the second "
         "and are symmetric.",
         "src/parser.y:299-304; src/treeevaluator.cpp:412-432"),
    Fact("intervals",
         "Intervals carry real interval arithmetic: x+y=[a+c,b+d], "
         "x-y=[a-d,b-c], and x*y / x/y take the min and max of the four "
         "corner products.",
         "src/intervalvalue.cpp:58-98"),
    Fact("quaternions",
         "<w,x,y,z> is a quaternion literal (RapCAD calls it a complex value) "
         "and its * is the Hamilton product, hence associative but not "
         "commutative.",
         "src/parser.y:305-306; src/complexvalue.cpp:107-119"),
    Fact("quaternions",
         "ang(angle, axis) is the axis-angle constructor: w=cos(angle/2), "
         "xyz=axis*sin(angle/2), angle in degrees, and it is evaluated with "
         "right-angle-exact trig so ang(0, axis) is exactly <1,0,0,0>.",
         "src/function/angfunction.cpp:27-48; src/rmath.cpp:221-235"),
)


# --------------------------------------------------------------------------- #
# selfcheck / CLI                                                             #
# --------------------------------------------------------------------------- #
def _selfcheck() -> int:
    failures = 0

    def check(label: str, got, want) -> None:
        nonlocal failures
        ok = got == want
        if not ok:
            failures += 1
        print("%-7s %-46s got=%s want=%s"
              % ("ok" if ok else "FAIL", label, got, want))

    # 1. unit-suffixed literals, exact, millimetre base.
    check("5mm is 5", parse_number("5mm"), Fraction(5))
    check("5 (bare) is 5mm", parse_number("5"), parse_number("5mm"))
    check("1in is exactly 127/5 mm", parse_number("1in"), Fraction(127, 5))
    check("1th is 1/1000 in", parse_number("1th") * 1000, parse_number("1in"))
    check("1ft is 12in", parse_number("1ft"), parse_number("12in"))
    check("1m is 100cm", parse_number("1m"), parse_number("100cm"))
    check("1m is 1000000um", parse_number("1m"), parse_number("1000000um"))
    # the suffix-ordering trap: cm must beat the bare m.
    check("5cm is 50, not 5c metres", parse_number("5cm"), Fraction(50))

    # 2. rationals compose with units -- one literal, not a division.
    check("1/2m is 500", parse_number("1/2m"), Fraction(500))
    check("1/2/4mm is 1/8 (left assoc)", parse_number("1/2/4mm"),
          Fraction(1, 8))
    check("1 1/2in is 381/10", parse_number("1 1/2in"), Fraction(381, 10))
    check("1/0m is undef", parse_number("1/0m"), None)

    # 3. exactness: the discriminator, and a decimal that no double can hold.
    check("1.5 is exactly 15/10", parse_number("1.5"), Fraction(15, 10))
    check("1.0000000000000001 != 1.0 (exact tower)",
          parse_number("1.0000000000000001") != parse_number("1.0"), True)
    check("...and doubles cannot tell them apart",
          float("1.0000000000000001") == float("1.0"), True)
    check("is_exact_rational_dialect(this module)",
          is_exact_rational_dialect(
              lambda a, b: parse_number(a) != parse_number(b)), True)
    check("is_exact_rational_dialect(a double dialect)",
          is_exact_rational_dialect(lambda a, b: float(a) != float(b)), False)
    check("2e-3mm is exactly 1/500", parse_number("2e-3mm"), Fraction(1, 500))

    # 4. intervals. N[a,b] == [N-b, N+a] -- deviations are upper-then-lower.
    check("10[1,2] is [8,11]", Interval.from_literal(10, 1, 2),
          Interval(Fraction(8), Fraction(11)))
    check("10[1] is [9,11]", Interval.from_literal(10, 1),
          Interval(Fraction(9), Fraction(11)))
    check("10 PM 1 == 10[1]", Interval.from_literal(10, 1),
          Interval.from_literal(10, 1, 1))
    check("10[1,2] is NOT [9,12]",
          Interval.from_literal(10, 1, 2) == Interval(Fraction(9),
                                                      Fraction(12)), False)
    a = Interval.from_literal(10, 1, 2)     # [8, 11]
    b = Interval.from_literal(5, "1/2")     # [9/2, 11/2]
    check("[8,11]+[4.5,5.5] is [12.5,16.5]", a + b,
          Interval(Fraction(25, 2), Fraction(33, 2)))
    check("[8,11]-[4.5,5.5] is [2.5,6.5]", a - b,
          Interval(Fraction(5, 2), Fraction(13, 2)))
    check("subtraction is not symmetric-difference", a - a,
          Interval(Fraction(-3), Fraction(3)))
    check("[8,11]*[4.5,5.5] is [36,60.5]", a * b,
          Interval(Fraction(36), Fraction(121, 2)))
    check("negation swaps the ends", -a, Interval(Fraction(-11), Fraction(-8)))
    check("midpoint of 10[1,2] is 19/2", a.midpoint, Fraction(19, 2))
    check("tolerance of 10[1,2] is 3/2", a.tolerance, Fraction(3, 2))
    # a signed-corner case: min/max over corners, not endpoint pairing.
    neg = Interval(Fraction(-2), Fraction(3))
    check("[-2,3]*[-2,3] is [-6,9]", neg * neg,
          Interval(Fraction(-6), Fraction(9)))

    # 5. quaternions: Hamilton, not componentwise.
    i = Quaternion.of(0, 1, 0, 0)
    j = Quaternion.of(0, 0, 1, 0)
    k = Quaternion.of(0, 0, 0, 1)
    check("i*j == k", i * j, k)
    check("j*i == -k", j * i, Quaternion.of(0, 0, 0, -1))
    check("i*j != j*i (non-commutative)", (i * j) == (j * i), False)
    check("i*i == -1", i * i, Quaternion.of(-1, 0, 0, 0))
    check("(i*j)*k == i*(j*k) (associative)", (i * j) * k, i * (j * k))
    check("identity is a left unit", Quaternion.identity() * i, i)
    check("q*conj(q) is |q|^2", (i * i.conjugate()),
          Quaternion.of(i.norm_squared(), 0, 0, 0))
    check("ang(0, axis) is <1,0,0,0>", ang(0, (0, 0, 1)),
          Quaternion.identity())
    check("ang(180, z) is exactly <0,0,0,1>", ang(180, (0, 0, 1)), k)
    # Composing two 90-degree turns gives the 180-degree one. The half-angle is
    # 45 degrees, which is irrational, so this holds only to the working
    # precision -- as it does in the dialect, whose exact-rounding path also
    # applies only at right angles. Assert it numerically.
    composed = ang(90, (0, 0, 1)) * ang(90, (0, 0, 1))
    half_turn = ang(180, (0, 0, 1))
    drift = max(abs(float(getattr(composed, c) - getattr(half_turn, c)))
                for c in ("w", "x", "y", "z"))
    check("ang(90,z)*ang(90,z) ~= ang(180,z)", drift < 1e-9, True)

    # 6. every fact carries a citation.
    check("all facts cite a path and line",
          all(f.citation and ":" in f.citation and f.claim for f in FACTS),
          True)

    if failures:
        print("selfcheck: %d failure(s)" % failures)
        return 1
    print("selfcheck: all checks passed; %d facts recorded" % len(FACTS))
    return 0


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        prog="rapcad_language",
        description="RapCAD numeric/literal semantics: exact-rational "
                    "unit literals, tolerance intervals, quaternions.")
    parser.add_argument("--selfcheck", action="store_true",
                        help="prove the recorded semantics on worked examples")
    parser.add_argument("--facts", action="store_true",
                        help="print the recorded facts with their citations")
    parser.add_argument("--number", metavar="LITERAL",
                        help="parse one RapCAD numeric literal to exact mm")
    args = parser.parse_args(argv)
    if args.selfcheck:
        return _selfcheck()
    if args.facts:
        for fact in FACTS:
            print(fact.render())
        return 0
    if args.number:
        value = parse_number(args.number)
        if value is None:
            print("undef")
            return 1
        print("%s mm (exact) = %s" % (value, float(value)))
        return 0
    parser.print_usage()
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
