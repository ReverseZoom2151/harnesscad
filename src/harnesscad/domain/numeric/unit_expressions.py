"""Unit-aware length/angle expression evaluator (CodeToCAD-style ``LengthExp``).

CodeToCAD lets a user write dimensions as strings -- ``"2mm + 1m"``, ``"6in + 2ft"``,
``"90deg + 0.5rad"`` -- and normalises them to SI base units (metres, radians).  The
upstream implementation does this with a regex substitution followed by ``eval``,
which is neither safe nor able to type-check the arithmetic (it happily returns
``m**2`` for ``1mm * 1in``).

This module reimplements the idea properly and deterministically:

* a tokeniser that recognises decimal numbers, imperial fractions (``"1/2in"``),
  mixed numbers (``"1-1/2in"``), percentages (``"50%"``), unit suffixes and the
  operators ``+ - * / ``;
* a recursive-descent parser (no ``eval``, no ``__builtins__`` exposure);
* a small dimensional-analysis layer: a :class:`Quantity` carries a *kind*
  (``scalar`` / ``length`` / ``angle`` / ``percent``) and the evaluator rejects
  nonsense such as ``length + angle`` or ``length * length``;
* percentages resolve against an optional ``base`` value, giving the
  proportional / relative-dimension behaviour ("make it 50% of the parent").

Everything is exact w.r.t. the IEEE-754 arithmetic performed; no randomness.

Public API
----------
``parse_length(expr, base=None) -> float``      metres
``parse_angle(expr) -> float``                  radians
``parse_quantity(expr, base=None) -> Quantity``
``convert_length(metres, unit) -> float``
``format_length(metres, unit, ndigits=6) -> str``
``LENGTH_UNITS`` / ``ANGLE_UNITS`` -- unit -> base-unit factor.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass

__all__ = [
    "LENGTH_UNITS",
    "ANGLE_UNITS",
    "Quantity",
    "ExpressionError",
    "tokenize",
    "parse_quantity",
    "parse_length",
    "parse_angle",
    "convert_length",
    "convert_angle",
    "format_length",
]


# Metres per unit.
LENGTH_UNITS: dict[str, float] = {
    "km": 1000.0,
    "m": 1.0,
    "dm": 0.1,
    "cm": 0.01,
    "mm": 0.001,
    "um": 1e-6,
    "nm": 1e-9,
    "mil": 0.0000254,
    "thou": 0.0000254,
    "in": 0.0254,
    '"': 0.0254,
    "ft": 0.3048,
    "'": 0.3048,
    "yd": 0.9144,
}

# Radians per unit.
ANGLE_UNITS: dict[str, float] = {
    "rad": 1.0,
    "deg": math.pi / 180.0,
    "d": math.pi / 180.0,
    "grad": math.pi / 200.0,
    "turn": 2.0 * math.pi,
    "rev": 2.0 * math.pi,
}

SCALAR = "scalar"
LENGTH = "length"
ANGLE = "angle"
PERCENT = "percent"


class ExpressionError(ValueError):
    """Raised for malformed expressions or invalid dimensional arithmetic."""


@dataclass(frozen=True)
class Quantity:
    """A number plus a dimensional *kind*.

    ``value`` is always in the base unit of the kind: metres for ``length``,
    radians for ``angle``, a plain number for ``scalar``, and a *fraction*
    (0.5 for "50%") for ``percent``.
    """

    value: float
    kind: str = SCALAR

    def __post_init__(self) -> None:
        if self.kind not in (SCALAR, LENGTH, ANGLE, PERCENT):
            raise ExpressionError("unknown kind: " + str(self.kind))

    # -- dimensional arithmetic -------------------------------------------------
    def _additive(self, other: "Quantity", sign: float) -> "Quantity":
        a, b = self, other
        if a.kind == b.kind:
            return Quantity(a.value + sign * b.value, a.kind)
        # scalar + percent is meaningless; length + scalar is a common typo.
        raise ExpressionError(
            "cannot add/subtract {0} and {1}".format(a.kind, b.kind)
        )

    def __add__(self, other: "Quantity") -> "Quantity":
        return self._additive(other, 1.0)

    def __sub__(self, other: "Quantity") -> "Quantity":
        return self._additive(other, -1.0)

    def __mul__(self, other: "Quantity") -> "Quantity":
        a, b = self, other
        # percent behaves like a scalar multiplier: "50% * 2m" -> 1m
        akind = SCALAR if a.kind == PERCENT else a.kind
        bkind = SCALAR if b.kind == PERCENT else b.kind
        if akind == SCALAR:
            return Quantity(a.value * b.value, bkind)
        if bkind == SCALAR:
            return Quantity(a.value * b.value, akind)
        raise ExpressionError(
            "cannot multiply {0} by {1}".format(a.kind, b.kind)
        )

    def __truediv__(self, other: "Quantity") -> "Quantity":
        a, b = self, other
        bkind = SCALAR if b.kind == PERCENT else b.kind
        if b.value == 0.0:
            raise ExpressionError("division by zero")
        if bkind == SCALAR:
            return Quantity(a.value / b.value, SCALAR if a.kind == PERCENT else a.kind)
        if a.kind == b.kind:
            # length / length -> dimensionless ratio
            return Quantity(a.value / b.value, SCALAR)
        raise ExpressionError("cannot divide {0} by {1}".format(a.kind, b.kind))

    def __neg__(self) -> "Quantity":
        return Quantity(-self.value, self.kind)

    def resolve(self, base: float | None = None, base_kind: str = LENGTH) -> "Quantity":
        """Turn a bare percentage into a concrete quantity of ``base_kind``."""
        if self.kind != PERCENT or base is None:
            return self
        return Quantity(self.value * base, base_kind)


# ---------------------------------------------------------------------------
# tokenizer
# ---------------------------------------------------------------------------

_UNIT_NAMES = sorted(
    set(LENGTH_UNITS) | set(ANGLE_UNITS), key=lambda u: (-len(u), u)
)
_UNIT_ALT = "|".join(re.escape(u) for u in _UNIT_NAMES)
_NUM = r"\d+(?:\.\d*)?|\.\d+"

# Ordered: mixed number, fraction, plain number+unit, plain number.
_TAIL = r")(?![A-Za-z])"
_MIXED_RE = re.compile(
    r"(?P<w>\d+)\s*-\s*(?P<n>\d+)\s*/\s*(?P<d>\d+)\s*(?P<u>" + _UNIT_ALT + _TAIL
)
_FRAC_RE = re.compile(
    r"(?P<n>\d+)\s*/\s*(?P<d>\d+)\s*(?P<u>" + _UNIT_ALT + _TAIL
)
_PCT_RE = re.compile(r"(?P<v>" + _NUM + r")\s*%")
_NUMU_RE = re.compile(r"(?P<v>" + _NUM + r")\s*(?P<u>" + _UNIT_ALT + _TAIL)
_NUM_RE = re.compile(_NUM)
_OP_RE = re.compile(r"[+\-*/()]")


def _unit_quantity(value: float, unit: str) -> Quantity:
    if unit in LENGTH_UNITS:
        return Quantity(value * LENGTH_UNITS[unit], LENGTH)
    return Quantity(value * ANGLE_UNITS[unit], ANGLE)


def tokenize(expr: str) -> list[tuple[str, object]]:
    """Return a list of ``(kind, payload)`` tokens.

    Kinds: ``"q"`` (payload :class:`Quantity`) and ``"op"`` (payload str).
    """
    text = str(expr)
    tokens: list[tuple[str, object]] = []
    i = 0
    n = len(text)
    while i < n:
        ch = text[i]
        if ch.isspace():
            i += 1
            continue
        for regex, build in (
            (_MIXED_RE, "mixed"),
            (_FRAC_RE, "frac"),
            (_PCT_RE, "pct"),
            (_NUMU_RE, "numu"),
        ):
            m = regex.match(text, i)
            if not m:
                continue
            if build == "mixed":
                val = float(m.group("w")) + float(m.group("n")) / float(m.group("d"))
                tokens.append(("q", _unit_quantity(val, m.group("u"))))
            elif build == "frac":
                val = float(m.group("n")) / float(m.group("d"))
                tokens.append(("q", _unit_quantity(val, m.group("u"))))
            elif build == "pct":
                tokens.append(("q", Quantity(float(m.group("v")) / 100.0, PERCENT)))
            else:
                tokens.append(("q", _unit_quantity(float(m.group("v")), m.group("u"))))
            i = m.end()
            break
        else:
            m = _NUM_RE.match(text, i)
            if m:
                tokens.append(("q", Quantity(float(m.group(0)), SCALAR)))
                i = m.end()
                continue
            m = _OP_RE.match(text, i)
            if m:
                tokens.append(("op", m.group(0)))
                i = m.end()
                continue
            raise ExpressionError(
                "unexpected character {0!r} at index {1}".format(ch, i)
            )
    if not tokens:
        raise ExpressionError("empty expression")
    return tokens


# ---------------------------------------------------------------------------
# recursive-descent parser:  expr := term (('+'|'-') term)*
#                            term := unary (('*'|'/') unary)*
#                            unary := ('-'|'+')? primary
#                            primary := QUANTITY | '(' expr ')'
# ---------------------------------------------------------------------------


class _Parser:
    def __init__(self, tokens: list[tuple[str, object]]) -> None:
        self.tokens = tokens
        self.pos = 0

    def peek(self) -> tuple[str, object] | None:
        if self.pos < len(self.tokens):
            return self.tokens[self.pos]
        return None

    def eat_op(self, *ops: str) -> str | None:
        tok = self.peek()
        if tok is not None and tok[0] == "op" and tok[1] in ops:
            self.pos += 1
            return str(tok[1])
        return None

    def parse(self) -> Quantity:
        value = self.expr()
        if self.pos != len(self.tokens):
            raise ExpressionError("trailing tokens in expression")
        return value

    def expr(self) -> Quantity:
        value = self.term()
        while True:
            op = self.eat_op("+", "-")
            if op is None:
                return value
            rhs = self.term()
            value = value + rhs if op == "+" else value - rhs

    def term(self) -> Quantity:
        value = self.unary()
        while True:
            op = self.eat_op("*", "/")
            if op is None:
                return value
            rhs = self.unary()
            value = value * rhs if op == "*" else value / rhs

    def unary(self) -> Quantity:
        op = self.eat_op("-", "+")
        value = self.primary()
        return -value if op == "-" else value

    def primary(self) -> Quantity:
        tok = self.peek()
        if tok is None:
            raise ExpressionError("unexpected end of expression")
        if tok[0] == "q":
            self.pos += 1
            assert isinstance(tok[1], Quantity)
            return tok[1]
        if tok[1] == "(":
            self.pos += 1
            value = self.expr()
            if self.eat_op(")") is None:
                raise ExpressionError("missing closing parenthesis")
            return value
        raise ExpressionError("unexpected operator {0!r}".format(tok[1]))


def parse_quantity(expr, base: float | None = None, base_kind: str = LENGTH) -> Quantity:
    """Evaluate ``expr`` into a :class:`Quantity`.

    ``int``/``float`` inputs are treated as *base-unit* quantities of ``base_kind``
    (metres for lengths -- matching CodeToCAD's convention).  A bare percentage
    resolves against ``base`` when supplied.
    """
    if isinstance(expr, Quantity):
        return expr.resolve(base, base_kind)
    if isinstance(expr, bool):
        raise ExpressionError("bool is not a dimension")
    if isinstance(expr, (int, float)):
        return Quantity(float(expr), base_kind)
    tokens = tokenize(expr)
    if base is not None:
        # Resolve percentages *before* evaluation so that "50% + 1mm" type-checks.
        tokens = [
            ("q", token.resolve(base, base_kind)) if kind == "q" else (kind, token)
            for kind, token in tokens
        ]
    quantity = _Parser(tokens).parse()
    return quantity.resolve(base, base_kind)


def parse_length(expr, base: float | None = None) -> float:
    """Evaluate ``expr`` and return metres."""
    quantity = parse_quantity(expr, base=base, base_kind=LENGTH)
    if quantity.kind == PERCENT:
        raise ExpressionError("percentage used without a base value")
    if quantity.kind == SCALAR:
        # bare number in a length context == metres (CodeToCAD convention)
        return quantity.value
    if quantity.kind != LENGTH:
        raise ExpressionError("expected a length, got " + quantity.kind)
    return quantity.value


def parse_angle(expr) -> float:
    """Evaluate ``expr`` and return radians (bare numbers are radians)."""
    quantity = parse_quantity(expr, base_kind=ANGLE)
    if quantity.kind == SCALAR:
        return quantity.value
    if quantity.kind != ANGLE:
        raise ExpressionError("expected an angle, got " + quantity.kind)
    return quantity.value


def convert_length(metres: float, unit: str) -> float:
    """Convert metres into ``unit``."""
    if unit not in LENGTH_UNITS:
        raise ExpressionError("unknown length unit: " + str(unit))
    return metres / LENGTH_UNITS[unit]


def convert_angle(radians: float, unit: str) -> float:
    if unit not in ANGLE_UNITS:
        raise ExpressionError("unknown angle unit: " + str(unit))
    return radians / ANGLE_UNITS[unit]


def format_length(metres: float, unit: str = "mm", ndigits: int = 6) -> str:
    """Render ``metres`` in ``unit`` with trailing zeros trimmed."""
    value = round(convert_length(metres, unit), ndigits)
    text = "{0:.{1}f}".format(value, ndigits).rstrip("0").rstrip(".")
    if text in ("", "-0"):
        text = "0"
    return text + unit
