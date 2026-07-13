"""Linear-form algebra and an arithmetic-expression parser for programming-based CAD.

Gonzalez et al., *Facilitating the Parametric Definition of Geometric Properties
in Programming-Based CAD* (UIST '24), makes one central empirical claim: in
programming-based CAD (OpenSCAD), the positions and sizes of model elements are
"mainly linear combinations of variables". Formalised in the paper (Section 4),
a translate component is

    t_i = sum_j (alpha_j * x_j) + c

with ``alpha_j`` and ``c`` constant and ``x_j`` a variable. The paper's whole
"position" / "delta vector" machinery — deriving the parametric position of a
handle by walking the CSG tree and *adding up* the translate definitions along a
branch — is exactly arithmetic on such linear forms, followed by an "expression
simplification" step (the paper offloads this to a SymPy server; here it is a
deterministic, stdlib-only canonicaliser).

This module provides the two deterministic primitives that underpin everything
else in the ``paramgeom_*`` family:

* :class:`LinearForm` — a canonical ``sum(alpha_j x_j) + c`` over named
  variables, with exact :class:`~fractions.Fraction` coefficients, closed under
  add / subtract / scale / negate, plus evaluation and code rendering. This is
  both the paper's C3 "linear combination" model *and* its expression
  simplifier (like terms are collected, zero terms dropped, ``translate 0``
  contributions vanish — the paper flags un-simplified ``translate 0`` output as
  a readability problem).
* a tiny arithmetic-expression parser (``+ - * / unary-minus parens``, numeric
  literals, identifiers, and the ternary ``?:``) producing an :class:`Expr`
  AST, plus :func:`to_linear_form` which reduces an :class:`Expr` to a
  :class:`LinearForm` when it is affine and raises :class:`NonLinearError`
  otherwise. Downstream, :mod:`programs.paramgeom_classify` uses the AST to
  assign the paper's C1..C5 categories.

Pure stdlib, no I/O, fully deterministic.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from fractions import Fraction
from typing import Dict, List, Mapping, Optional, Tuple, Union

Number = Union[int, float, Fraction]


def _as_fraction(value: Number) -> Fraction:
    """Coerce an int/float/Fraction to an exact Fraction deterministically."""
    if isinstance(value, Fraction):
        return value
    if isinstance(value, int):
        return Fraction(value)
    if isinstance(value, float):
        return Fraction(value).limit_denominator(10 ** 9)
    raise TypeError(f"cannot coerce {type(value)!r} to Fraction")


def _fmt_coeff(fr: Fraction) -> str:
    """Render a Fraction as a compact code literal (int when integral)."""
    if fr.denominator == 1:
        return str(fr.numerator)
    return f"{fr.numerator}/{fr.denominator}"


class NonLinearError(ValueError):
    """Raised when an expression cannot be reduced to a linear form."""


@dataclass(frozen=True)
class LinearForm:
    """A canonical linear combination ``sum(coeff_v * v) + constant``.

    Variables are held in a mapping ``{name: Fraction}`` with zero coefficients
    already pruned, so equality is structural and rendering is stable. Instances
    are immutable; every operation returns a fresh form.
    """

    terms: Dict[str, Fraction] = field(default_factory=dict)
    constant: Fraction = Fraction(0)

    # -- constructors -----------------------------------------------------
    @staticmethod
    def const(value: Number) -> "LinearForm":
        return LinearForm({}, _as_fraction(value))

    @staticmethod
    def var(name: str, coeff: Number = 1) -> "LinearForm":
        c = _as_fraction(coeff)
        if c == 0:
            return LinearForm({}, Fraction(0))
        return LinearForm({name: c}, Fraction(0))

    @staticmethod
    def from_terms(terms: Mapping[str, Number], constant: Number = 0) -> "LinearForm":
        pruned: Dict[str, Fraction] = {}
        for name, coeff in terms.items():
            c = _as_fraction(coeff)
            if c != 0:
                pruned[name] = pruned.get(name, Fraction(0)) + c
        pruned = {k: v for k, v in pruned.items() if v != 0}
        return LinearForm(pruned, _as_fraction(constant))

    # -- algebra ----------------------------------------------------------
    def __add__(self, other: "LinearForm") -> "LinearForm":
        terms = dict(self.terms)
        for name, coeff in other.terms.items():
            terms[name] = terms.get(name, Fraction(0)) + coeff
        terms = {k: v for k, v in terms.items() if v != 0}
        return LinearForm(terms, self.constant + other.constant)

    def __neg__(self) -> "LinearForm":
        return LinearForm({k: -v for k, v in self.terms.items()}, -self.constant)

    def __sub__(self, other: "LinearForm") -> "LinearForm":
        return self + (-other)

    def scaled(self, factor: Number) -> "LinearForm":
        f = _as_fraction(factor)
        if f == 0:
            return LinearForm({}, Fraction(0))
        return LinearForm({k: v * f for k, v in self.terms.items()}, self.constant * f)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, LinearForm):
            return NotImplemented
        return self.terms == other.terms and self.constant == other.constant

    def __hash__(self) -> int:
        return hash((tuple(sorted(self.terms.items())), self.constant))

    # -- queries ----------------------------------------------------------
    @property
    def is_zero(self) -> bool:
        return not self.terms and self.constant == 0

    @property
    def is_constant(self) -> bool:
        return not self.terms

    @property
    def variables(self) -> Tuple[str, ...]:
        return tuple(sorted(self.terms))

    def coefficient(self, name: str) -> Fraction:
        return self.terms.get(name, Fraction(0))

    def evaluate(self, env: Mapping[str, Number]) -> Fraction:
        total = self.constant
        for name, coeff in self.terms.items():
            if name not in env:
                raise KeyError(f"variable {name!r} not bound in environment")
            total += coeff * _as_fraction(env[name])
        return total

    # -- rendering --------------------------------------------------------
    def to_code(self, var_order: Optional[List[str]] = None) -> str:
        """Render as a simplified arithmetic expression, e.g. ``3 + 2*var1 - var2``.

        Terms are ordered by ``var_order`` (any variables missing from it sort
        alphabetically after), then the constant is appended. A zero form
        renders as ``0``. This is the deterministic replacement for the paper's
        SymPy-backed simplification service.
        """
        names = list(self.terms)
        if var_order is not None:
            index = {n: i for i, n in enumerate(var_order)}
            names.sort(key=lambda n: (index.get(n, len(var_order)), n))
        else:
            names.sort()

        pieces: List[str] = []
        for name in names:
            coeff = self.terms[name]
            sign = "-" if coeff < 0 else "+"
            mag = -coeff if coeff < 0 else coeff
            if mag == 1:
                term = name
            else:
                term = f"{_fmt_coeff(mag)}*{name}"
            if not pieces:
                pieces.append(term if sign == "+" else f"-{term}")
            else:
                pieces.append(f"{sign} {term}")

        if self.constant != 0 or not pieces:
            c = self.constant
            if not pieces:
                pieces.append(_fmt_coeff(c))
            else:
                sign = "-" if c < 0 else "+"
                mag = -c if c < 0 else c
                pieces.append(f"{sign} {_fmt_coeff(mag)}")
        return " ".join(pieces)

    def __repr__(self) -> str:  # pragma: no cover - debug aid
        return f"LinearForm({self.to_code()!r})"


# ---------------------------------------------------------------------------
# Expression AST + parser
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Expr:
    """Base class for arithmetic-expression AST nodes."""


@dataclass(frozen=True)
class Num(Expr):
    value: Fraction


@dataclass(frozen=True)
class Var(Expr):
    name: str


@dataclass(frozen=True)
class BinOp(Expr):
    op: str  # one of + - * /
    left: Expr
    right: Expr


@dataclass(frozen=True)
class Neg(Expr):
    operand: Expr


@dataclass(frozen=True)
class Ternary(Expr):
    """A conditional ``cond ? a : b`` (OpenSCAD/C style); marks C5 territory."""

    cond: "Expr"
    then: Expr
    otherwise: Expr


@dataclass(frozen=True)
class Call(Expr):
    """A function call such as ``sin(x)``; also C5 territory."""

    name: str
    args: Tuple[Expr, ...]


# -- tokenizer --------------------------------------------------------------

_SYMBOLS = {"+", "-", "*", "/", "(", ")", "?", ":", ",", "<", ">", "="}


@dataclass
class _Token:
    kind: str  # num, id, op
    text: str


def _tokenize(text: str) -> List[_Token]:
    tokens: List[_Token] = []
    i, n = 0, len(text)
    while i < n:
        ch = text[i]
        if ch.isspace():
            i += 1
            continue
        if ch.isdigit() or (ch == "." and i + 1 < n and text[i + 1].isdigit()):
            j = i
            while j < n and (text[j].isdigit() or text[j] == "."):
                j += 1
            tokens.append(_Token("num", text[i:j]))
            i = j
            continue
        if ch.isalpha() or ch == "_":
            j = i
            while j < n and (text[j].isalnum() or text[j] == "_"):
                j += 1
            tokens.append(_Token("id", text[i:j]))
            i = j
            continue
        # multi-char comparison operators
        if ch in "<>=!" and i + 1 < n and text[i + 1] == "=":
            tokens.append(_Token("op", text[i : i + 2]))
            i += 2
            continue
        if ch in _SYMBOLS:
            tokens.append(_Token("op", ch))
            i += 1
            continue
        raise SyntaxError(f"unexpected character {ch!r} at position {i}")
    return tokens


class _Parser:
    """Recursive-descent parser: ternary > add > mul > unary > primary."""

    def __init__(self, tokens: List[_Token]) -> None:
        self._tokens = tokens
        self._pos = 0

    def _peek(self) -> Optional[_Token]:
        return self._tokens[self._pos] if self._pos < len(self._tokens) else None

    def _next(self) -> _Token:
        tok = self._tokens[self._pos]
        self._pos += 1
        return tok

    def _accept(self, text: str) -> bool:
        tok = self._peek()
        if tok is not None and tok.text == text:
            self._pos += 1
            return True
        return False

    def _expect(self, text: str) -> None:
        if not self._accept(text):
            raise SyntaxError(f"expected {text!r}")

    def parse(self) -> Expr:
        expr = self._ternary()
        if self._peek() is not None:
            raise SyntaxError(f"trailing tokens near {self._peek().text!r}")
        return expr

    def _ternary(self) -> Expr:
        cond = self._comparison()
        if self._accept("?"):
            then = self._ternary()
            self._expect(":")
            otherwise = self._ternary()
            return Ternary(cond, then, otherwise)
        return cond

    def _comparison(self) -> Expr:
        left = self._add()
        tok = self._peek()
        if tok is not None and tok.text in ("<", ">", "<=", ">=", "==", "!="):
            op = self._next().text
            right = self._add()
            # Comparisons are non-arithmetic; represent as an opaque Call so the
            # classifier routes them to C5.
            return Call(f"cmp{op}", (left, right))
        return left

    def _add(self) -> Expr:
        node = self._mul()
        while True:
            tok = self._peek()
            if tok is not None and tok.text in ("+", "-"):
                op = self._next().text
                node = BinOp(op, node, self._mul())
            else:
                return node

    def _mul(self) -> Expr:
        node = self._unary()
        while True:
            tok = self._peek()
            if tok is not None and tok.text in ("*", "/"):
                op = self._next().text
                node = BinOp(op, node, self._unary())
            else:
                return node

    def _unary(self) -> Expr:
        if self._accept("-"):
            return Neg(self._unary())
        if self._accept("+"):
            return self._unary()
        return self._primary()

    def _primary(self) -> Expr:
        tok = self._peek()
        if tok is None:
            raise SyntaxError("unexpected end of expression")
        if tok.text == "(":
            self._next()
            expr = self._ternary()
            self._expect(")")
            return expr
        if tok.kind == "num":
            self._next()
            return Num(Fraction(tok.text))
        if tok.kind == "id":
            self._next()
            if self._accept("("):
                args: List[Expr] = []
                if not self._accept(")"):
                    args.append(self._ternary())
                    while self._accept(","):
                        args.append(self._ternary())
                    self._expect(")")
                return Call(tok.text, tuple(args))
            return Var(tok.text)
        raise SyntaxError(f"unexpected token {tok.text!r}")


def parse_expression(text: str) -> Expr:
    """Parse an arithmetic expression string into an :class:`Expr` AST."""
    tokens = _tokenize(text)
    if not tokens:
        raise SyntaxError("empty expression")
    return _Parser(tokens).parse()


def to_linear_form(expr: Union[str, Expr]) -> LinearForm:
    """Reduce an affine :class:`Expr` (or string) to a :class:`LinearForm`.

    Raises :class:`NonLinearError` for anything that is not affine in its
    variables: variable*variable products, division by a variable, ternaries,
    comparisons, or general function calls.
    """
    if isinstance(expr, str):
        expr = parse_expression(expr)
    return _reduce(expr)


def _reduce(expr: Expr) -> LinearForm:
    if isinstance(expr, Num):
        return LinearForm.const(expr.value)
    if isinstance(expr, Var):
        return LinearForm.var(expr.name)
    if isinstance(expr, Neg):
        return -_reduce(expr.operand)
    if isinstance(expr, BinOp):
        left = _reduce(expr.left)
        right = _reduce(expr.right)
        if expr.op == "+":
            return left + right
        if expr.op == "-":
            return left - right
        if expr.op == "*":
            if left.is_constant:
                return right.scaled(left.constant)
            if right.is_constant:
                return left.scaled(right.constant)
            raise NonLinearError("product of two non-constant subexpressions")
        if expr.op == "/":
            if right.is_constant:
                if right.constant == 0:
                    raise NonLinearError("division by zero")
                return left.scaled(Fraction(1) / right.constant)
            raise NonLinearError("division by a non-constant subexpression")
    if isinstance(expr, (Ternary, Call)):
        raise NonLinearError(f"non-affine construct: {type(expr).__name__}")
    raise NonLinearError(f"unsupported expression node: {type(expr).__name__}")
