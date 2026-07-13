"""FreeCAD expression-language parser and evaluator (deterministic, stdlib).

FreeCAD makes a model parametric by binding an object property to an
*expression* rather than a literal: ``Pad.Length = Variables.height``,
``Box.Length = Variables.width * 2``, ``Sketch.Constraints[0] = spreadsheet.w``,
``Placement.Base.x = cos(30) * 10``. The freecad-ai workbench exposes this via
its ``set_expression`` tool (bind ``object_name.property_name`` to an expression
string) and its ``create_variable_set`` / ``create_spreadsheet`` tools that
create the ``Variables`` cells those expressions reference. The actual binding
and recompute happen inside FreeCAD; but the *expression language itself* -- its
grammar, how a reference like ``Variables.height`` or ``Object.Sub[3].x``
resolves, and how the arithmetic evaluates -- is a small deterministic language,
and that is what this module implements.

What it provides, all without FreeCAD:

* :func:`tokenize` / :func:`parse` -- a recursive-descent parser for the FreeCAD
  expression grammar: numbers (with an optional trailing unit like ``mm``),
  property references (dotted, with ``[index]`` subscripts), parenthesised
  sub-expressions, unary +/-, the binary operators ``+ - * / % ^``, and function
  calls (``sin``, ``cos``, ``sqrt``, ``min``, ``max`` ...). Trig is in DEGREES to
  match FreeCAD.
* :class:`Expression` -- the parsed AST, which can :meth:`~Expression.evaluate`
  against an environment mapping references to numbers, and can report its
  :meth:`~Expression.references` (the deterministic dependency set an editor
  needs to order recomputes and detect cycles).

Everything is stdlib-only and deterministic. No FreeCAD, no ``eval``, no network.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple, Union

__all__ = [
    "ExpressionError",
    "Token",
    "tokenize",
    "parse",
    "Expression",
    "Reference",
    "FUNCTIONS",
]


class ExpressionError(ValueError):
    """Raised for a malformed or unevaluable FreeCAD expression."""


# ── functions (degrees for trig, matching FreeCAD) ──────────────────────────

FUNCTIONS: Dict[str, Callable[..., float]] = {
    "sin": lambda x: math.sin(math.radians(x)),
    "cos": lambda x: math.cos(math.radians(x)),
    "tan": lambda x: math.tan(math.radians(x)),
    "asin": lambda x: math.degrees(math.asin(x)),
    "acos": lambda x: math.degrees(math.acos(x)),
    "atan": lambda x: math.degrees(math.atan(x)),
    "atan2": lambda y, x: math.degrees(math.atan2(y, x)),
    "sqrt": math.sqrt,
    "abs": abs,
    "floor": lambda x: float(math.floor(x)),
    "ceil": lambda x: float(math.ceil(x)),
    "round": lambda x: float(round(x)),
    "exp": math.exp,
    "log": math.log,
    "pow": lambda x, y: math.pow(x, y),
    "min": min,
    "max": max,
    "mod": lambda x, y: math.fmod(x, y),
    "hypot": math.hypot,
}


# ── tokenizer ───────────────────────────────────────────────────────────────

# Token kinds
NUM = "NUM"
IDENT = "IDENT"
OP = "OP"
LPAREN = "LPAREN"
RPAREN = "RPAREN"
LBRACK = "LBRACK"
RBRACK = "RBRACK"
DOT = "DOT"
COMMA = "COMMA"


@dataclass(frozen=True)
class Token:
    kind: str
    value: str
    pos: int


_OPS = set("+-*/%^")


def tokenize(text: str) -> List[Token]:
    """Lex a FreeCAD expression string into tokens."""
    tokens: List[Token] = []
    i, n = 0, len(text)
    while i < n:
        c = text[i]
        if c.isspace():
            i += 1
            continue
        if c.isdigit() or (c == "." and i + 1 < n and text[i + 1].isdigit()):
            j = i
            seen_dot = False
            while j < n and (text[j].isdigit() or text[j] == "."):
                if text[j] == ".":
                    if seen_dot:
                        break
                    seen_dot = True
                j += 1
            # optional exponent
            if j < n and text[j] in "eE":
                k = j + 1
                if k < n and text[k] in "+-":
                    k += 1
                if k < n and text[k].isdigit():
                    j = k
                    while j < n and text[j].isdigit():
                        j += 1
            tokens.append(Token(NUM, text[i:j], i))
            i = j
            continue
        if c.isalpha() or c == "_":
            j = i
            while j < n and (text[j].isalnum() or text[j] == "_"):
                j += 1
            tokens.append(Token(IDENT, text[i:j], i))
            i = j
            continue
        if c in _OPS:
            tokens.append(Token(OP, c, i))
            i += 1
            continue
        if c == "(":
            tokens.append(Token(LPAREN, c, i))
        elif c == ")":
            tokens.append(Token(RPAREN, c, i))
        elif c == "[":
            tokens.append(Token(LBRACK, c, i))
        elif c == "]":
            tokens.append(Token(RBRACK, c, i))
        elif c == ".":
            tokens.append(Token(DOT, c, i))
        elif c == ",":
            tokens.append(Token(COMMA, c, i))
        else:
            raise ExpressionError("unexpected character %r at %d" % (c, i))
        i += 1
    return tokens


# ── AST nodes ───────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class Number:
    value: float
    unit: Optional[str] = None


@dataclass(frozen=True)
class Reference:
    """A dotted property reference like ``Variables.height`` or ``B.Sub[3].x``.

    ``path`` is the tuple of dotted identifiers; ``subscripts`` maps the index
    of a path element to its integer subscript (``Constraints[0]`` -> element 1
    of ``Sketch.Constraints`` has subscript 0).
    """
    path: Tuple[str, ...]
    subscripts: Tuple[Tuple[int, int], ...] = ()

    def key(self) -> str:
        """Canonical string form used for environment lookup."""
        subs = dict(self.subscripts)
        parts = []
        for idx, name in enumerate(self.path):
            if idx in subs:
                parts.append("%s[%d]" % (name, subs[idx]))
            else:
                parts.append(name)
        return ".".join(parts)


@dataclass(frozen=True)
class UnaryOp:
    op: str
    operand: "Node"


@dataclass(frozen=True)
class BinOp:
    op: str
    left: "Node"
    right: "Node"


@dataclass(frozen=True)
class Call:
    name: str
    args: Tuple["Node", ...]


Node = Union[Number, Reference, UnaryOp, BinOp, Call]


# ── parser (recursive descent, standard precedence) ─────────────────────────

class _Parser:
    def __init__(self, tokens: List[Token]):
        self.toks = tokens
        self.i = 0

    def _peek(self) -> Optional[Token]:
        return self.toks[self.i] if self.i < len(self.toks) else None

    def _next(self) -> Token:
        tok = self._peek()
        if tok is None:
            raise ExpressionError("unexpected end of expression")
        self.i += 1
        return tok

    def _expect(self, kind: str) -> Token:
        tok = self._next()
        if tok.kind != kind:
            raise ExpressionError(
                "expected %s but got %r at %d" % (kind, tok.value, tok.pos))
        return tok

    def parse(self) -> Node:
        node = self._expr()
        if self._peek() is not None:
            tok = self._peek()
            raise ExpressionError(
                "trailing tokens from %r at %d" % (tok.value, tok.pos))
        return node

    # expr := term (('+'|'-') term)*
    def _expr(self) -> Node:
        node = self._term()
        while True:
            tok = self._peek()
            if tok and tok.kind == OP and tok.value in "+-":
                self._next()
                node = BinOp(tok.value, node, self._term())
            else:
                return node

    # term := factor (('*'|'/'|'%') factor)*
    def _term(self) -> Node:
        node = self._factor()
        while True:
            tok = self._peek()
            if tok and tok.kind == OP and tok.value in "*/%":
                self._next()
                node = BinOp(tok.value, node, self._factor())
            else:
                return node

    # factor := unary ('^' factor)?   (right-associative power)
    def _factor(self) -> Node:
        node = self._unary()
        tok = self._peek()
        if tok and tok.kind == OP and tok.value == "^":
            self._next()
            return BinOp("^", node, self._factor())
        return node

    # unary := ('+'|'-') unary | atom
    def _unary(self) -> Node:
        tok = self._peek()
        if tok and tok.kind == OP and tok.value in "+-":
            self._next()
            return UnaryOp(tok.value, self._unary())
        return self._atom()

    def _atom(self) -> Node:
        tok = self._peek()
        if tok is None:
            raise ExpressionError("unexpected end of expression")
        if tok.kind == LPAREN:
            self._next()
            node = self._expr()
            self._expect(RPAREN)
            return node
        if tok.kind == NUM:
            self._next()
            return self._number_with_unit(tok)
        if tok.kind == IDENT:
            return self._ident_atom()
        raise ExpressionError("unexpected token %r at %d" % (tok.value, tok.pos))

    def _number_with_unit(self, tok: Token) -> Number:
        value = float(tok.value)
        unit = None
        nxt = self._peek()
        if nxt and nxt.kind == IDENT and nxt.value in _UNITS:
            self._next()
            unit = nxt.value
        return Number(value, unit)

    def _ident_atom(self) -> Node:
        first = self._next()  # IDENT
        # function call?
        if self._peek() and self._peek().kind == LPAREN:
            self._next()
            args: List[Node] = []
            if self._peek() and self._peek().kind != RPAREN:
                args.append(self._expr())
                while self._peek() and self._peek().kind == COMMA:
                    self._next()
                    args.append(self._expr())
            self._expect(RPAREN)
            return Call(first.value, tuple(args))
        # otherwise a (possibly dotted, possibly subscripted) reference
        path: List[str] = [first.value]
        subs: List[Tuple[int, int]] = []
        self._maybe_subscript(len(path) - 1, subs)
        while self._peek() and self._peek().kind == DOT:
            self._next()
            name = self._expect(IDENT)
            path.append(name.value)
            self._maybe_subscript(len(path) - 1, subs)
        return Reference(tuple(path), tuple(subs))

    def _maybe_subscript(self, elem_idx: int, subs: List[Tuple[int, int]]):
        if self._peek() and self._peek().kind == LBRACK:
            self._next()
            num = self._expect(NUM)
            self._expect(RBRACK)
            subs.append((elem_idx, int(float(num.value))))


# Common FreeCAD length/angle units recognised after a number literal. Values
# are the multiplier to FreeCAD's base unit (mm for length, degree for angle).
_UNITS: Dict[str, float] = {
    "mm": 1.0, "cm": 10.0, "m": 1000.0, "in": 25.4, "ft": 304.8,
    "deg": 1.0, "rad": 57.29577951308232,
}


def parse(text: str) -> "Expression":
    """Parse a FreeCAD expression string into an :class:`Expression`."""
    tokens = tokenize(text)
    if not tokens:
        raise ExpressionError("empty expression")
    ast = _Parser(tokens).parse()
    return Expression(text, ast)


# ── evaluation ──────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class Expression:
    source: str
    ast: Node

    def references(self) -> List[Reference]:
        """Every property reference the expression depends on (dedup, ordered)."""
        out: List[Reference] = []
        seen = set()

        def walk(node: Node):
            if isinstance(node, Reference):
                k = node.key()
                if k not in seen:
                    seen.add(k)
                    out.append(node)
            elif isinstance(node, UnaryOp):
                walk(node.operand)
            elif isinstance(node, BinOp):
                walk(node.left)
                walk(node.right)
            elif isinstance(node, Call):
                for a in node.args:
                    walk(a)

        walk(self.ast)
        return out

    def reference_keys(self) -> List[str]:
        return [r.key() for r in self.references()]

    def evaluate(self, env: Optional[Dict[str, float]] = None) -> float:
        """Evaluate the expression against ``env`` (reference-key -> number).

        Units on number literals convert to FreeCAD base units (mm / degree).
        """
        env = env or {}
        return _eval(self.ast, env)


def _eval(node: Node, env: Dict[str, float]) -> float:
    if isinstance(node, Number):
        if node.unit is not None:
            return node.value * _UNITS[node.unit]
        return node.value
    if isinstance(node, Reference):
        key = node.key()
        if key not in env:
            raise ExpressionError("unresolved reference %r" % key)
        return float(env[key])
    if isinstance(node, UnaryOp):
        val = _eval(node.operand, env)
        return -val if node.op == "-" else +val
    if isinstance(node, BinOp):
        left = _eval(node.left, env)
        right = _eval(node.right, env)
        return _apply(node.op, left, right)
    if isinstance(node, Call):
        fn = FUNCTIONS.get(node.name)
        if fn is None:
            raise ExpressionError("unknown function %r" % node.name)
        args = [_eval(a, env) for a in node.args]
        try:
            return float(fn(*args))
        except TypeError as exc:
            raise ExpressionError(
                "bad arity for %r: %s" % (node.name, exc))
    raise ExpressionError("cannot evaluate node %r" % (node,))


def _apply(op: str, a: float, b: float) -> float:
    if op == "+":
        return a + b
    if op == "-":
        return a - b
    if op == "*":
        return a * b
    if op == "/":
        if b == 0:
            raise ExpressionError("division by zero")
        return a / b
    if op == "%":
        if b == 0:
            raise ExpressionError("modulo by zero")
        return math.fmod(a, b)
    if op == "^":
        return math.pow(a, b)
    raise ExpressionError("unknown operator %r" % op)
