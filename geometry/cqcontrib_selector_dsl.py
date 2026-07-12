"""Deterministic parser + evaluator for the CadQuery string-selector DSL.

The ``cadquery-contrib`` example corpus uses string selectors pervasively
(``>Z``, ``<Y``, ``|Z``, ``#Z``, ``%CIRCLE``, ``>Z[1]``,
``not(<X or >X or <Y or >Y)``).  Anything that consumes, generates, or
validates CadQuery programs needs to understand this mini-language; the
harness had no implementation of it.

This module is a stdlib-only, runtime-free implementation:

* :func:`parse` turns a selector string into an expression tree.
* :func:`evaluate` filters a list of :class:`Entity` (a face / edge / vertex
  abstracted to a centre point, an axis, and a geometry type).

Grammar (recursive descent, left-associative binary operators)::

    expr    := term (('or' | 'exc' | '+' | '-') term)*
    term    := factor (('and' | '*') factor)*
    factor  := 'not' factor | '(' expr ')' | atom
    atom    := ('>' | '<') AXIS index? | ('|' | '#' | '+' | '-') AXIS | '%' TYPE

``AXIS`` is ``X``/``Y``/``Z`` or a parenthesised vector ``(x,y,z)``.

Semantics (matching CadQuery):

* ``>Z`` / ``<Z``  -- max / min of the centre projected on the axis
  (``DirectionMinMaxSelector``); an optional ``[n]`` index groups distinct
  projections in ascending order and picks the n-th (negative allowed).
* ``|Z``  -- entity axis parallel to Z (either sense).
* ``#Z``  -- entity axis perpendicular to Z.
* ``+Z`` / ``-Z``  -- entity axis parallel to Z, same / opposite sense.
* ``%CIRCLE``  -- geometry type equals CIRCLE (case-insensitive).
* ``not``/``and``/``or``/``exc`` are set complement / intersection / union /
  difference over the input entity list; ``exc`` is CadQuery's subtraction.

All results preserve input order and are deterministic.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Sequence, Tuple

__all__ = [
    "Entity",
    "SelectorError",
    "Node",
    "DirMinMax",
    "Parallel",
    "Perpendicular",
    "Directional",
    "TypeSel",
    "Not",
    "And",
    "Or",
    "Exc",
    "tokenize",
    "parse",
    "evaluate",
    "select",
]

TOL = 1e-6


class SelectorError(ValueError):
    """Raised for malformed selector strings."""


@dataclass(frozen=True)
class Entity:
    """A face / edge / vertex abstracted for selection.

    ``center``: centre of mass.  ``axis``: face normal or edge tangent
    (``(0, 0, 0)`` for vertices / non-axial shapes).  ``geom_type``: e.g.
    ``"PLANE"``, ``"CIRCLE"``, ``"LINE"``.  ``name`` is an opaque label.
    """

    center: Tuple[float, float, float]
    axis: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    geom_type: str = ""
    name: str = ""


# --------------------------------------------------------------------------
# vector helpers
# --------------------------------------------------------------------------

def _dot(a: Sequence[float], b: Sequence[float]) -> float:
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


def _norm(a: Sequence[float]) -> float:
    return math.sqrt(_dot(a, a))


def _unit(a: Sequence[float]) -> Tuple[float, float, float]:
    n = _norm(a)
    if n <= TOL:
        raise SelectorError("zero-length axis vector")
    return (a[0] / n, a[1] / n, a[2] / n)


# --------------------------------------------------------------------------
# AST
# --------------------------------------------------------------------------

class Node:
    """Base class of selector expression nodes."""

    def filter(self, entities: Sequence[Entity]) -> List[Entity]:  # pragma: no cover
        raise NotImplementedError


@dataclass(frozen=True)
class DirMinMax(Node):
    axis: Tuple[float, float, float]
    maximize: bool
    index: int | None = None

    def filter(self, entities: Sequence[Entity]) -> List[Entity]:
        if not entities:
            return []
        u = _unit(self.axis)
        projs = [_dot(e.center, u) for e in entities]
        if self.index is None:
            target = max(projs) if self.maximize else min(projs)
        else:
            groups: List[float] = []
            for p in sorted(projs):
                if not groups or abs(p - groups[-1]) > TOL:
                    groups.append(p)
            if self.maximize:
                groups.reverse()
            i = self.index
            if i < 0:
                i += len(groups)
            if not (0 <= i < len(groups)):
                return []
            target = groups[i]
        return [e for e, p in zip(entities, projs) if abs(p - target) <= TOL]


@dataclass(frozen=True)
class Parallel(Node):
    axis: Tuple[float, float, float]

    def filter(self, entities: Sequence[Entity]) -> List[Entity]:
        u = _unit(self.axis)
        out = []
        for e in entities:
            if _norm(e.axis) <= TOL:
                continue
            if abs(abs(_dot(_unit(e.axis), u)) - 1.0) <= 1e-6:
                out.append(e)
        return out


@dataclass(frozen=True)
class Perpendicular(Node):
    axis: Tuple[float, float, float]

    def filter(self, entities: Sequence[Entity]) -> List[Entity]:
        u = _unit(self.axis)
        out = []
        for e in entities:
            if _norm(e.axis) <= TOL:
                continue
            if abs(_dot(_unit(e.axis), u)) <= 1e-6:
                out.append(e)
        return out


@dataclass(frozen=True)
class Directional(Node):
    axis: Tuple[float, float, float]
    positive: bool

    def filter(self, entities: Sequence[Entity]) -> List[Entity]:
        u = _unit(self.axis)
        want = 1.0 if self.positive else -1.0
        out = []
        for e in entities:
            if _norm(e.axis) <= TOL:
                continue
            if abs(_dot(_unit(e.axis), u) - want) <= 1e-6:
                out.append(e)
        return out


@dataclass(frozen=True)
class TypeSel(Node):
    geom_type: str

    def filter(self, entities: Sequence[Entity]) -> List[Entity]:
        want = self.geom_type.upper()
        return [e for e in entities if e.geom_type.upper() == want]


@dataclass(frozen=True)
class Not(Node):
    child: Node

    def filter(self, entities: Sequence[Entity]) -> List[Entity]:
        keep = {id(e) for e in self.child.filter(entities)}
        return [e for e in entities if id(e) not in keep]


@dataclass(frozen=True)
class And(Node):
    left: Node
    right: Node

    def filter(self, entities: Sequence[Entity]) -> List[Entity]:
        keep = {id(e) for e in self.right.filter(entities)}
        return [e for e in self.left.filter(entities) if id(e) in keep]


@dataclass(frozen=True)
class Or(Node):
    left: Node
    right: Node

    def filter(self, entities: Sequence[Entity]) -> List[Entity]:
        keep = {id(e) for e in self.left.filter(entities)}
        keep |= {id(e) for e in self.right.filter(entities)}
        return [e for e in entities if id(e) in keep]


@dataclass(frozen=True)
class Exc(Node):
    left: Node
    right: Node

    def filter(self, entities: Sequence[Entity]) -> List[Entity]:
        drop = {id(e) for e in self.right.filter(entities)}
        return [e for e in self.left.filter(entities) if id(e) not in drop]


# --------------------------------------------------------------------------
# tokenizer
# --------------------------------------------------------------------------

_PUNCT = {">", "<", "|", "#", "%", "+", "-", "(", ")", ",", "[", "]"}


def tokenize(text: str) -> List[str]:
    """Split a selector string into tokens (deterministic, whitespace-free)."""
    tokens: List[str] = []
    i = 0
    n = len(text)
    while i < n:
        c = text[i]
        if c.isspace():
            i += 1
            continue
        if c in _PUNCT:
            tokens.append(c)
            i += 1
            continue
        j = i
        while j < n and not text[j].isspace() and text[j] not in _PUNCT:
            j += 1
        tokens.append(text[i:j])
        i = j
    return tokens


class _Parser:
    def __init__(self, tokens: Sequence[str]) -> None:
        self.toks = list(tokens)
        self.pos = 0

    def peek(self) -> str | None:
        return self.toks[self.pos] if self.pos < len(self.toks) else None

    def next(self) -> str:
        tok = self.peek()
        if tok is None:
            raise SelectorError("unexpected end of selector")
        self.pos += 1
        return tok

    def expect(self, tok: str) -> None:
        got = self.next()
        if got != tok:
            raise SelectorError(f"expected {tok!r}, got {got!r}")

    # expr := term (('or'|'exc'|'+'|'-') term)*
    def expr(self) -> Node:
        node = self.term()
        while True:
            tok = self.peek()
            low = tok.lower() if tok else None
            if low == "or":
                self.next()
                node = Or(node, self.term())
            elif low == "exc":
                self.next()
                node = Exc(node, self.term())
            else:
                return node

    # term := factor (('and'|'*') factor)*
    def term(self) -> Node:
        node = self.factor()
        while True:
            tok = self.peek()
            if tok is not None and tok.lower() == "and":
                self.next()
                node = And(node, self.factor())
            else:
                return node

    def factor(self) -> Node:
        tok = self.peek()
        if tok is None:
            raise SelectorError("unexpected end of selector")
        if tok.lower() == "not":
            self.next()
            return Not(self.factor())
        if tok == "(":
            self.next()
            node = self.expr()
            self.expect(")")
            return node
        return self.atom()

    def _axis(self) -> Tuple[float, float, float]:
        tok = self.next()
        if tok == "(":
            comps: List[float] = []
            while True:
                t = self.next()
                if t == ",":
                    continue
                if t == ")":
                    break
                try:
                    comps.append(float(t))
                except ValueError as exc:
                    raise SelectorError(f"bad vector component {t!r}") from exc
            if len(comps) != 3:
                raise SelectorError("vector axis needs 3 components")
            return (comps[0], comps[1], comps[2])
        name = tok.upper()
        base = {"X": (1.0, 0.0, 0.0), "Y": (0.0, 1.0, 0.0), "Z": (0.0, 0.0, 1.0)}
        if name not in base:
            raise SelectorError(f"unknown axis {tok!r}")
        return base[name]

    def _index(self) -> int | None:
        if self.peek() != "[":
            return None
        self.next()
        sign = 1
        tok = self.next()
        if tok == "-":
            sign = -1
            tok = self.next()
        elif tok == "+":
            tok = self.next()
        try:
            value = int(tok)
        except ValueError as exc:
            raise SelectorError(f"bad selector index {tok!r}") from exc
        self.expect("]")
        return sign * value

    def atom(self) -> Node:
        tok = self.next()
        if tok in (">", "<"):
            axis = self._axis()
            return DirMinMax(axis, maximize=(tok == ">"), index=self._index())
        if tok == "|":
            return Parallel(self._axis())
        if tok == "#":
            return Perpendicular(self._axis())
        if tok in ("+", "-"):
            return Directional(self._axis(), positive=(tok == "+"))
        if tok == "%":
            return TypeSel(self.next())
        raise SelectorError(f"unexpected token {tok!r}")


def parse(text: str) -> Node:
    """Parse a CadQuery selector string into an expression tree."""
    tokens = tokenize(text)
    if not tokens:
        raise SelectorError("empty selector")
    p = _Parser(tokens)
    node = p.expr()
    if p.peek() is not None:
        raise SelectorError(f"trailing tokens: {p.toks[p.pos:]}")
    return node


def evaluate(node: Node, entities: Sequence[Entity]) -> List[Entity]:
    """Apply a parsed selector to an entity list, preserving input order."""
    return node.filter(list(entities))


def select(text: str, entities: Sequence[Entity]) -> List[Entity]:
    """Parse and apply a selector in one step."""
    return evaluate(parse(text), entities)
