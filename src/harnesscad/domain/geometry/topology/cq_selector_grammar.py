"""Grammar-faithful compiler from CadQuery selector strings to the object algebra.

This complements :mod:`geometry.cqcontrib_selector_dsl` (the harness's existing
string-selector parser) by matching the *real* CadQuery grammar in
``cadquery/selectors.py`` -- including the corners that module omitted or got
wrong.  It produces :class:`geometry.cq_selector_algebra.Selector` objects, so
the parsed result is a first-class composable selector.

Grammar (mirrors ``_makeGrammar`` + ``_makeExpressionGrammar``)::

    expr   := term  ('or'  term)*          # SumSelector
    diff   := ...                          # handled at 'exc'/'except' level
    term   := unot ('and' unot)*           # AndSelector
    unot   := 'not' unot | atom            # InverseSelector  (LOWEST precedence!)
    atom   := '(' expr ')'
            | ('>>'|'<<') dir index?       # CenterNthSelector / min-max
            | ('>'|'<')   dir index?       # DirectionMinMax / DirectionNth
            | ('|'|'#'|'+'|'-') dir        # Parallel/Perp/Direction(+/-)
            | '%' TYPE                      # TypeSelector
            | NAMEDVIEW                     # front/back/left/right/top/bottom
            | dir                           # bare direction == DirectionSelector

    dir    := X|Y|Z|XY|XZ|YZ | '(' f ',' f ',' f ')'

Precedence, from tightest to loosest, exactly as the reference's
``infix_notation`` operator list orders it: ``and`` > ``or`` >
``exc``/``except`` > ``not``.

See module-level ``GRAMMAR_FINDINGS`` for the concrete points where
``cqcontrib_selector_dsl`` diverges from this.
"""

from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

from harnesscad.domain.geometry.topology.cq_selector_algebra import (
    AndSelector,
    CenterNthSelector,
    DirectionMinMaxSelector,
    DirectionNthSelector,
    DirectionSelector,
    InverseSelector,
    ParallelDirSelector,
    PerpendicularDirSelector,
    Selector,
    SelectorError,
    SubtractSelector,
    SumSelector,
    TypeSelector,
)

__all__ = ["parse_selector", "tokenize", "GRAMMAR_FINDINGS"]

# Concrete divergences of geometry.cqcontrib_selector_dsl from the real grammar.
GRAMMAR_FINDINGS: Tuple[str, ...] = (
    "not-precedence: cqcontrib_selector_dsl parses 'not' in factor() (tightest "
    "binding), but CadQuery's infix_notation lists 'not' LAST, making it the "
    "loosest operator. So 'not >X and #XY' is (not(>X)) and #XY in the DSL but "
    "not(>X and #XY) in CadQuery -- a real semantic bug.",
    "center-nth '>>'/'<<': CadQuery has a distinct center_nth_op ('>>','<<' -> "
    "CenterNthSelector). The DSL tokenises '>'/'<' as single chars only, so "
    "'>>Z' raises instead of selecting.",
    "named views: front/back/left/right/top/bottom map to DirectionMinMax on a "
    "fixed axis in CadQuery. The DSL has no named views.",
    "bare direction: a lone 'X' or '(1,0,0)' is DirectionSelector(vec) in "
    "CadQuery. The DSL requires a leading operator and raises on a bare axis.",
    "compound axes XY/XZ/YZ: valid simple_dir tokens in CadQuery. The DSL only "
    "accepts single X/Y/Z (or a vector), so '>XY' raises.",
    "'except' spelling: CadQuery accepts both 'exc' and 'except' for set "
    "difference. The DSL accepts only 'exc'.",
    "phantom '+'/'-'/'*' binary ops: the DSL docstring advertises '+'/'-' as "
    "binary expr operators and '*' as a binary term operator; none exist in the "
    "CadQuery string grammar (there '+'/'-' are only unary direction prefixes).",
)

_AXES = {
    "X": (1.0, 0.0, 0.0),
    "Y": (0.0, 1.0, 0.0),
    "Z": (0.0, 0.0, 1.0),
    "XY": (1.0, 1.0, 0.0),
    "YZ": (0.0, 1.0, 1.0),
    "XZ": (1.0, 0.0, 1.0),
}

# named view -> (axis, directionMax) exactly as CadQuery's namedViews table
_NAMED_VIEWS = {
    "front": ((0.0, 0.0, 1.0), True),
    "back": ((0.0, 0.0, 1.0), False),
    "left": ((1.0, 0.0, 0.0), False),
    "right": ((1.0, 0.0, 0.0), True),
    "top": ((0.0, 1.0, 0.0), True),
    "bottom": ((0.0, 1.0, 0.0), False),
}

_PUNCT_MULTI = (">>", "<<")
_PUNCT = {">", "<", "|", "#", "%", "+", "-", "(", ")", ",", "[", "]"}


def tokenize(text: str) -> List[str]:
    """Split a selector string into tokens, recognising the ``>>``/``<<`` digraphs."""
    tokens: List[str] = []
    i, n = 0, len(text)
    while i < n:
        c = text[i]
        if c.isspace():
            i += 1
            continue
        two = text[i : i + 2]
        if two in _PUNCT_MULTI:
            tokens.append(two)
            i += 2
            continue
        if c in _PUNCT:
            tokens.append(c)
            i += 1
            continue
        j = i
        while j < n and not text[j].isspace() and text[j] not in _PUNCT:
            # stop a word before a '>>'/'<<' digraph too
            if text[j : j + 2] in _PUNCT_MULTI:
                break
            j += 1
        tokens.append(text[i:j])
        i = j
    return tokens


class _Parser:
    def __init__(self, tokens: Sequence[str]):
        self.toks = list(tokens)
        self.pos = 0

    def peek(self) -> Optional[str]:
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

    # expr := not_expr  (loosest level, the parse entry point)
    # 'not' is the LOOSEST operator (last in CadQuery's infix_notation list),
    # so it is handled outermost and only leads an expression or follows '('.
    def expr(self) -> Selector:
        if self._peek_kw("not"):
            self.next()
            return InverseSelector(self.expr())
        return self._exc()

    # exc := or (('exc'|'except') or)*
    def _exc(self) -> Selector:
        node = self._or()
        while self._peek_kw("exc") or self._peek_kw("except"):
            self.next()
            node = SubtractSelector(node, self._or())
        return node

    # or := and ('or' and)*
    def _or(self) -> Selector:
        node = self._and()
        while self._peek_kw("or"):
            self.next()
            node = SumSelector(node, self._and())
        return node

    # and := primary ('and' primary)*  -- 'and' operands are primaries (base
    # atoms or a parenthesised expression), never a bare 'not' (matches
    # CadQuery: 'A and not B' is a parse error; use 'A and (not B)').
    def _and(self) -> Selector:
        node = self._primary()
        while self._peek_kw("and"):
            self.next()
            node = AndSelector(node, self._primary())
        return node

    def _primary(self) -> Selector:
        tok = self.peek()
        if tok is None:
            raise SelectorError("unexpected end of selector")
        if tok == "(":
            self.next()
            node = self.expr()
            self.expect(")")
            return node
        return self.atom()

    def _peek_kw(self, kw: str) -> bool:
        tok = self.peek()
        return tok is not None and tok.lower() == kw

    def _dir(self) -> Tuple[float, float, float]:
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
                raise SelectorError("vector direction needs 3 components")
            return (comps[0], comps[1], comps[2])
        name = tok.upper()
        if name not in _AXES:
            raise SelectorError(f"unknown direction {tok!r}")
        return _AXES[name]

    def _index(self) -> Optional[int]:
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

    def atom(self) -> Selector:
        tok = self.peek()
        if tok is None:
            raise SelectorError("unexpected end of selector")

        low = tok.lower()
        if low in _NAMED_VIEWS:
            self.next()
            axis, dmax = _NAMED_VIEWS[low]
            return DirectionMinMaxSelector(axis, directionMax=dmax)

        if tok in (">>", "<<"):
            self.next()
            axis = self._dir()
            idx = self._index()
            dmax = tok == ">>"
            if idx is None:
                return CenterNthSelector(axis, -1, directionMax=dmax)
            return CenterNthSelector(axis, idx, directionMax=dmax)

        if tok in (">", "<"):
            self.next()
            axis = self._dir()
            idx = self._index()
            dmax = tok == ">"
            if idx is None:
                return DirectionMinMaxSelector(axis, directionMax=dmax)
            return DirectionNthSelector(axis, idx, directionMax=dmax)

        if tok == "|":
            self.next()
            return ParallelDirSelector(self._dir())
        if tok == "#":
            self.next()
            return PerpendicularDirSelector(self._dir())
        if tok == "+":
            self.next()
            return DirectionSelector(self._dir())
        if tok == "-":
            self.next()
            ax = self._dir()
            return DirectionSelector((-ax[0], -ax[1], -ax[2]))
        if tok == "%":
            self.next()
            return TypeSelector(self.next())

        # bare direction (only_dir) -> DirectionSelector
        axis = self._dir()
        return DirectionSelector(axis)


def parse_selector(text: str) -> Selector:
    """Compile a CadQuery selector string into a composable ``Selector`` object."""
    tokens = tokenize(text)
    if not tokens:
        raise SelectorError("empty selector")
    p = _Parser(tokens)
    node = p.expr()
    if p.peek() is not None:
        raise SelectorError(f"trailing tokens: {p.toks[p.pos:]}")
    return node
