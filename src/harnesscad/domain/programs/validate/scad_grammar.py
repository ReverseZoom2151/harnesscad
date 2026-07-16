"""OpenSCAD grammar validator -- an FSA / structural checker for .scad text.

The grammar enforced here is transliterated from the OpenSCAD language as
documented in RapCAD's ``doc/openscad.bnf`` (RapCAD is GPL-3). The BNF file
was read as a *reference for facts about a public language*: the grammar rules
of OpenSCAD are properties of the language itself, not creative expression of
RapCAD. The rules have been reimplemented from scratch in this module's own
structure (token tables, an explicit delimiter FSA, and a recursive-descent
checker); no file content was copied. The underlying standard is the OpenSCAD
language reference (https://openscad.org/documentation.html).

Coverage (statement level):

  * ``use <path>`` / ``include <path>`` headers;
  * assignments ``name = expr;`` (including ``$special`` variables);
  * ``module name(params) statement`` and ``function name(params) = expr;``;
  * module instantiations with the ``!``, ``#``, ``%``, ``*`` modifiers,
    argument lists, and child statements (``;``, blocks, nested chains);
  * control flow: ``if``/``else``, ``for``, ``intersection_for``, ``let``,
    and ``children()`` as an ordinary instantiation;
  * blocks ``{ ... }`` with balanced-delimiter positions.

Coverage (expression level, by precedence):

  * ternary ``a ? b : c`` (right associative);
  * ``||``, ``&&``, comparisons, ``+ -``, ``* / %``, unary ``! + -``;
  * postfix indexing ``a[i]``, member access ``a.x``, calls ``f(...)``;
  * vectors ``[a, b, c]``, ranges ``[a:b]`` / ``[a:b:c]`` (a fourth ``:`` is
    rejected as a malformed range), list comprehensions with ``for`` /
    ``if`` / ``else`` / ``let`` / ``each``, and function literals
    ``function (params) expr``.

Complementarity with :mod:`harnesscad.domain.programs.validate.openscad_check`:
that module is a *semantic* gate built on a full AST (unknown builtins, bad
arguments, undefined variables, degenerate booleans). This module is the layer
below it: a pure *syntactic* gate that needs no AST and answers "is this even
shaped like OpenSCAD?" with positioned diagnostics. Run this first; run
``openscad_check`` on programs that pass. The FSA table style, state naming,
and diagnostic conventions mirror :mod:`harnesscad.core.grammar_fsa`.

Deterministic: no clock, no randomness, nothing executed; identical input
yields identical diagnostics in identical order.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from enum import Enum
from typing import List, Optional, Sequence, Tuple

__all__ = [
    "Rule",
    "Token",
    "Diagnostic",
    "Result",
    "KEYWORDS",
    "MODIFIERS",
    "tokenize",
    "check_delimiters",
    "validate",
    "main",
]


# --------------------------------------------------------------------------
# rules / states (diagnostics carry the rule that was active when they fired)
# --------------------------------------------------------------------------

class Rule(str, Enum):
    TOKEN = "token"
    DELIM = "delimiters"
    INPUT = "input"
    STATEMENT = "statement"
    ASSIGNMENT = "assignment"
    MODULE_DEF = "module_definition"
    FUNCTION_DEF = "function_definition"
    MODULE_INST = "module_instantiation"
    MODIFIER = "modifier"
    CHILD = "child_statement"
    ARGS_CALL = "arguments_call"
    ARGS_DECL = "arguments_decl"
    EXPR = "expr"
    RANGE = "range"
    VECTOR = "vector"
    COMPREHENSION = "list_comprehension"
    DEPTH = "nesting_depth"


# --------------------------------------------------------------------------
# tokens
# --------------------------------------------------------------------------

KEYWORDS = frozenset({
    "module", "function", "if", "else", "for", "let", "each",
    "true", "false", "undef", "use", "include", "intersection_for",
})

# module instantiation prefix modifiers: root / debug / background / disable
MODIFIERS = frozenset({"!", "#", "%", "*"})

_TWO_CHAR_OPS = ("<=", ">=", "==", "!=", "&&", "||")
_ONE_CHAR_OPS = frozenset("+-*/%<>!?:;,=.(){}[]#^")

_OPEN = {"(": ")", "[": "]", "{": "}"}
_CLOSE = {v: k for k, v in _OPEN.items()}

_MAX_DEPTH = 200


@dataclass(frozen=True)
class Token:
    kind: str        # "ident" | "number" | "string" | "keyword" |
                     # "special" | "op" | "path" | "eof"
    value: str
    line: int
    col: int


@dataclass(frozen=True)
class Diagnostic:
    line: int
    col: int
    rule: str        # Rule value active when the diagnostic fired
    expected: str
    found: str

    def __post_init__(self) -> None:
        # accept a Rule enum member but store its plain string value so that
        # render() and equality behave the same on every Python version
        object.__setattr__(self, "rule",
                           str(getattr(self.rule, "value", self.rule)))

    def render(self) -> str:
        return "line %d col %d [%s]: expected %s, found %s" % (
            self.line, self.col, self.rule, self.expected, self.found)


@dataclass(frozen=True)
class Result:
    ok: bool
    diagnostics: Tuple[Diagnostic, ...] = ()

    def render(self) -> str:
        if self.ok:
            return "OK"
        return "\n".join(d.render() for d in self.diagnostics)


class _LexError(Exception):
    def __init__(self, diag: Diagnostic) -> None:
        super().__init__(diag.render())
        self.diag = diag


def tokenize(source: str) -> List[Token]:
    """Tokenize OpenSCAD source. Raises :class:`_LexError` on lexical faults."""
    toks: List[Token] = []
    i, line, col = 0, 1, 1
    n = len(source)

    def advance(count: int) -> None:
        nonlocal i, line, col
        for _ in range(count):
            if i < n and source[i] == "\n":
                line += 1
                col = 1
            else:
                col += 1
            i += 1

    while i < n:
        ch = source[i]
        # whitespace
        if ch in " \t\r\n":
            advance(1)
            continue
        # line comment
        if source.startswith("//", i):
            while i < n and source[i] != "\n":
                advance(1)
            continue
        # block comment
        if source.startswith("/*", i):
            start_line, start_col = line, col
            advance(2)
            while i < n and not source.startswith("*/", i):
                advance(1)
            if i >= n:
                raise _LexError(Diagnostic(start_line, start_col, Rule.TOKEN,
                                           "closing '*/'", "end of input"))
            advance(2)
            continue
        # use/include path: <...> immediately after those keywords
        if ch == "<" and toks and toks[-1].kind == "keyword" \
                and toks[-1].value in ("use", "include"):
            start_line, start_col = line, col
            advance(1)
            j = i
            while j < n and source[j] not in ">\n":
                j += 1
            if j >= n or source[j] != ">":
                raise _LexError(Diagnostic(start_line, start_col, Rule.TOKEN,
                                           "closing '>' in file path",
                                           "end of line"))
            path = source[i:j]
            advance(j - i + 1)
            toks.append(Token("path", path, start_line, start_col))
            continue
        # string
        if ch == '"':
            start_line, start_col = line, col
            advance(1)
            buf: List[str] = []
            while i < n and source[i] != '"':
                if source[i] == "\\":
                    if i + 1 >= n:
                        break
                    buf.append(source[i:i + 2])
                    advance(2)
                else:
                    buf.append(source[i])
                    advance(1)
            if i >= n:
                raise _LexError(Diagnostic(start_line, start_col, Rule.TOKEN,
                                           "closing '\"'", "end of input"))
            advance(1)
            toks.append(Token("string", "".join(buf), start_line, start_col))
            continue
        # number: 12, 12.5, .5, 1e-3
        if ch.isdigit() or (ch == "." and i + 1 < n and source[i + 1].isdigit()):
            start_line, start_col = line, col
            j = i
            while j < n and source[j].isdigit():
                j += 1
            if j < n and source[j] == ".":
                j += 1
                while j < n and source[j].isdigit():
                    j += 1
            if j < n and source[j] in "eE":
                k = j + 1
                if k < n and source[k] in "+-":
                    k += 1
                if k < n and source[k].isdigit():
                    j = k
                    while j < n and source[j].isdigit():
                        j += 1
            text = source[i:j]
            advance(j - i)
            toks.append(Token("number", text, start_line, start_col))
            continue
        # $special variable
        if ch == "$":
            start_line, start_col = line, col
            j = i + 1
            while j < n and (source[j].isalnum() or source[j] == "_"):
                j += 1
            if j == i + 1:
                raise _LexError(Diagnostic(start_line, start_col, Rule.TOKEN,
                                           "identifier after '$'",
                                           repr(source[j]) if j < n
                                           else "end of input"))
            text = source[i:j]
            advance(j - i)
            toks.append(Token("special", text, start_line, start_col))
            continue
        # identifier / keyword
        if ch.isalpha() or ch == "_":
            start_line, start_col = line, col
            j = i
            while j < n and (source[j].isalnum() or source[j] == "_"):
                j += 1
            text = source[i:j]
            advance(j - i)
            kind = "keyword" if text in KEYWORDS else "ident"
            toks.append(Token(kind, text, start_line, start_col))
            continue
        # operators (longest match first)
        matched = False
        for op in _TWO_CHAR_OPS:
            if source.startswith(op, i):
                toks.append(Token("op", op, line, col))
                advance(2)
                matched = True
                break
        if matched:
            continue
        if ch in _ONE_CHAR_OPS:
            toks.append(Token("op", ch, line, col))
            advance(1)
            continue
        raise _LexError(Diagnostic(line, col, Rule.TOKEN,
                                   "a valid OpenSCAD token", repr(ch)))

    toks.append(Token("eof", "", line, col))
    return toks


# --------------------------------------------------------------------------
# delimiter FSA: a stack machine over ( ) [ ] { } with positions
# --------------------------------------------------------------------------

def check_delimiters(tokens: Sequence[Token]) -> List[Diagnostic]:
    """Balanced-delimiter pass. Reports every unmatched open/close position."""
    diags: List[Diagnostic] = []
    stack: List[Token] = []
    for tok in tokens:
        if tok.kind != "op":
            continue
        if tok.value in _OPEN:
            stack.append(tok)
        elif tok.value in _CLOSE:
            if not stack:
                diags.append(Diagnostic(tok.line, tok.col, Rule.DELIM,
                                        "no unmatched delimiters",
                                        "unmatched %r" % tok.value))
            elif _OPEN[stack[-1].value] != tok.value:
                opener = stack.pop()
                diags.append(Diagnostic(
                    tok.line, tok.col, Rule.DELIM,
                    "%r closing %r opened at line %d col %d"
                    % (_OPEN[opener.value], opener.value,
                       opener.line, opener.col),
                    repr(tok.value)))
            else:
                stack.pop()
    for opener in stack:
        diags.append(Diagnostic(opener.line, opener.col, Rule.DELIM,
                                "%r to close %r" % (_OPEN[opener.value],
                                                    opener.value),
                                "end of input"))
    return diags


# --------------------------------------------------------------------------
# recursive-descent grammar checker
# --------------------------------------------------------------------------

class _ParseError(Exception):
    def __init__(self, diag: Diagnostic) -> None:
        super().__init__(diag.render())
        self.diag = diag


class _Parser:
    def __init__(self, tokens: Sequence[Token]) -> None:
        self.toks = list(tokens)
        self.pos = 0
        self.diags: List[Diagnostic] = []
        self.depth = 0

    # -- primitives --
    def peek(self, offset: int = 0) -> Token:
        idx = min(self.pos + offset, len(self.toks) - 1)
        return self.toks[idx]

    def next(self) -> Token:
        tok = self.toks[self.pos]
        if tok.kind != "eof":
            self.pos += 1
        return tok

    def at_op(self, *values: str) -> bool:
        tok = self.peek()
        return tok.kind == "op" and tok.value in values

    def at_kw(self, *values: str) -> bool:
        tok = self.peek()
        return tok.kind == "keyword" and tok.value in values

    def fail(self, rule: Rule, expected: str, tok: Optional[Token] = None):
        tok = tok or self.peek()
        found = "end of input" if tok.kind == "eof" else repr(tok.value)
        raise _ParseError(Diagnostic(tok.line, tok.col, rule, expected, found))

    def expect_op(self, value: str, rule: Rule) -> Token:
        if not self.at_op(value):
            self.fail(rule, repr(value))
        return self.next()

    def enter(self, rule: Rule) -> None:
        self.depth += 1
        if self.depth > _MAX_DEPTH:
            self.fail(Rule.DEPTH, "nesting shallower than %d" % _MAX_DEPTH)

    def leave(self) -> None:
        self.depth -= 1

    def sync(self) -> None:
        """Skip to the next statement boundary after an error."""
        while self.peek().kind != "eof":
            tok = self.next()
            if tok.kind == "op" and tok.value in (";", "}"):
                return

    # -- input ::= (use | include | statement)* --
    def parse_input(self) -> None:
        while self.peek().kind != "eof":
            start = self.pos
            try:
                self.parse_statement()
            except _ParseError as exc:
                self.diags.append(exc.diag)
                self.depth = 0
                if self.pos == start:
                    self.next()
                self.sync()

    # -- statement --
    def parse_statement(self) -> None:
        self.enter(Rule.STATEMENT)
        try:
            tok = self.peek()
            if self.at_op(";"):
                self.next()
            elif self.at_op("{"):
                self.next()
                while not self.at_op("}") and self.peek().kind != "eof":
                    self.parse_statement()
                self.expect_op("}", Rule.STATEMENT)
            elif self.at_kw("use", "include"):
                self.next()
                if self.peek().kind != "path":
                    self.fail(Rule.STATEMENT, "'<file>' path after %r"
                              % tok.value)
                self.next()
                if self.at_op(";"):        # tolerated, not required
                    self.next()
            elif self.at_kw("module"):
                self.parse_module_def()
            elif self.at_kw("function"):
                self.parse_function_def()
            elif tok.kind in ("ident", "special") \
                    and self.peek(1).kind == "op" and self.peek(1).value == "=" \
                    and not (self.peek(2).kind == "op"
                             and self.peek(2).value == "="):
                self.parse_assignment()
            elif tok.kind == "ident" or self.at_kw("if", "for", "let",
                                                   "intersection_for") \
                    or (tok.kind == "op" and tok.value in MODIFIERS):
                self.parse_module_instantiation()
            else:
                self.fail(Rule.STATEMENT,
                          "a statement (';', '{', assignment, module/function "
                          "definition, or module instantiation)")
        finally:
            self.leave()

    # -- assignment ::= identifier '=' expr ';' --
    def parse_assignment(self) -> None:
        self.next()                        # ident / $special
        self.expect_op("=", Rule.ASSIGNMENT)
        self.parse_expr()
        self.expect_op(";", Rule.ASSIGNMENT)

    # -- "module" identifier '(' arguments_decl ')' statement --
    def parse_module_def(self) -> None:
        self.next()                        # module
        if self.peek().kind != "ident":
            self.fail(Rule.MODULE_DEF, "module name")
        self.next()
        self.expect_op("(", Rule.MODULE_DEF)
        self.parse_arguments_decl()
        self.expect_op(")", Rule.MODULE_DEF)
        self.parse_statement()

    # -- "function" identifier '(' arguments_decl ')' '=' expr ';' --
    def parse_function_def(self) -> None:
        self.next()                        # function
        if self.peek().kind != "ident":
            self.fail(Rule.FUNCTION_DEF, "function name")
        self.next()
        self.expect_op("(", Rule.FUNCTION_DEF)
        self.parse_arguments_decl()
        self.expect_op(")", Rule.FUNCTION_DEF)
        self.expect_op("=", Rule.FUNCTION_DEF)
        self.parse_expr()
        self.expect_op(";", Rule.FUNCTION_DEF)

    # -- module_instantiation ::= modifier* (single_inst child | if/for/let) --
    def parse_module_instantiation(self) -> None:
        self.enter(Rule.MODULE_INST)
        try:
            while self.peek().kind == "op" and self.peek().value in MODIFIERS:
                mod = self.next()
                tok = self.peek()
                starts_inst = tok.kind == "ident" \
                    or self.at_kw("if", "for", "let", "intersection_for") \
                    or (tok.kind == "op" and tok.value in MODIFIERS)
                if not starts_inst:
                    self.fail(Rule.MODIFIER,
                              "a module instantiation after modifier %r"
                              % mod.value)
            if self.at_kw("if"):
                self.parse_if()
                return
            if self.at_kw("for", "intersection_for", "let"):
                self.next()
                self.expect_op("(", Rule.MODULE_INST)
                self.parse_arguments_call()
                self.expect_op(")", Rule.MODULE_INST)
                self.parse_child_statement()
                return
            if self.peek().kind != "ident":
                self.fail(Rule.MODULE_INST, "module name")
            self.next()
            self.expect_op("(", Rule.MODULE_INST)
            self.parse_arguments_call()
            self.expect_op(")", Rule.MODULE_INST)
            self.parse_child_statement()
        finally:
            self.leave()

    # -- ifelse_statement --
    def parse_if(self) -> None:
        self.next()                        # if
        self.expect_op("(", Rule.MODULE_INST)
        self.parse_expr()
        self.expect_op(")", Rule.MODULE_INST)
        self.parse_child_statement()
        if self.at_kw("else"):
            self.next()
            if self.at_kw("if"):
                self.parse_if()
            else:
                self.parse_child_statement()

    # -- child_statement ::= ';' | '{' child_statements '}' | instantiation --
    def parse_child_statement(self) -> None:
        self.enter(Rule.CHILD)
        try:
            if self.at_op(";"):
                self.next()
            elif self.at_op("{"):
                self.next()
                while not self.at_op("}") and self.peek().kind != "eof":
                    self.parse_statement()   # child blocks allow assignments
                self.expect_op("}", Rule.CHILD)
            else:
                tok = self.peek()
                starts_inst = tok.kind == "ident" \
                    or self.at_kw("if", "for", "let", "intersection_for") \
                    or (tok.kind == "op" and tok.value in MODIFIERS)
                if not starts_inst:
                    self.fail(Rule.CHILD,
                              "';', '{', or a module instantiation")
                self.parse_module_instantiation()
        finally:
            self.leave()

    # -- arguments_decl ::= [param (',' param)*] with tolerated extra commas --
    def parse_arguments_decl(self) -> None:
        while True:
            while self.at_op(","):         # <optional_commas>
                self.next()
            if self.at_op(")") or self.peek().kind == "eof":
                return
            if self.peek().kind not in ("ident", "special"):
                self.fail(Rule.ARGS_DECL, "parameter name")
            self.next()
            if self.at_op("="):
                self.next()
                self.parse_expr()
            if self.at_op(","):
                continue
            return

    # -- arguments_call ::= [arg (',' arg)*] where arg is expr | name=expr --
    def parse_arguments_call(self) -> None:
        while True:
            while self.at_op(","):         # <optional_commas>
                self.next()
            if self.at_op(")") or self.peek().kind == "eof":
                return
            if self.peek().kind in ("ident", "special") \
                    and self.peek(1).kind == "op" \
                    and self.peek(1).value == "=" \
                    and not (self.peek(2).kind == "op"
                             and self.peek(2).value == "="):
                self.next()
                self.next()
                self.parse_expr()
            else:
                self.parse_expr()
            if self.at_op(","):
                continue
            return

    # -- expression grammar, precedence low -> high --
    def parse_expr(self) -> None:
        self.enter(Rule.EXPR)
        try:
            self.parse_ternary()
        finally:
            self.leave()

    def parse_ternary(self) -> None:
        self.parse_or()
        if self.at_op("?"):
            self.next()
            self.parse_ternary()
            self.expect_op(":", Rule.EXPR)
            self.parse_ternary()           # right associative

    def parse_or(self) -> None:
        self.parse_and()
        while self.at_op("||"):
            self.next()
            self.parse_and()

    def parse_and(self) -> None:
        self.parse_comparison()
        while self.at_op("&&"):
            self.next()
            self.parse_comparison()

    def parse_comparison(self) -> None:
        self.parse_additive()
        while self.at_op("==", "!=", "<", "<=", ">", ">="):
            self.next()
            self.parse_additive()

    def parse_additive(self) -> None:
        self.parse_multiplicative()
        while self.at_op("+", "-"):
            self.next()
            self.parse_multiplicative()

    def parse_multiplicative(self) -> None:
        self.parse_unary()
        while self.at_op("*", "/", "%"):
            self.next()
            self.parse_unary()

    def parse_unary(self) -> None:
        self.enter(Rule.EXPR)
        try:
            if self.at_op("!", "+", "-"):
                self.next()
                self.parse_unary()
            else:
                self.parse_postfix()
        finally:
            self.leave()

    def parse_postfix(self) -> None:
        self.parse_primary()
        while True:
            if self.at_op("["):
                self.next()
                self.parse_expr()
                self.expect_op("]", Rule.EXPR)
            elif self.at_op("."):
                self.next()
                if self.peek().kind != "ident":
                    self.fail(Rule.EXPR, "member name after '.'")
                self.next()
            elif self.at_op("("):
                self.next()
                self.parse_arguments_call()
                self.expect_op(")", Rule.EXPR)
            else:
                return

    def parse_primary(self) -> None:
        tok = self.peek()
        if tok.kind in ("number", "string", "ident", "special"):
            self.next()
            return
        if self.at_kw("true", "false", "undef"):
            self.next()
            return
        if self.at_kw("let"):              # "let" '(' args ')' expr
            self.next()
            self.expect_op("(", Rule.EXPR)
            self.parse_arguments_call()
            self.expect_op(")", Rule.EXPR)
            self.parse_expr()
            return
        if self.at_kw("function"):         # function literal (2021 language)
            self.next()
            self.expect_op("(", Rule.EXPR)
            self.parse_arguments_decl()
            self.expect_op(")", Rule.EXPR)
            self.parse_expr()
            return
        if self.at_op("("):
            self.next()
            self.parse_expr()
            self.expect_op(")", Rule.EXPR)
            return
        if self.at_op("["):
            self.parse_bracket()
            return
        self.fail(Rule.EXPR, "an expression")

    # -- '[' ... ']' : empty vector, range, vector, or comprehension --
    def parse_bracket(self) -> None:
        self.enter(Rule.VECTOR)
        try:
            self.next()                    # '['
            if self.at_op("]"):
                self.next()
                return
            while self.at_op(","):         # leading <optional_commas>
                self.next()
                if self.at_op("]"):
                    self.next()
                    return
            if self.at_kw("for", "let", "if", "each"):
                self.parse_comprehension_elements()
            else:
                self.parse_expr()
                if self.at_op(":"):
                    # range: '[' expr ':' expr [':' expr] ']'
                    self.next()
                    self.parse_expr()
                    if self.at_op(":"):
                        self.next()
                        self.parse_expr()
                    if self.at_op(":"):
                        self.fail(Rule.RANGE,
                                  "']' after at most three range parts "
                                  "([start:end] or [start:step:end])")
                    self.expect_op("]", Rule.RANGE)
                    return
            # vector: (',' <optional_commas> element)* ']'
            while self.at_op(","):
                self.next()
                while self.at_op(","):
                    self.next()
                if self.at_op("]"):
                    break
                if self.at_kw("for", "let", "if", "each"):
                    self.parse_comprehension_elements()
                else:
                    self.parse_expr()
            self.expect_op("]", Rule.VECTOR)
        finally:
            self.leave()

    # -- list_comprehension_elements --
    def parse_comprehension_elements(self) -> None:
        self.enter(Rule.COMPREHENSION)
        try:
            if self.at_kw("let"):
                self.next()
                self.expect_op("(", Rule.COMPREHENSION)
                self.parse_arguments_call()
                self.expect_op(")", Rule.COMPREHENSION)
                self.parse_comprehension_body()
                return
            if self.at_kw("for"):
                self.next()
                self.expect_op("(", Rule.COMPREHENSION)
                self.parse_arguments_call()
                if self.at_op(";"):        # C-style: for (init; cond; update)
                    self.next()
                    self.parse_expr()
                    self.expect_op(";", Rule.COMPREHENSION)
                    self.parse_arguments_call()
                self.expect_op(")", Rule.COMPREHENSION)
                self.parse_comprehension_body()
                return
            if self.at_kw("if"):
                self.next()
                self.expect_op("(", Rule.COMPREHENSION)
                self.parse_expr()
                self.expect_op(")", Rule.COMPREHENSION)
                self.parse_comprehension_body()
                if self.at_kw("else"):
                    self.next()
                    self.parse_comprehension_body()
                return
            if self.at_kw("each"):
                self.next()
                self.parse_comprehension_body()
                return
            self.fail(Rule.COMPREHENSION,
                      "'for', 'if', 'let', or 'each'")
        finally:
            self.leave()

    def parse_comprehension_body(self) -> None:
        """list_comprehension_elements_or_expr, with '(' grouping allowed."""
        if self.at_kw("for", "let", "if", "each"):
            self.parse_comprehension_elements()
            return
        if self.at_op("("):
            # could be a parenthesized comprehension element or an expression;
            # look one token ahead to decide
            if self.peek(1).kind == "keyword" \
                    and self.peek(1).value in ("for", "let", "if", "each"):
                self.next()
                self.parse_comprehension_elements()
                self.expect_op(")", Rule.COMPREHENSION)
                return
        self.parse_expr()


# --------------------------------------------------------------------------
# public API
# --------------------------------------------------------------------------

def validate(source: str) -> Result:
    """Validate ``source`` against the OpenSCAD grammar.

    Returns a :class:`Result` whose ``ok`` is True when no diagnostic fired.
    Lexical faults short-circuit; unbalanced delimiters short-circuit (with
    every unmatched position reported); otherwise the recursive-descent pass
    reports each statement-level fault it can recover past. Deterministic.
    """
    try:
        tokens = tokenize(source)
    except _LexError as exc:
        return Result(False, (exc.diag,))
    delim = check_delimiters(tokens)
    if delim:
        return Result(False, tuple(delim))
    parser = _Parser(tokens)
    parser.parse_input()
    return Result(not parser.diags, tuple(parser.diags))


# --------------------------------------------------------------------------
# selfcheck / CLI
# --------------------------------------------------------------------------

_VALID_SNIPPETS: Tuple[Tuple[str, str], ...] = (
    ("primitive chain",
     "translate([10, 0, 0]) cube([1, 2, 3], center = true);"),
    ("module def with children()",
     "module ring(r = 5, n = 6) {\n"
     "    for (i = [0 : n - 1])\n"
     "        rotate([0, 0, i * 360 / n]) translate([r, 0, 0]) children();\n"
     "    children(0);\n"
     "}\n"
     "ring(r = 8) sphere(r = 1, $fn = 32);"),
    ("for loop over range",
     "for (i = [0 : 2 : 10]) translate([i, 0, 0]) cylinder(h = i + 1, r = 0.5);"),
    ("ternary and range assignment",
     "n = $preview ? 16 : 64;\n"
     "steps = [0 : 360 / n : 359];\n"
     "x = n > 8 ? -n : +n;"),
    ("function def, let, comprehension, each",
     "function sq(x) = x * x;\n"
     "vals = [ for (i = [0 : 4]) if (i % 2 == 0) sq(i) else -i ];\n"
     "flat = [ each vals, each [10, 11] ];\n"
     "y = let (a = 2, b = 3) a * b;"),
    ("modifiers, if/else, booleans",
     "if (true) { #cube(1); } else { %sphere(2); }\n"
     "!difference() { cube(4); *translate([1, 1, -1]) cylinder(h = 6, r = 1); }"),
    ("use/include, function literal, indexing, member",
     "use <lib/shapes.scad>\n"
     "include <MCAD/units.scad>\n"
     "f = function (x, y = 1) x + y;\n"
     "p = [[1, 2], [3, 4]];\n"
     "q = p[1][0] + p[0].x;"),
)

_INVALID_SNIPPETS: Tuple[Tuple[str, str, str], ...] = (
    # (label, source, rule expected among diagnostics)
    ("unbalanced brace",
     "module a() { cube(1);", Rule.DELIM),
    ("bad modifier target",
     "# = 5;", Rule.MODIFIER),
    ("malformed range",
     "a = [1 : 2 : 3 : 4];", Rule.RANGE),
    ("assignment missing semicolon",
     "x = 1 y = 2;", Rule.ASSIGNMENT),
    ("function def without body",
     "function f(x);", Rule.FUNCTION_DEF),
    ("instantiation without argument parens",
     "cube;", Rule.MODULE_INST),
    ("dangling ternary",
     "x = true ? 1;", Rule.EXPR),
)


def _selfcheck() -> int:
    failures = 0
    for label, src in _VALID_SNIPPETS:
        result = validate(src)
        status = "ok" if result.ok else "FAIL"
        print("valid   [%s] %s" % (status, label))
        if not result.ok:
            failures += 1
            for d in result.diagnostics:
                print("    " + d.render())
    for label, src, rule in _INVALID_SNIPPETS:
        result = validate(src)
        hit = any(d.rule == rule for d in result.diagnostics)
        rejected = not result.ok and hit
        status = "ok" if rejected else "FAIL"
        print("invalid [%s] %s" % (status, label))
        for d in result.diagnostics:
            print("    " + d.render())
        if not rejected:
            failures += 1
    if failures:
        print("selfcheck: %d failure(s)" % failures)
        return 1
    print("selfcheck: all %d valid and %d invalid snippets behaved as expected"
          % (len(_VALID_SNIPPETS), len(_INVALID_SNIPPETS)))
    return 0


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        prog="scad_grammar",
        description="OpenSCAD grammar validator (FSA + recursive descent).")
    parser.add_argument("path", nargs="?",
                        help="a .scad file to validate")
    parser.add_argument("--selfcheck", action="store_true",
                        help="run the built-in valid/invalid snippet suite")
    args = parser.parse_args(argv)
    if args.selfcheck:
        return _selfcheck()
    if not args.path:
        parser.print_usage()
        return 2
    with open(args.path, "r", encoding="utf-8") as handle:
        source = handle.read()
    result = validate(source)
    print(result.render())
    return 0 if result.ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
