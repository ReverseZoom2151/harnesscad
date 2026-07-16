"""KCL production-rule grammar and recursive-descent structural checker.

Reimplementation of the KCL *phrase* grammar from Zoo's ``modeling-app``
(MIT, (c) 2023 The Zoo Authors), transliterated from the Lezer grammar at
``packages/codemirror-lang-kcl/src/kcl.grammar``.  The grammar rules
themselves are facts about the KCL language; this module restates them as a
Python production table plus a hand-written recursive-descent checker -- it
copies no source text from the reference file.

Relationship to :mod:`harnesscad.domain.spec.kcl_grammar`
---------------------------------------------------------
That module is the *lexical* layer: a lossless tokeniser mirrored from the
Rust ``kcl-syntax`` crate, with token kinds, keywords and AST vocabulary.
This module is the *syntactic* layer on top of it:

*   :data:`PRODUCTIONS` -- every production rule from ``kcl.grammar``
    (statement kinds, the expression grammar, types, imports, annotations),
    written as ``rule -> tuple of alternatives`` in EBNF-ish notation.
*   :data:`PRECEDENCE` -- the ``@precedence`` block, tightest binding first,
    with each level's associativity.
*   :func:`check` -- a recursive-descent structural checker driven by
    :func:`kcl_grammar.lex`.  It validates statement forms, balanced
    delimiters and expression well-formedness (precedence climbing over the
    declared operator levels) and returns typed diagnostics.  It builds no
    AST and evaluates nothing.

Disambiguation notes (the Lezer grammar is GLR; this checker is LL with two
small lookahead decisions, both documented here):

*   ``sketch`` is only treated as the ``SketchBlockExpression`` keyword when
    followed by an argument list *and* a ``{`` body; otherwise it parses as a
    plain identifier.
*   ``identifier =`` at statement level is a ``VariableDeclaration``; inside
    an argument list it is a ``LabeledArgument``.  (``==`` lexes as one token
    so comparison never collides with these.)

Stdlib only, deterministic, no model or kernel calls.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple, Union

from harnesscad.domain.spec import kcl_grammar
from harnesscad.domain.spec.kcl_grammar import Token

__all__ = [
    "PRODUCTIONS",
    "PRECEDENCE",
    "STATEMENT_RULES",
    "EXPRESSION_RULES",
    "BINARY_LEVELS",
    "Diagnostic",
    "CheckResult",
    "check",
    "main",
]

# ---------------------------------------------------------------------------
# Production table (transliteration of kcl.grammar).
#
# Notation: rule name -> tuple of alternatives. ``?`` optional, ``*`` zero or
# more, ``+`` one or more, ``|`` separates inline alternatives, quoted strings
# are literal tokens, ``kw(x)`` is the identifier x specialised to a keyword.
# ``commaSep<t>`` allows an empty list and a trailing comma;
# ``commaSep1NoTrailingComma<t>`` requires one item and forbids a trailer.
# ---------------------------------------------------------------------------

PRODUCTIONS: Dict[str, Tuple[str, ...]] = {
    "Program": ("Shebang? statement*",),
    "statement": (
        "ImportStatement",
        "FunctionDeclaration",
        "VariableDeclaration",
        "TypeDeclaration",
        "ReturnStatement",
        "ExpressionStatement",
        "Annotation",
    ),
    "ImportStatement": ("kw(import) ImportItems ImportFrom String",),
    "FunctionDeclaration": ("kw(export)? kw(fn) VariableDefinition ParamList Body",),
    "VariableDeclaration": ("kw(export)? VariableDefinition Equals expression",),
    "TypeDeclaration": ("kw(export)? kw(type) identifier (Equals type)?",),
    "ReturnStatement": ("kw(return) expression",),
    "ExpressionStatement": ("expression",),
    "Annotation": ("AnnotationName AnnotationList?",),
    "AnnotationList": ("'(' commaSep<AnnotationProperty> ')'",),
    "AnnotationProperty": (
        "PropertyName (AddOp | MultOp | ExpOp | LogicOp | BangOp | CompOp"
        " | Equals | PipeOperator | PipeSubstitution) expression",
    ),
    "ParamList": ("'(' commaSep<Parameter> ')'",),
    "Parameter": ("VariableDefinition '?'? (':' type)?",),
    "Body": ("'{' statement* '}'",),
    "ImportItems": ("commaSep1NoTrailingComma<ImportItem>",),
    "ImportItem": ("identifier (ImportItemAs identifier)?",),
    "expression": (
        "String",
        "Number",
        "SketchVar",
        "VariableName",
        "TagDeclarator",
        "kw(true)",
        "kw(false)",
        "kw(nil)",
        "PipeSubstitution",
        "BinaryExpression",
        "UnaryExpression",
        "ParenthesizedExpression",
        "IfExpression",
        "SketchBlockExpression",
        "CallExpression",
        "ArrayExpression",
        "ObjectExpression",
        "MemberExpression",
        "SubscriptExpression",
        "PipeExpression",
    ),
    "SketchVar": ("kw(var) AddOp? Number",),
    "BinaryExpression": (
        "expression !add AddOp expression",
        "expression !mult MultOp expression",
        "expression !exp ExpOp expression",
        "expression !comp CompOp expression",
        "expression !logic LogicOp expression",
    ),
    "UnaryExpression": ("UnaryOp expression",),
    "UnaryOp": ("AddOp", "BangOp"),
    "ParenthesizedExpression": ("'(' expression ')'",),
    "IfExpression": ("kw(if) expression Body kw(else) Body",),
    "SketchBlockExpression": ("kw(sketch) ArgumentList Body",),
    "CallExpression": ("expression !call ArgumentList",),
    "ArrayExpression": ("'[' commaSep<expression | IntegerRange> ']'",),
    "IntegerRange": ("expression !range '..' expression",),
    "ObjectExpression": ("'{' commaSep<ObjectProperty> '}'",),
    "ObjectProperty": ("PropertyName (':' | Equals) expression",),
    "MemberExpression": ("expression !member '.' PropertyName",),
    "SubscriptExpression": ("expression !member '[' expression ']'",),
    "PipeExpression": ("expression (!pipe PipeOperator expression)+",),
    "LabeledArgument": ("ArgumentLabel Equals expression",),
    "ArgumentList": ("'(' commaSep<LabeledArgument | expression> ')'",),
    "type": ("PrimitiveType", "ArrayType", "ObjectType"),
    "PrimitiveType": ("identifier",),
    "ArrayType": ("'[' type !member (';' Number '+'?)? ']'",),
    "ObjectType": ("'{' commaSep<TypeProperty> '}'",),
    "TypeProperty": ("PropertyName ':' type",),
    "VariableDefinition": ("identifier",),
    "VariableName": ("identifier ('::' identifier)*",),
    "ArgumentLabel": ("identifier",),
    # Terminal spellings carried by kcl_grammar's lexer (informational).
    "AddOp": ("'+'", "'-'"),
    "MultOp": ("'/'", "'*'", "'\\\\'"),
    "ExpOp": ("'^'",),
    "LogicOp": ("'|'", "'&'"),
    "BangOp": ("'!'",),
    "CompOp": ("'=='", "'!='", "'<='", "'>='", "'<'", "'>'"),
    "Equals": ("'='",),
    "PipeOperator": ("'|>'",),
    "PipeSubstitution": ("'%'",),
    "TagDeclarator": ("'$' identifier",),
    "AnnotationName": ("'@' identifier?",),
}

#: The @precedence block, tightest binding first, as (name, associativity).
#: ``None`` associativity means the level is a marker (postfix/prefix/range),
#: not a left/right infix declaration.
PRECEDENCE: Tuple[Tuple[str, Optional[str]], ...] = (
    ("annotation", None),
    ("member", None),
    ("call", None),
    ("exp", "left"),
    ("mult", "left"),
    ("add", "left"),
    ("comp", "left"),
    ("logic", "left"),
    ("pipe", "left"),
    ("range", None),
)

#: Statement-level rule names (the @isGroup=Statement group).
STATEMENT_RULES: Tuple[str, ...] = PRODUCTIONS["statement"]

#: Expression-level rule names (the @isGroup=Expression group).
EXPRESSION_RULES: Tuple[str, ...] = PRODUCTIONS["expression"]

# ---------------------------------------------------------------------------
# Token-kind sets (kcl_grammar SyntaxKind names) per operator class.
# ---------------------------------------------------------------------------

_ADD_OPS = frozenset({"Plus", "Minus"})
_MULT_OPS = frozenset({"Star", "Slash", "Backslash"})
_EXP_OPS = frozenset({"Caret"})
_COMP_OPS = frozenset({"EqEq", "BangEq", "LtEq", "GtEq", "Lt", "Gt"})
_LOGIC_OPS = frozenset({"Pipe", "Amp"})
_PIPE_OPS = frozenset({"PipeGt"})
_UNARY_OPS = frozenset({"Plus", "Minus", "Bang"})

#: Infix levels for precedence climbing: binding power (higher binds tighter)
#: -> (level name, token-kind set). All levels are left-associative, exactly
#: as the @precedence block declares.
BINARY_LEVELS: Tuple[Tuple[int, str, frozenset], ...] = (
    (6, "exp", _EXP_OPS),
    (5, "mult", _MULT_OPS),
    (4, "add", _ADD_OPS),
    (3, "comp", _COMP_OPS),
    (2, "logic", _LOGIC_OPS),
    (1, "pipe", _PIPE_OPS),
)

_KIND_TO_LEVEL: Dict[str, Tuple[int, str]] = {}
for _bp, _name, _kinds in BINARY_LEVELS:
    for _k in _kinds:
        _KIND_TO_LEVEL[_k] = (_bp, _name)

#: Token kinds that may begin an expression (first-set of ``expression``).
_EXPR_FIRST = frozenset(
    {
        "String", "Number", "Word", "VarKw", "TrueKw", "FalseKw", "NilKw",
        "Percent", "Dollar", "OpenParen", "IfKw", "OpenBracket", "OpenBrace",
        "Plus", "Minus", "Bang",
    }
)

#: Token kinds that reliably begin a new statement (used for error recovery).
_STATEMENT_SYNC = frozenset(
    {"ImportKw", "FnKw", "ExportKw", "TypeKw", "ReturnKw", "At"}
)


# ---------------------------------------------------------------------------
# Diagnostics.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Diagnostic:
    """One structural problem. ``position`` is a UTF-8 byte offset."""

    rule: str        # production rule being matched when the error occurred
    position: int    # byte offset of the offending token (or end of input)
    expected: str    # human-readable description of what the rule needed
    found: str       # token kind actually seen, or "end of input"
    found_text: str  # source text of the offending token ("" at end of input)

    def __str__(self) -> str:
        got = self.found if not self.found_text else (
            "%s %r" % (self.found, self.found_text)
        )
        return "at byte %d in %s: expected %s, found %s" % (
            self.position, self.rule, self.expected, got,
        )


@dataclass(frozen=True)
class CheckResult:
    """Outcome of :func:`check`: ``ok`` iff no diagnostics were produced."""

    ok: bool
    diagnostics: Tuple[Diagnostic, ...] = field(default=())
    statement_count: int = 0


class _ParseError(Exception):
    """Internal control flow; carries the Diagnostic to the statement level."""

    def __init__(self, diagnostic: Diagnostic) -> None:
        super().__init__(str(diagnostic))
        self.diagnostic = diagnostic


# ---------------------------------------------------------------------------
# Recursive-descent checker.
# ---------------------------------------------------------------------------


class _Checker:
    def __init__(self, tokens: Sequence[Token], end_offset: int) -> None:
        self.tokens: List[Token] = list(tokens)
        self.pos = 0
        self.end_offset = end_offset
        self.diagnostics: List[Diagnostic] = []
        self.statement_count = 0

    # -- token plumbing -----------------------------------------------------

    def peek(self, offset: int = 0) -> Optional[Token]:
        i = self.pos + offset
        return self.tokens[i] if i < len(self.tokens) else None

    def at(self, kind: str, offset: int = 0) -> bool:
        tok = self.peek(offset)
        return tok is not None and tok.kind == kind

    def at_end(self) -> bool:
        return self.pos >= len(self.tokens)

    def advance(self) -> Token:
        tok = self.tokens[self.pos]
        self.pos += 1
        return tok

    def _error(self, rule: str, expected: str) -> _ParseError:
        tok = self.peek()
        if tok is None:
            return _ParseError(
                Diagnostic(rule, self.end_offset, expected, "end of input", "")
            )
        return _ParseError(
            Diagnostic(rule, tok.start, expected, tok.kind, tok.text)
        )

    def expect(self, kind: str, rule: str, expected: str) -> Token:
        if self.at(kind):
            return self.advance()
        raise self._error(rule, expected)

    # -- program ------------------------------------------------------------

    def check_program(self) -> None:
        while not self.at_end():
            start = self.pos
            try:
                self.parse_statement()
                self.statement_count += 1
            except _ParseError as exc:
                self.diagnostics.append(exc.diagnostic)
                self._synchronize(start)

    def _synchronize(self, error_start: int) -> None:
        """Skip to a plausible next statement so later errors still surface."""
        if self.pos == error_start:
            self.pos += 1
        while not self.at_end():
            tok = self.peek()
            if tok.kind in _STATEMENT_SYNC:
                return
            if tok.kind == "Word" and self.at("Eq", 1):
                return
            self.pos += 1

    # -- statements ---------------------------------------------------------

    def parse_statement(self) -> None:
        tok = self.peek()
        assert tok is not None
        kind = tok.kind
        if kind == "At":
            self.parse_annotation()
        elif kind == "ImportKw":
            self.parse_import_statement()
        elif kind == "ExportKw":
            self.advance()
            nxt = self.peek()
            if nxt is None:
                raise self._error(
                    "statement", "fn, type, or a variable declaration after export"
                )
            if nxt.kind == "FnKw":
                self.parse_function_declaration()
            elif nxt.kind == "TypeKw":
                self.parse_type_declaration()
            elif nxt.kind == "Word" and self.at("Eq", 1):
                self.parse_variable_declaration()
            else:
                raise self._error(
                    "statement", "fn, type, or a variable declaration after export"
                )
        elif kind == "FnKw":
            self.parse_function_declaration()
        elif kind == "TypeKw":
            self.parse_type_declaration()
        elif kind == "ReturnKw":
            self.advance()
            self.parse_expression("ReturnStatement")
        elif kind == "Word" and self.at("Eq", 1):
            self.parse_variable_declaration()
        elif kind in _EXPR_FIRST:
            self.parse_expression("ExpressionStatement")
        else:
            raise self._error("statement", "a statement or expression")

    def parse_annotation(self) -> None:
        self.expect("At", "Annotation", "'@'")
        # AnnotationName { "@" identifier? } -- name is optional.
        if self.at("Word"):
            self.advance()
        if self.at("OpenParen"):
            self.advance()
            self._comma_sep(
                "AnnotationList", "CloseParen", self._parse_annotation_property
            )
            self.expect("CloseParen", "AnnotationList", "')'")

    _ANNOTATION_OPS = frozenset(
        _ADD_OPS | _MULT_OPS | _EXP_OPS | _LOGIC_OPS | _COMP_OPS
        | {"Bang", "Eq", "PipeGt", "Percent"}
    )

    def _parse_annotation_property(self) -> None:
        self.expect("Word", "AnnotationProperty", "a property name")
        tok = self.peek()
        if tok is None or tok.kind not in self._ANNOTATION_OPS:
            raise self._error(
                "AnnotationProperty", "an operator or '=' after the property name"
            )
        self.advance()
        self.parse_expression("AnnotationProperty")

    def parse_import_statement(self) -> None:
        self.expect("ImportKw", "ImportStatement", "'import'")
        # ImportItems: one or more, no trailing comma.
        self._parse_import_item()
        while self.at("Comma"):
            self.advance()
            self._parse_import_item()
        tok = self.peek()
        if tok is None or not (tok.kind == "Word" and tok.text == "from"):
            raise self._error("ImportStatement", "'from'")
        self.advance()
        self.expect("String", "ImportStatement", "a string path after 'from'")

    def _parse_import_item(self) -> None:
        self.expect("Word", "ImportItem", "an identifier to import")
        if self.at("AsKw"):
            self.advance()
            self.expect("Word", "ImportItem", "an alias identifier after 'as'")

    def parse_function_declaration(self) -> None:
        self.expect("FnKw", "FunctionDeclaration", "'fn'")
        self.expect("Word", "FunctionDeclaration", "a function name")
        self.parse_param_list()
        self.parse_body("FunctionDeclaration")

    def parse_param_list(self) -> None:
        self.expect("OpenParen", "ParamList", "'('")
        self._comma_sep("ParamList", "CloseParen", self._parse_parameter)
        self.expect("CloseParen", "ParamList", "')'")

    def _parse_parameter(self) -> None:
        self.expect("Word", "Parameter", "a parameter name")
        if self.at("QuestionMark"):
            self.advance()
        if self.at("Colon"):
            self.advance()
            self.parse_type()

    def parse_type_declaration(self) -> None:
        self.expect("TypeKw", "TypeDeclaration", "'type'")
        self.expect("Word", "TypeDeclaration", "a type name")
        if self.at("Eq"):
            self.advance()
            self.parse_type()

    def parse_variable_declaration(self) -> None:
        self.expect("Word", "VariableDeclaration", "a variable name")
        self.expect("Eq", "VariableDeclaration", "'='")
        self.parse_expression("VariableDeclaration")

    def parse_body(self, rule: str) -> None:
        self.expect("OpenBrace", rule, "'{' to open the body")
        while not self.at_end() and not self.at("CloseBrace"):
            start = self.pos
            try:
                self.parse_statement()
                self.statement_count += 1
            except _ParseError as exc:
                self.diagnostics.append(exc.diagnostic)
                if self.pos == start:
                    self.pos += 1
                while not self.at_end() and not self.at("CloseBrace"):
                    tok = self.peek()
                    if tok.kind in _STATEMENT_SYNC:
                        break
                    if tok.kind == "Word" and self.at("Eq", 1):
                        break
                    self.pos += 1
        self.expect("CloseBrace", "Body", "'}' to close the body")

    # -- types ---------------------------------------------------------------

    def parse_type(self) -> None:
        tok = self.peek()
        if tok is None:
            raise self._error("type", "a type")
        if tok.kind == "Word":
            self.advance()  # PrimitiveType
        elif tok.kind == "OpenBracket":
            self.advance()
            self.parse_type()
            if self.at("SemiColon"):
                self.advance()
                self.expect("Number", "ArrayType", "an array length")
                if self.at("Plus"):
                    self.advance()
            self.expect("CloseBracket", "ArrayType", "']'")
        elif tok.kind == "OpenBrace":
            self.advance()
            self._comma_sep("ObjectType", "CloseBrace", self._parse_type_property)
            self.expect("CloseBrace", "ObjectType", "'}'")
        else:
            raise self._error("type", "an identifier, '[' or '{' starting a type")

    def _parse_type_property(self) -> None:
        self.expect("Word", "ObjectType", "a property name")
        self.expect("Colon", "ObjectType", "':' after the property name")
        self.parse_type()

    # -- expressions ----------------------------------------------------------

    def parse_expression(self, rule: str, min_bp: int = 1) -> None:
        """Precedence climbing over BINARY_LEVELS (pipe is the loosest)."""
        if self.peek() is None or self.peek().kind not in _EXPR_FIRST:
            raise self._error(rule, "an expression")
        self._parse_unary(rule)
        while True:
            tok = self.peek()
            if tok is None:
                return
            level = _KIND_TO_LEVEL.get(tok.kind)
            if level is None:
                return
            bp, name = level
            if bp < min_bp:
                return
            self.advance()
            # Left-associative: the right operand climbs one level tighter.
            child_rule = "PipeExpression" if name == "pipe" else "BinaryExpression"
            self.parse_expression(child_rule, bp + 1)

    def _parse_unary(self, rule: str) -> None:
        tok = self.peek()
        if tok is not None and tok.kind in _UNARY_OPS:
            self.advance()
            self._parse_unary("UnaryExpression")
            return
        self._parse_primary(rule)
        self._parse_postfix()

    def _parse_postfix(self) -> None:
        """CallExpression, MemberExpression, SubscriptExpression (tightest)."""
        while True:
            if self.at("OpenParen"):
                self.parse_argument_list()
            elif self.at("Period"):
                self.advance()
                self.expect("Word", "MemberExpression", "a property name after '.'")
            elif self.at("OpenBracket"):
                self.advance()
                self.parse_expression("SubscriptExpression")
                self.expect("CloseBracket", "SubscriptExpression", "']'")
            else:
                return

    def _parse_primary(self, rule: str) -> None:
        tok = self.peek()
        if tok is None:
            raise self._error(rule, "an expression")
        kind = tok.kind
        if kind in ("String", "Number", "TrueKw", "FalseKw", "NilKw", "Percent"):
            self.advance()
        elif kind == "VarKw":
            # SketchVar { var AddOp? Number }
            self.advance()
            if self.peek() is not None and self.peek().kind in _ADD_OPS:
                self.advance()
            self.expect("Number", "SketchVar", "a number after 'var'")
        elif kind == "Dollar":
            # TagDeclarator { "$" identifier } -- the two tokens are adjacent.
            dollar = self.advance()
            nxt = self.peek()
            if nxt is None or nxt.kind != "Word" or nxt.start != dollar.end:
                raise self._error("TagDeclarator", "an identifier joined to '$'")
            self.advance()
        elif kind == "OpenParen":
            self.advance()
            self.parse_expression("ParenthesizedExpression")
            self.expect("CloseParen", "ParenthesizedExpression", "')'")
        elif kind == "IfKw":
            self.advance()
            self.parse_expression("IfExpression")
            self.parse_body("IfExpression")
            self.expect("ElseKw", "IfExpression", "'else'")
            self.parse_body("IfExpression")
        elif kind == "OpenBracket":
            self.parse_array_expression()
        elif kind == "OpenBrace":
            self.parse_object_expression()
        elif kind == "Word":
            if (
                tok.text == "sketch"
                and self.at("OpenParen", 1)
                and self._sketch_block_ahead()
            ):
                self.advance()
                self.parse_argument_list()
                self.parse_body("SketchBlockExpression")
            else:
                # VariableName { identifier ("::" identifier)* }
                self.advance()
                while self.at("DoubleColon"):
                    self.advance()
                    self.expect("Word", "VariableName", "an identifier after '::'")
        else:
            raise self._error(rule, "an expression")

    def _sketch_block_ahead(self) -> bool:
        """True when 'sketch' '(' ... ')' is immediately followed by '{'."""
        depth = 0
        i = self.pos + 1  # at the '('
        while i < len(self.tokens):
            k = self.tokens[i].kind
            if k == "OpenParen":
                depth += 1
            elif k == "CloseParen":
                depth -= 1
                if depth == 0:
                    return i + 1 < len(self.tokens) and (
                        self.tokens[i + 1].kind == "OpenBrace"
                    )
            i += 1
        return False

    def parse_argument_list(self) -> None:
        self.expect("OpenParen", "ArgumentList", "'('")
        self._comma_sep("ArgumentList", "CloseParen", self._parse_argument)
        self.expect("CloseParen", "ArgumentList", "')'")

    def _parse_argument(self) -> None:
        # LabeledArgument { identifier "=" expression } | expression
        if self.at("Word") and self.at("Eq", 1):
            self.advance()
            self.advance()
            self.parse_expression("LabeledArgument")
        else:
            self.parse_expression("ArgumentList")

    def parse_array_expression(self) -> None:
        self.expect("OpenBracket", "ArrayExpression", "'['")
        def element() -> None:
            self.parse_expression("ArrayExpression")
            if self.at("DoublePeriod"):
                self.advance()
                self.parse_expression("IntegerRange")
        self._comma_sep("ArrayExpression", "CloseBracket", element)
        self.expect("CloseBracket", "ArrayExpression", "']'")

    def parse_object_expression(self) -> None:
        self.expect("OpenBrace", "ObjectExpression", "'{'")
        self._comma_sep(
            "ObjectExpression", "CloseBrace", self._parse_object_property
        )
        self.expect("CloseBrace", "ObjectExpression", "'}'")

    def _parse_object_property(self) -> None:
        self.expect("Word", "ObjectProperty", "a property name")
        if self.at("Colon") or self.at("Eq"):
            self.advance()
        else:
            raise self._error("ObjectProperty", "':' or '=' after the property name")
        self.parse_expression("ObjectProperty")

    # -- shared list helper ---------------------------------------------------

    def _comma_sep(self, rule: str, close_kind: str, item) -> None:
        """commaSep<item>: empty ok, trailing comma ok, stops before close."""
        if self.at(close_kind) or self.at_end():
            return
        item()
        while self.at("Comma"):
            self.advance()
            if self.at(close_kind) or self.at_end():
                return  # trailing comma
            item()


# ---------------------------------------------------------------------------
# Public entry point.
# ---------------------------------------------------------------------------


def check(source_or_tokens: Union[str, Sequence[Token]]) -> CheckResult:
    """Structurally validate KCL source (or a pre-lexed token sequence).

    Accepts either a source string (lexed with :func:`kcl_grammar.lex`; a
    leading shebang line is skipped, as the grammar's ``Shebang?`` allows) or
    a sequence of :class:`kcl_grammar.Token` (trivia is filtered out; shebang
    handling is only available for string input).

    Returns a :class:`CheckResult` whose diagnostics carry the production
    rule, byte position, and expected/found descriptions. Lexical recovery
    tokens (``Unknown``, ``UnterminatedString``, ``UnterminatedBlockComment``)
    are reported as diagnostics against the ``@tokens`` layer.
    """
    if isinstance(source_or_tokens, str):
        source = source_or_tokens
        if source.startswith("#!"):
            # Shebang? -- blank the first line, preserving byte offsets.
            nl = source.find("\n")
            head_end = len(source) if nl == -1 else nl
            source = " " * head_end + source[head_end:]
        tokens = kcl_grammar.lex(source)
        end_offset = tokens[-1].end if tokens else 0
    else:
        tokens = list(source_or_tokens)
        end_offset = tokens[-1].end if tokens else 0

    lexical: List[Diagnostic] = []
    significant: List[Token] = []
    for tok in tokens:
        if tok.kind in ("Unknown", "UnterminatedString", "UnterminatedBlockComment"):
            lexical.append(
                Diagnostic(
                    "@tokens", tok.start, "a valid token", tok.kind, tok.text
                )
            )
        elif not tok.is_trivia:
            significant.append(tok)

    checker = _Checker(significant, end_offset)
    checker.check_program()
    diagnostics = tuple(lexical + checker.diagnostics)
    return CheckResult(
        ok=not diagnostics,
        diagnostics=diagnostics,
        statement_count=checker.statement_count,
    )


# ---------------------------------------------------------------------------
# Self-check.
# ---------------------------------------------------------------------------

_VALID_SNIPPETS: Tuple[Tuple[str, str], ...] = (
    ("variable declaration", "width = 10mm"),
    ("exported variable", "export origin = [0, 0]"),
    (
        "function declaration",
        "export fn cube(size, center?: Point2d) {\n"
        "  return size * 2 + 1\n"
        "}",
    ),
    (
        "pipe expression chain",
        "part = startSketchOn('XY')\n"
        "  |> startProfileAt([0, 0], %)\n"
        "  |> line([0, 10], %, $edge1)\n"
        "  |> angledLine([45deg, segLen(edge1)], %)\n"
        "  |> close(%)\n"
        "  |> extrude(10, %)",
    ),
    (
        "sketch-like call chain with member/subscript",
        "d = getSketch().paths[0].length + foo::bar(x = 1, 2)",
    ),
    ("if expression", "h = if a > 2 { y = 1 } else { y = 2 }"),
    ("import statement", 'import cube, sphere as ball from "lib.kcl"'),
    ("object and array with range", "o = { a: 1, b = [1..5, 2,] }"),
    ("annotation", "@settings(defaultLengthUnit = mm)"),
    ("type declaration", "export type Point2d = { x: number, y: number }"),
    ("sketch block", "s = sketch(on = 'XY') { profile = line(end = [1, 2]) }"),
    ("sketch var and unary", "j = var +4 - -2 ^ 3"),
    ("shebang program", "#!/usr/bin/env kcl\nx = 1"),
    ("empty program", ""),
)

# (description, snippet, expected rule substring, expected byte position)
_INVALID_SNIPPETS: Tuple[Tuple[str, str, str, int], ...] = (
    ("missing initializer", "x = ", "VariableDeclaration", 4),
    ("fn without a name", "fn (a) { return a }", "FunctionDeclaration", 3),
    ("dangling pipe operator", "a |> ", "PipeExpression", 5),
    ("unbalanced bracket", "x = [1, 2", "ArrayExpression", 9),
    # 'from' itself lexes as an identifier, so it is consumed as the (only)
    # ImportItem and the failure surfaces at the string literal.
    ("import without items", 'import from "x.kcl"', "ImportStatement", 12),
    ("unbalanced paren", "x = (1 + ", "BinaryExpression", 9),
    ("if without else", "x = if a { b = 1 }", "IfExpression", 18),
    ("object missing separator", "o = { a 1 }", "ObjectProperty", 8),
    ("unterminated string", 'msg = "hello', "@tokens", 6),
)


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        prog="kcl_productions",
        description="KCL production-rule table and structural checker "
        "(grammar from Zoo modeling-app, MIT).",
    )
    parser.add_argument(
        "--selfcheck",
        action="store_true",
        help="parse known-good and known-bad KCL snippets and verify results",
    )
    args = parser.parse_args(argv)

    if not args.selfcheck:
        parser.print_help()
        return 0

    failures = 0

    for name, snippet in _VALID_SNIPPETS:
        result = check(snippet)
        if result.ok:
            print("ok   valid   %s" % name)
        else:
            failures += 1
            print("FAIL valid   %s" % name)
            for diag in result.diagnostics:
                print("       %s" % diag)

    for name, snippet, rule, position in _INVALID_SNIPPETS:
        result = check(snippet)
        if result.ok:
            failures += 1
            print("FAIL invalid %s: checker accepted it" % name)
            continue
        diag = result.diagnostics[0]
        if rule not in diag.rule:
            failures += 1
            print(
                "FAIL invalid %s: expected rule %s, got %s" % (name, rule, diag.rule)
            )
        elif diag.position != position:
            failures += 1
            print(
                "FAIL invalid %s: expected byte %d, got %d"
                % (name, position, diag.position)
            )
        else:
            print("ok   invalid %s (%s at byte %d)" % (name, diag.rule, diag.position))

    total_alts = sum(len(alts) for alts in PRODUCTIONS.values())
    print(
        "productions: %d rules, %d alternatives; precedence levels: %d"
        % (len(PRODUCTIONS), total_alts, len(PRECEDENCE))
    )
    if failures:
        print("selfcheck FAILED: %d problem(s)" % failures)
        return 1
    print("selfcheck passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
