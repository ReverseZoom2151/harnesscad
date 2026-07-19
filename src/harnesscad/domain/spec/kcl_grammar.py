"""Deterministic KCL modeling-language tokeniser and grammar tables.

This module is a stdlib-only, deterministic reference implementation of the KCL
lexer. It is NOT an engine backend and does NOT talk to any API -- it exists so
the rest of the harness (and in particular a backend author) has a checked,
importable model of KCL's *lexical* grammar, keyword set, AST node vocabulary,
standard-library function catalogue, and engine op set, without needing a
separate native toolchain.

What is modelled here
---------------------
*   :data:`SYNTAX_KINDS` -- every ``SyntaxKind`` variant (token kind).
*   :data:`KEYWORDS` -- the reserved-word table.
*   :func:`lex` -- a lossless tokeniser: whitespace and comments are preserved as
    tokens, and lexical errors surface as recovery token kinds
    (``Unknown``, ``UnterminatedString``, ``UnterminatedBlockComment``), exactly
    like any lossless lexer. Byte ranges are UTF-8 byte offsets.
*   :data:`AST_NODES` -- the AST node/enum vocabulary (statements, expressions,
    operators).
*   :data:`BINARY_OPERATORS` / :data:`UNARY_OPERATORS` -- operator spellings.

The number-suffix unit set (``mm``, ``cm``, ``m``, ``inch``, ``in``, ``ft``,
``yd``, ``deg``, ``rad``, ``?``) is carried in :data:`NUMBER_SUFFIXES`; it is part
of the *number token*, not a separate token, in KCL.

Everything is deterministic: same input -> byte-identical token sequence.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import List, Tuple

__all__ = [
    "SYNTAX_KINDS",
    "KEYWORDS",
    "TRIVIA_KINDS",
    "RECOVERY_KINDS",
    "NUMBER_SUFFIXES",
    "BINARY_OPERATORS",
    "UNARY_OPERATORS",
    "AST_NODES",
    "Token",
    "lex",
    "lex_significant",
    "keyword_or_word",
]

# ---------------------------------------------------------------------------
# Token kinds (mirror of syntax_kind.rs SyntaxKind).
# ---------------------------------------------------------------------------

SYNTAX_KINDS: Tuple[str, ...] = (
    "Number", "Word", "String", "UnterminatedString",
    "IfKw", "ElseKw", "ForKw", "WhileKw", "ReturnKw", "BreakKw", "ContinueKw",
    "FnKw", "LetKw", "MutKw", "AsKw", "LoopKw", "TrueKw", "FalseKw", "NilKw",
    "AndKw", "OrKw", "NotKw", "VarKw", "ConstKw", "ImportKw", "ExportKw",
    "TypeKw", "InterfaceKw", "NewKw", "SelfKw", "RecordKw", "StructKw", "ObjectKw",
    "GtEq", "LtEq", "EqEq", "FatArrow", "BangEq", "PipeGt",
    "Star", "Plus", "Minus", "Slash", "Percent", "Eq", "Lt", "Gt",
    "Backslash", "Caret", "PipePipe", "AmpAmp", "Pipe", "Amp",
    "OpenParen", "CloseParen", "OpenBrace", "CloseBrace", "OpenBracket", "CloseBracket",
    "Hash", "Bang", "Dollar", "Whitespace", "Comma", "Colon", "DoubleColon",
    "Period", "DoublePeriod", "DoublePeriodLessThan",
    "LineComment", "BlockComment", "UnterminatedBlockComment",
    "Unknown", "QuestionMark", "At", "SemiColon",
)

#: Reserved words -> their keyword SyntaxKind (mirror of ``keyword_or_word``).
KEYWORDS = {
    "if": "IfKw", "else": "ElseKw", "for": "ForKw", "while": "WhileKw",
    "return": "ReturnKw", "break": "BreakKw", "continue": "ContinueKw",
    "fn": "FnKw", "let": "LetKw", "mut": "MutKw", "as": "AsKw", "loop": "LoopKw",
    "true": "TrueKw", "false": "FalseKw", "nil": "NilKw",
    "and": "AndKw", "or": "OrKw", "not": "NotKw",
    "var": "VarKw", "const": "ConstKw", "import": "ImportKw", "export": "ExportKw",
    "type": "TypeKw", "interface": "InterfaceKw", "new": "NewKw", "self": "SelfKw",
    "record": "RecordKw", "struct": "StructKw", "object": "ObjectKw",
}

#: Token kinds that carry no program semantics (whitespace and comments).
TRIVIA_KINDS = frozenset(
    {"Whitespace", "LineComment", "BlockComment", "UnterminatedBlockComment"}
)

#: Token kinds emitted for lexical errors (the lexer never raises; it recovers).
RECOVERY_KINDS = frozenset({"Unknown", "UnterminatedString", "UnterminatedBlockComment"})

#: Numeric-literal unit suffixes, part of the number token itself.
NUMBER_SUFFIXES = ("mm", "cm", "m", "inch", "in", "ft", "yd", "deg", "rad", "?")

# ---------------------------------------------------------------------------
# AST / operator vocabulary (mirror of parsing/ast/types/mod.rs).
# ---------------------------------------------------------------------------

#: BinaryOperator variants -> their source spelling.
BINARY_OPERATORS = {
    "Add": "+", "Sub": "-", "Mul": "*", "Div": "/", "Mod": "%", "Pow": "^",
    "Eq": "==", "Neq": "!=", "Gt": ">", "Gte": ">=", "Lt": "<", "Lte": "<=",
    "And": "&", "Or": "|",
}

#: UnaryOperator variants -> their source spelling.
UNARY_OPERATORS = {"Neg": "-", "Not": "!", "Plus": "+"}

#: AST node / enum names grouped by role (from ``parsing/ast/types/mod.rs``).
AST_NODES = {
    "program": ("Program", "Shebang", "BodyItem"),
    "statements": (
        "ImportStatement", "ExpressionStatement", "VariableDeclaration",
        "TypeDeclaration", "ReturnStatement", "VariableDeclarator", "VariableKind",
    ),
    "expressions": (
        "Literal", "NumericLiteral", "Name", "Identifier", "TagDeclarator",
        "BinaryExpression", "UnaryExpression", "FunctionExpression",
        "CallExpressionKw", "LabeledArg", "PipeExpression", "PipeSubstitution",
        "ArrayExpression", "ArrayRangeExpression", "ObjectExpression",
        "ObjectProperty", "MemberExpression", "IfExpression", "LabelledExpression",
        "AscribedExpression", "SketchBlock", "SketchVar", "KclNone",
    ),
    "types": (
        "PrimitiveType", "FunctionType", "Type", "Parameter", "DefaultParamVal",
    ),
    "operators": ("BinaryOperator", "UnaryOperator", "Associativity"),
    "trivia": ("NonCodeNode", "NonCodeValue", "NonCodeMeta", "CommentStyle", "Annotation"),
    "imports": ("ImportItem", "ImportSelector", "ImportPath"),
}


@dataclass(frozen=True)
class Token:
    """One lexed token. ``start``/``end`` are UTF-8 byte offsets into the source."""

    kind: str
    text: str
    start: int
    end: int

    @property
    def is_trivia(self) -> bool:
        return self.kind in TRIVIA_KINDS

    @property
    def is_recovery(self) -> bool:
        return self.kind in RECOVERY_KINDS


def keyword_or_word(text: str) -> str:
    """SyntaxKind for a word token: a keyword kind if reserved, else ``Word``."""
    return KEYWORDS.get(text, "Word")


# ---------------------------------------------------------------------------
# Lexer.
#
# Longest-match tokenisation. The ordering below only breaks ties (equal match
# length), which mirrors logos' priority resolution for KCL's token set.
# ---------------------------------------------------------------------------

_WHITESPACE = re.compile(r"[ \t\n\r]+")
# Closed strings: content is any char except quote/backslash, or a backslash-escape
# of ANY char (including a newline) -- KCL strings may span lines.
_STRING_DQ = re.compile(r'"([^"\\]|\\[\s\S])*"')
_STRING_SQ = re.compile(r"'([^'\\]|\\[\s\S])*'")
_LINE_COMMENT = re.compile(r"//[^\n\r]*")
_NUMBER = re.compile(
    r"(?:[0-9]+(?:\.[0-9]+)?|\.[0-9]+)_?(?:mm|cm|m|inch|in|ft|yd|deg|rad|\?)?"
)
_WORD = re.compile(r"[^\W\d][\w]*", re.UNICODE)

# Multi-char then single-char punctuation. Longer spellings first so a plain
# scan is also longest-match for these fixed tokens.
_PUNCT: Tuple[Tuple[str, str], ...] = (
    ("..<", "DoublePeriodLessThan"),
    ("..", "DoublePeriod"),
    ("::", "DoubleColon"),
    (">=", "GtEq"), ("<=", "LtEq"), ("==", "EqEq"), ("=>", "FatArrow"),
    ("!=", "BangEq"), ("|>", "PipeGt"), ("||", "PipePipe"), ("&&", "AmpAmp"),
    ("*", "Star"), ("+", "Plus"), ("-", "Minus"), ("/", "Slash"),
    ("%", "Percent"), ("=", "Eq"), ("<", "Lt"), (">", "Gt"),
    ("\\", "Backslash"), ("^", "Caret"), ("|", "Pipe"), ("&", "Amp"),
    ("(", "OpenParen"), (")", "CloseParen"), ("{", "OpenBrace"), ("}", "CloseBrace"),
    ("[", "OpenBracket"), ("]", "CloseBracket"), ("#", "Hash"), ("!", "Bang"),
    ("$", "Dollar"), (",", "Comma"), (":", "Colon"), (".", "Period"),
    ("?", "QuestionMark"), ("@", "At"), (";", "SemiColon"),
)


def _byte_len(s: str) -> int:
    return len(s.encode("utf-8"))


def lex(source: str) -> List[Token]:
    """Losslessly tokenise KCL ``source`` into a list of :class:`Token`.

    Whitespace and comments are preserved as tokens. Lexical errors do not raise;
    they appear as recovery token kinds. Byte offsets are UTF-8 byte positions so
    they agree with the Rust reference lexer.
    """
    tokens: List[Token] = []
    i = 0
    n = len(source)
    byte_pos = 0
    while i < n:
        ch = source[i]
        # -- whitespace
        m = _WHITESPACE.match(source, i)
        if m:
            text = m.group(0)
            tokens.append(Token("Whitespace", text, byte_pos, byte_pos + _byte_len(text)))
            byte_pos += _byte_len(text)
            i = m.end()
            continue
        # -- comments
        if ch == "/" and i + 1 < n and source[i + 1] == "/":
            m = _LINE_COMMENT.match(source, i)
            text = m.group(0)
            tokens.append(Token("LineComment", text, byte_pos, byte_pos + _byte_len(text)))
            byte_pos += _byte_len(text)
            i = m.end()
            continue
        if ch == "/" and i + 1 < n and source[i + 1] == "*":
            end = source.find("*/", i + 2)
            if end == -1:
                text = source[i:]
                kind = "UnterminatedBlockComment"
                nexti = n
            else:
                text = source[i:end + 2]
                kind = "BlockComment"
                nexti = end + 2
            tokens.append(Token(kind, text, byte_pos, byte_pos + _byte_len(text)))
            byte_pos += _byte_len(text)
            i = nexti
            continue
        # -- strings (closed, possibly multiline), else unterminated recovery
        if ch == '"' or ch == "'":
            rx = _STRING_DQ if ch == '"' else _STRING_SQ
            m = rx.match(source, i)
            if m:
                text = m.group(0)
                tokens.append(Token("String", text, byte_pos, byte_pos + _byte_len(text)))
                byte_pos += _byte_len(text)
                i = m.end()
                continue
            # unterminated: consume up to (not including) the next line break
            j = i + 1
            while j < n and source[j] not in "\n\r":
                # allow backslash-escape of a non-newline
                if source[j] == "\\" and j + 1 < n and source[j + 1] not in "\n\r":
                    j += 2
                    continue
                j += 1
            text = source[i:j]
            tokens.append(Token("UnterminatedString", text, byte_pos, byte_pos + _byte_len(text)))
            byte_pos += _byte_len(text)
            i = j
            continue
        # -- number (must beat a leading '.' Period and '..' handled below)
        if ch.isdigit() or (ch == "." and i + 1 < n and source[i + 1].isdigit()):
            m = _NUMBER.match(source, i)
            if m and m.end() > i:
                text = m.group(0)
                tokens.append(Token("Number", text, byte_pos, byte_pos + _byte_len(text)))
                byte_pos += _byte_len(text)
                i = m.end()
                continue
        # -- word / keyword
        m = _WORD.match(source, i)
        if m:
            text = m.group(0)
            tokens.append(Token(keyword_or_word(text), text, byte_pos, byte_pos + _byte_len(text)))
            byte_pos += _byte_len(text)
            i = m.end()
            continue
        # -- punctuation (longest fixed spelling first)
        matched = False
        for spelling, kind in _PUNCT:
            if source.startswith(spelling, i):
                tokens.append(Token(kind, spelling, byte_pos, byte_pos + _byte_len(spelling)))
                byte_pos += _byte_len(spelling)
                i += len(spelling)
                matched = True
                break
        if matched:
            continue
        # -- unknown single character (recovery)
        tokens.append(Token("Unknown", ch, byte_pos, byte_pos + _byte_len(ch)))
        byte_pos += _byte_len(ch)
        i += 1
    return tokens


def lex_significant(source: str) -> List[Token]:
    """Like :func:`lex` but drops trivia (whitespace and comments)."""
    return [t for t in lex(source) if not t.is_trivia]
