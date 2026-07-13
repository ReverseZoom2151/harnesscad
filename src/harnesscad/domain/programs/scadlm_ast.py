"""OpenSCAD tokenizer, recursive-descent parser, AST and unparser.

ScadLM (Runpod hackathon, KrishKrosh) drives an LLM -> OpenSCAD -> render ->
feedback loop: the model emits OpenSCAD source, the backend shells out to the
``openscad`` binary to find out whether it even compiles, and re-prompts with a
"does not compile" message when it does not. The compile gate is the only
verifier in the loop, and it is an external binary.

To do that class of check locally the harness needs the piece ScadLM never had:
a real front end for the OpenSCAD language. The existing harness modules only
*segment* SCAD source into top-level blocks (``programs.cadreview_blocks``) or
scrape Customizer annotations off declaration lines
(``programs.cadam_scad_customizer``); neither understands expressions, module
definitions, children, ranges, or list comprehensions.

This module supplies a full lexer + recursive-descent parser producing a typed
AST, plus a deterministic unparser (source -> AST -> source is stable, i.e.
``unparse(parse(unparse(parse(s)))) == unparse(parse(s))``).

Grammar covered (OpenSCAD as of the cheat sheet shipped in ScadLM's prompt):

  statements     assignment, ``module``/``function`` definition, module
                 instantiation with children, ``{}`` blocks, ``if``/``else``,
                 ``for``, ``intersection_for``, ``let``, ``include``/``use``,
                 the ``* ! # %`` modifier characters, stray ``;``
  expressions    ternary ``?:``, ``||``, ``&&``, equality, relational, ``+ -``,
                 ``* / %``, ``^`` (right assoc), unary ``- !``, postfix index
                 ``[i]``, dot access ``.x/.y/.z``, calls, literals (number,
                 string, ``true``/``false``/``undef``), vectors, ranges
                 ``[a:b]`` / ``[a:s:b]``, ``let(...)`` expressions, anonymous
                 ``function (x) expr`` literals, and list comprehensions with
                 ``for`` / ``if`` / ``else`` / ``each`` / ``let``.

Pure stdlib, no execution, deterministic. :func:`parse` raises
:class:`ScadSyntaxError` (carrying line/column) on malformed input -- which is
exactly the local, binary-free replacement for ScadLM's ``openscad`` compile
gate at the *syntax* level (semantic checks live in
``programs.scadlm_check``).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, List, Optional, Sequence, Tuple, Union

__all__ = [
    "ScadSyntaxError",
    "Token",
    "tokenize",
    "Num",
    "Str",
    "Bool",
    "Undef",
    "Name",
    "Vector",
    "Range",
    "Unary",
    "Binary",
    "Ternary",
    "Index",
    "Member",
    "Call",
    "LetExpr",
    "FunctionLiteral",
    "Comprehension",
    "Argument",
    "Parameter",
    "Assign",
    "ModuleDef",
    "FunctionDef",
    "ModuleCall",
    "Block",
    "IfStmt",
    "ForStmt",
    "LetStmt",
    "Include",
    "NoOp",
    "parse",
    "parse_expression",
    "unparse",
    "walk",
]


class ScadSyntaxError(Exception):
    """Raised when OpenSCAD source cannot be parsed."""

    def __init__(self, message: str, line: int = 0, column: int = 0) -> None:
        super().__init__("line %d col %d: %s" % (line, column, message))
        self.message = message
        self.line = line
        self.column = column


# ---------------------------------------------------------------------------
# Lexer
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Token:
    kind: str  # NUMBER STRING NAME OP EOF
    value: str
    line: int
    column: int


_OPERATORS = [
    "<=", ">=", "==", "!=", "&&", "||",
    "+", "-", "*", "/", "%", "^", "<", ">", "!", "=", "?", ":",
    "(", ")", "[", "]", "{", "}", ",", ";", ".", "#",
]
_OPERATORS.sort(key=len, reverse=True)

_ID_START = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ_$")
_ID_BODY = _ID_START | set("0123456789")
_DIGITS = set("0123456789")


def tokenize(source: str) -> List[Token]:
    """Lex OpenSCAD source into tokens (comments stripped)."""
    tokens: List[Token] = []
    i = 0
    line = 1
    col = 1
    n = len(source)

    def advance(count: int) -> None:
        nonlocal i, line, col
        for _ in range(count):
            if source[i] == "\n":
                line += 1
                col = 1
            else:
                col += 1
            i += 1

    while i < n:
        ch = source[i]
        if ch in " \t\r\n":
            advance(1)
            continue
        if source.startswith("//", i):
            j = source.find("\n", i)
            advance((n if j < 0 else j) - i)
            continue
        if source.startswith("/*", i):
            j = source.find("*/", i + 2)
            if j < 0:
                raise ScadSyntaxError("unterminated block comment", line, col)
            advance(j + 2 - i)
            continue
        if ch == '"':
            start_line, start_col = line, col
            j = i + 1
            buf: List[str] = []
            while j < n and source[j] != '"':
                if source[j] == "\\" and j + 1 < n:
                    esc = source[j + 1]
                    buf.append({"n": "\n", "t": "\t", "r": "\r",
                                '"': '"', "\\": "\\"}.get(esc, esc))
                    j += 2
                    continue
                buf.append(source[j])
                j += 1
            if j >= n:
                raise ScadSyntaxError("unterminated string", start_line, start_col)
            advance(j + 1 - i)
            tokens.append(Token("STRING", "".join(buf), start_line, start_col))
            continue
        if ch in _DIGITS or (ch == "." and i + 1 < n and source[i + 1] in _DIGITS):
            start_line, start_col = line, col
            j = i
            seen_dot = False
            seen_exp = False
            while j < n:
                c = source[j]
                if c in _DIGITS:
                    j += 1
                elif c == "." and not seen_dot and not seen_exp:
                    seen_dot = True
                    j += 1
                elif c in "eE" and not seen_exp and j + 1 < n and (
                        source[j + 1] in _DIGITS or (
                            source[j + 1] in "+-" and j + 2 < n and source[j + 2] in _DIGITS)):
                    seen_exp = True
                    j += 2 if source[j + 1] in _DIGITS else 3
                else:
                    break
            text = source[i:j]
            advance(j - i)
            tokens.append(Token("NUMBER", text, start_line, start_col))
            continue
        if ch in _ID_START:
            start_line, start_col = line, col
            j = i
            while j < n and source[j] in _ID_BODY:
                j += 1
            text = source[i:j]
            advance(j - i)
            tokens.append(Token("NAME", text, start_line, start_col))
            continue
        for op in _OPERATORS:
            if source.startswith(op, i):
                start_line, start_col = line, col
                advance(len(op))
                tokens.append(Token("OP", op, start_line, start_col))
                break
        else:
            raise ScadSyntaxError("unexpected character %r" % ch, line, col)
    tokens.append(Token("EOF", "", line, col))
    return tokens


# ---------------------------------------------------------------------------
# AST
# ---------------------------------------------------------------------------

@dataclass
class Num:
    value: float


@dataclass
class Str:
    value: str


@dataclass
class Bool:
    value: bool


@dataclass
class Undef:
    pass


@dataclass
class Name:
    ident: str


@dataclass
class Vector:
    items: List[Any] = field(default_factory=list)


@dataclass
class Range:
    start: Any
    end: Any
    step: Optional[Any] = None


@dataclass
class Unary:
    op: str
    operand: Any


@dataclass
class Binary:
    op: str
    left: Any
    right: Any


@dataclass
class Ternary:
    cond: Any
    if_true: Any
    if_false: Any


@dataclass
class Index:
    target: Any
    index: Any


@dataclass
class Member:
    target: Any
    name: str


@dataclass
class Argument:
    value: Any
    name: Optional[str] = None


@dataclass
class Call:
    name: Any            # Name or arbitrary expression (function literal value)
    args: List[Argument] = field(default_factory=list)


@dataclass
class LetExpr:
    bindings: List[Tuple[str, Any]]
    body: Any


@dataclass
class Parameter:
    name: str
    default: Optional[Any] = None


@dataclass
class FunctionLiteral:
    params: List[Parameter]
    body: Any


@dataclass
class Comprehension:
    """A ``for`` / ``if`` / ``let`` / ``each`` element inside a vector literal."""

    kind: str                                  # for | if | let | each
    bindings: List[Tuple[str, Any]] = field(default_factory=list)  # for/let
    cond: Optional[Any] = None                 # if
    body: Optional[Any] = None                 # for/let/if-then/each
    orelse: Optional[Any] = None               # if-else


# --- statements ---

@dataclass
class Assign:
    name: str
    value: Any


@dataclass
class ModuleDef:
    name: str
    params: List[Parameter]
    body: Any                # statement


@dataclass
class FunctionDef:
    name: str
    params: List[Parameter]
    body: Any                # expression


@dataclass
class ModuleCall:
    name: str
    args: List[Argument] = field(default_factory=list)
    children: List[Any] = field(default_factory=list)
    modifier: str = ""       # one of "", "*", "!", "#", "%"


@dataclass
class Block:
    body: List[Any] = field(default_factory=list)


@dataclass
class IfStmt:
    cond: Any
    then: Any
    orelse: Optional[Any] = None


@dataclass
class ForStmt:
    bindings: List[Tuple[str, Any]]
    body: Any
    intersect: bool = False


@dataclass
class LetStmt:
    bindings: List[Tuple[str, Any]]
    body: Any


@dataclass
class Include:
    kind: str    # include | use
    path: str


@dataclass
class NoOp:
    pass


Expr = Union[Num, Str, Bool, Undef, Name, Vector, Range, Unary, Binary,
             Ternary, Index, Member, Call, LetExpr, FunctionLiteral]
Stmt = Union[Assign, ModuleDef, FunctionDef, ModuleCall, Block, IfStmt,
             ForStmt, LetStmt, Include, NoOp]

_KEYWORDS = {"module", "function", "if", "else", "for", "intersection_for",
             "let", "each", "true", "false", "undef", "include", "use"}


# ---------------------------------------------------------------------------
# Parser
# ---------------------------------------------------------------------------

class _Parser:
    def __init__(self, source: str) -> None:
        self.source = source
        self.tokens = tokenize(source)
        self.pos = 0

    # -- token helpers --
    def peek(self, offset: int = 0) -> Token:
        idx = min(self.pos + offset, len(self.tokens) - 1)
        return self.tokens[idx]

    def at_op(self, *ops: str) -> bool:
        t = self.peek()
        return t.kind == "OP" and t.value in ops

    def at_name(self, *names: str) -> bool:
        t = self.peek()
        return t.kind == "NAME" and t.value in names

    def next(self) -> Token:
        t = self.tokens[self.pos]
        if t.kind != "EOF":
            self.pos += 1
        return t

    def expect_op(self, op: str) -> Token:
        t = self.peek()
        if t.kind != "OP" or t.value != op:
            raise ScadSyntaxError("expected %r, found %r" % (op, t.value or "EOF"),
                                  t.line, t.column)
        return self.next()

    def expect_name(self) -> Token:
        t = self.peek()
        if t.kind != "NAME":
            raise ScadSyntaxError("expected identifier, found %r" % (t.value or "EOF"),
                                  t.line, t.column)
        return self.next()

    # -- entry --
    def parse_program(self) -> List[Any]:
        body: List[Any] = []
        while self.peek().kind != "EOF":
            body.append(self.parse_statement())
        return body

    # -- statements --
    def parse_statement(self) -> Any:
        t = self.peek()
        if t.kind == "OP" and t.value == ";":
            self.next()
            return NoOp()
        if t.kind == "OP" and t.value == "{":
            return self.parse_block()
        if t.kind == "OP" and t.value in ("*", "!", "#", "%"):
            self.next()
            inner = self.parse_statement()
            if isinstance(inner, ModuleCall):
                inner.modifier = t.value
                return inner
            raise ScadSyntaxError("modifier %r must precede a module call" % t.value,
                                  t.line, t.column)
        if t.kind == "NAME":
            if t.value in ("include", "use"):
                return self.parse_include()
            if t.value == "module":
                return self.parse_module_def()
            if t.value == "function" and self.peek(1).kind == "NAME":
                return self.parse_function_def()
            if t.value == "if":
                return self.parse_if()
            if t.value in ("for", "intersection_for"):
                return self.parse_for()
            if t.value == "let":
                return self.parse_let_stmt()
            nxt = self.peek(1)
            if nxt.kind == "OP" and nxt.value == "=":
                name = self.next().value
                self.next()  # '='
                value = self.parse_expr()
                self.expect_op(";")
                return Assign(name, value)
            return self.parse_module_call()
        raise ScadSyntaxError("unexpected token %r" % (t.value or "EOF"), t.line, t.column)

    def parse_include(self) -> Include:
        kind = self.next().value
        # path is delimited by < ... > which the lexer sees as OP '<' + names.
        t = self.expect_op("<")
        buf: List[str] = []
        while not self.at_op(">"):
            tok = self.peek()
            if tok.kind == "EOF":
                raise ScadSyntaxError("unterminated include path", t.line, t.column)
            buf.append(self.next().value)
        self.expect_op(">")
        return Include(kind, "".join(buf))

    def parse_block(self) -> Block:
        self.expect_op("{")
        body: List[Any] = []
        while not self.at_op("}"):
            if self.peek().kind == "EOF":
                t = self.peek()
                raise ScadSyntaxError("unterminated block", t.line, t.column)
            body.append(self.parse_statement())
        self.expect_op("}")
        return Block(body)

    def parse_params(self) -> List[Parameter]:
        self.expect_op("(")
        params: List[Parameter] = []
        while not self.at_op(")"):
            name = self.expect_name().value
            default = None
            if self.at_op("="):
                self.next()
                default = self.parse_expr()
            params.append(Parameter(name, default))
            if self.at_op(","):
                self.next()
            elif not self.at_op(")"):
                t = self.peek()
                raise ScadSyntaxError("expected ',' or ')' in parameter list",
                                      t.line, t.column)
        self.expect_op(")")
        return params

    def parse_module_def(self) -> ModuleDef:
        self.next()  # module
        name = self.expect_name().value
        params = self.parse_params()
        body = self.parse_statement()
        return ModuleDef(name, params, body)

    def parse_function_def(self) -> FunctionDef:
        self.next()  # function
        name = self.expect_name().value
        params = self.parse_params()
        self.expect_op("=")
        body = self.parse_expr()
        self.expect_op(";")
        return FunctionDef(name, params, body)

    def parse_if(self) -> IfStmt:
        self.next()  # if
        self.expect_op("(")
        cond = self.parse_expr()
        self.expect_op(")")
        then = self.parse_statement()
        orelse = None
        if self.at_name("else"):
            self.next()
            orelse = self.parse_statement()
        return IfStmt(cond, then, orelse)

    def parse_bindings(self) -> List[Tuple[str, Any]]:
        self.expect_op("(")
        bindings: List[Tuple[str, Any]] = []
        while not self.at_op(")"):
            name = self.expect_name().value
            self.expect_op("=")
            bindings.append((name, self.parse_expr()))
            if self.at_op(","):
                self.next()
            elif not self.at_op(")"):
                t = self.peek()
                raise ScadSyntaxError("expected ',' or ')' in binding list",
                                      t.line, t.column)
        self.expect_op(")")
        return bindings

    def parse_for(self) -> ForStmt:
        kw = self.next().value
        bindings = self.parse_bindings()
        body = self.parse_statement()
        return ForStmt(bindings, body, intersect=(kw == "intersection_for"))

    def parse_let_stmt(self) -> LetStmt:
        self.next()  # let
        bindings = self.parse_bindings()
        body = self.parse_statement()
        return LetStmt(bindings, body)

    def parse_args(self) -> List[Argument]:
        self.expect_op("(")
        args: List[Argument] = []
        while not self.at_op(")"):
            name = None
            t = self.peek()
            nxt = self.peek(1)
            if t.kind == "NAME" and nxt.kind == "OP" and nxt.value == "=":
                name = self.next().value
                self.next()
            args.append(Argument(self.parse_expr(), name))
            if self.at_op(","):
                self.next()
            elif not self.at_op(")"):
                tt = self.peek()
                raise ScadSyntaxError("expected ',' or ')' in argument list",
                                      tt.line, tt.column)
        self.expect_op(")")
        return args

    def parse_module_call(self) -> ModuleCall:
        name = self.expect_name().value
        args = self.parse_args()
        children: List[Any] = []
        if self.at_op(";"):
            self.next()
        elif self.at_op("{"):
            children = self.parse_block().body
        else:
            children = [self.parse_statement()]
        return ModuleCall(name, args, children)

    # -- expressions --
    def parse_expr(self) -> Any:
        return self.parse_ternary()

    def parse_ternary(self) -> Any:
        cond = self.parse_or()
        if self.at_op("?"):
            self.next()
            a = self.parse_ternary()
            self.expect_op(":")
            b = self.parse_ternary()
            return Ternary(cond, a, b)
        return cond

    def _binary(self, sub, ops: Sequence[str]):
        left = sub()
        while self.at_op(*ops):
            op = self.next().value
            left = Binary(op, left, sub())
        return left

    def parse_or(self) -> Any:
        return self._binary(self.parse_and, ("||",))

    def parse_and(self) -> Any:
        return self._binary(self.parse_equality, ("&&",))

    def parse_equality(self) -> Any:
        return self._binary(self.parse_relational, ("==", "!="))

    def parse_relational(self) -> Any:
        return self._binary(self.parse_additive, ("<", "<=", ">", ">="))

    def parse_additive(self) -> Any:
        return self._binary(self.parse_multiplicative, ("+", "-"))

    def parse_multiplicative(self) -> Any:
        return self._binary(self.parse_unary, ("*", "/", "%"))

    def parse_unary(self) -> Any:
        if self.at_op("-", "!", "+"):
            op = self.next().value
            operand = self.parse_unary()
            if op == "+":
                return operand
            return Unary(op, operand)
        return self.parse_power()

    def parse_power(self) -> Any:
        base = self.parse_postfix()
        if self.at_op("^"):
            self.next()
            return Binary("^", base, self.parse_unary())  # right associative
        return base

    def parse_postfix(self) -> Any:
        node = self.parse_primary()
        while True:
            if self.at_op("["):
                self.next()
                idx = self.parse_expr()
                self.expect_op("]")
                node = Index(node, idx)
            elif self.at_op("."):
                self.next()
                node = Member(node, self.expect_name().value)
            elif self.at_op("(") and isinstance(node, (Name, FunctionLiteral, Call,
                                                       Index, Member)):
                node = Call(node, self.parse_args())
            else:
                return node

    def parse_primary(self) -> Any:
        t = self.peek()
        if t.kind == "NUMBER":
            self.next()
            return Num(float(t.value))
        if t.kind == "STRING":
            self.next()
            return Str(t.value)
        if t.kind == "NAME":
            if t.value == "true":
                self.next()
                return Bool(True)
            if t.value == "false":
                self.next()
                return Bool(False)
            if t.value == "undef":
                self.next()
                return Undef()
            if t.value == "let":
                self.next()
                bindings = self.parse_bindings()
                return LetExpr(bindings, self.parse_expr())
            if t.value == "function":
                self.next()
                params = self.parse_params()
                return FunctionLiteral(params, self.parse_expr())
            self.next()
            return Name(t.value)
        if t.kind == "OP" and t.value == "(":
            self.next()
            inner = self.parse_expr()
            self.expect_op(")")
            return inner
        if t.kind == "OP" and t.value == "[":
            return self.parse_bracket()
        raise ScadSyntaxError("unexpected token %r in expression" % (t.value or "EOF"),
                              t.line, t.column)

    def parse_bracket(self) -> Any:
        self.expect_op("[")
        if self.at_op("]"):
            self.next()
            return Vector([])
        items: List[Any] = []
        first = self.parse_vector_element()
        # range?  [a : b]  or  [a : s : b]
        if self.at_op(":"):
            self.next()
            second = self.parse_expr()
            if self.at_op(":"):
                self.next()
                third = self.parse_expr()
                self.expect_op("]")
                return Range(first, third, second)
            self.expect_op("]")
            return Range(first, second)
        items.append(first)
        while self.at_op(","):
            self.next()
            if self.at_op("]"):
                break
            items.append(self.parse_vector_element())
        self.expect_op("]")
        return Vector(items)

    def parse_vector_element(self) -> Any:
        """A vector element: an expression, or a comprehension clause."""
        if self.at_name("for"):
            self.next()
            bindings = self.parse_bindings()
            body = self.parse_vector_element()
            return Comprehension("for", bindings=bindings, body=body)
        if self.at_name("if"):
            self.next()
            self.expect_op("(")
            cond = self.parse_expr()
            self.expect_op(")")
            body = self.parse_vector_element()
            orelse = None
            if self.at_name("else"):
                self.next()
                orelse = self.parse_vector_element()
            return Comprehension("if", cond=cond, body=body, orelse=orelse)
        if self.at_name("each"):
            self.next()
            return Comprehension("each", body=self.parse_vector_element())
        if self.at_name("let") and self.peek(1).kind == "OP" and self.peek(1).value == "(":
            # A `let` inside a comprehension may be followed by another clause.
            save = self.pos
            self.next()
            bindings = self.parse_bindings()
            if self.at_name("for", "if", "each", "let"):
                return Comprehension("let", bindings=bindings,
                                     body=self.parse_vector_element())
            self.pos = save
        return self.parse_expr()


def parse(source: str) -> List[Any]:
    """Parse an OpenSCAD program into a list of statements."""
    return _Parser(source).parse_program()


def parse_expression(source: str) -> Any:
    """Parse a single OpenSCAD expression."""
    p = _Parser(source)
    node = p.parse_expr()
    t = p.peek()
    if t.kind != "EOF":
        raise ScadSyntaxError("trailing input %r" % t.value, t.line, t.column)
    return node


# ---------------------------------------------------------------------------
# Unparser
# ---------------------------------------------------------------------------

def _fmt_number(v: float) -> str:
    if v == int(v) and abs(v) < 1e15:
        return str(int(v))
    return repr(v)


def _escape(s: str) -> str:
    out = s.replace("\\", "\\\\").replace('"', '\\"')
    return out.replace("\n", "\\n").replace("\t", "\\t").replace("\r", "\\r")


_PRECEDENCE = {
    "||": 1, "&&": 2,
    "==": 3, "!=": 3,
    "<": 4, "<=": 4, ">": 4, ">=": 4,
    "+": 5, "-": 5,
    "*": 6, "/": 6, "%": 6,
    "^": 7,
}


def _expr(node: Any, parent_prec: int = 0) -> str:
    if isinstance(node, Num):
        return _fmt_number(node.value)
    if isinstance(node, Str):
        return '"%s"' % _escape(node.value)
    if isinstance(node, Bool):
        return "true" if node.value else "false"
    if isinstance(node, Undef):
        return "undef"
    if isinstance(node, Name):
        return node.ident
    if isinstance(node, Vector):
        return "[%s]" % ", ".join(_expr(i) for i in node.items)
    if isinstance(node, Range):
        if node.step is None:
            return "[%s : %s]" % (_expr(node.start), _expr(node.end))
        return "[%s : %s : %s]" % (_expr(node.start), _expr(node.step), _expr(node.end))
    if isinstance(node, Unary):
        return "%s%s" % (node.op, _expr(node.operand, 8))
    if isinstance(node, Binary):
        prec = _PRECEDENCE[node.op]
        text = "%s %s %s" % (_expr(node.left, prec), node.op,
                             _expr(node.right, prec + (0 if node.op == "^" else 1)))
        return "(%s)" % text if prec < parent_prec else text
    if isinstance(node, Ternary):
        text = "%s ? %s : %s" % (_expr(node.cond, 1), _expr(node.if_true),
                                 _expr(node.if_false))
        return "(%s)" % text if parent_prec > 0 else text
    if isinstance(node, Index):
        return "%s[%s]" % (_expr(node.target, 9), _expr(node.index))
    if isinstance(node, Member):
        return "%s.%s" % (_expr(node.target, 9), node.name)
    if isinstance(node, Call):
        return "%s(%s)" % (_expr(node.name, 9), _args(node.args))
    if isinstance(node, LetExpr):
        return "let(%s) %s" % (_bindings(node.bindings), _expr(node.body))
    if isinstance(node, FunctionLiteral):
        return "function (%s) %s" % (_params(node.params), _expr(node.body))
    if isinstance(node, Comprehension):
        if node.kind == "for":
            return "for (%s) %s" % (_bindings(node.bindings), _expr(node.body))
        if node.kind == "let":
            return "let (%s) %s" % (_bindings(node.bindings), _expr(node.body))
        if node.kind == "each":
            return "each %s" % _expr(node.body)
        text = "if (%s) %s" % (_expr(node.cond), _expr(node.body))
        if node.orelse is not None:
            text += " else %s" % _expr(node.orelse)
        return text
    raise TypeError("cannot unparse expression node %r" % (node,))


def _args(args: Sequence[Argument]) -> str:
    parts = []
    for a in args:
        parts.append(("%s = %s" % (a.name, _expr(a.value))) if a.name
                     else _expr(a.value))
    return ", ".join(parts)


def _params(params: Sequence[Parameter]) -> str:
    parts = []
    for p in params:
        parts.append(("%s = %s" % (p.name, _expr(p.default))) if p.default is not None
                     else p.name)
    return ", ".join(parts)


def _bindings(bindings: Sequence[Tuple[str, Any]]) -> str:
    return ", ".join("%s = %s" % (k, _expr(v)) for k, v in bindings)


def _stmt(node: Any, indent: int) -> List[str]:
    pad = "    " * indent
    if isinstance(node, NoOp):
        return []
    if isinstance(node, Include):
        return ["%s%s <%s>" % (pad, node.kind, node.path)]
    if isinstance(node, Assign):
        return ["%s%s = %s;" % (pad, node.name, _expr(node.value))]
    if isinstance(node, FunctionDef):
        return ["%sfunction %s(%s) = %s;" % (pad, node.name, _params(node.params),
                                             _expr(node.body))]
    if isinstance(node, ModuleDef):
        head = "%smodule %s(%s)" % (pad, node.name, _params(node.params))
        return _attach(head, node.body, indent)
    if isinstance(node, Block):
        lines = ["%s{" % pad]
        for s in node.body:
            lines.extend(_stmt(s, indent + 1))
        lines.append("%s}" % pad)
        return lines
    if isinstance(node, IfStmt):
        lines = _attach("%sif (%s)" % (pad, _expr(node.cond)), node.then, indent)
        if node.orelse is not None:
            else_lines = _attach("%selse" % pad, node.orelse, indent)
            lines.extend(else_lines)
        return lines
    if isinstance(node, ForStmt):
        kw = "intersection_for" if node.intersect else "for"
        return _attach("%s%s (%s)" % (pad, kw, _bindings(node.bindings)),
                       node.body, indent)
    if isinstance(node, LetStmt):
        return _attach("%slet (%s)" % (pad, _bindings(node.bindings)),
                       node.body, indent)
    if isinstance(node, ModuleCall):
        head = "%s%s%s(%s)" % (pad, node.modifier, node.name, _args(node.args))
        if not node.children:
            return [head + ";"]
        if len(node.children) == 1 and not isinstance(node.children[0], Block):
            child = _stmt(node.children[0], indent + 1)
            if len(child) == 1:
                return [head + " " + child[0].strip()]
        lines = [head + " {"]
        for c in node.children:
            lines.extend(_stmt(c, indent + 1))
        lines.append("%s}" % pad)
        return lines
    raise TypeError("cannot unparse statement node %r" % (node,))


def _attach(head: str, body: Any, indent: int) -> List[str]:
    """Emit ``head`` followed by ``body`` (inline when the body is one line)."""
    body_lines = _stmt(body, indent + 1)
    if isinstance(body, Block):
        pad = "    " * indent
        lines = [head + " {"]
        for s in body.body:
            lines.extend(_stmt(s, indent + 1))
        lines.append("%s}" % pad)
        return lines
    if len(body_lines) == 1:
        return [head + " " + body_lines[0].strip()]
    return [head] + body_lines


def unparse(nodes: Union[Sequence[Any], Any]) -> str:
    """Render an AST (statement list, single statement, or expression) to source."""
    if isinstance(nodes, (list, tuple)):
        lines: List[str] = []
        for s in nodes:
            lines.extend(_stmt(s, 0))
        return "\n".join(lines)
    if isinstance(nodes, (Assign, ModuleDef, FunctionDef, ModuleCall, Block,
                          IfStmt, ForStmt, LetStmt, Include, NoOp)):
        return "\n".join(_stmt(nodes, 0))
    return _expr(nodes)


# ---------------------------------------------------------------------------
# Traversal
# ---------------------------------------------------------------------------

def walk(node: Any):
    """Yield ``node`` and every AST node beneath it (pre-order, deterministic)."""
    yield node
    for child in _children(node):
        for sub in walk(child):
            yield sub


def _children(node: Any) -> List[Any]:
    out: List[Any] = []
    if isinstance(node, (list, tuple)):
        for item in node:
            out.append(item)
        return out
    if isinstance(node, Vector):
        out.extend(node.items)
    elif isinstance(node, Range):
        out.extend([node.start, node.end] + ([node.step] if node.step is not None else []))
    elif isinstance(node, Unary):
        out.append(node.operand)
    elif isinstance(node, Binary):
        out.extend([node.left, node.right])
    elif isinstance(node, Ternary):
        out.extend([node.cond, node.if_true, node.if_false])
    elif isinstance(node, Index):
        out.extend([node.target, node.index])
    elif isinstance(node, Member):
        out.append(node.target)
    elif isinstance(node, Call):
        out.append(node.name)
        out.extend(a.value for a in node.args)
    elif isinstance(node, LetExpr):
        out.extend(v for _, v in node.bindings)
        out.append(node.body)
    elif isinstance(node, FunctionLiteral):
        out.extend(p.default for p in node.params if p.default is not None)
        out.append(node.body)
    elif isinstance(node, Comprehension):
        out.extend(v for _, v in node.bindings)
        if node.cond is not None:
            out.append(node.cond)
        if node.body is not None:
            out.append(node.body)
        if node.orelse is not None:
            out.append(node.orelse)
    elif isinstance(node, Assign):
        out.append(node.value)
    elif isinstance(node, (ModuleDef, FunctionDef)):
        out.extend(p.default for p in node.params if p.default is not None)
        out.append(node.body)
    elif isinstance(node, ModuleCall):
        out.extend(a.value for a in node.args)
        out.extend(node.children)
    elif isinstance(node, Block):
        out.extend(node.body)
    elif isinstance(node, IfStmt):
        out.append(node.cond)
        out.append(node.then)
        if node.orelse is not None:
            out.append(node.orelse)
    elif isinstance(node, (ForStmt, LetStmt)):
        out.extend(v for _, v in node.bindings)
        out.append(node.body)
    return out
