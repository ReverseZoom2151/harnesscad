"""Static validity checker for OpenSCAD source -- a local compile gate.

ScadLM's agent loop asks the ``openscad`` binary "does this compile?" and, when
it does not, re-prompts the model with ``no_compile_prompt`` ("The code you
provided does not compile. Identify what is wrong..."). That single bit of
feedback is the whole verifier, and it needs an external binary and a temp
directory.

This module reproduces that gate deterministically and locally, on top of the
AST from :mod:`programs.scadlm_ast`. It reports:

  * **syntax errors** -- with line/column, straight from the parser;
  * **unknown module / function calls** -- checked against the OpenSCAD builtin
    vocabulary (the very cheat sheet ScadLM pastes into its prompt) plus the
    program's own ``module``/``function`` definitions;
  * **bad arguments** -- named arguments a builtin does not accept, too many
    positional arguments, duplicate parameter names, positional-after-named;
  * **undefined variables** -- scope-aware (OpenSCAD hoists assignments within a
    scope), knowing module/function parameters, ``for``/``let`` bindings, list
    comprehension bindings and the ``$``-special variables;
  * **semantic warnings** the compiler accepts but which are almost always the
    bug an LLM just made: a ``difference()``/``intersection()`` with fewer than
    two children, children attached to a primitive (silently ignored by
    OpenSCAD), a zero or negative literal dimension, a program that defines
    modules but instantiates no geometry, and unused module definitions.

:func:`is_valid` gives the ScadLM compile bit; :func:`format_report` gives the
"identify what is wrong" text to feed back into a repair prompt. Deterministic:
issues come out in a fixed order, no clock, no randomness, nothing executed.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

from harnesscad.domain.programs.ast.openscad import (
    Assign,
    Binary,
    Block,
    Bool,
    Call,
    Comprehension,
    ForStmt,
    FunctionDef,
    FunctionLiteral,
    IfStmt,
    Include,
    Index,
    LetExpr,
    LetStmt,
    Member,
    ModuleCall,
    ModuleDef,
    Name,
    NoOp,
    Num,
    Range,
    ScadSyntaxError,
    Str,
    Ternary,
    Unary,
    Undef,
    Vector,
    parse,
)

__all__ = [
    "Issue",
    "BUILTIN_MODULES",
    "BUILTIN_FUNCTION_NAMES",
    "SPECIAL_VARIABLES",
    "check",
    "is_valid",
    "format_report",
]


@dataclass(frozen=True)
class Issue:
    severity: str      # "error" | "warning"
    code: str
    message: str
    line: int = 0
    column: int = 0

    def render(self) -> str:
        where = " (line %d)" % self.line if self.line else ""
        return "[%s] %s: %s%s" % (self.severity, self.code, self.message, where)


# name -> (positional parameter names in order, extra accepted named arguments)
BUILTIN_MODULES: Dict[str, Tuple[Tuple[str, ...], Tuple[str, ...]]] = {
    # 3D
    "cube": (("size", "center"), ()),
    "sphere": (("r",), ("d", "$fn", "$fa", "$fs")),
    "cylinder": (("h", "r1", "r2", "center"),
                 ("r", "d", "d1", "d2", "$fn", "$fa", "$fs")),
    "polyhedron": (("points", "faces", "convexity"), ("triangles",)),
    "import": (("file", "convexity"), ("layer", "center", "dpi", "$fn")),
    "surface": (("file", "center", "invert", "convexity"), ()),
    "linear_extrude": (("height", "center", "convexity", "twist", "slices"),
                       ("scale", "$fn", "$fa", "$fs")),
    "rotate_extrude": (("angle", "convexity"), ("$fn", "$fa", "$fs")),
    # 2D
    "circle": (("r",), ("d", "$fn", "$fa", "$fs")),
    "square": (("size", "center"), ()),
    "polygon": (("points", "paths", "convexity"), ()),
    "text": (("text", "size", "font"),
             ("halign", "valign", "spacing", "direction", "language", "script",
              "$fn")),
    "projection": (("cut",), ()),
    "offset": (("r", "delta", "chamfer"), ("$fn", "$fa", "$fs")),
    # transforms
    "translate": (("v",), ()),
    "rotate": (("a", "v"), ()),
    "scale": (("v",), ()),
    "resize": (("newsize", "auto", "convexity"), ()),
    "mirror": (("v",), ()),
    "multmatrix": (("m",), ()),
    "color": (("c", "alpha"), ()),
    # booleans / grouping
    "union": ((), ()),
    "difference": ((), ()),
    "intersection": ((), ()),
    "hull": ((), ()),
    "minkowski": (("convexity",), ()),
    "group": ((), ()),
    "render": (("convexity",), ()),
    "children": (("index",), ()),
    "echo": ((), ()),
    "assert": (("condition", "message"), ()),
}

BUILTIN_FUNCTION_NAMES: Set[str] = {
    "abs", "sign", "sin", "cos", "tan", "asin", "acos", "atan", "atan2",
    "floor", "round", "ceil", "ln", "len", "log", "pow", "sqrt", "exp",
    "rands", "min", "max", "norm", "cross", "concat", "lookup", "str", "chr",
    "ord", "search", "version", "version_num", "parent_module", "is_undef",
    "is_bool", "is_num", "is_string", "is_list", "is_function", "assert",
    "echo",
}

SPECIAL_VARIABLES: Set[str] = {
    "$fa", "$fs", "$fn", "$t", "$vpr", "$vpt", "$vpd", "$vpf", "$children",
    "$preview", "PI", "undef",
}

_PRIMITIVE_MODULES = {"cube", "sphere", "cylinder", "polyhedron", "square",
                      "circle", "polygon", "text", "import", "surface"}
_VARARG_MODULES = {"echo", "union", "difference", "intersection", "hull",
                   "group"}
_SIZE_ARGS = {"cube": ("size",), "sphere": ("r", "d"),
              "cylinder": ("h", "r", "r1", "r2", "d", "d1", "d2"),
              "circle": ("r", "d"), "square": ("size",)}


class _Checker:
    def __init__(self) -> None:
        self.issues: List[Issue] = []
        self.modules: Dict[str, ModuleDef] = {}
        self.functions: Dict[str, FunctionDef] = {}
        self.called_modules: Set[str] = set()
        self.geometry_count = 0

    # -- reporting --
    def error(self, code: str, message: str) -> None:
        self.issues.append(Issue("error", code, message))

    def warn(self, code: str, message: str) -> None:
        self.issues.append(Issue("warning", code, message))

    # -- entry --
    def run(self, stmts: Sequence[Any]) -> List[Issue]:
        self._collect_defs(stmts)
        scope = set(SPECIAL_VARIABLES)
        self._check_stmts(stmts, scope, in_module=False)
        for name, defn in self.modules.items():
            if name not in self.called_modules:
                self.warn("unused-module", "module %r is defined but never used"
                          % name)
        if self.geometry_count == 0:
            self.warn("no-geometry",
                      "program instantiates no geometry (nothing would render)")
        return self.issues

    def _collect_defs(self, stmts: Sequence[Any]) -> None:
        """Definitions are visible program-wide within their scope; hoist them."""
        for s in stmts:
            if isinstance(s, ModuleDef):
                self.modules[s.name] = s
                self._collect_defs([s.body])
            elif isinstance(s, FunctionDef):
                self.functions[s.name] = s
            elif isinstance(s, Block):
                self._collect_defs(s.body)
            elif isinstance(s, (ForStmt, LetStmt)):
                self._collect_defs([s.body])
            elif isinstance(s, IfStmt):
                self._collect_defs([s.then] + ([s.orelse] if s.orelse else []))
            elif isinstance(s, ModuleCall):
                self._collect_defs(s.children)

    # -- statements --
    def _hoisted(self, stmts: Sequence[Any], scope: Set[str]) -> Set[str]:
        inner = set(scope)
        for s in stmts:
            if isinstance(s, Assign):
                inner.add(s.name)
        return inner

    def _check_stmts(self, stmts: Sequence[Any], scope: Set[str],
                     in_module: bool) -> None:
        inner = self._hoisted(stmts, scope)
        for s in stmts:
            self._check_stmt(s, inner, in_module)

    def _check_stmt(self, stmt: Any, scope: Set[str], in_module: bool) -> None:
        if isinstance(stmt, (NoOp, Include)):
            return
        if isinstance(stmt, Assign):
            self._check_expr(stmt.value, scope)
            scope.add(stmt.name)
            return
        if isinstance(stmt, Block):
            self._check_stmts(stmt.body, scope, in_module)
            return
        if isinstance(stmt, ModuleDef):
            self._check_params(stmt.name, stmt.params, scope)
            body_scope = set(scope) | {p.name for p in stmt.params} | {"$children"}
            self._check_stmt(stmt.body, body_scope, in_module=True)
            return
        if isinstance(stmt, FunctionDef):
            self._check_params(stmt.name, stmt.params, scope)
            body_scope = set(scope) | {p.name for p in stmt.params}
            self._check_expr(stmt.body, body_scope)
            return
        if isinstance(stmt, IfStmt):
            self._check_expr(stmt.cond, scope)
            self._check_stmt(stmt.then, set(scope), in_module)
            if stmt.orelse is not None:
                self._check_stmt(stmt.orelse, set(scope), in_module)
            return
        if isinstance(stmt, (ForStmt, LetStmt)):
            body_scope = set(scope)
            for name, value in stmt.bindings:
                self._check_expr(value, body_scope)
                body_scope.add(name)
            self._check_stmt(stmt.body, body_scope, in_module)
            return
        if isinstance(stmt, ModuleCall):
            self._check_module_call(stmt, scope, in_module)
            return
        self.error("unknown-statement",
                   "unsupported statement %s" % type(stmt).__name__)

    def _check_params(self, owner: str, params, scope: Set[str]) -> None:
        seen: Set[str] = set()
        for p in params:
            if p.name in seen:
                self.error("duplicate-parameter",
                           "%r declares parameter %r twice" % (owner, p.name))
            seen.add(p.name)
            if p.default is not None:
                self._check_expr(p.default, scope)

    def _check_module_call(self, call: ModuleCall, scope: Set[str],
                           in_module: bool) -> None:
        name = call.name
        self.called_modules.add(name)

        for a in call.args:
            self._check_expr(a.value, scope)

        seen_named = False
        for a in call.args:
            if a.name is not None:
                seen_named = True
            elif seen_named:
                self.error("positional-after-named",
                           "%s() has a positional argument after a named one" % name)
                break

        if name in BUILTIN_MODULES:
            positional, extra = BUILTIN_MODULES[name]
            accepted = set(positional) | set(extra)
            n_pos = sum(1 for a in call.args if a.name is None)
            if name not in _VARARG_MODULES and n_pos > len(positional):
                self.error("too-many-arguments",
                           "%s() accepts %d positional argument(s), got %d"
                           % (name, len(positional), n_pos))
            for a in call.args:
                if a.name is not None and a.name not in accepted and \
                        not a.name.startswith("$"):
                    self.error("unknown-argument",
                               "%s() has no argument %r" % (name, a.name))
            if name == "children" and not in_module:
                self.error("children-outside-module",
                           "children() used outside a module definition")
            self._check_dimensions(call)
            if name in _PRIMITIVE_MODULES:
                self.geometry_count += 1
                if call.children:
                    self.warn("ignored-children",
                              "children attached to primitive %s() are ignored"
                              % name)
            if name in ("difference", "intersection") and \
                    len(_geometry_children(call)) < 2:
                self.warn("degenerate-boolean",
                          "%s() with fewer than two children has no effect" % name)
            if name in ("union", "difference", "intersection", "hull",
                        "minkowski") and not call.children:
                self.warn("empty-boolean", "%s() has no children" % name)
        elif name in self.modules:
            defn = self.modules[name]
            known = {p.name for p in defn.params}
            n_pos = sum(1 for a in call.args if a.name is None)
            if n_pos > len(defn.params):
                self.error("too-many-arguments",
                           "module %s() takes %d parameter(s), got %d positional"
                           % (name, len(defn.params), n_pos))
            for a in call.args:
                if a.name is not None and a.name not in known and \
                        not a.name.startswith("$"):
                    self.error("unknown-argument",
                               "module %s() has no parameter %r" % (name, a.name))
            self.geometry_count += 1
        else:
            self.error("unknown-module", "call to undefined module %r" % name)

        for child in call.children:
            self._check_stmt(child, set(scope), in_module)

    def _check_dimensions(self, call: ModuleCall) -> None:
        names = _SIZE_ARGS.get(call.name)
        if not names:
            return
        positional, _extra = BUILTIN_MODULES[call.name]
        for i, a in enumerate(call.args):
            arg_name = a.name
            if arg_name is None:
                arg_name = positional[i] if i < len(positional) else None
            if arg_name not in names:
                continue
            for value in _literal_numbers(a.value):
                if value <= 0:
                    self.warn("nonpositive-dimension",
                              "%s() argument %s is %s (non-positive dimension)"
                              % (call.name, arg_name,
                                 int(value) if value == int(value) else value))

    # -- expressions --
    def _check_expr(self, node: Any, scope: Set[str]) -> None:
        if node is None or isinstance(node, (Num, Str, Bool, Undef)):
            return
        if isinstance(node, Name):
            if node.ident not in scope and node.ident not in self.functions and \
                    node.ident not in BUILTIN_FUNCTION_NAMES and \
                    not node.ident.startswith("$"):
                self.error("undefined-variable",
                           "variable %r is used but never defined" % node.ident)
            return
        if isinstance(node, Vector):
            for item in node.items:
                self._check_expr(item, scope)
            return
        if isinstance(node, Range):
            for part in (node.start, node.end, node.step):
                if part is not None:
                    self._check_expr(part, scope)
            return
        if isinstance(node, Unary):
            self._check_expr(node.operand, scope)
            return
        if isinstance(node, Binary):
            self._check_expr(node.left, scope)
            self._check_expr(node.right, scope)
            return
        if isinstance(node, Ternary):
            self._check_expr(node.cond, scope)
            self._check_expr(node.if_true, scope)
            self._check_expr(node.if_false, scope)
            return
        if isinstance(node, Index):
            self._check_expr(node.target, scope)
            self._check_expr(node.index, scope)
            return
        if isinstance(node, Member):
            self._check_expr(node.target, scope)
            if node.name not in ("x", "y", "z"):
                self.error("bad-member",
                           "dot access %r is not one of .x/.y/.z" % node.name)
            return
        if isinstance(node, LetExpr):
            inner = set(scope)
            for name, value in node.bindings:
                self._check_expr(value, inner)
                inner.add(name)
            self._check_expr(node.body, inner)
            return
        if isinstance(node, FunctionLiteral):
            self._check_params("function literal", node.params, scope)
            self._check_expr(node.body, set(scope) | {p.name for p in node.params})
            return
        if isinstance(node, Comprehension):
            inner = set(scope)
            for name, value in node.bindings:
                self._check_expr(value, inner)
                inner.add(name)
            if node.cond is not None:
                self._check_expr(node.cond, inner)
            if node.body is not None:
                self._check_expr(node.body, inner)
            if node.orelse is not None:
                self._check_expr(node.orelse, inner)
            return
        if isinstance(node, Call):
            self._check_call(node, scope)
            return
        self.error("unknown-expression",
                   "unsupported expression %s" % type(node).__name__)

    def _check_call(self, node: Call, scope: Set[str]) -> None:
        for a in node.args:
            self._check_expr(a.value, scope)
        callee = node.name
        if not isinstance(callee, Name):
            self._check_expr(callee, scope)
            return
        name = callee.ident
        if name in self.functions:
            defn = self.functions[name]
            known = {p.name for p in defn.params}
            n_pos = sum(1 for a in node.args if a.name is None)
            if n_pos > len(defn.params):
                self.error("too-many-arguments",
                           "function %s() takes %d parameter(s), got %d positional"
                           % (name, len(defn.params), n_pos))
            for a in node.args:
                if a.name is not None and a.name not in known:
                    self.error("unknown-argument",
                               "function %s() has no parameter %r" % (name, a.name))
            return
        if name in BUILTIN_FUNCTION_NAMES or name in scope:
            return
        self.error("unknown-function", "call to undefined function %r" % name)


def _geometry_children(call: ModuleCall) -> List[Any]:
    out: List[Any] = []
    for c in call.children:
        if isinstance(c, Block):
            out.extend(_geometry_children(ModuleCall("", [], c.body)))
        elif isinstance(c, NoOp):
            continue
        else:
            out.append(c)
    return out


def _literal_numbers(node: Any) -> List[float]:
    """Literal numbers directly present in ``node`` (no variable resolution)."""
    if isinstance(node, Num):
        return [node.value]
    if isinstance(node, Vector):
        out: List[float] = []
        for item in node.items:
            if isinstance(item, Num):
                out.append(item.value)
        return out
    if isinstance(node, Unary) and node.op == "-" and isinstance(node.operand, Num):
        return [-node.operand.value]
    return []


def check(source: str) -> List[Issue]:
    """Check OpenSCAD source; returns issues in a deterministic order."""
    try:
        stmts = parse(source)
    except ScadSyntaxError as exc:
        return [Issue("error", "syntax", exc.message, exc.line, exc.column)]
    return _Checker().run(stmts)


def is_valid(source: str) -> bool:
    """True when the source has no *errors* (warnings are allowed).

    This is the local stand-in for ScadLM's ``openscad`` compile bit.
    """
    return not any(i.severity == "error" for i in check(source))


def format_report(issues: Sequence[Issue]) -> str:
    """Render issues as feedback text for a repair prompt."""
    if not issues:
        return "No issues found."
    return "\n".join(i.render() for i in issues)
