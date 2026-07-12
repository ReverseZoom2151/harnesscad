"""OpenECAD editable CAD-script format (Yuan, Shi & Huang, 2024).

OpenECAD's distinct contribution is not its VLM but the *editable CAD code
representation* the model emits: a Python-like script of named CAD operations
whose variable names encode the model's Sketch-Extrusion structure, making the
program directly human-editable (paper Sec. 4.2, Algorithm 1 & 3, Table 1).

This module is the deterministic core of that representation. It provides:

* a small value/expression model (numbers, vectors, strings, booleans, empty
  lists and *variable references*),
* the command vocabulary of Table 1,
* an **emitter** that renders a program to the exact OpenECAD code style, and
* a **parser** (built on the stdlib :mod:`ast`) that reads such code back, so
  the representation round-trips.

Command vocabulary (Table 1)::

    add_line(start, end)                        curve
    add_arc(start, end, mid)                    curve
    add_circle(center, radius)                  curve
    add_loop(curves)                            sketch helper (Algorithm 1)
    add_profile(loops)                          sketch helper
    add_sketchplane(origin, normal, x_axis)     reference plane, direct
    add_sketchplane_ref(extrude, origin, type,  reference plane from an
                        ...optional values)     existing extrude feature
    add_sketch(sketchplane, profile, position, size)
    add_extrude(sketch, operation, type, extent_one, extent_two)

Unlike DeepCAD's flat, scaled 16-vector (``reconstruction.deepcad_command_spec``)
this is a higher-level *named-variable script* whose coordinates live in absolute
space and whose reference planes may depend on previously created features -- the
properties the paper credits for editability. Pure and deterministic; the VLM
that generates such code is out of scope.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass, field

# --- command vocabulary (Table 1) ------------------------------------------
ADD_LINE = "add_line"
ADD_ARC = "add_arc"
ADD_CIRCLE = "add_circle"
ADD_LOOP = "add_loop"
ADD_PROFILE = "add_profile"
ADD_SKETCHPLANE = "add_sketchplane"
ADD_SKETCHPLANE_REF = "add_sketchplane_ref"
ADD_SKETCH = "add_sketch"
ADD_EXTRUDE = "add_extrude"

CURVE_FUNCS: frozenset[str] = frozenset({ADD_LINE, ADD_ARC, ADD_CIRCLE})
COMMAND_FUNCS: frozenset[str] = frozenset({
    ADD_LINE, ADD_ARC, ADD_CIRCLE, ADD_LOOP, ADD_PROFILE,
    ADD_SKETCHPLANE, ADD_SKETCHPLANE_REF, ADD_SKETCH, ADD_EXTRUDE,
})

# Reference-plane types for add_sketchplane_ref (paper Sec. 4.2 / Algorithm 3).
REF_SAMEPLANE = "sameplan"
REF_TOPFACE = "topface"
REF_SIDEFACE = "line"  # a side face, identified by a boundary line


@dataclass(frozen=True)
class Ref:
    """A reference to a previously assigned variable (e.g. ``Curves0_0``)."""

    name: str

    def __post_init__(self):
        if not self.name or not self.name.isidentifier():
            raise ValueError(f"invalid variable reference: {self.name!r}")


@dataclass(frozen=True)
class Arg:
    """A call argument: keyword ``key=value`` when *key* is set, else positional."""

    value: object
    key: str | None = None


@dataclass(frozen=True)
class Call:
    """A single CAD operation call, e.g. ``add_line(start=[...], end=[...])``."""

    func: str
    args: tuple[Arg, ...] = ()

    def __post_init__(self):
        if self.func not in COMMAND_FUNCS:
            raise ValueError(f"unknown OpenECAD command: {self.func!r}")

    def keyword(self, key: str, default: object = None) -> object:
        """Return the value of keyword argument *key* (or *default*)."""
        for a in self.args:
            if a.key == key:
                return a.value
        return default

    def positional(self) -> tuple[object, ...]:
        """The positional (non-keyword) argument values, in order."""
        return tuple(a.value for a in self.args if a.key is None)


@dataclass(frozen=True)
class Assign:
    """One statement ``t0, t1, ... = v0, v1, ...``.

    A command assignment has a single target and a single :class:`Call` value
    (``Sketch0 = add_sketch(...)``). A tuple init (``Loops0, Curves0_0 = [], []``)
    has parallel targets and literal values.
    """

    targets: tuple[str, ...]
    values: tuple[object, ...]

    def __post_init__(self):
        if not self.targets:
            raise ValueError("assignment needs at least one target")
        if len(self.targets) != len(self.values):
            raise ValueError("targets and values count mismatch")

    @property
    def call(self) -> Call | None:
        """The RHS :class:`Call` when this is a single command assignment."""
        if len(self.values) == 1 and isinstance(self.values[0], Call):
            return self.values[0]
        return None


@dataclass
class Program:
    """An ordered OpenECAD script (a list of :class:`Assign` statements)."""

    statements: list[Assign] = field(default_factory=list)

    def calls(self):
        """Yield ``(target_name, Call)`` for every single-command assignment."""
        for st in self.statements:
            c = st.call
            if c is not None:
                yield st.targets[0], c

    def calls_of(self, *funcs: str) -> list[Call]:
        """All calls whose function name is one of *funcs*, in program order."""
        wanted = set(funcs)
        return [c for _, c in self.calls() if c.func in wanted]


# --- emitter ----------------------------------------------------------------
def emit_value(value: object) -> str:
    """Render one value to OpenECAD-style source text."""
    if isinstance(value, Ref):
        return value.name
    if isinstance(value, bool):  # before int: bool is an int subclass
        return "True" if value else "False"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        # Canonical: integral floats render as ``N.0`` (paper writes ``1000.``).
        if value == int(value):
            return f"{int(value)}.0"
        return repr(value)
    if isinstance(value, str):
        return '"' + value + '"'
    if isinstance(value, (list, tuple)):
        return "[" + ", ".join(emit_value(v) for v in value) + "]"
    raise TypeError(f"cannot emit value of type {type(value).__name__}")


def emit_call(call: Call) -> str:
    parts = []
    for a in call.args:
        text = emit_value(a.value)
        parts.append(f"{a.key}={text}" if a.key is not None else text)
    return f"{call.func}(" + ", ".join(parts) + ")"


def emit_assign(assign: Assign) -> str:
    lhs = ", ".join(assign.targets)
    rhs = ", ".join(
        emit_call(v) if isinstance(v, Call) else emit_value(v)
        for v in assign.values)
    return f"{lhs} = {rhs}"


def emit(program: Program) -> str:
    """Render a whole program to source text (one statement per line)."""
    return "\n".join(emit_assign(st) for st in program.statements)


# --- parser (stdlib ast) ----------------------------------------------------
def _node_to_value(node: ast.AST) -> object:
    if isinstance(node, ast.Constant):
        return node.value
    if isinstance(node, ast.Name):
        return Ref(node.id)
    if isinstance(node, ast.List):
        return [_node_to_value(e) for e in node.elts]
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
        inner = _node_to_value(node.operand)
        if isinstance(inner, (int, float)) and not isinstance(inner, bool):
            return -inner
        raise ValueError("unary minus on non-numeric value")
    if isinstance(node, ast.Call):
        return _node_to_call(node)
    raise ValueError(f"unsupported expression node: {type(node).__name__}")


def _node_to_call(node: ast.Call) -> Call:
    if not isinstance(node.func, ast.Name):
        raise ValueError("only simple function-name calls are supported")
    args: list[Arg] = []
    for a in node.args:
        args.append(Arg(_node_to_value(a)))
    for kw in node.keywords:
        if kw.arg is None:
            raise ValueError("**kwargs not supported")
        args.append(Arg(_node_to_value(kw.value), kw.arg))
    return Call(node.func.id, tuple(args))


def _target_names(target: ast.AST) -> tuple[str, ...]:
    if isinstance(target, ast.Name):
        return (target.id,)
    if isinstance(target, ast.Tuple):
        names = []
        for e in target.elts:
            if not isinstance(e, ast.Name):
                raise ValueError("assignment targets must be plain names")
            names.append(e.id)
        return tuple(names)
    raise ValueError("unsupported assignment target")


def parse(code: str) -> Program:
    """Parse OpenECAD script *code* into a :class:`Program`.

    Accepts the subset of Python the format uses: single or tuple assignments
    whose right-hand side is a command call or literal(s). Raises
    :class:`ValueError` on anything outside that grammar.
    """
    module = ast.parse(code)
    statements: list[Assign] = []
    for node in module.body:
        if not isinstance(node, ast.Assign):
            raise ValueError(f"only assignments are allowed, got "
                             f"{type(node).__name__}")
        if len(node.targets) != 1:
            raise ValueError("chained assignment (a = b = ...) not supported")
        targets = _target_names(node.targets[0])
        rhs = node.value
        if isinstance(rhs, ast.Tuple):
            values = tuple(_node_to_value(e) for e in rhs.elts)
        else:
            values = (_node_to_value(rhs),)
        statements.append(Assign(targets, values))
    return Program(statements)


def round_trip(program: Program) -> Program:
    """Convenience: ``parse(emit(program))``."""
    return parse(emit(program))
