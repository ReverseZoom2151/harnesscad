"""CadQuery-subset structured program (AST): build / serialize / parse / validate.

Paper 171 -- *Text-to-CadQuery: A New Paradigm for CAD Generation with Scalable
Large Model Capabilities* (Xie & Ju, ASU, 2025). The paper's central idea is to
make the text-to-CAD **target** ordinary CadQuery Python code (a real, pure-Python
CAD API) instead of a bespoke command sequence, so that pretrained LLMs -- already
fluent in Python -- can emit executable 3D-generative code directly (Sec. 1, and
the modeling examples in Appendix A.2 / A.3).

This module is the deterministic core the paper leaves implicit: a *structured
representation* of the CadQuery subset the paper actually uses. Rather than treat
generated CadQuery as an opaque string, we model it as a small typed AST -- a list
of variable assignments, each a **method chain** rooted at ``cq.Workplane(plane)``
or a previously-defined variable, e.g.::

    part_1 = cq.Workplane("XY").moveTo(0.0, 0.0).lineTo(0.75, 0.0).close().extrude(0.5)
    result = part_1

The subset of chain methods mirrors exactly the operations that appear in the
paper's CadQuery examples (Appendix A.2/A.3) and its stated abstractions "box, arc,
circle, and extrude" (Sec. 2): ``box``, ``cylinder``, ``sphere``, ``rect``,
``circle``, ``moveTo``, ``lineTo``, ``line``, ``threePointArc``, ``radiusArc``,
``close``, ``extrude``, ``cut``, ``cutBlind``, ``union``, ``intersect``, ``fillet``,
``chamfer``, ``hole``, ``center``, ``workplane``, ``faces``, ``edges``,
``pushPoints``, ``translate``, ``rotate``.

The module provides four deterministic capabilities:

  * **build**  -- construct programs with small dataclasses (:class:`Workplane`,
    :class:`VarRef`, :class:`Call`, :class:`Chain`, :class:`Assign`,
    :class:`CqProgram`);
  * **serialize** -- emit runnable CadQuery Python source (:func:`serialize`);
  * **parse** -- read CadQuery source back into a :class:`CqProgram` via the stdlib
    ``ast`` module (:func:`parse_program`), so source round-trips through the AST;
  * **validate** -- structural checks (:func:`validate`): known methods, argument
    arity, workplane plane names, and use-before-definition of variables.

Pure stdlib (``ast`` only). No CadQuery/OCCT is imported or executed -- this is a
static representation, distinct from :mod:`backends.cadquery_backend` (which drives
the real kernel) and from :mod:`datagen.cadquery_codegen` (a one-shot CISP-dict
string emitter with no AST, parser or validator). It is the CadQuery analogue of
:mod:`programs.openecad_script` (paper 138's editable DSL, a different language).
"""

from __future__ import annotations

import ast
from dataclasses import dataclass, field

# --- allowed CadQuery-subset chain methods -> (min positional, max positional) --
# The subset covers exactly the operations used in the paper's CadQuery examples
# (Appendix A.2/A.3) plus the robust-arc alternative it recommends (radiusArc,
# Appendix A.4). Tuple-valued arguments (e.g. threePointArc points) count as one
# positional argument each.
CHAIN_METHODS: dict[str, tuple[int, int]] = {
    "box": (3, 4),
    "cylinder": (2, 3),
    "sphere": (1, 2),
    "rect": (2, 3),
    "circle": (1, 1),
    # CadQuery's signature is moveTo(x=0, y=0), so nought/one/two positional
    # args are all legal; the corpus in resources/cadbible/cadquery-contrib
    # calls it with one. A (2, 2) arity here rejected valid programs.
    "moveTo": (0, 2),
    "lineTo": (2, 2),
    "line": (2, 2),
    "vLine": (1, 1),
    "hLine": (1, 1),
    "threePointArc": (2, 2),
    "radiusArc": (2, 2),
    "sagittaArc": (2, 2),
    "close": (0, 0),
    "extrude": (1, 2),
    "revolve": (0, 4),
    "cut": (1, 1),
    "cutBlind": (1, 1),
    "cutThruAll": (0, 1),
    "union": (1, 1),
    "intersect": (1, 1),
    "fillet": (1, 1),
    "chamfer": (1, 2),
    "hole": (1, 2),
    "center": (2, 2),
    "workplane": (0, 3),
    "faces": (0, 1),
    "edges": (0, 1),
    "vertices": (0, 1),
    "pushPoints": (1, 1),
    "translate": (1, 1),
    "rotate": (3, 3),
    "rotateAboutCenter": (2, 2),
    "mirror": (0, 3),
}

# Canonical CadQuery basis planes accepted by ``cq.Workplane("...")``.
WORKPLANES: frozenset[str] = frozenset({"XY", "YZ", "XZ", "YX", "ZY", "ZX", "front",
                                        "back", "left", "right", "top", "bottom"})


@dataclass(frozen=True)
class Workplane:
    """Chain root ``cq.Workplane(plane)``."""

    plane: str = "XY"


@dataclass(frozen=True)
class VarRef:
    """Reference to a previously-assigned variable (chain root or call argument)."""

    name: str


@dataclass(frozen=True)
class Call:
    """A single chained method call ``.method(*args)``.

    ``args`` is a tuple of literal values (``int``/``float``/``str``), point tuples
    (``tuple`` of numbers), or :class:`VarRef` (for boolean operands like
    ``.union(part_2)``).
    """

    method: str
    args: tuple = ()


@dataclass(frozen=True)
class Chain:
    """A method chain: a ``root`` followed by zero or more :class:`Call`."""

    root: object                       # Workplane | VarRef
    calls: tuple[Call, ...] = ()


@dataclass(frozen=True)
class Assign:
    """A statement ``var = <chain>``."""

    var: str
    chain: Chain


@dataclass(frozen=True)
class CqProgram:
    """An ordered list of assignments plus the name of the final result variable."""

    statements: tuple[Assign, ...] = ()
    result_var: str | None = None


# --- serialization ----------------------------------------------------------
def _fmt_number(value) -> str:
    if isinstance(value, bool):
        return "True" if value else "False"
    if isinstance(value, int):
        return str(value)
    return repr(float(value))


def format_arg(arg) -> str:
    """Serialize a single call argument to a Python source fragment."""
    if isinstance(arg, VarRef):
        return arg.name
    if isinstance(arg, str):
        return repr(arg)
    if isinstance(arg, tuple):
        inner = ", ".join(format_arg(a) for a in arg)
        return f"({inner})" if len(arg) != 1 else f"({inner},)"
    if isinstance(arg, bool):
        return "True" if arg else "False"
    if isinstance(arg, (int, float)):
        return _fmt_number(arg)
    raise TypeError(f"unsupported argument type: {type(arg).__name__}")


def _serialize_root(root) -> str:
    if isinstance(root, Workplane):
        return f'cq.Workplane("{root.plane}")'
    if isinstance(root, VarRef):
        return root.name
    raise TypeError(f"unsupported chain root: {type(root).__name__}")


def serialize_chain(chain: Chain) -> str:
    """Serialize a :class:`Chain` to a single-line CadQuery expression."""
    parts = [_serialize_root(chain.root)]
    for call in chain.calls:
        args = ", ".join(format_arg(a) for a in call.args)
        parts.append(f".{call.method}({args})")
    return "".join(parts)


def serialize(program: CqProgram, import_header: bool = True) -> str:
    """Serialize a whole program to runnable CadQuery Python source."""
    lines: list[str] = []
    if import_header:
        lines.append("import cadquery as cq")
    for stmt in program.statements:
        lines.append(f"{stmt.var} = {serialize_chain(stmt.chain)}")
    if program.result_var is not None:
        assigned = {s.var for s in program.statements}
        if program.result_var not in assigned:
            raise ValueError(f"result_var {program.result_var!r} is never assigned")
        if not program.statements or program.statements[-1].var != program.result_var:
            lines.append(f"result = {program.result_var}")
    return "\n".join(lines) + "\n"


# --- parsing (source -> AST) ------------------------------------------------
def _arg_from_ast(node) -> object:
    """Convert an ``ast`` argument node into an AST argument value."""
    if isinstance(node, ast.Name):
        return VarRef(node.id)
    if isinstance(node, ast.Tuple):
        return tuple(_arg_from_ast(e) for e in node.elts)
    try:
        return ast.literal_eval(node)
    except (ValueError, SyntaxError, TypeError) as exc:
        raise ValueError(f"unsupported argument expression: {ast.dump(node)}") from exc


def _chain_from_ast(node) -> Chain:
    """Unwind a (possibly chained) call/attribute expression into a :class:`Chain`."""
    calls: list[Call] = []
    cur = node
    while True:
        if isinstance(cur, ast.Call) and isinstance(cur.func, ast.Attribute):
            method = cur.func.attr
            if cur.keywords:
                raise ValueError(f"keyword arguments unsupported in call to {method!r}")
            args = tuple(_arg_from_ast(a) for a in cur.args)
            # cq.Workplane(...) is the root, not a chain method.
            if (method == "Workplane" and isinstance(cur.func.value, ast.Name)
                    and cur.func.value.id == "cq"):
                plane = args[0] if args else "XY"
                if not isinstance(plane, str):
                    raise ValueError("Workplane plane argument must be a string")
                calls.reverse()
                return Chain(Workplane(plane), tuple(calls))
            calls.append(Call(method, args))
            cur = cur.func.value
            continue
        if isinstance(cur, ast.Name):
            calls.reverse()
            return Chain(VarRef(cur.id), tuple(calls))
        raise ValueError(f"unsupported chain expression: {ast.dump(cur)}")


def parse_program(code: str) -> CqProgram:
    """Parse CadQuery source into a :class:`CqProgram` using the stdlib ``ast``.

    Accepts an optional ``import cadquery as cq`` header and any number of
    single-target assignments of method chains. The last assignment's variable is
    taken as :attr:`CqProgram.result_var`. Raises :class:`SyntaxError` on invalid
    Python and :class:`ValueError` on constructs outside the CadQuery subset.
    """
    module = ast.parse(code)
    statements: list[Assign] = []
    for node in module.body:
        if isinstance(node, ast.Import):
            continue  # tolerate the `import cadquery as cq` header
        if not isinstance(node, ast.Assign):
            raise ValueError(f"unsupported top-level statement: {ast.dump(node)}")
        if len(node.targets) != 1 or not isinstance(node.targets[0], ast.Name):
            raise ValueError("only single-name assignment targets are supported")
        statements.append(Assign(node.targets[0].id, _chain_from_ast(node.value)))
    result_var = statements[-1].var if statements else None
    return CqProgram(tuple(statements), result_var)


# --- validation -------------------------------------------------------------
def validate(program: CqProgram) -> list[str]:
    """Structural validation: return a list of human-readable error strings.

    Checks: (1) each chain method is in the CadQuery subset; (2) positional arg
    counts fall within the method's arity; (3) ``Workplane`` roots use a known
    basis plane; (4) every :class:`VarRef` (as a root or argument) refers to a
    variable assigned *earlier* in the program (no use-before-definition). An empty
    list means the program is structurally valid.
    """
    errors: list[str] = []
    defined: set[str] = set()

    def check_varref(ref: VarRef, where: str) -> None:
        if ref.name not in defined:
            errors.append(f"{where}: variable {ref.name!r} used before assignment")

    def check_arg(arg, where: str) -> None:
        if isinstance(arg, VarRef):
            check_varref(arg, where)
        elif isinstance(arg, tuple):
            for a in arg:
                check_arg(a, where)

    for stmt in program.statements:
        chain = stmt.chain
        if isinstance(chain.root, Workplane):
            if chain.root.plane not in WORKPLANES:
                errors.append(
                    f"{stmt.var}: unknown workplane {chain.root.plane!r}")
        elif isinstance(chain.root, VarRef):
            check_varref(chain.root, stmt.var)
        for call in chain.calls:
            if call.method not in CHAIN_METHODS:
                errors.append(f"{stmt.var}: unknown method {call.method!r}")
            else:
                lo, hi = CHAIN_METHODS[call.method]
                n = len(call.args)
                if not (lo <= n <= hi):
                    errors.append(
                        f"{stmt.var}: {call.method}() takes {lo}..{hi} args, got {n}")
            for arg in call.args:
                check_arg(arg, f"{stmt.var}.{call.method}")
        defined.add(stmt.var)

    if program.result_var is not None and program.result_var not in defined:
        errors.append(f"result_var {program.result_var!r} is never assigned")
    return errors


def is_valid(program: CqProgram) -> bool:
    """Convenience boolean wrapper over :func:`validate`."""
    return not validate(program)
