"""AST-based CadQuery-code validity check (execution-free API + arity checking).

Paper 171 -- *Text-to-CadQuery* -- evaluates generated code with an **Invalid Rate**
(IR): "the percentage of generated CadQuery scripts that fail during execution, due
to syntax errors, incomplete expressions, or invalid operations" (Sec. 4.2). The
paper measures IR by actually *running* every script in a subprocess. This module
provides the deterministic, execution-free counterpart: it decides, purely by
static analysis of the source, whether a CadQuery script is *plausibly* valid --
catching the syntactic and API-shape errors that dominate the paper's IR without
importing CadQuery or invoking the OpenCascade kernel.

It parses the source with the stdlib ``ast`` module and reports:

  * **syntax errors / incomplete expressions** -- ``ast.parse`` failure (the paper's
    first IR category);
  * **unknown API calls** -- chained ``.method(...)`` names outside the CadQuery
    subset defined in :mod:`programs.t2cq_ast` (``CHAIN_METHODS``);
  * **argument-arity errors** -- calls whose positional-argument count falls outside
    the method's arity band (the "incomplete expressions / invalid operations" the
    paper cites);
  * **bad Workplane usage** -- ``cq.Workplane(...)`` called with the wrong number or
    type of arguments, or with an unknown basis-plane name;
  * **missing import** -- no ``import cadquery`` header.

Each finding is an :class:`Issue` with a stable ``code`` and a source ``line``, so
the check is reproducible. :func:`invalid_rate` aggregates :func:`is_valid` over a
corpus to reproduce the paper's IR metric statically.

Pure stdlib (``ast`` only). Complements :func:`programs.t2cq_ast.validate` (which
checks an already-parsed :class:`~programs.t2cq_ast.CqProgram`): this module works
on arbitrary source text, including code an external LLM emitted directly.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass

from harnesscad.domain.programs.ast.t2cq_ast import CHAIN_METHODS, WORKPLANES


@dataclass(frozen=True)
class Issue:
    """A single static-validity finding."""

    code: str          # stable machine-readable category
    message: str
    line: int


# Non-chain helper calls that are legitimate but are not sketch/solid methods
# (so their arity is not checked against CHAIN_METHODS).
_ALLOWED_FREE_CALLS = frozenset({"export", "show", "exporters"})


def check_code(code: str) -> list[Issue]:
    """Statically analyse CadQuery source and return a list of :class:`Issue`.

    An empty list means the script passes all static checks (it is *plausibly*
    executable). This never runs the code.
    """
    try:
        tree = ast.parse(code)
    except SyntaxError as exc:
        return [Issue("syntax", f"syntax error: {exc.msg}", exc.lineno or 0)]

    issues: list[Issue] = []

    # (1) require an `import cadquery` header.
    has_import = any(
        (isinstance(n, ast.Import) and any(a.name.split(".")[0] == "cadquery"
                                           for a in n.names))
        or (isinstance(n, ast.ImportFrom) and (n.module or "").split(".")[0] == "cadquery")
        for n in ast.walk(tree)
    )
    if not has_import:
        issues.append(Issue("missing_import", "no `import cadquery` found", 1))

    # (2) walk every call expression.
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if not isinstance(func, ast.Attribute):
            continue
        method = func.attr
        line = getattr(node, "lineno", 0)

        # cq.Workplane("XY") -- the chain root.
        if method == "Workplane":
            _check_workplane(node, issues, line)
            continue
        if method in _ALLOWED_FREE_CALLS:
            continue
        if method not in CHAIN_METHODS:
            issues.append(Issue("unknown_api", f"unknown CadQuery method {method!r}", line))
            continue
        lo, hi = CHAIN_METHODS[method]
        n_pos = len(node.args)
        if not (lo <= n_pos <= hi):
            issues.append(Issue(
                "arity",
                f"{method}() expects {lo}..{hi} positional args, got {n_pos}", line))

    return issues


def _check_workplane(node: ast.Call, issues: list[Issue], line: int) -> None:
    if len(node.args) > 1:
        issues.append(Issue("workplane_arity",
                            f"Workplane() takes 0 or 1 args, got {len(node.args)}", line))
        return
    if node.args:
        arg = node.args[0]
        if isinstance(arg, ast.Constant) and isinstance(arg.value, str):
            if arg.value not in WORKPLANES:
                issues.append(Issue("workplane_plane",
                                    f"unknown workplane {arg.value!r}", line))
        elif not isinstance(arg, ast.Constant):
            # Non-literal (e.g. a variable) -- can't verify statically; allowed.
            return


def is_valid(code: str) -> bool:
    """True iff :func:`check_code` finds no issues."""
    return not check_code(code)


def invalid_rate(codes: list[str]) -> float:
    """Static analogue of the paper's Invalid Rate: fraction of scripts with issues.

    Returns a value in ``[0, 1]`` (the paper reports it as a percentage). An empty
    corpus yields ``0.0``.
    """
    if not codes:
        return 0.0
    invalid = sum(1 for c in codes if not is_valid(c))
    return invalid / len(codes)
