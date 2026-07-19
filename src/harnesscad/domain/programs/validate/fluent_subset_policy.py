"""Straight-line fluent-subset policy for model-generated CAD scripts.

A fluent-CAD runtime executes model-generated fluent scripts (``Part``/``Sketch``
chains) in-process, and its gate is stricter than an import/call blocklist: it
constrains the *shape of the program itself*. A generated script may only be a
straight-line sequence of assignments and fluent expression chains -- no function
or class definitions, no loops, no ``try``, no ``with``, no ``lambda``, no
``raise``, no ``global``. The insight is that a CAD build script has no business
owning control flow: every legitimate output of the code-generation prompt is a
linear recipe, so any control-flow node in the AST is either a prompt violation
or an attempted escape, and both get refused before ``exec``.

The full rule set:

* **statement-shape allowlist** -- every node in :data:`BLOCKED_STATEMENTS` is
  refused;
* **single import form** -- exactly ``from <module> import <allowed names>``,
  no ``import x``, no relative imports, no aliasing (``as``), no names outside
  the allowlist;
* **name hygiene** -- no dunder-prefixed names anywhere, no references to
  process/OS module names even as bare identifiers (so ``os`` smuggled in via a
  variable is refused, not just ``import os``);
* **attribute hygiene** -- no dunder attributes, and the *root* of every
  attribute chain (walked through call results, so ``x().os.system`` is seen)
  must not be a blocked module name;
* **call hygiene** -- no ``eval``/``exec``/``getattr``/``open``/... by name,
  whether bare or as an attribute.

It also ships the matching execution namespace recipe
(:func:`execution_namespace`): a minimal builtins dict whose ``__import__``
only honours the declared module/names, so even code that slips the static
gate cannot import anything else at runtime.

Relation to neighbours: this is deliberately distinct from
:mod:`harnesscad.domain.programs.validate.code_safety` (the Text23D gate),
which allows arbitrary Python statements and blocks only dangerous
imports/calls -- the right contract for full CadQuery/FreeCAD scripts with
loops and functions. This policy is the tighter contract for *fluent-chain*
generation prompts, where the grammar itself is part of the spec and any
control flow is evidence of a prompt escape. Both are static; neither replaces
:mod:`harnesscad.domain.programs.runtime.module_sandbox` (runtime module
isolation).

Stdlib-only (``ast``), deterministic.

Public API
----------
``check_fluent_script``, ``assert_fluent_script``, ``execution_namespace``
``FluentPolicy``, ``PolicyViolation``, ``PolicyReport``, ``FluentPolicyError``
``BLOCKED_STATEMENTS``, ``BLOCKED_CALLS``, ``BLOCKED_MODULE_NAMES``
"""

from __future__ import annotations

import argparse
import ast
from dataclasses import dataclass, field
from typing import Dict, FrozenSet, List, Optional, Sequence, Tuple

__all__ = [
    "BLOCKED_STATEMENTS",
    "BLOCKED_CALLS",
    "BLOCKED_MODULE_NAMES",
    "FluentPolicy",
    "PolicyViolation",
    "PolicyReport",
    "FluentPolicyError",
    "check_fluent_script",
    "assert_fluent_script",
    "execution_namespace",
]

#: Statement/expression node types a straight-line fluent script never needs.
BLOCKED_STATEMENTS: Tuple[type, ...] = (
    ast.AsyncFor,
    ast.AsyncFunctionDef,
    ast.AsyncWith,
    ast.Await,
    ast.ClassDef,
    ast.Delete,
    ast.For,
    ast.FunctionDef,
    ast.Global,
    ast.Lambda,
    ast.Match,
    ast.Nonlocal,
    ast.Raise,
    ast.Try,
    ast.While,
    ast.With,
    ast.Yield,
    ast.YieldFrom,
)

#: Callables refused by name, bare or as an attribute.
BLOCKED_CALLS: FrozenSet[str] = frozenset({
    "__import__", "breakpoint", "compile", "delattr", "dir", "eval", "exec",
    "getattr", "globals", "input", "locals", "open", "setattr", "vars",
})

#: Module-ish names refused wherever they appear (name or attribute root).
BLOCKED_MODULE_NAMES: FrozenSet[str] = frozenset({
    "builtins", "ctypes", "importlib", "os", "pathlib", "pickle", "requests",
    "shutil", "socket", "subprocess", "sys", "urllib",
})

#: Builtins the execution namespace keeps (everything else is absent).
_SAFE_BUILTINS: Tuple[str, ...] = ("abs", "max", "min", "round")


class FluentPolicyError(ValueError):
    """The script violates the fluent-subset policy."""


@dataclass(frozen=True)
class FluentPolicy:
    """What one generation prompt permits.

    ``module`` is the only importable module; ``names`` the only importable
    symbols from it (e.g. ``("Part", "Sketch")``).
    """

    module: str
    names: Tuple[str, ...]

    def __post_init__(self) -> None:
        if not self.module:
            raise FluentPolicyError("Policy requires a module name.")
        if not self.names:
            raise FluentPolicyError("Policy requires at least one allowed name.")


@dataclass(frozen=True)
class PolicyViolation:
    code: str
    message: str
    line: int = 0


@dataclass
class PolicyReport:
    ok: bool
    violations: List[PolicyViolation] = field(default_factory=list)

    def codes(self) -> List[str]:
        return [v.code for v in self.violations]


def _call_name(node: ast.AST) -> Optional[str]:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return node.attr
    return None


def _attribute_root(node: ast.AST) -> Optional[str]:
    """The base identifier of an attribute chain, walked through calls."""
    current = node
    while isinstance(current, ast.Attribute):
        current = current.value
    if isinstance(current, ast.Call):
        return _attribute_root(current.func)
    if isinstance(current, ast.Name):
        return current.id
    return None


def check_fluent_script(code: str, policy: FluentPolicy) -> PolicyReport:
    """Statically check *code* against the fluent-subset policy.

    Collects EVERY violation (with line numbers) rather than stopping at the
    first, so a generation loop can hand the full list back as one diagnostic.
    """
    violations: List[PolicyViolation] = []

    try:
        tree = ast.parse(code, filename="<generated-cad-code>", mode="exec")
    except SyntaxError as exc:
        return PolicyReport(ok=False, violations=[PolicyViolation(
            code="syntax", line=exc.lineno or 0,
            message="not valid Python: %s" % (exc.msg,),
        )])

    allowed = set(policy.names)

    for node in ast.walk(tree):
        line = getattr(node, "lineno", 0)

        if isinstance(node, BLOCKED_STATEMENTS):
            violations.append(PolicyViolation(
                code="blocked-statement", line=line,
                message="%s is outside the fluent subset"
                        % type(node).__name__,
            ))
            continue

        if isinstance(node, ast.Import):
            violations.append(PolicyViolation(
                code="import-form", line=line,
                message="only 'from %s import %s' is allowed"
                        % (policy.module, ", ".join(policy.names)),
            ))
            continue

        if isinstance(node, ast.ImportFrom):
            if node.level != 0 or node.module != policy.module:
                violations.append(PolicyViolation(
                    code="import-module", line=line,
                    message="imports may only come from '%s'" % policy.module,
                ))
                continue
            imported = {alias.name for alias in node.names}
            extra = sorted(imported - allowed)
            if not imported or extra:
                violations.append(PolicyViolation(
                    code="import-name", line=line,
                    message="only %s may be imported (got: %s)"
                            % (", ".join(policy.names),
                               ", ".join(sorted(imported)) or "nothing"),
                ))
            if any(alias.asname is not None for alias in node.names):
                violations.append(PolicyViolation(
                    code="import-alias", line=line,
                    message="imports may not be aliased",
                ))
            continue

        if isinstance(node, ast.Name):
            if node.id.startswith("__") or node.id in BLOCKED_MODULE_NAMES:
                violations.append(PolicyViolation(
                    code="blocked-name", line=line,
                    message="reference to %r is refused" % node.id,
                ))
            continue

        if isinstance(node, ast.Attribute):
            if node.attr.startswith("__"):
                violations.append(PolicyViolation(
                    code="dunder-attribute", line=line,
                    message="dunder attribute %r is refused" % node.attr,
                ))
            root = _attribute_root(node)
            if root in BLOCKED_MODULE_NAMES:
                violations.append(PolicyViolation(
                    code="blocked-root", line=line,
                    message="attribute access on %r is refused" % root,
                ))
            continue

        if isinstance(node, ast.Call):
            name = _call_name(node.func)
            if name in BLOCKED_CALLS:
                violations.append(PolicyViolation(
                    code="blocked-call", line=line,
                    message="call to %r is refused" % name,
                ))
            root = _attribute_root(node.func)
            if root in BLOCKED_MODULE_NAMES:
                violations.append(PolicyViolation(
                    code="blocked-root", line=line,
                    message="call on %r is refused" % root,
                ))

    return PolicyReport(ok=not violations, violations=violations)


def assert_fluent_script(code: str, policy: FluentPolicy) -> ast.Module:
    """Raise :class:`FluentPolicyError` on the first violation; return the AST."""
    report = check_fluent_script(code, policy)
    if not report.ok:
        first = report.violations[0]
        raise FluentPolicyError(
            "line %d [%s]: %s" % (first.line, first.code, first.message))
    return ast.parse(code, filename="<generated-cad-code>", mode="exec")


def execution_namespace(policy: FluentPolicy,
                        resolver=None) -> Dict[str, object]:
    """Minimal namespace for executing a policy-checked script.

    Only :data:`_SAFE_BUILTINS` plus the literals survive; ``__import__`` is a
    guard that honours exactly ``from <policy.module> import <policy.names>``.
    *resolver* maps a module name to the module object (defaults to
    ``importlib.import_module``); inject a stub in tests so nothing real is
    imported.
    """
    import importlib

    resolve = resolver if resolver is not None else importlib.import_module
    allowed = set(policy.names)

    def guarded_import(name, globals=None, locals=None, fromlist=(), level=0):
        del globals, locals
        if level == 0 and name == policy.module and set(fromlist) <= allowed:
            return resolve(name)
        raise ImportError(
            "only 'from %s import %s' is allowed"
            % (policy.module, ", ".join(policy.names)))

    safe_builtins: Dict[str, object] = {
        "__import__": guarded_import,
        "False": False,
        "None": None,
        "True": True,
    }
    import builtins as _builtins
    for name in _SAFE_BUILTINS:
        safe_builtins[name] = getattr(_builtins, name)
    return {"__builtins__": safe_builtins, "__name__": "__main__"}


# ── selfcheck ───────────────────────────────────────────────────────


def selfcheck(verbose: bool = False) -> bool:
    """Accept a straight-line fluent script; refuse every escape shape."""
    checks: List[Tuple[str, bool]] = []
    policy = FluentPolicy(module="opencad", names=("Part", "Sketch"))

    good = (
        "from opencad import Part, Sketch\n"
        "profile = Sketch().rect(30, 20).circle(4, subtract=True)\n"
        "part = Part().extrude(profile, depth=8).fillet(radius=0.5)\n"
        "final = part.cut(Part().cylinder(2, 8))\n"
    )
    checks.append(("fluent chain accepted", check_fluent_script(good, policy).ok))

    def refused(name: str, code: str, expect: str) -> None:
        report = check_fluent_script(code, policy)
        checks.append((name, not report.ok and expect in report.codes()))

    refused("loop refused",
            "from opencad import Part\nfor i in range(3):\n    pass\n",
            "blocked-statement")
    refused("def refused",
            "from opencad import Part\ndef f():\n    return 1\n",
            "blocked-statement")
    refused("lambda refused",
            "from opencad import Part\nf = lambda: 1\n",
            "blocked-statement")
    refused("try refused",
            "from opencad import Part\ntry:\n    x = 1\nexcept Exception:\n"
            "    x = 2\n",
            "blocked-statement")
    refused("with refused",
            "from opencad import Part\nwith x:\n    pass\n",
            "blocked-statement")
    refused("bare import refused", "import opencad\n", "import-form")
    refused("foreign module refused", "from os import path\n", "import-module")
    refused("foreign name refused",
            "from opencad import Part, Runtime\n", "import-name")
    refused("alias refused",
            "from opencad import Part as P\n", "import-alias")
    refused("blocked call refused",
            "from opencad import Part\nx = eval('1')\n", "blocked-call")
    refused("blocked name refused",
            "from opencad import Part\ny = os\n", "blocked-name")
    refused("dunder attribute refused",
            "from opencad import Part\nz = Part().__class__\n",
            "dunder-attribute")
    refused("attr-root through call refused",
            "from opencad import Part\nsys.exit()\n", "blocked-name")
    refused("syntax error refused", "part = = Part()", "syntax")

    # All violations are collected, not just the first.
    multi = check_fluent_script(
        "import os\nx = eval('1')\nfor i in []:\n    pass\n", policy)
    checks.append(("all violations collected", len(multi.violations) >= 3))

    # assert_ raises with the first violation.
    try:
        assert_fluent_script("import os\n", policy)
        checks.append(("assert raises", False))
    except FluentPolicyError:
        checks.append(("assert raises", True))

    # Execution namespace: guarded import honours only the policy.
    sentinel = object()
    ns = execution_namespace(policy, resolver=lambda name: sentinel)
    imp = ns["__builtins__"]["__import__"]  # type: ignore[index]
    checks.append(("guarded import allows policy",
                   imp("opencad", fromlist=("Part",)) is sentinel))
    try:
        imp("os", fromlist=("path",))
        checks.append(("guarded import refuses others", False))
    except ImportError:
        checks.append(("guarded import refuses others", True))
    checks.append(("no open in namespace",
                   "open" not in ns["__builtins__"]))  # type: ignore[operator]

    ok = all(passed for _, passed in checks)
    if verbose:
        for name, passed in checks:
            print("  %-32s %s" % (name, "ok" if passed else "FAIL"))
        print("fluent_subset_policy selfcheck: %s" % ("ok" if ok else "FAILED"))
    return ok


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m harnesscad.domain.programs.validate.fluent_subset_policy",
        description="Straight-line fluent-subset policy for generated CAD "
                    "scripts.",
    )
    parser.add_argument("source", nargs="?",
                        help="path to a generated script to check")
    parser.add_argument("--module", default="opencad",
                        help="the single allowed import module")
    parser.add_argument("--names", default="Part,Sketch",
                        help="comma-separated allowed import names")
    parser.add_argument("--selfcheck", action="store_true",
                        help="run the policy self-check (no real data)")
    args = parser.parse_args(list(argv) if argv is not None else None)

    if args.selfcheck:
        return 0 if selfcheck(verbose=True) else 1

    if not args.source:
        parser.print_help()
        return 0

    policy = FluentPolicy(
        module=args.module,
        names=tuple(n.strip() for n in args.names.split(",") if n.strip()),
    )
    with open(args.source, "r", encoding="utf-8") as handle:
        code = handle.read()
    report = check_fluent_script(code, policy)
    for violation in report.violations:
        print("line %d [%s]: %s"
              % (violation.line, violation.code, violation.message))
    print("ok" if report.ok else "REFUSED (%d violations)"
          % len(report.violations))
    return 0 if report.ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
