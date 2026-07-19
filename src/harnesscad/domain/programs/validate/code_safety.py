"""Static safety gate for model-generated CAD scripts (Text23D).

**Text23D** is a text->3D backend that executes model-written CAD code in-process
against CadQuery and FreeCAD. Before any ``exec``, it runs a static AST allowlist
(``backend/app/validation.py``): the generated code must define a known entry
point, may only import from a per-kernel allowlist, and must not touch dangerous
builtins (``eval``/``exec``/``open``/...) or process/OS modules
(``os``/``subprocess``/``socket``/...). This is the deterministic *pre-execution*
gate that makes running an LLM's code merely risky rather than reckless.

This module reimplements that check as a reusable, kernel-parameterised safety
gate. It is deliberately distinct from:

* :mod:`harnesscad.domain.programs.runtime.module_sandbox` -- that isolates a run
  at *runtime* (snapshot/restore of ``sys.modules``); this rejects unsafe code
  *before* it ever runs, purely by reading the AST.
* the ``programs/validate`` API-correctness checks (``cadquery_validity`` etc.) --
  those ask "will this build valid geometry?"; this asks "is this code safe to
  execute at all?".

Two entry points: :func:`check_cad_code` returns a structured
:class:`SafetyReport`; :func:`assert_cad_code_safe` raises
:class:`CodeSafetyError` on the first violation (Text23D's original contract).

Pure stdlib (``ast``), deterministic.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass, field
from typing import List, Optional, Set

__all__ = [
    "CodeSafetyError",
    "CADQUERY_IMPORT_ROOTS",
    "BUILD123D_IMPORT_ROOTS",
    "FREECAD_IMPORT_ROOTS",
    "BLOCKED_ATTRS",
    "BLOCKED_CALLS",
    "BLOCKED_NAMES",
    "SafetyViolation",
    "SafetyReport",
    "allowed_import_roots",
    "check_cad_code",
    "assert_cad_code_safe",
]

CADQUERY_IMPORT_ROOTS: Set[str] = {"cadquery", "math"}
BUILD123D_IMPORT_ROOTS: Set[str] = {"build123d", "math"}
FREECAD_IMPORT_ROOTS: Set[str] = {"Draft", "FreeCAD", "Mesh", "Part", "Sketcher", "math"}

BLOCKED_CALLS: Set[str] = {
    "__import__",
    "breakpoint",  # drops into pdb == arbitrary interactive execution
    "compile",
    "eval",
    "exec",
    "getattr",  # dynamic attribute access defeats the static dunder block below
    "globals",
    "input",
    "locals",
    "open",
    "setattr",
    "vars",
}
#: Attribute names whose ACCESS is an introspection escape, independent of the
#: object they hang off. ``().__class__.__base__.__subclasses__()`` reaches
#: ``Popen`` without ever naming ``subprocess`` -- an import-allowlist AST gate
#: provably cannot see that by watching imports, so the traversal itself is what
#: gets blocked. Real CAD-generation code never walks the type hierarchy or a
#: function's globals/closure; blocking these has no legitimate false positive.
BLOCKED_ATTRS: Set[str] = {
    "__base__",
    "__bases__",
    "__class__",
    "__closure__",
    "__code__",
    "__dict__",
    "__func__",
    "__globals__",
    "__mro__",
    "__self__",
    "__subclasses__",
}
BLOCKED_NAMES: Set[str] = {
    "__builtins__",
    "__file__",
    "__loader__",
    "__package__",
    "__spec__",
    "ctypes",
    "multiprocessing",
    "os",
    "pathlib",
    "pickle",
    "shutil",
    "socket",
    "subprocess",
    "sys",
    "threading",
}

_KERNEL_ROOTS = {
    "cadquery": CADQUERY_IMPORT_ROOTS,
    "build123d": BUILD123D_IMPORT_ROOTS,
    "freecad": FREECAD_IMPORT_ROOTS,
}


@dataclass(frozen=True)
class SafetyViolation:
    """One reason the code is unsafe to execute."""

    code: str
    message: str


@dataclass
class SafetyReport:
    """The outcome of the static safety gate."""

    ok: bool
    violations: List[SafetyViolation] = field(default_factory=list)

    def codes(self) -> List[str]:
        return [v.code for v in self.violations]

    def first_message(self) -> Optional[str]:
        return self.violations[0].message if self.violations else None


class CodeSafetyError(ValueError):
    """Raised by :func:`assert_cad_code_safe` for unsafe code."""


def allowed_import_roots(kernel: str) -> Set[str]:
    """Return the import allowlist for a CAD kernel (defaults to cadquery)."""
    return _KERNEL_ROOTS.get(kernel, CADQUERY_IMPORT_ROOTS)


def _attribute_root(node: ast.Attribute) -> Optional[str]:
    current: ast.AST = node
    while isinstance(current, ast.Attribute):
        current = current.value
    if isinstance(current, ast.Name):
        return current.id
    return None


def check_cad_code(
    code: str,
    *,
    kernel: str = "cadquery",
    required_def: Optional[str] = "build_model",
    allow_async: bool = False,
) -> SafetyReport:
    """Statically vet ``code`` and collect every safety violation.

    * ``syntax`` -- the code does not parse.
    * ``missing_entrypoint`` -- ``required_def`` (if given) is not defined.
    * ``async_forbidden`` -- an ``async def`` when ``allow_async`` is False.
    * ``import_not_allowed`` -- an import outside the kernel allowlist.
    * ``blocked_name`` -- a reference to a blocked module/dunder.
    * ``blocked_call`` -- a call to a blocked builtin, or a call routed through a
      blocked module (``os.system(...)``).
    """
    violations: List[SafetyViolation] = []

    try:
        tree = ast.parse(code)
    except SyntaxError as exc:
        return SafetyReport(ok=False, violations=[SafetyViolation("syntax", f"Python syntax error: {exc}")])

    roots = allowed_import_roots(kernel)

    if required_def is not None:
        has_def = any(
            isinstance(node, ast.FunctionDef) and node.name == required_def
            for node in tree.body
        )
        if not has_def:
            violations.append(
                SafetyViolation("missing_entrypoint", f"Generated code must define {required_def}().")
            )

    for node in ast.walk(tree):
        if isinstance(node, ast.AsyncFunctionDef) and not allow_async:
            violations.append(
                SafetyViolation("async_forbidden", "Async functions are not allowed in CAD scripts.")
            )

        if isinstance(node, ast.Import):
            for alias in node.names:
                root = alias.name.split(".", 1)[0]
                if root not in roots:
                    violations.append(
                        SafetyViolation("import_not_allowed", f"Import is not allowed: {alias.name}")
                    )

        if isinstance(node, ast.ImportFrom):
            root = (node.module or "").split(".", 1)[0]
            if root not in roots:
                violations.append(
                    SafetyViolation("import_not_allowed", f"Import is not allowed: {node.module}")
                )

        if isinstance(node, ast.Name) and node.id in BLOCKED_NAMES:
            violations.append(SafetyViolation("blocked_name", f"Blocked name used: {node.id}"))

        if isinstance(node, ast.Attribute) and node.attr in BLOCKED_ATTRS:
            violations.append(
                SafetyViolation(
                    "blocked_attr",
                    f"Blocked introspection attribute: .{node.attr}",
                )
            )

        if isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name) and node.func.id in BLOCKED_CALLS:
                violations.append(
                    SafetyViolation("blocked_call", f"Blocked function call: {node.func.id}()")
                )
            if isinstance(node.func, ast.Attribute):
                attr_root = _attribute_root(node.func)
                if attr_root in BLOCKED_NAMES:
                    violations.append(
                        SafetyViolation(
                            "blocked_call",
                            f"Blocked module call through {attr_root}.{node.func.attr}().",
                        )
                    )

    return SafetyReport(ok=not violations, violations=violations)


def assert_cad_code_safe(
    code: str,
    *,
    kernel: str = "cadquery",
    required_def: Optional[str] = "build_model",
    allow_async: bool = False,
) -> None:
    """Raise :class:`CodeSafetyError` on the first violation (Text23D contract)."""
    report = check_cad_code(
        code, kernel=kernel, required_def=required_def, allow_async=allow_async
    )
    if not report.ok:
        raise CodeSafetyError(report.first_message() or "unsafe CAD code")
