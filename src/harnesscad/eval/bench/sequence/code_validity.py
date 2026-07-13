"""Static safety/output contract checks and VSR aggregation; never executes code."""

from __future__ import annotations

import ast

_FORBIDDEN = (ast.With, ast.AsyncWith, ast.Try, ast.Raise, ast.Lambda,
              ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Delete,
              ast.Global, ast.Nonlocal, ast.Await, ast.Yield, ast.YieldFrom)
_BANNED_ROOTS = {"os", "sys", "subprocess", "socket", "requests", "urllib",
                 "pathlib", "shutil", "builtins"}


def validate_cad_code(source):
    try:
        tree = ast.parse(source)
    except SyntaxError as exc:
        return {"valid": False, "category": "syntax", "message": str(exc)}
    for node in ast.walk(tree):
        if isinstance(node, _FORBIDDEN):
            return {"valid": False, "category": "unsafe-ast",
                    "message": type(node).__name__}
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            names = [alias.name.split(".")[0] for alias in node.names] \
                if isinstance(node, ast.Import) else [(node.module or "").split(".")[0]]
            if any(name in _BANNED_ROOTS for name in names):
                return {"valid": False, "category": "unsafe-import",
                        "message": ",".join(names)}
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) \
                and node.func.id in {"eval", "exec", "compile", "open", "__import__"}:
            return {"valid": False, "category": "unsafe-call", "message": node.func.id}
    assigned = any(isinstance(node, ast.Assign)
                   and any(isinstance(target, ast.Name) and target.id == "solid"
                           for target in node.targets) for node in tree.body)
    if not assigned:
        return {"valid": False, "category": "output-contract",
                "message": "missing final solid assignment"}
    return {"valid": True, "category": "static-valid", "message": ""}


def valid_syntax_rate(results):
    values = tuple(results)
    return sum(bool(item.get("valid")) for item in values) / len(values) if values else None
