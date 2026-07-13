"""AST metrics for reusable parameters and high-level CAD abstractions."""

from __future__ import annotations

import ast
from collections import Counter


def code_modularity(source):
    tree = ast.parse(source)
    literals = Counter()
    names_defined, names_used, calls = set(), Counter(), Counter()
    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            literals[float(node.value)] += 1
        elif isinstance(node, ast.Name):
            if isinstance(node.ctx, ast.Store): names_defined.add(node.id)
            elif isinstance(node.ctx, ast.Load): names_used[node.id] += 1
        elif isinstance(node, ast.Call):
            name = node.func.attr if isinstance(node.func, ast.Attribute) else \
                node.func.id if isinstance(node.func, ast.Name) else "?"
            calls[name] += 1
    high = sum(calls[name] for name in ("rect", "box", "cylinder"))
    low = sum(calls[name] for name in ("segment", "line", "arc", "circle"))
    return {"repeated_literals": tuple(sorted(value for value, count in literals.items()
                                               if count > 1)),
            "reused_variables": tuple(sorted(name for name, count in names_used.items()
                                              if name in names_defined and count > 1)),
            "dead_definitions": tuple(sorted(names_defined - names_used.keys())),
            "high_level_calls": high, "low_level_calls": low,
            "abstraction_ratio": high/(high+low) if high+low else None}
