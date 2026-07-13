"""CAD-code complexity records and explicit overflow routing."""

from __future__ import annotations

import ast
from dataclasses import dataclass


@dataclass(frozen=True)
class CodeComplexity:
    tokens: int
    statements: int
    calls: int
    max_depth: int
    bin: str


def analyze_code(source, count_tokens=lambda text: len(text.split())):
    tree = ast.parse(source)
    depth = 0
    for node in ast.walk(tree):
        current = 0
        probe = node
        while hasattr(probe, "parent"):
            current += 1
            probe = probe.parent
        depth = max(depth, current)
        for child in ast.iter_child_nodes(node):
            child.parent = node
    tokens = count_tokens(source)
    label = "long" if tokens > 3000 else "medium" if tokens > 1000 else "short"
    return CodeComplexity(tokens, len(tree.body),
                          sum(isinstance(node, ast.Call) for node in ast.walk(tree)),
                          depth, label)


def overflow_route(tokens, maximum, policy="reject"):
    if policy not in {"reject", "chunk", "long-context-route"}:
        raise ValueError("unknown overflow policy")
    return "accept" if tokens <= maximum else policy
