"""Completion-budget and structural-completeness diagnostics."""

from __future__ import annotations

import ast
from dataclasses import dataclass


@dataclass(frozen=True)
class GenerationStatus:
    complete: bool
    truncated: bool
    issues: tuple[str, ...]
    route: str


def assess_generation(text, *, finish_reason, output_tokens, maximum_tokens,
                      require_solid=False):
    issues = []
    truncated = finish_reason in {"length", "max_tokens"} or output_tokens >= maximum_tokens
    if truncated:
        issues.append("token-truncation")
    try:
        tree = ast.parse(text)
    except SyntaxError:
        tree = None
        issues.append("incomplete-syntax")
    if require_solid and tree is not None:
        found = any(isinstance(node, ast.Assign)
                    and any(isinstance(target, ast.Name) and target.id == "solid"
                            for target in node.targets)
                    for node in tree.body)
        if not found:
            issues.append("missing-solid")
    route = "long-context-route" if truncated else "retry" if issues else "accept"
    return GenerationStatus(not issues, truncated, tuple(issues), route)
