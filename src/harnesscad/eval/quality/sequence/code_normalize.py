"""Safe AST-only normalization of generated CAD Python."""

from __future__ import annotations

import ast


def normalize_cad_code(source: str) -> str:
    tree = ast.parse(source)
    imports, body = [], []
    seen_imports = set()
    for node in tree.body:
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            text = ast.unparse(node)
            if text not in seen_imports:
                imports.append(node)
                seen_imports.add(text)
        else:
            body.append(node)
    imports.sort(key=ast.unparse)
    tree.body = imports + body
    ast.fix_missing_locations(tree)
    return ast.unparse(tree).strip() + "\n"
