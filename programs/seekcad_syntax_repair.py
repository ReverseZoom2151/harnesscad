"""Pattern-template syntax repair for generated SSR CAD code.

Deterministic re-implementation of the "pattern template Q" from "Seek-CAD" (Li
et al., ICLR 2026), Section 3.1(4)/(5).

Initially generated CAD code I_0 is occasionally subject to compilation failures
attributed to *syntax* errors E (as opposed to logical/geometric errors).  Before
handing code to the geometry kernel, the paper applies a pattern template Q that
automatically rectifies E in I_0 -- specifically "issues such as mismatched
parentheses and incorrect capitalization of variable names":

    I_0 ~ P(I_0 | Q)                                                    (Sec 3.1)

and the same check re-runs on each refinement iterate I_k (I_k ~ P(I_k | Q)).

This module implements a deterministic, model-free Q:

  * balance round brackets by trimming unmatched trailing ``)`` and appending
    ``)`` for unmatched opening ``(`` (ignoring brackets inside string
    literals);
  * normalise the capitalisation of variable-name references to the casing they
    were first *assigned* with, fixing later mis-cased uses (a common LLM
    slip).

Repairs are reported so callers can log what Q changed.  The template is a
surface-level fixer; it does not attempt to correct logical errors.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Dict, List, Tuple

_ASSIGN_RE = re.compile(r"^(\s*)([A-Za-z_]\w*)(\s*=)(?!=)")
_IDENT_RE = re.compile(r"[A-Za-z_]\w*")


@dataclass
class RepairReport:
    """What pattern template Q changed."""

    code: str
    paren_fixes: int = 0
    case_fixes: int = 0
    renamed: List[Tuple[str, str]] = field(default_factory=list)

    @property
    def changed(self) -> bool:
        return self.paren_fixes > 0 or self.case_fixes > 0


def _mask_strings(line: str) -> str:
    """Replace string-literal contents with spaces so bracket/identifier scans
    ignore them.  Handles single- and double-quoted literals without escapes
    (sufficient for the CAD DSL of Listing 1)."""
    out = list(line)
    quote = None
    for i, ch in enumerate(line):
        if quote is None:
            if ch in ("'", '"'):
                quote = ch
        else:
            if ch == quote:
                quote = None
            else:
                out[i] = " "
    return "".join(out)


def balance_parentheses(code: str) -> Tuple[str, int]:
    """Balance round brackets line by line, ignoring those in string literals.

    Returns the repaired code and the number of bracket insert/trim fixes.
    """
    fixed_lines: List[str] = []
    fixes = 0
    for line in code.splitlines():
        masked = _mask_strings(line)
        depth = 0
        drop_positions = set()
        for i, ch in enumerate(masked):
            if ch == "(":
                depth += 1
            elif ch == ")":
                if depth == 0:
                    drop_positions.add(i)  # unmatched closing -> remove
                else:
                    depth -= 1
        if drop_positions:
            line = "".join(c for i, c in enumerate(line) if i not in drop_positions)
            fixes += len(drop_positions)
        if depth > 0:
            line = line + (")" * depth)  # unmatched opening -> append
            fixes += depth
        fixed_lines.append(line)
    sep = "\n" if "\n" in code or code.endswith("\n") else "\n"
    result = sep.join(fixed_lines)
    if code.endswith("\n"):
        result += "\n"
    return result, fixes


def normalise_identifier_case(code: str) -> Tuple[str, int, List[Tuple[str, str]]]:
    """Fix later mis-cased uses of assigned variable names.

    The canonical casing of a variable is the one used at its first assignment
    (``name = ...``).  Any later identifier that differs only by case is
    rewritten to the canonical form.  Returns repaired code, fix count, and the
    list of (wrong, canonical) rewrites performed.
    """
    canonical: Dict[str, str] = {}  # lower -> canonical casing
    # First pass: collect canonical names from assignments.
    for line in code.splitlines():
        masked = _mask_strings(line)
        m = _ASSIGN_RE.match(masked)
        if m:
            name = m.group(2)
            canonical.setdefault(name.lower(), name)

    fixes = 0
    renamed: List[Tuple[str, str]] = []
    out_lines: List[str] = []
    for line in code.splitlines():
        masked = _mask_strings(line)
        # Build the rewritten line using match spans from the masked view so we
        # never touch identifiers inside string literals.
        pieces: List[str] = []
        last = 0
        for m in _IDENT_RE.finditer(masked):
            tok = line[m.start():m.end()]
            canon = canonical.get(tok.lower())
            pieces.append(line[last:m.start()])
            # A token immediately followed by '(' is a call target (a class /
            # feature constructor such as Sketch(...), Extrude(...)), not a
            # variable reference, and must not be re-cased to a same-spelled
            # variable name.
            is_call = m.end() < len(masked) and masked[m.end()] == "("
            if canon is not None and tok != canon and not is_call:
                pieces.append(canon)
                fixes += 1
                renamed.append((tok, canon))
            else:
                pieces.append(tok)
            last = m.end()
        pieces.append(line[last:])
        out_lines.append("".join(pieces))
    result = "\n".join(out_lines)
    if code.endswith("\n"):
        result += "\n"
    return result, fixes, renamed


def repair(code: str) -> RepairReport:
    """Apply pattern template Q: case normalisation then bracket balancing."""
    cased, case_fixes, renamed = normalise_identifier_case(code)
    balanced, paren_fixes = balance_parentheses(cased)
    return RepairReport(
        code=balanced,
        paren_fixes=paren_fixes,
        case_fixes=case_fixes,
        renamed=renamed,
    )
