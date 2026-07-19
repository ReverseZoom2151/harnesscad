"""clarify_leakage -- detect CAD-script/Python code leakage in NL descriptions.

A data-annotation pipeline that generates natural-language descriptions from
ground-truth CAD-script programs risks *leaking* raw code into the text. This
module is a precise, rule-based auditor: the description may (and should) repeat
the same *geometric information* -- dimensions, coordinate tuples, plane names,
feature ordering -- but any CAD-script/Python *surface form* (API tokens,
method-call syntax, imports, code identifiers) is leakage.

The checker is deterministic (stdlib-only), applying HARD-FAIL rules (A-D) and
an explicit allow-list, including the rule that the bare English words
``origin`` / ``workplane`` are allowed and only leak when in code form
(``cq.Workplane``, ``Workplane(``, a method chain). It returns the JSON shape
``{contains_code, detected_code_snippets, explanation}``.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import List, Optional, Tuple

# -- A) CAD-script / API surface form -------------------------------------- #
_IMPORT_RE = re.compile(r"\bimport\s+cadquery\b|\bfrom\s+cadquery\b", re.I)
_CQ_ALIAS_RE = re.compile(r"\bcq\.\w+")
_WORKPLANE_CALL_RE = re.compile(r"\b(?:cq\.)?Workplane\s*\(")
#: Method-call / method-chain syntax: ``.name(``.
_METHOD_CALL_RE = re.compile(r"\.(?:" + "|".join([
    "extrude", "circle", "rect", "cut", "union", "faces", "edges", "fillet",
    "chamfer", "translate", "rotate", "workplane", "sketch", "finalize",
    "segment", "polyline", "box", "sphere", "cylinder", "revolve", "loft",
    "sweep", "moveTo", "lineTo", "close", "assemble", "center", "face",
]) + r")\s*\(", re.I)

# -- B) Python code surface form ------------------------------------------- #
_PY_KEYWORD_RE = re.compile(r"\b(?:def|return|lambda|class|import)\b")
_CODE_FENCE_RE = re.compile(r"```|~~~")

# -- C) code-like object assignments (but NOT `origin = (...)`/`radius = 10`) #
_ASSIGN_RE = re.compile(
    r"\b(\w+)\s*=\s*(?:cq\.)?(?:Workplane|Sketch|Assembly|Solid)\b")
#: Any assignment whose right-hand side contains API surface form.
_ASSIGN_API_RE = re.compile(r"\b\w+\s*=\s*.*(?:cq\.|Workplane\s*\(|\.\w+\s*\()")

# Allowed descriptive spec assignments (do not flag by themselves).
_ALLOWED_ASSIGN_RE = re.compile(
    r"\b(?:origin|radius|width|height|length|thickness|depth|diameter)\s*=\s*"
    r"(?:\(|-?\d)", re.I)


@dataclass(frozen=True)
class LeakageResult:
    contains_code: bool
    detected_code_snippets: Tuple[str, ...]
    explanation: str

    def to_json(self) -> dict:
        return {
            "contains_code": self.contains_code,
            "detected_code_snippets": list(self.detected_code_snippets),
            "explanation": self.explanation,
        }


def check_leakage(description: str,
                  original_identifiers: Optional[List[str]] = None
                  ) -> LeakageResult:
    """Audit ``description`` for CAD-script/Python code leakage.

    ``original_identifiers`` are variable/function names from the source script
    (rule D): if any appear verbatim (and are not the generic English words
    ``origin``/``workplane``), that is leakage.
    """
    snippets: List[str] = []

    def add(pat: re.Pattern, text: str) -> None:
        for m in pat.finditer(text):
            frag = m.group(0).strip()
            if frag and frag not in snippets:
                snippets.append(frag)

    # A) CAD-script / API surface form.
    add(_IMPORT_RE, description)
    add(_CQ_ALIAS_RE, description)
    add(_WORKPLANE_CALL_RE, description)
    add(_METHOD_CALL_RE, description)
    # B) Python surface form.
    add(_PY_KEYWORD_RE, description)
    if _CODE_FENCE_RE.search(description):
        snippets.append("```")
    # C) code-object assignments (skip allowed geometry specs).
    for m in _ASSIGN_RE.finditer(description):
        frag = m.group(0).strip()
        if frag not in snippets:
            snippets.append(frag)
    for m in _ASSIGN_API_RE.finditer(description):
        frag = m.group(0).strip()
        if _ALLOWED_ASSIGN_RE.match(frag):
            continue
        if frag not in snippets:
            snippets.append(frag)
    # D) verbatim reuse of source identifiers.
    for ident in (original_identifiers or ()):
        if ident.lower() in ("origin", "workplane"):
            continue
        if re.search(r"\b" + re.escape(ident) + r"\b", description):
            frag = ident
            if frag not in snippets:
                snippets.append(frag)

    contains = bool(snippets)
    if contains:
        expl = ("Description leaks CadQuery/Python surface form: "
                + ", ".join(snippets[:6]) + ".")
    else:
        expl = ("No raw code leakage; geometric information "
                "(dimensions, coordinates, plane names) is allowed.")
    return LeakageResult(contains, tuple(snippets), expl)


# --------------------------------------------------------------------------- #
# Style-warning branch (paper: overly code-styled text without a HARD FAIL)
# --------------------------------------------------------------------------- #

_STYLE_WARN_RE = re.compile(r"\b[XYZ]{2}\s*@\s*\(")  # e.g. "ZX @ (-64, 9, -36)"


def style_warnings(description: str) -> List[str]:
    """Return non-leakage style warnings (the ``detected_code_snippets``)."""
    return [m.group(0).strip() for m in _STYLE_WARN_RE.finditer(description)]
