"""t2cdean_scad_extract -- recover bare OpenSCAD source from an LLM reply.

The PrintX / Text-to-CAD (dean) app asks the model to "only return your openscad
code and nothing else whatsoever" and then pipes the raw completion straight into
a ``.scad`` file.  Prompting is not a parser: models still wrap the answer in
```` ```scad ```` fences, prepend "Sure! Here's the code:", and append a "This
creates a 10mm cube..." paragraph.  Every one of those tokens is a syntax error
for the OpenSCAD compiler, and the failure surfaces far downstream as an empty
render.

This module is the deterministic recovery layer that belongs between the model
and the compiler:

* ``extract_code_blocks`` -- all fenced blocks with their language tags,
  tolerating an unterminated final fence (a truncated completion), which a naive
  ``split("```")`` silently turns into garbage.
* ``extract_scad`` -- pick the right block: prefer a block tagged ``scad`` /
  ``openscad``, else an untagged block that *looks* like OpenSCAD, else fall back
  to line-level prose stripping when the model emitted no fences at all.
* ``strip_prose_lines`` -- keep only lines that plausibly belong to a ``.scad``
  file, used for the unfenced case.
* ``looks_like_scad`` -- a cheap content heuristic (OpenSCAD keywords being
  called, a statement terminator, no Python tell-tales) used to rank candidates.

The harness has fence handling only as a private helper inside
``quality.cadclaw_claim_audit`` (for auditing prose claims) and a format-reward
regex in ``dataengine.cmecad_reward`` that merely *checks* the shape; neither
returns compilable source.  Extraction itself is deterministic and stdlib-only.
"""

from __future__ import annotations

import re
from typing import List, NamedTuple, Optional

# OpenSCAD's built-in primitives, transforms and control keywords. Presence of
# any of these is the primary evidence that a block is really OpenSCAD.
SCAD_KEYWORDS = frozenset(
    {
        "cube",
        "sphere",
        "cylinder",
        "polyhedron",
        "square",
        "circle",
        "polygon",
        "text",
        "translate",
        "rotate",
        "scale",
        "mirror",
        "resize",
        "multmatrix",
        "color",
        "hull",
        "minkowski",
        "union",
        "difference",
        "intersection",
        "linear_extrude",
        "rotate_extrude",
        "offset",
        "projection",
        "surface",
        "import",
        "module",
        "function",
        "include",
        "use",
        "for",
        "if",
        "let",
        "children",
    }
)

# Languages an LLM plausibly tags an OpenSCAD block with.
SCAD_LANGUAGES = frozenset({"scad", "openscad"})

_FENCE_RE = re.compile(
    r"^[ \t]*(?P<fence>`{3,}|~{3,})[ \t]*(?P<lang>[A-Za-z0-9_+-]*)[ \t]*$"
)

_IDENT_CALL_RE = re.compile(r"\b([A-Za-z_][A-Za-z0-9_]*)\s*\(")

# Lines that are unambiguously OpenSCAD rather than English prose.
_SCAD_LINE_RE = re.compile(
    r"^\s*("
    r"//"  # comment
    r"|/\*|\*/"  # block comment
    r"|\}|\{"  # brace
    r"|\$\w+\s*="  # special variable
    r"|[A-Za-z_]\w*\s*="  # assignment
    r"|[A-Za-z_]\w*\s*\("  # call / module instantiation
    r"|module\b|function\b|include\b|use\b|for\b|if\b|else\b|let\b"
    r"|[-+*!#%]"  # modifier chars / transforms prefix
    r")"
)


class CodeBlock(NamedTuple):
    """One fenced block: its language tag (lowercased, possibly '') and body."""

    language: str
    code: str


class ScadExtractionError(ValueError):
    """Raised when no plausible OpenSCAD source can be recovered."""


def extract_code_blocks(text: str) -> List[CodeBlock]:
    """All fenced code blocks, in order.

    An unterminated final fence -- what a length-truncated completion looks like
    -- is closed at end-of-text rather than discarded, because the partial code
    is usually still the best candidate available.
    """
    blocks: List[CodeBlock] = []
    lines = (text or "").splitlines()
    i = 0
    n = len(lines)
    while i < n:
        m = _FENCE_RE.match(lines[i])
        if not m:
            i += 1
            continue
        fence = m.group("fence")
        lang = m.group("lang").lower()
        marker = fence[0] * 3
        i += 1
        body: List[str] = []
        closed = False
        while i < n:
            close = _FENCE_RE.match(lines[i])
            # A closing fence is the same character, at least as long, no lang.
            if close and close.group("fence")[0] * 3 == marker and not close.group("lang"):
                closed = True
                i += 1
                break
            body.append(lines[i])
            i += 1
        blocks.append(CodeBlock(lang, "\n".join(body).strip("\n")))
        if not closed:
            break  # truncated completion: nothing sane follows
    return blocks


def looks_like_scad(code: str) -> bool:
    """Heuristic: does this text plausibly compile as OpenSCAD?

    Requires at least one known OpenSCAD identifier being *called* or a
    statement terminator, and rejects obvious Python/JS lookalikes.
    """
    if not code or not code.strip():
        return False
    if re.search(r"^\s*(import\s+\w+\s*$|from\s+\w+\s+import\b|def\s+\w+\s*\()", code, re.M):
        return False  # Python
    calls = {m.group(1) for m in _IDENT_CALL_RE.finditer(code)}
    if calls & SCAD_KEYWORDS:
        return True
    # A user-defined module call plus a terminator is still valid OpenSCAD.
    return bool(calls) and ";" in code


def scad_score(block: CodeBlock) -> int:
    """Rank a candidate block; higher is more likely the intended source."""
    score = 0
    if block.language in SCAD_LANGUAGES:
        score += 100
    elif block.language == "":
        score += 10
    elif block.language in {"python", "py", "js", "javascript", "json", "bash", "sh"}:
        score -= 50
    if looks_like_scad(block.code):
        score += 50
    calls = {m.group(1) for m in _IDENT_CALL_RE.finditer(block.code)}
    score += len(calls & SCAD_KEYWORDS)
    return score


def strip_prose_lines(text: str) -> str:
    """Drop English commentary from an unfenced reply, keeping SCAD-ish lines.

    Blank lines *inside* the kept region are preserved so the source stays
    readable; leading/trailing blanks are trimmed.
    """
    kept: List[str] = []
    for line in (text or "").splitlines():
        if not line.strip():
            if kept:
                kept.append("")
            continue
        if _SCAD_LINE_RE.match(line):
            kept.append(line.rstrip())
    while kept and not kept[-1].strip():
        kept.pop()
    return "\n".join(kept)


def extract_scad(reply: str, strict: bool = False) -> str:
    """Recover compilable OpenSCAD source from a raw model reply.

    Strategy, in order:
      1. fenced blocks, ranked (``scad``/``openscad`` tag wins, then content);
      2. no fences -> strip prose lines and keep the code-looking remainder;
      3. nothing plausible -> raise :class:`ScadExtractionError`.

    With ``strict=True`` the result must additionally pass
    :func:`looks_like_scad`, so a confident-sounding refusal cannot slip a
    paragraph of English into the compiler.
    """
    blocks = extract_code_blocks(reply)
    candidate: Optional[str] = None
    if blocks:
        best = max(blocks, key=scad_score)
        if scad_score(best) > 0 and best.code.strip():
            candidate = best.code.strip()
    if candidate is None:
        stripped = strip_prose_lines(reply)
        if stripped.strip():
            candidate = stripped.strip()
    if candidate is None or not candidate.strip():
        raise ScadExtractionError("no OpenSCAD source found in reply")
    if strict and not looks_like_scad(candidate):
        raise ScadExtractionError("extracted text does not look like OpenSCAD")
    return candidate


def normalise_scad(code: str) -> str:
    """Canonicalise recovered source: LF endings, no trailing spaces, final NL.

    Makes the content-addressed digest in
    ``fabrication.t2cdean_openscad_export`` stable across model reruns that
    differ only in whitespace.
    """
    text = (code or "").replace("\r\n", "\n").replace("\r", "\n")
    lines = [line.rstrip() for line in text.split("\n")]
    while lines and not lines[0]:
        lines.pop(0)
    while lines and not lines[-1]:
        lines.pop()
    return "\n".join(lines) + "\n" if lines else ""
