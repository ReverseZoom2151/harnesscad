"""Honesty linter for README / docs text — flags overclaims in prose.

A deterministic, stdlib-only text linter that scans documentation strings
for three classes of untrustworthy language, so a project's claims stay
as honest as its verification harness:

  * **Forbidden absolutes** — marketing superlatives that assert a
    guarantee the geometry checks can't back ("production-ready",
    "guaranteed", "bulletproof", ...). Case-insensitive substring match.
    Severity: fail.
  * **Untagged numeric claims** — lines that assert a load / deflection /
    safety-factor number but carry no evidence tag (``[analysis]`` /
    ``[measured-*]``). Regex match; severity: warn. A number that is
    tagged with its provenance is fine.
  * **Stale terms** — project-supplied strings that should no longer
    appear (a dropped adhesive, a renamed part). Severity: fail.

Two features keep the linter from crying wolf, both deterministic:

  * **Negation awareness** — a negation token ("no", "not", "without",
    "instead of", ...) within a short look-back window, bounded by
    sentence punctuation, suppresses a forbidden/stale match. "no longer
    production-ready" and "we do not use JB Weld" do not fire.
  * **Context stripping** — Markdown fenced code blocks and license /
    attribution lines (CC BY, MIT, SPDX, Copyright, ...) are blanked
    before the forbidden/stale scan, so sample code and a license notice
    mentioning a stale term don't trip the linter. The numeric-claim scan
    still sees the original text — a number in a license block still
    wants a tag.

This is intentionally not a geometry check: it is the text-honesty gate
that surrounds the geometry harness. It returns structured findings
(line-numbered) rather than printing, so a caller can format or fail on
them.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Tuple

DEFAULT_FORBIDDEN_ABSOLUTES: Tuple[str, ...] = (
    "production-ready",
    "production-capable",
    "guaranteed",
    "fully automated",
    "no risk",
    "100% reliable",
    "industry-leading",
    "best-in-class",
    "bulletproof",
    "complete solution",
)

DEFAULT_NUMERIC_PATTERNS: Tuple[str, ...] = (
    r"flex(?:ure)?\s+(?:under|of|=|<=|<)\s*\d+(?:\.\d+)?\s*(?:mm|kg)",
    r"deflect(?:ion|s)?\s+(?:of|=|<=|<|under)\s*\d+(?:\.\d+)?\s*mm",
    r"safety\s+factor\s+(?:of|=)?\s*\d+(?:\.\d+)?",
    r"loads?\s+(?:up\s+to|of|to)\s*\d+(?:\.\d+)?\s*kg",
)

DEFAULT_EVIDENCE_TAGS: Tuple[str, ...] = (
    "[analysis]", "[measured]", "[measured-prototype]", "[tested]",
)

_NEGATION_PATTERN = re.compile(
    r"\b(?:not|no|never|do not|don't|doesn't|replaces|instead of|"
    r"rather than|without|excludes|no longer)\b",
    re.IGNORECASE,
)
_SENTENCE_BOUNDARIES = (".", "!", "?", "\n")
_NEGATION_LOOKBACK_CHARS = 30

_LICENSE_LINE_PATTERN = re.compile(
    r"(?i)(?:CC[-\s]?BY[-\s]|CC0\b|Creative\s+Commons|MIT\s+License|"
    r"Apache\s+License|BSD\s+License|\bGPL[v\d-]?|CERN-OHL|"
    r"SPDX-License-Identifier:|Copyright\s*\((?:c|C)\)|Copyright\s*©|"
    r"licensed\s+under)"
)


@dataclass
class ClaimFinding:
    """One flagged line."""
    rule: str                 # "forbidden_absolute" | "untagged_numeric" | "stale_term"
    severity: str             # "fail" | "warn"
    line: int
    message: str
    match: str = ""


@dataclass
class ClaimReport:
    findings: List[ClaimFinding] = field(default_factory=list)
    lines_scanned: int = 0

    @property
    def failed(self) -> bool:
        return any(f.severity == "fail" for f in self.findings)

    @property
    def n_fail(self) -> int:
        return sum(1 for f in self.findings if f.severity == "fail")

    @property
    def n_warn(self) -> int:
        return sum(1 for f in self.findings if f.severity == "warn")


def _is_negated(haystack: str, term_index: int) -> bool:
    start = max(0, term_index - _NEGATION_LOOKBACK_CHARS)
    window = haystack[start:term_index]
    last_boundary = max((window.rfind(b) for b in _SENTENCE_BOUNDARIES),
                        default=-1)
    if last_boundary >= 0:
        window = window[last_boundary + 1:]
    return bool(_NEGATION_PATTERN.search(window))


def _find_unnegated(line: str, needle: str) -> int:
    """Index of the first non-negated occurrence of ``needle``, else -1."""
    h = line.lower()
    n = needle.lower()
    if not n:
        return -1
    idx = h.find(n)
    while idx >= 0:
        if not _is_negated(line, idx):
            return idx
        idx = h.find(n, idx + 1)
    return -1


def _strip_code_fences(text: str) -> List[str]:
    """Blank fenced code-block lines (preserving line numbers)."""
    out: List[str] = []
    in_fence = False
    for line in text.splitlines():
        if line.strip().startswith("```"):
            in_fence = not in_fence
            out.append("")
            continue
        out.append("" if in_fence else line)
    return out


def audit_text(text: str,
               forbidden_absolutes: Optional[Sequence[str]] = None,
               stale_terms: Optional[Sequence[str]] = None,
               numeric_patterns: Optional[Sequence[str]] = None,
               evidence_tags: Optional[Sequence[str]] = None,
               is_markdown: bool = True) -> ClaimReport:
    """Audit a block of prose and return structured findings.

    ``is_markdown`` enables fenced-code-block stripping. Forbidden and
    stale scans run over a context-stripped view (code fences + license
    lines blanked); the numeric-claim scan runs over the original text.
    """
    forbidden = list(forbidden_absolutes) if forbidden_absolutes is not None \
        else list(DEFAULT_FORBIDDEN_ABSOLUTES)
    stale = list(stale_terms or [])
    numeric = [re.compile(p, re.IGNORECASE)
               for p in (numeric_patterns
                         if numeric_patterns is not None
                         else DEFAULT_NUMERIC_PATTERNS)]
    tags = list(evidence_tags if evidence_tags is not None
                else DEFAULT_EVIDENCE_TAGS)

    numeric_lines = text.splitlines()
    if is_markdown:
        claim_lines = _strip_code_fences(text)
    else:
        claim_lines = list(numeric_lines)
    # Blank license/attribution lines for the forbidden/stale scan only.
    claim_lines = ["" if _LICENSE_LINE_PATTERN.search(ln) else ln
                   for ln in claim_lines]

    findings: List[ClaimFinding] = []
    total = max(len(claim_lines), len(numeric_lines))
    for i in range(total):
        line_no = i + 1
        claim_line = claim_lines[i] if i < len(claim_lines) else ""
        numeric_line = numeric_lines[i] if i < len(numeric_lines) else ""

        for word in forbidden:
            if _find_unnegated(claim_line, word) >= 0:
                findings.append(ClaimFinding(
                    rule="forbidden_absolute", severity="fail", line=line_no,
                    message=f"line {line_no}: forbidden absolute {word!r}",
                    match=word))
        for term in stale:
            if _find_unnegated(claim_line, term) >= 0:
                findings.append(ClaimFinding(
                    rule="stale_term", severity="fail", line=line_no,
                    message=f"line {line_no}: stale term {term!r}",
                    match=term))
        for rx in numeric:
            if rx.search(numeric_line):
                if not any(tag in numeric_line for tag in tags):
                    findings.append(ClaimFinding(
                        rule="untagged_numeric", severity="warn", line=line_no,
                        message=(f"line {line_no}: numeric claim missing an "
                                 f"evidence tag"),
                        match=rx.pattern))
                break  # one warning per line

    return ClaimReport(findings=findings, lines_scanned=len(numeric_lines))
