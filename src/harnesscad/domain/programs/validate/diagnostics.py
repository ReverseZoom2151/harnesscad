"""Multi-language code-CAD diagnostic parser (deterministic, stdlib-only).

Ported and generalised from CadHub, whose per-language controllers each hand the
raw toolchain stderr to the IDE console after a one-line clean-up
(``cleanOpenScadError`` in ``openScadController.ts`` /
``app/api/src/docker/openscad/runScad.ts``: strip the sandbox temp path so the
user sees ``'main.scad'``). CadHub stops there -- the message stays an opaque
string. For a text-to-CAD *repair* loop the useful artifact is the structured
diagnostic: which file, which line, which column, which severity, which message,
so the offending source line can be quoted back to the model.

The harness already classifies OpenSCAD stderr *lines* into error/warning
buckets (``fabrication/t2cdean_openscad_export.py``) and statically checks
OpenSCAD / CadQuery source (``programs/scadlm_check``, ``programs/t2cq_validity``).
Missing was a parser that turns raw toolchain output from ANY of the four
code-CAD dialects into line/column-resolved ``Diagnostic`` records under one
schema -- which is what a language-agnostic repair loop needs.

Dialects handled (keys match ``adapters.cadhub_language_registry`` diagnostic_dialect):

* ``openscad`` -- ``ERROR: Parser error: syntax error in file main.scad, line 12``
* ``cadquery`` -- CPython tracebacks (``File "main.py", line 5`` + trailing
  ``NameError: ...``), including ``SyntaxError`` caret columns
* ``jscad``    -- JS engine errors with ``at <source>:LINE:COL`` stack frames
* ``curv``     -- ``ERROR: msg`` + ``at file "main.curv":3(5)`` locations

Deterministic: pure regex over text, stable ordering, no clock, no I/O.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

SEVERITY_ERROR = "error"
SEVERITY_WARNING = "warning"
SEVERITY_INFO = "info"

_SEVERITY_RANK = {SEVERITY_ERROR: 0, SEVERITY_WARNING: 1, SEVERITY_INFO: 2}


class UnknownDialect(KeyError):
    """Raised for a dialect with no parser."""


@dataclass(frozen=True)
class Diagnostic:
    """One structured message from a code-CAD toolchain."""

    language: str
    severity: str
    message: str
    file: Optional[str] = None
    line: Optional[int] = None
    column: Optional[int] = None
    raw: str = ""

    def location(self) -> str:
        """``file:line:col`` with the parts that are known."""
        parts = [self.file or "?"]
        if self.line is not None:
            parts.append(str(self.line))
            if self.column is not None:
                parts.append(str(self.column))
        return ":".join(parts)

    def as_dict(self) -> Dict[str, object]:
        return {
            "language": self.language,
            "severity": self.severity,
            "message": self.message,
            "file": self.file,
            "line": self.line,
            "column": self.column,
        }


# ---------------------------------------------------------------------------
# Sandbox path scrubbing (generalisation of CadHub's cleanOpenScadError)
# ---------------------------------------------------------------------------

# Any absolute posix/windows path whose basename we keep.
_PATH_RE = re.compile(r"""(['"]?)((?:[A-Za-z]:)?[\\/][^\s'"()]*[\\/])([^\s'"()\\/]+)\1""")


def scrub_paths(text: str) -> str:
    """Replace absolute sandbox paths with their bare basename.

    ``ERROR: syntax error in file "/tmp/aX9/main.scad", line 3`` becomes
    ``ERROR: syntax error in file "main.scad", line 3`` -- stable across runs,
    which makes diagnostics comparable and cacheable.
    """

    def repl(match: "re.Match[str]") -> str:
        quote = match.group(1)
        return quote + match.group(3) + quote

    return _PATH_RE.sub(repl, text)


def _basename(path: str) -> str:
    cleaned = path.strip().strip("'\"")
    for sep in ("/", "\\"):
        if sep in cleaned:
            cleaned = cleaned.rsplit(sep, 1)[-1]
    return cleaned


def _int(value: Optional[str]) -> Optional[int]:
    if value is None or value == "":
        return None
    try:
        return int(value)
    except ValueError:  # pragma: no cover - guarded by regex
        return None


# ---------------------------------------------------------------------------
# OpenSCAD
# ---------------------------------------------------------------------------

_SCAD_SEVERITY = re.compile(r"^\s*(ERROR|WARNING|TRACE|INFO)\s*:\s*(.*)$", re.IGNORECASE)
# "... in file main.scad, line 12" (file optionally quoted); also "file main.scad line 12"
_SCAD_LOC = re.compile(
    r"""in\s+file\s+['"]?([^'",]+?)['"]?\s*,?\s*line\s+(\d+)""", re.IGNORECASE
)

_SCAD_SEVERITY_MAP = {
    "error": SEVERITY_ERROR,
    "warning": SEVERITY_WARNING,
    "trace": SEVERITY_INFO,
    "info": SEVERITY_INFO,
}


def parse_openscad(text: str) -> List[Diagnostic]:
    out: List[Diagnostic] = []
    for raw_line in text.splitlines():
        match = _SCAD_SEVERITY.match(raw_line)
        if not match:
            continue
        severity = _SCAD_SEVERITY_MAP[match.group(1).lower()]
        body = match.group(2).strip()
        loc = _SCAD_LOC.search(body)
        file_name = _basename(loc.group(1)) if loc else None
        line = _int(loc.group(2)) if loc else None
        if loc:
            body = (body[: loc.start()] + body[loc.end() :]).strip()
            body = body.strip(" ,:")
        out.append(
            Diagnostic(
                language="openscad",
                severity=severity,
                message=body,
                file=file_name,
                line=line,
                raw=raw_line.rstrip(),
            )
        )
    return out


# ---------------------------------------------------------------------------
# CadQuery (CPython traceback)
# ---------------------------------------------------------------------------

_PY_FRAME = re.compile(r"""^\s*File\s+"([^"]+)",\s+line\s+(\d+)""")
_PY_EXC = re.compile(r"^(\w[\w.]*(?:Error|Exception|Warning|Exit|Interrupt))\s*:\s*(.*)$")
_PY_SYNTAX_LOC = re.compile(r"^\s*\^+\s*$")


def parse_cadquery(text: str) -> List[Diagnostic]:
    """Take the *deepest user frame* of a traceback plus the exception line."""
    lines = text.splitlines()
    last_frame: Optional[Diagnostic] = None
    caret_col: Optional[int] = None
    frame_file: Optional[str] = None
    frame_line: Optional[int] = None
    out: List[Diagnostic] = []

    for raw_line in lines:
        frame = _PY_FRAME.match(raw_line)
        if frame:
            frame_file = _basename(frame.group(1))
            frame_line = _int(frame.group(2))
            caret_col = None
            continue
        if _PY_SYNTAX_LOC.match(raw_line):
            # column is 1-based index of the first caret
            caret_col = raw_line.index("^") + 1
            continue
        exc = _PY_EXC.match(raw_line)
        if exc:
            name = exc.group(1)
            severity = SEVERITY_WARNING if name.endswith("Warning") else SEVERITY_ERROR
            message = name + ": " + exc.group(2).strip() if exc.group(2).strip() else name
            out.append(
                Diagnostic(
                    language="cadquery",
                    severity=severity,
                    message=message,
                    file=frame_file,
                    line=frame_line,
                    column=caret_col,
                    raw=raw_line.rstrip(),
                )
            )
    if out:
        return out
    if last_frame:  # pragma: no cover - defensive
        return [last_frame]
    return []


# ---------------------------------------------------------------------------
# JSCAD (JS engine error + stack)
# ---------------------------------------------------------------------------

_JS_HEAD = re.compile(r"^\s*(\w*(?:Error|Warning))\s*:\s*(.*)$")
_JS_FRAME = re.compile(r"at\s+(?:.*?\()?([^\s():]+):(\d+):(\d+)\)?")


def parse_jscad(text: str) -> List[Diagnostic]:
    lines = text.splitlines()
    head: Optional[str] = None
    severity = SEVERITY_ERROR
    raw_head = ""
    for raw_line in lines:
        match = _JS_HEAD.match(raw_line)
        if match:
            name = match.group(1)
            severity = SEVERITY_WARNING if name.endswith("Warning") else SEVERITY_ERROR
            body = match.group(2).strip()
            head = name + ": " + body if body else name
            raw_head = raw_line.rstrip()
            break
    frame = None
    for raw_line in lines:
        frame = _JS_FRAME.search(raw_line)
        if frame:
            break
    if head is None and frame is None:
        return []
    if head is None:
        head = lines[0].strip() if lines else ""
        raw_head = head
    return [
        Diagnostic(
            language="jscad",
            severity=severity,
            message=head,
            file=_basename(frame.group(1)) if frame else None,
            line=_int(frame.group(2)) if frame else None,
            column=_int(frame.group(3)) if frame else None,
            raw=raw_head,
        )
    ]


# ---------------------------------------------------------------------------
# curv
# ---------------------------------------------------------------------------

_CURV_HEAD = re.compile(r"^\s*(ERROR|WARNING)\s*:\s*(.*)$", re.IGNORECASE)
_CURV_LOC = re.compile(r"""at\s+file\s+["']([^"']+)["']\s*:\s*(\d+)\((\d+)\)""")


def parse_curv(text: str) -> List[Diagnostic]:
    lines = text.splitlines()
    out: List[Diagnostic] = []
    pending: Optional[Diagnostic] = None
    for raw_line in lines:
        head = _CURV_HEAD.match(raw_line)
        if head:
            if pending is not None:
                out.append(pending)
            severity = (
                SEVERITY_WARNING
                if head.group(1).lower() == "warning"
                else SEVERITY_ERROR
            )
            pending = Diagnostic(
                language="curv",
                severity=severity,
                message=head.group(2).strip(),
                raw=raw_line.rstrip(),
            )
            continue
        loc = _CURV_LOC.search(raw_line)
        if loc and pending is not None:
            pending = Diagnostic(
                language="curv",
                severity=pending.severity,
                message=pending.message,
                file=_basename(loc.group(1)),
                line=_int(loc.group(2)),
                column=_int(loc.group(3)),
                raw=pending.raw,
            )
    if pending is not None:
        out.append(pending)
    return out


_PARSERS = {
    "openscad": parse_openscad,
    "cadquery": parse_cadquery,
    "jscad": parse_jscad,
    "curv": parse_curv,
}


# ---------------------------------------------------------------------------
# Facade
# ---------------------------------------------------------------------------


def parse(dialect: str, text: str) -> List[Diagnostic]:
    """Parse toolchain output for ``dialect`` into structured diagnostics."""
    key = dialect.strip().lower()
    if key not in _PARSERS:
        raise UnknownDialect(dialect)
    return _PARSERS[key](scrub_paths(text or ""))


def errors(diagnostics: Sequence[Diagnostic]) -> List[Diagnostic]:
    return [d for d in diagnostics if d.severity == SEVERITY_ERROR]


def first_error(diagnostics: Sequence[Diagnostic]) -> Optional[Diagnostic]:
    found = errors(diagnostics)
    return found[0] if found else None


def is_success(diagnostics: Sequence[Diagnostic]) -> bool:
    return not errors(diagnostics)


def sort_key(diag: Diagnostic) -> tuple:
    """Severity first, then position -- a stable ordering for display."""
    return (
        _SEVERITY_RANK.get(diag.severity, 9),
        diag.line if diag.line is not None else 10**9,
        diag.column if diag.column is not None else 10**9,
        diag.message,
    )


def sorted_diagnostics(diagnostics: Sequence[Diagnostic]) -> List[Diagnostic]:
    return sorted(diagnostics, key=sort_key)


def summarize(diagnostics: Sequence[Diagnostic]) -> Dict[str, object]:
    counts = {SEVERITY_ERROR: 0, SEVERITY_WARNING: 0, SEVERITY_INFO: 0}
    for diag in diagnostics:
        counts[diag.severity] = counts.get(diag.severity, 0) + 1
    top = first_error(sorted_diagnostics(diagnostics))
    return {
        "ok": counts[SEVERITY_ERROR] == 0,
        "errors": counts[SEVERITY_ERROR],
        "warnings": counts[SEVERITY_WARNING],
        "info": counts[SEVERITY_INFO],
        "first_error": top.as_dict() if top else None,
    }


def annotate_source(source: str, diag: Diagnostic, context: int = 1) -> str:
    """Quote the offending source line(s) with a caret -- feedback for a repair prompt."""
    if diag.line is None:
        return ""
    lines = source.splitlines()
    index = diag.line - 1
    if index < 0 or index >= len(lines):
        return ""
    start = max(0, index - context)
    end = min(len(lines), index + context + 1)
    width = len(str(end))
    out: List[str] = []
    for i in range(start, end):
        out.append("%*d| %s" % (width, i + 1, lines[i]))
        if i == index and diag.column:
            pad = " " * (width + 2 + max(0, diag.column - 1))
            out.append(pad + "^")
    return "\n".join(out)
