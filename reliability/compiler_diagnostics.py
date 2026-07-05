"""Stable CAD compiler diagnostic normalization retaining raw evidence."""

from __future__ import annotations

from dataclasses import dataclass
import re

_RULES = (
    ("missing-terminator", re.compile(r"(missing|expected).*(end|terminator)", re.I)),
    ("open-loop", re.compile(
        r"((open|unclosed|not closed).*(wire|loop|profile)|"
        r"(wire|loop|profile).*(open|unclosed|not closed))", re.I)),
    ("invalid-extrusion", re.compile(r"(extrud).*(invalid|negative|zero|fail)", re.I)),
    ("invalid-boolean", re.compile(r"(boolean|fuse|cut|common).*(invalid|fail|null)", re.I)),
    ("syntax-error", re.compile(r"(syntax|parse).*(error|fail)", re.I)),
)


@dataclass(frozen=True)
class CompilerDiagnostic:
    code: str
    raw: str
    provider: str = ""


def normalize_compiler_error(error, *, provider=""):
    raw = str(error)
    code = next((code for code, pattern in _RULES if pattern.search(raw)),
                "unknown-compiler-error")
    return CompilerDiagnostic(code, raw, provider)
