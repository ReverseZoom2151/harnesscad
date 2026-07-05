"""Validate atomic corpus ideas and determine auditable closure."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional


ALLOWED_DISPOSITIONS = frozenset({
    "implemented",
    "external",
    "research-heavy",
    "rejected",
    "duplicate",
})


@dataclass(frozen=True)
class AuditIssue:
    code: str
    message: str
    idea_id: Optional[str] = None


@dataclass
class AuditReport:
    idea_count: int = 0
    issues: List[AuditIssue] = field(default_factory=list)
    disposition_counts: Dict[str, int] = field(default_factory=dict)

    @property
    def closed(self) -> bool:
        return not self.issues

    def to_dict(self) -> dict:
        return {
            "closed": self.closed,
            "idea_count": self.idea_count,
            "disposition_counts": dict(sorted(self.disposition_counts.items())),
            "issues": [
                {"code": issue.code, "message": issue.message, "idea_id": issue.idea_id}
                for issue in self.issues
            ],
        }


def load_register(path: str | Path) -> dict:
    with Path(path).open(encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError("register root must be an object")
    return value


def validate_register(
    register: Mapping[str, Any],
    *,
    repo_root: str | Path,
    corpus_root: str | Path,
) -> AuditReport:
    """Validate register structure, evidence and closure invariants."""
    repo = Path(repo_root)
    corpus = Path(corpus_root)
    ideas = register.get("ideas")
    coverage = register.get("coverage")
    report = AuditReport()

    if not isinstance(ideas, list):
        report.issues.append(AuditIssue("ideas-shape", "ideas must be a list"))
        return report
    report.idea_count = len(ideas)
    if not ideas:
        report.issues.append(AuditIssue(
            "empty-register", "register must contain at least one atomic idea"
        ))
    if not isinstance(coverage, list):
        report.issues.append(AuditIssue("coverage-shape", "coverage must be a list"))
        coverage = []

    _validate_coverage(coverage, corpus, report)

    seen = set()
    for raw in ideas:
        if not isinstance(raw, Mapping):
            report.issues.append(AuditIssue("idea-shape", "idea must be an object"))
            continue
        idea_id = str(raw.get("id", "")).strip()
        if not idea_id:
            report.issues.append(AuditIssue("missing-id", "idea has no id"))
            continue
        if idea_id in seen:
            report.issues.append(AuditIssue("duplicate-id", "idea id is duplicated", idea_id))
        seen.add(idea_id)

        statement = str(raw.get("statement", "")).strip()
        if not statement:
            report.issues.append(AuditIssue(
                "missing-statement", "idea has no atomic statement", idea_id
            ))

        disposition = str(raw.get("disposition", "")).strip()
        report.disposition_counts[disposition] = (
            report.disposition_counts.get(disposition, 0) + 1
        )
        if disposition not in ALLOWED_DISPOSITIONS:
            report.issues.append(AuditIssue(
                "open-disposition",
                f"disposition {disposition!r} is not a closed disposition",
                idea_id,
            ))

        sources = raw.get("sources")
        if not isinstance(sources, list) or not sources:
            report.issues.append(AuditIssue(
                "missing-source", "idea requires at least one source locator", idea_id
            ))
        else:
            for source in sources:
                _validate_source(source, corpus, report, idea_id)

        if disposition == "implemented":
            _validate_paths(raw.get("code"), repo, "missing-code", report, idea_id)
            _validate_paths(raw.get("tests"), repo, "missing-tests", report, idea_id)
        elif disposition in {"external", "research-heavy", "rejected"}:
            if not str(raw.get("rationale", "")).strip():
                report.issues.append(AuditIssue(
                    "missing-rationale",
                    f"{disposition} idea requires a rationale",
                    idea_id,
                ))
        elif disposition == "duplicate":
            duplicate_of = str(raw.get("duplicate_of", "")).strip()
            if not duplicate_of:
                report.issues.append(AuditIssue(
                    "missing-duplicate-target",
                    "duplicate idea requires duplicate_of",
                    idea_id,
                ))

        children = raw.get("children", [])
        if children:
            report.issues.append(AuditIssue(
                "undecomposed-parent",
                "atomic ideas cannot retain child slices; register children separately",
                idea_id,
            ))

    for raw in ideas:
        if isinstance(raw, Mapping) and raw.get("disposition") == "duplicate":
            target = raw.get("duplicate_of")
            if target not in seen:
                report.issues.append(AuditIssue(
                    "bad-duplicate-target",
                    f"duplicate target {target!r} does not exist",
                    str(raw.get("id", "")) or None,
                ))
    return report


def _validate_coverage(
    coverage: Iterable[Any], corpus: Path, report: AuditReport
) -> None:
    actual = {
        str(path.relative_to(corpus)).replace("\\", "/")
        for path in corpus.rglob("*") if path.is_file()
    }
    declared = set()
    for raw in coverage:
        if not isinstance(raw, Mapping):
            report.issues.append(AuditIssue("coverage-entry", "coverage entry is not an object"))
            continue
        path = str(raw.get("path", "")).strip()
        path_glob = str(raw.get("glob", "")).strip()
        if path:
            declared.add(path)
        elif path_glob:
            matches = {
                str(match.relative_to(corpus)).replace("\\", "/")
                for match in corpus.glob(path_glob) if match.is_file()
            }
            if not matches:
                report.issues.append(AuditIssue(
                    "empty-coverage-glob", f"coverage glob has no files: {path_glob}"
                ))
            declared.update(matches)
        else:
            report.issues.append(AuditIssue(
                "missing-coverage-path", "coverage entry needs path or glob"
            ))
        label = path or path_glob
        if raw.get("status") != "reviewed":
            report.issues.append(AuditIssue(
                "unreviewed-source", f"source {label!r} is not marked reviewed"
            ))
        if not str(raw.get("method", "")).strip():
            report.issues.append(AuditIssue(
                "missing-review-method", f"source {label!r} has no review method"
            ))
    for path in sorted(actual - declared):
        report.issues.append(AuditIssue("uncovered-file", f"corpus file not covered: {path}"))
    for path in sorted(declared - actual):
        report.issues.append(AuditIssue("missing-corpus-file", f"declared file is absent: {path}"))


def _validate_source(
    source: Any,
    corpus: Path,
    report: AuditReport,
    idea_id: str,
) -> None:
    if not isinstance(source, Mapping):
        report.issues.append(AuditIssue(
            "source-shape", "source locator must be an object", idea_id
        ))
        return
    path = str(source.get("path", "")).strip()
    locator = str(source.get("locator", "")).strip()
    if not path or not (corpus / path).is_file():
        report.issues.append(AuditIssue(
            "bad-source-path", f"source path is absent: {path!r}", idea_id
        ))
    if not locator:
        report.issues.append(AuditIssue(
            "missing-locator", "source requires a page/line/section locator", idea_id
        ))


def _validate_paths(
    value: Any,
    root: Path,
    code: str,
    report: AuditReport,
    idea_id: str,
) -> None:
    if not isinstance(value, list) or not value:
        report.issues.append(AuditIssue(code, f"{code} evidence is empty", idea_id))
        return
    for raw in value:
        path = str(raw).split(":", 1)[0]
        if not (root / path).is_file():
            report.issues.append(AuditIssue(
                code, f"evidence path is absent: {raw!r}", idea_id
            ))
