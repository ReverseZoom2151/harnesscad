"""Namespace-aware project iteration and self-correction engine.

It revises a project document through an LLM: compacts the document for context, builds a
revise-the-whole-document prompt (optionally focused on one namespace), rejects
placeholder output, reruns validation, bumps revision and per-namespace
versions, appends a history entry, and includes a self-correction agent that
turns stored validation issues and metadata/output findings into a
smallest-coherent-mutation repair instruction.

The implementation includes iteration metadata and self-correction plan records,
instruction normalization, revision helpers, placeholder detection, document
compaction (reusing the redaction from
``harnesscad.domain.spec.project_object``), the iteration prompt builder with
an optional target-namespace payload block, deduplicated history appends,
``finalize_iteration`` with an injectable validator, metadata-only iterations
that record pending instructions without pretending content changed, the
``ProjectIterator`` driver, and the generic metadata/output finding checks
(failed or pending operations, requested-but-not-succeeded outputs, stale
errors after success).

The implementation is generic over a plain ``dict`` document with
injectable functions -- ``llm`` is any callable ``prompt -> dict``,
``validate`` is any callable ``document -> (summary_dict, is_valid)``, and
``now`` is a caller-injected timestamp string, so the module stays
deterministic and stdlib-only.

Harness gap filled: the harness's correction loop
(``harnesscad.agents.agent.runner`` with ``iterative_edit_policy`` and
``edit_session``) revises *op programs* against kernel feedback; nothing
revises the *project document itself* -- overview, geometry namespaces, docs,
history -- as one versioned artifact. This module supplies that document-level
iteration layer. It complements (does not duplicate) ``edit_session`` and
``iterative_edit_policy``: those decide how to mutate CISP ops inside a
session, while this module governs whole-document revisions, revision lineage,
and validation-driven self-correction planning on top of the namespace view
from ``harnesscad.domain.spec.project_object``.
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Tuple

from harnesscad.domain.spec.project_object import (
    DEFAULT_PROJECT_NAMESPACES,
    attach_project_object_metadata,
    namespace_payload,
    normalize_project_namespace,
    project_object_version,
    redact_payload_value,
)


PLACEHOLDER_TEXT_VALUES = {"", "unknown", "n/a", "na", "none", "null", "new", "tbd", "placeholder"}
DEFAULT_CONTEXT_MAX_STRING_CHARS = 4000

LLMCallable = Callable[[str], Dict[str, Any]]
ValidateCallable = Callable[[Dict[str, Any]], Tuple[Dict[str, Any], bool]]
PayloadCallable = Callable[[Mapping[str, Any], str], Dict[str, Any]]


@dataclass(frozen=True)
class IterationMetadata:
    """Debuggable summary of one project iteration operation."""

    mode: str
    revision: int
    previous_revision: int
    instruction: str
    provider: str = ""
    model: str = ""
    target_namespace: Optional[str] = None
    validation_error: Optional[str] = None

    def as_dict(self) -> Dict[str, Any]:
        return {
            "mode": self.mode,
            "revision": self.revision,
            "previous_revision": self.previous_revision,
            "instruction": self.instruction,
            "provider": self.provider,
            "model": self.model,
            "target_namespace": self.target_namespace,
            "validation_error": self.validation_error,
        }


@dataclass(frozen=True)
class SelfCorrectionPlan:
    """A repair instruction derived from validation issues and output findings."""

    target_namespace: str
    instruction: str
    critical_issue_count: int
    warning_issue_count: int
    output_issue_count: int = 0

    def as_dict(self) -> Dict[str, Any]:
        return {
            "target_namespace": self.target_namespace,
            "instruction": self.instruction,
            "critical_issue_count": self.critical_issue_count,
            "warning_issue_count": self.warning_issue_count,
            "output_issue_count": self.output_issue_count,
        }


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------


def normalize_iteration_instruction(value: str) -> str:
    instruction = (value or "").strip()
    if not instruction:
        raise ValueError("Iteration instruction is required.")
    return instruction


def document_revision(document: Mapping[str, Any]) -> int:
    """Current revision: metadata["revision"], else history length, min 1."""
    return project_object_version(document)


def next_revision_number(document: Mapping[str, Any]) -> int:
    return document_revision(document) + 1


def is_placeholder_text(value: Optional[str]) -> bool:
    return str(value or "").strip().lower() in PLACEHOLDER_TEXT_VALUES


def _document_metadata(document: Mapping[str, Any]) -> Dict[str, Any]:
    metadata = document.get("metadata")
    return dict(metadata) if isinstance(metadata, dict) else {}


def _project_id(document: Mapping[str, Any], override: Optional[str] = None) -> str:
    value = override or _document_metadata(document).get("project_id")
    text = str(value).strip() if value not in (None, "") else ""
    return text or "project"


def compact_document_for_iteration(
    document: Mapping[str, Any],
    *,
    max_string_chars: int = DEFAULT_CONTEXT_MAX_STRING_CHARS,
) -> Dict[str, Any]:
    """LLM-safe project context: data URLs redacted, huge strings truncated."""
    if not isinstance(document, Mapping):
        raise TypeError("Project document must be a mapping.")
    return redact_payload_value(dict(document), max_string_chars=max_string_chars)


# ---------------------------------------------------------------------------
# Prompt construction
# ---------------------------------------------------------------------------


def build_iteration_prompt(
    document: Mapping[str, Any],
    instruction: str,
    *,
    original_prompt: Optional[str] = None,
    project_id: Optional[str] = None,
    target_namespace: Optional[str] = None,
    payload_fn: Optional[PayloadCallable] = None,
) -> str:
    """Build the revise-the-whole-document prompt, optionally namespace-focused."""
    instruction = normalize_iteration_instruction(instruction)
    compact = compact_document_for_iteration(document)
    previous_revision = document_revision(document)
    next_revision = previous_revision + 1
    normalized_namespace = normalize_project_namespace(target_namespace)

    namespace_block = ""
    if normalized_namespace:
        resolve_payload = payload_fn or namespace_payload
        namespace_block = (
            f"\nTarget namespace: {normalized_namespace}\n"
            "Focus the requested change on this namespace. Preserve other "
            "namespaces unless they must change to keep the project coherent.\n"
            "Current target namespace payload:\n"
            f"{json.dumps(resolve_payload(document, normalized_namespace), indent=2, sort_keys=True, default=str)}\n"
        )

    return (
        "You are HarnessCAD's project iteration engine. Revise an existing CAD "
        "project document.\n"
        "Return one complete project JSON document, not a patch and not markdown.\n"
        "Preserve every part of the project that the instruction does not "
        "explicitly change.\n"
        "Keep the existing project_id, op ids, and stable item ids unless a "
        "requested change requires updates.\n"
        "If you add, remove, or replace geometry, update ops, sketches, "
        "features, assembly steps, fabrication notes, and validation-related "
        "fields together.\n"
        "Never claim unsupported geometry or functionality without adding the "
        "required ops and feature changes.\n"
        f"Set metadata.revision to {next_revision} and append a history entry "
        "for this iteration.\n\n"
        f"Project id: {project_id or _document_metadata(document).get('project_id') or 'unknown'}\n"
        f"Original prompt: {original_prompt or 'unknown'}\n"
        f"Iteration instruction: {instruction}\n\n"
        f"{namespace_block}"
        "Current project JSON:\n"
        f"{json.dumps(compact, indent=2, sort_keys=True, default=str)}"
    )


# ---------------------------------------------------------------------------
# History and finalization
# ---------------------------------------------------------------------------


def append_history_entry(
    document: Dict[str, Any],
    *,
    instruction: str,
    revision: int,
    previous_revision: int,
    mode: str,
    provider: str = "",
    model: str = "",
    now: str = "",
) -> Dict[str, Any]:
    """Append a deduplicated iteration entry to ``document["history"]`` in place."""
    version = f"0.{revision}"
    history = [
        dict(item)
        for item in (document.get("history") or [])
        if isinstance(item, dict)
        and item.get("revision") != revision
        and item.get("version") != version
    ]
    history.append(
        {
            "version": version,
            "revision": revision,
            "previous_revision": previous_revision,
            "description": instruction,
            "change_type": "iteration",
            "mode": mode,
            "provider": provider,
            "model": model,
            "created_at": now,
        }
    )
    document["history"] = history
    return document


def finalize_iteration(
    revised: Mapping[str, Any],
    *,
    base: Mapping[str, Any],
    instruction: str,
    provider: str = "",
    model: str = "",
    target_namespace: Optional[str] = None,
    mode: str = "llm",
    validate: Optional[ValidateCallable] = None,
    now: str = "",
) -> Tuple[Dict[str, Any], IterationMetadata]:
    """Validate, stamp, and record one iteration; returns (document, metadata).

    Rejects placeholder title/description in ``revised["overview"]`` with
    ValueError, reruns the injected ``validate`` (which returns a
    ``(validation_summary_dict, is_valid)`` pair), normalizes revision and
    per-namespace versions via ``attach_project_object_metadata``, and appends
    a history entry. ``now`` is a caller-injected timestamp string.
    """
    if not isinstance(revised, Mapping) or not isinstance(base, Mapping):
        raise TypeError("Project documents must be mappings.")
    instruction = normalize_iteration_instruction(instruction)
    normalized_namespace = normalize_project_namespace(target_namespace)
    document = copy.deepcopy(dict(revised))

    overview = document.get("overview")
    if isinstance(overview, dict) and (
        is_placeholder_text(overview.get("title"))
        or is_placeholder_text(overview.get("description"))
    ):
        raise ValueError(
            "Project iteration output was unusable: placeholder overview fields "
            f"(title={overview.get('title')!r}, "
            f"description={overview.get('description')!r})."
        )

    validation_error: Optional[str] = None
    if validate is not None:
        try:
            summary, is_valid = validate(document)
        except Exception as exc:
            raise ValueError(f"Injected validator failed: {exc}") from exc
        document["validation"] = dict(summary) if isinstance(summary, Mapping) else summary
        document["is_valid"] = bool(is_valid)
        if not is_valid:
            validation_error = "Validation reported the revised document invalid."

    previous_revision = document_revision(base)
    revision = previous_revision + 1
    project_id = _project_id(base)

    metadata = _document_metadata(document)
    metadata.update(
        {
            "project_id": project_id,
            "revision": revision,
            "previous_revision": previous_revision,
            "iterated_at": now,
            "iteration_instruction": instruction,
            "iteration_target_namespace": normalized_namespace,
            "iteration_mode": mode,
            "iteration_provider": provider,
            "iteration_model": model,
            "last_iteration": {
                "instruction": instruction,
                "mode": mode,
                "target_namespace": normalized_namespace,
                "previous_revision": previous_revision,
                "revision": revision,
                "provider": provider,
                "model": model,
                "created_at": now,
            },
        }
    )
    document["metadata"] = metadata
    document = attach_project_object_metadata(
        document, target_namespace=normalized_namespace, updated_at=now
    )
    append_history_entry(
        document,
        instruction=instruction,
        revision=revision,
        previous_revision=previous_revision,
        mode=mode,
        provider=provider,
        model=model,
        now=now,
    )
    return document, IterationMetadata(
        mode=mode,
        revision=revision,
        previous_revision=previous_revision,
        instruction=instruction,
        provider=provider,
        model=model,
        target_namespace=normalized_namespace,
        validation_error=validation_error,
    )


def build_metadata_only_iteration(
    document: Mapping[str, Any],
    instruction: str,
    *,
    provider: str = "",
    model: str = "",
    project_id: Optional[str] = None,
    target_namespace: Optional[str] = None,
    validate: Optional[ValidateCallable] = None,
    now: str = "",
) -> Tuple[Dict[str, Any], IterationMetadata]:
    """Record an iteration request without pretending project content changed."""
    if not isinstance(document, Mapping):
        raise TypeError("Project document must be a mapping.")
    base = copy.deepcopy(dict(document))
    revised = copy.deepcopy(base)
    metadata = _document_metadata(revised)
    if project_id:
        metadata["project_id"] = str(project_id)
    requested = metadata.get("pending_iteration_instructions")
    pending = (
        [item for item in requested if isinstance(item, dict)]
        if isinstance(requested, list)
        else []
    )
    pending.append(
        {
            "instruction": normalize_iteration_instruction(instruction),
            "target_namespace": normalize_project_namespace(target_namespace),
            "created_at": now,
        }
    )
    metadata["pending_iteration_instructions"] = pending
    revised["metadata"] = metadata
    return finalize_iteration(
        revised,
        base=base,
        instruction=instruction,
        provider=provider,
        model=model,
        target_namespace=target_namespace,
        mode="metadata-only",
        validate=validate,
        now=now,
    )


# ---------------------------------------------------------------------------
# Iterator
# ---------------------------------------------------------------------------


class ProjectIterator:
    """Provider-agnostic document revision engine over plain dict documents.

    ``llm`` is any callable ``prompt -> dict`` (the complete revised document).
    Without an llm the iterate call degrades to a metadata-only iteration that
    records the pending instruction honestly instead of faking a revision.
    """

    def __init__(
        self,
        *,
        provider: str = "",
        model: str = "",
        validate: Optional[ValidateCallable] = None,
    ) -> None:
        self.provider = provider
        self.model = model
        self.validate = validate

    def iterate(
        self,
        document: Mapping[str, Any],
        instruction: str,
        *,
        llm: Optional[LLMCallable] = None,
        original_prompt: Optional[str] = None,
        project_id: Optional[str] = None,
        target_namespace: Optional[str] = None,
        payload_fn: Optional[PayloadCallable] = None,
        now: str = "",
    ) -> Tuple[Dict[str, Any], IterationMetadata]:
        instruction = normalize_iteration_instruction(instruction)
        normalized_namespace = normalize_project_namespace(target_namespace)

        if llm is None:
            return build_metadata_only_iteration(
                document,
                instruction,
                provider=self.provider,
                model=self.model,
                project_id=project_id,
                target_namespace=normalized_namespace,
                validate=self.validate,
                now=now,
            )

        prompt = build_iteration_prompt(
            document,
            instruction,
            original_prompt=original_prompt,
            project_id=project_id,
            target_namespace=normalized_namespace,
            payload_fn=payload_fn,
        )
        revised = llm(prompt)
        if not isinstance(revised, Mapping):
            raise TypeError(
                "Iteration llm callable must return a dict document, got "
                f"{type(revised).__name__}."
            )
        return finalize_iteration(
            revised,
            base=document,
            instruction=instruction,
            provider=self.provider,
            model=self.model,
            target_namespace=normalized_namespace,
            mode="llm",
            validate=self.validate,
            now=now,
        )


# ---------------------------------------------------------------------------
# Metadata/output findings and self-correction planning
# ---------------------------------------------------------------------------


def metadata_output_findings(document: Mapping[str, Any]) -> List[str]:
    """Generic port of Forma-OSS's operation-status and output checks.

    Findings: operations in metadata["operation_statuses"] that are failed or
    pending; any ``<stem>_requested`` flag whose ``<stem>_status`` is not
    "succeeded" (a requested-but-not-succeeded output); and any
    ``<stem>_error`` still present after ``<stem>_status`` reads "succeeded"
    (a stale error after success).
    """
    metadata = _document_metadata(document)
    findings: List[str] = []

    for operation in metadata.get("operation_statuses") or []:
        if not isinstance(operation, dict):
            continue
        status = str(operation.get("status") or "").lower()
        if status in {"failed", "pending"}:
            label = operation.get("label") or operation.get("id") or "operation"
            error = operation.get("error") or operation.get("reason") or "no detail"
            findings.append(f"Operation {label!r} is {status}: {error}")

    for key in sorted(metadata):
        text_key = str(key)
        if not text_key.endswith("_requested") or not metadata.get(key):
            continue
        stem = text_key[: -len("_requested")]
        status = str(metadata.get(f"{stem}_status") or "").lower()
        error = metadata.get(f"{stem}_error")
        if status != "succeeded":
            detail = f": {error}" if error else "."
            findings.append(
                f"Output {stem!r} was requested but did not succeed"
                f" (status={status or 'missing'}){detail}"
            )
        elif error:
            findings.append(
                f"Output {stem!r} still carries a stale error after success: {error}"
            )

    return findings


def _issue_field(issue: Any, name: str) -> str:
    if isinstance(issue, Mapping):
        return str(issue.get(name) or "")
    return str(getattr(issue, name, "") or "")


def _dedupe_issues(issues: Iterable[Any]) -> List[Any]:
    deduped: List[Any] = []
    seen: set[Tuple[str, str, str]] = set()
    for issue in issues:
        key = (
            _issue_field(issue, "severity").upper(),
            _issue_field(issue, "category"),
            _issue_field(issue, "description"),
        )
        if key in seen:
            continue
        seen.add(key)
        deduped.append(issue)
    return deduped


def _stored_validation_issues(document: Mapping[str, Any]) -> List[Any]:
    validation = document.get("validation")
    if not isinstance(validation, Mapping):
        return []
    issues: List[Any] = []
    for bucket in ("critical", "warning", "issues"):
        for item in validation.get(bucket) or []:
            issues.append(item)
    return issues


class SelfCorrectionPlanner:
    """Plans validation-driven repairs as smallest-coherent-mutation instructions."""

    max_issue_lines = 8
    max_output_lines = 8

    def plan_correction(
        self,
        document: Mapping[str, Any],
        *,
        issues: Optional[Iterable[Any]] = None,
        output_findings: Optional[Iterable[str]] = None,
        target_namespace: Optional[str] = None,
    ) -> SelfCorrectionPlan:
        if not isinstance(document, Mapping):
            raise TypeError("Project document must be a mapping.")
        supplied = list(issues) if issues is not None else []
        all_issues = _dedupe_issues([*supplied, *_stored_validation_issues(document)])
        critical = [
            issue for issue in all_issues
            if _issue_field(issue, "severity").upper() == "CRITICAL"
        ]
        warnings = [
            issue for issue in all_issues
            if _issue_field(issue, "severity").upper() == "WARNING"
        ]
        findings = (
            list(output_findings)
            if output_findings is not None
            else metadata_output_findings(document)
        )
        normalized_namespace = normalize_project_namespace(target_namespace)
        if normalized_namespace is None:
            normalized_namespace = "product.geometry" if all_issues else "project.docs"

        if all_issues or findings:
            issue_lines = [
                "- {severity} {category}: {description} Remediation: {troubleshooting}".format(
                    severity=_issue_field(issue, "severity") or "ISSUE",
                    category=_issue_field(issue, "category") or "general",
                    description=_issue_field(issue, "description") or "no description",
                    troubleshooting=_issue_field(issue, "troubleshooting")
                    or "no remediation provided",
                )
                for issue in all_issues[: self.max_issue_lines]
            ]
            output_lines = [
                f"- Metadata/output: {finding}"
                for finding in findings[: self.max_output_lines]
            ]
            instruction = (
                "Self-correct the project by resolving these validation and "
                "output/metadata issues while preserving the user's intent.\n"
                "Make the smallest coherent mutation that improves the current "
                "revision. If an external service failed, do not fabricate "
                "sources; record the limitation clearly in project docs/history "
                "and remove unsupported claims.\n"
                + "\n".join([*issue_lines, *output_lines])
            )
        else:
            instruction = (
                "Self-review this project namespace, metadata, and generated "
                "outputs for consistency. Preserve the current design unless a "
                "small correction is needed to keep the project document "
                "internally coherent."
            )

        return SelfCorrectionPlan(
            target_namespace=normalized_namespace,
            instruction=instruction,
            critical_issue_count=len(critical),
            warning_issue_count=len(warnings),
            output_issue_count=len(findings),
        )


# ---------------------------------------------------------------------------
# Selfcheck
# ---------------------------------------------------------------------------


def _sample_document() -> Dict[str, Any]:
    return {
        "overview": {"title": "Bracket", "description": "L-bracket with slots"},
        "ops": [
            {"op": "sketch_rect", "id": "op1", "w": 40, "h": 20},
            {"op": "extrude", "id": "op2", "depth": 5},
        ],
        "validation": {"critical": [], "warning": []},
        "is_valid": True,
        "history": [{"version": "0.1", "revision": 1, "description": "initial"}],
        "metadata": {"revision": 1, "project_id": "bracket-42"},
    }


def _run_selfcheck() -> int:
    stamp = "2026-01-01T00:00:00Z"
    later = "2026-01-02T00:00:00Z"

    # Instruction normalization and placeholder detection.
    assert normalize_iteration_instruction("  add a fillet ") == "add a fillet"
    try:
        normalize_iteration_instruction("   ")
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError for empty instruction")
    assert is_placeholder_text("Unknown") and is_placeholder_text("")
    assert not is_placeholder_text("Bracket")

    # Compaction reuses redaction.
    compacted = compact_document_for_iteration(
        {"metadata": {"render_image": "data:image/png;base64,AAAA"}}
    )
    assert compacted["metadata"]["render_image"].startswith("<redacted data url:")

    # Prompt embeds the namespace payload and revision bump.
    base = _sample_document()
    prompt = build_iteration_prompt(
        base, "widen the slots", target_namespace="product.geometry"
    )
    assert "Target namespace: product.geometry" in prompt
    assert "sketch_rect" in prompt
    assert "Set metadata.revision to 2" in prompt

    # Metadata-only iteration: pending instruction recorded, revision bumped.
    iterator = ProjectIterator(provider="stub", model="none")
    doc1, meta1 = iterator.iterate(base, "widen the slots", now=stamp)
    assert meta1.mode == "metadata-only"
    assert meta1.revision == 2 and meta1.previous_revision == 1
    pending = doc1["metadata"]["pending_iteration_instructions"]
    assert pending and pending[0]["instruction"] == "widen the slots"
    assert doc1["metadata"]["project_object"]["object_id"] == "bracket-42"
    assert doc1["history"][-1]["revision"] == 2
    assert doc1["history"][-1]["created_at"] == stamp
    assert base["metadata"]["revision"] == 1  # base untouched

    # Full round trip with an injected llm and validator.
    def fake_llm(prompt_text: str) -> Dict[str, Any]:
        revised = copy.deepcopy(base)
        revised["ops"].append({"op": "fillet", "id": "op3", "radius": 2})
        return revised

    def fake_validate(document: Dict[str, Any]) -> Tuple[Dict[str, Any], bool]:
        ok = len(document.get("ops") or []) >= 3
        return {"critical": [] if ok else [{"severity": "CRITICAL"}], "warning": []}, ok

    iterator2 = ProjectIterator(provider="stub", model="fake-1", validate=fake_validate)
    doc2, meta2 = iterator2.iterate(
        base,
        "add a fillet",
        llm=fake_llm,
        target_namespace="product.geometry",
        now=later,
    )
    assert meta2.mode == "llm" and meta2.revision == 2
    assert meta2.validation_error is None
    assert doc2["is_valid"] is True
    assert doc2["metadata"]["revision"] == 2
    versions = doc2["metadata"]["project_object"]["namespace_versions"]
    assert versions["product.geometry"] == 2
    assert doc2["history"][-1]["mode"] == "llm"
    assert doc2["history"][-1]["model"] == "fake-1"
    assert "product.geometry" in DEFAULT_PROJECT_NAMESPACES.names

    # Placeholder output is rejected.
    try:
        finalize_iteration(
            {"overview": {"title": "unknown", "description": "x"}},
            base=base,
            instruction="noop",
            now=stamp,
        )
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError for placeholder overview")

    # Metadata/output findings: failed op, unmet request, stale error.
    findings_doc = {
        "metadata": {
            "operation_statuses": [
                {"id": "mesh", "status": "failed", "error": "boolean failed"}
            ],
            "render_requested": True,
            "render_status": "pending",
            "export_requested": True,
            "export_status": "succeeded",
            "export_error": "old timeout",
        }
    }
    findings = metadata_output_findings(findings_doc)
    assert len(findings) == 3
    assert any("failed" in finding for finding in findings)
    assert any("did not succeed" in finding for finding in findings)
    assert any("stale error" in finding for finding in findings)

    # Self-correction plan from sample issues, deduped.
    planner = SelfCorrectionPlanner()
    issues = [
        {
            "severity": "CRITICAL",
            "category": "geometry",
            "description": "Extrude op2 has zero depth.",
            "troubleshooting": "Set a positive depth.",
        },
        {
            "severity": "CRITICAL",
            "category": "geometry",
            "description": "Extrude op2 has zero depth.",
            "troubleshooting": "Set a positive depth.",
        },
        {
            "severity": "WARNING",
            "category": "docs",
            "description": "Missing fabrication note.",
            "troubleshooting": "Add a note.",
        },
    ]
    plan = planner.plan_correction(findings_doc, issues=issues)
    assert plan.target_namespace == "product.geometry"
    assert plan.critical_issue_count == 1  # duplicates collapsed
    assert plan.warning_issue_count == 1
    assert plan.output_issue_count == 3
    assert "smallest coherent mutation" in plan.instruction
    assert "Extrude op2 has zero depth." in plan.instruction

    # No issues, no findings: docs namespace and review instruction.
    clean_plan = planner.plan_correction({"metadata": {}})
    assert clean_plan.target_namespace == "project.docs"
    assert clean_plan.critical_issue_count == 0
    assert "Self-review" in clean_plan.instruction

    print(
        "PASS project_iteration selfcheck: metadata_only_rev=%d llm_rev=%d "
        "findings=%d plan_ns=%s"
        % (meta1.revision, meta2.revision, len(findings), plan.target_namespace)
    )
    return 0


def main(argv: Optional[Iterable[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        prog="harnesscad.agents.agent.project_iteration",
        description=(
            "Namespace-aware project iteration and self-correction engine "
            "(ported from Forma-OSS)."
        ),
    )
    parser.add_argument(
        "--selfcheck",
        action="store_true",
        help="run deterministic iteration/self-correction assertions with "
        "injected llm and validator doubles.",
    )
    args = parser.parse_args(list(argv) if argv is not None else None)

    if not args.selfcheck:
        parser.print_help()
        return 0
    try:
        return _run_selfcheck()
    except AssertionError as exc:
        print(f"SELFCHECK FAILED: {exc}", file=sys.stderr)
        return 1


__all__ = [
    "DEFAULT_CONTEXT_MAX_STRING_CHARS",
    "IterationMetadata",
    "PLACEHOLDER_TEXT_VALUES",
    "ProjectIterator",
    "SelfCorrectionPlan",
    "SelfCorrectionPlanner",
    "append_history_entry",
    "build_iteration_prompt",
    "build_metadata_only_iteration",
    "compact_document_for_iteration",
    "document_revision",
    "finalize_iteration",
    "is_placeholder_text",
    "main",
    "metadata_output_findings",
    "next_revision_number",
    "normalize_iteration_instruction",
]


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
