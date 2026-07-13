"""Deterministic policy gate for sensitive CAD files and metadata.

This module does not claim encryption, federation, or a secure enclave. It
provides the enforceable boundary the local harness can test today: file-type
and size policy, path traversal rejection, metadata redaction, execution-mode
restrictions, content hashing, and append-only audit events.
"""

from __future__ import annotations

import hashlib
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional


_SECRET_KEY = re.compile(
    r"(api[_-]?key|token|secret|password|credential|authorization)", re.I
)
_PII_KEY = re.compile(r"(email|phone|owner|author|customer|client|user)", re.I)


@dataclass(frozen=True)
class DataPolicy:
    allowed_extensions: frozenset[str] = frozenset(
        {".step", ".stp", ".iges", ".igs", ".brep", ".stl", ".dxf", ".json"}
    )
    max_bytes: int = 100 * 1024 * 1024
    execution_mode: str = "on_prem"
    allow_network: bool = False
    redact_pii: bool = True
    redact_secrets: bool = True

    def __post_init__(self) -> None:
        if self.max_bytes <= 0:
            raise ValueError("max_bytes must be positive")
        if self.execution_mode not in {"on_prem", "private_cloud", "public_cloud"}:
            raise ValueError(f"unknown execution_mode {self.execution_mode!r}")
        if self.execution_mode == "on_prem" and self.allow_network:
            raise ValueError("on_prem mode cannot allow network egress")


@dataclass(frozen=True)
class PolicyDecision:
    allowed: bool
    code: str
    reason: str
    content_sha256: Optional[str] = None
    redacted_metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class AuditEvent:
    sequence: int
    action: str
    allowed: bool
    code: str
    path: str
    content_sha256: Optional[str]
    execution_mode: str

    def to_dict(self) -> dict:
        return {
            "sequence": self.sequence,
            "action": self.action,
            "allowed": self.allowed,
            "code": self.code,
            "path": self.path,
            "content_sha256": self.content_sha256,
            "execution_mode": self.execution_mode,
        }


def _redacted_value(value: Any) -> str:
    raw = str(value).encode("utf-8", errors="replace")
    digest = hashlib.sha256(raw).hexdigest()[:12]
    return f"[REDACTED:{digest}]"


def redact_metadata(metadata: Mapping[str, Any], policy: DataPolicy) -> Dict[str, Any]:
    """Return a recursively redacted copy without mutating caller data."""

    def clean(key: str, value: Any) -> Any:
        if policy.redact_secrets and _SECRET_KEY.search(key):
            return _redacted_value(value)
        if policy.redact_pii and _PII_KEY.search(key):
            return _redacted_value(value)
        if isinstance(value, Mapping):
            return {str(k): clean(str(k), v) for k, v in value.items()}
        if isinstance(value, list):
            return [clean(key, item) for item in value]
        if isinstance(value, tuple):
            return tuple(clean(key, item) for item in value)
        return value

    return {str(k): clean(str(k), v) for k, v in metadata.items()}


class SecureIngestGate:
    """Validate local CAD inputs and retain deterministic audit provenance."""

    def __init__(self, policy: Optional[DataPolicy] = None) -> None:
        self.policy = policy or DataPolicy()
        self._events: List[AuditEvent] = []

    @property
    def events(self) -> List[AuditEvent]:
        return list(self._events)

    def _record(self, path: Path, decision: PolicyDecision) -> PolicyDecision:
        self._events.append(AuditEvent(
            sequence=len(self._events) + 1,
            action="ingest",
            allowed=decision.allowed,
            code=decision.code,
            path=str(path),
            content_sha256=decision.content_sha256,
            execution_mode=self.policy.execution_mode,
        ))
        return decision

    def inspect(
        self,
        path: os.PathLike[str] | str,
        metadata: Optional[Mapping[str, Any]] = None,
        *,
        root: Optional[os.PathLike[str] | str] = None,
    ) -> PolicyDecision:
        candidate = Path(path)
        redacted = redact_metadata(metadata or {}, self.policy)

        if root is not None:
            base = Path(root).resolve()
            try:
                candidate.resolve().relative_to(base)
            except (OSError, ValueError):
                return self._record(candidate, PolicyDecision(
                    False, "path-outside-root",
                    "input path resolves outside the authorized root",
                    redacted_metadata=redacted,
                ))

        suffix = candidate.suffix.casefold()
        if suffix not in self.policy.allowed_extensions:
            return self._record(candidate, PolicyDecision(
                False, "extension-denied",
                f"file extension {suffix or '(none)'} is not allowed",
                redacted_metadata=redacted,
            ))
        try:
            stat = candidate.stat()
        except OSError as exc:
            return self._record(candidate, PolicyDecision(
                False, "file-unavailable", str(exc), redacted_metadata=redacted
            ))
        if not candidate.is_file():
            return self._record(candidate, PolicyDecision(
                False, "not-a-file", "input is not a regular file",
                redacted_metadata=redacted,
            ))
        if stat.st_size > self.policy.max_bytes:
            return self._record(candidate, PolicyDecision(
                False, "file-too-large",
                f"{stat.st_size} bytes exceeds {self.policy.max_bytes}",
                redacted_metadata=redacted,
            ))

        digest = hashlib.sha256()
        with candidate.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
        return self._record(candidate, PolicyDecision(
            True, "allowed", "input satisfies local ingestion policy",
            content_sha256=digest.hexdigest(),
            redacted_metadata=redacted,
        ))

    def audit_log(self) -> List[dict]:
        return [event.to_dict() for event in self._events]
