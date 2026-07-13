"""Independent compiler-judged desirable/undesirable preference records."""

from __future__ import annotations

from dataclasses import dataclass, asdict
import hashlib
import json


@dataclass(frozen=True)
class BinaryPreference:
    prompt: str
    candidate: object
    desirable: bool
    candidate_digest: str
    reference_digest: str
    reason: str
    metrics: dict
    provenance: dict

    @classmethod
    def create(cls, prompt, candidate, desirable, *, reference,
               reason, metrics=None, provenance=None):
        digest = lambda value: hashlib.sha256(json.dumps(
            value, sort_keys=True, separators=(",", ":"), default=repr).encode()).hexdigest()
        candidate_digest, reference_digest = digest(candidate), digest(reference)
        if not desirable and candidate_digest == reference_digest:
            raise ValueError("reference cannot be labelled undesirable")
        return cls(prompt, candidate, bool(desirable), candidate_digest,
                   reference_digest, reason, dict(metrics or {}),
                   dict(provenance or {}))


def audit_preferences(records):
    labels, issues = {}, []
    for record in records:
        key = (record.prompt, record.candidate_digest, record.reference_digest)
        if key in labels and labels[key] != record.desirable:
            issues.append(("conflicting-label", key))
        labels[key] = record.desirable
    return tuple(issues)
