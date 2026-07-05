"""Conformance-certificate exporter — roll a set of verifiers up into a
signed-by-content, rule-by-rule pass/fail record.

The blueprint's plural verifier produces many independent
:class:`verify.VerifyReport` s (constraint, B-rep, DFM, standards, compliance,
assembly, interference, completeness, kinematics, functional …). A human — or a
downstream gate — needs one *certificate* that aggregates them: which rule
passed, which failed, what was measured, and a stable fingerprint that ties the
verdict to the exact model it was computed against.

:class:`ConformanceReport` is that exporter. It is **not** a
:class:`verify.Verifier`: it *runs* verifiers and aggregates their output.
:meth:`ConformanceReport.from_verifiers` executes each verifier once, rolls every
diagnostic into a structured rule record
``{rule, severity, message, where, verdict}`` grouped per check, records the
``query('metrics')`` measurements, and computes an op-DAG provenance hash plus a
model digest.

No cryptographic signing is needed: the certificate is deterministic, so a
SHA-256 *content hash* of the report body IS the signature — recompute it and it
either matches (authentic, unmodified) or it does not. :meth:`to_dict` /
:meth:`to_json` / :meth:`to_markdown` render it; the markdown is a readable
certificate (title, model digest, per-check verdicts, summary pass/fail).
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import List, Optional

from verifiers.verify import Severity, VerifyReport


# --------------------------------------------------------------------------- #
# Rule-by-rule records
# --------------------------------------------------------------------------- #
def _verdict_for(severity: str) -> str:
    """A single diagnostic's verdict: ERROR -> 'fail', otherwise 'pass'."""
    return "fail" if severity == Severity.ERROR.value else "pass"


@dataclass
class RuleRecord:
    """One diagnostic seen as a rule outcome: ``{rule, severity, message,
    where, verdict}``."""

    rule: str
    severity: str
    message: str
    where: Optional[str]
    verdict: str

    def to_dict(self) -> dict:
        return {
            "rule": self.rule,
            "severity": self.severity,
            "message": self.message,
            "where": self.where,
            "verdict": self.verdict,
        }


@dataclass
class CheckRecord:
    """The rolled-up outcome of a single verifier."""

    name: str
    verdict: str                       # 'pass' | 'fail'
    severity: str                      # worst severity seen ('error'|'warning'|'info'|'none')
    rules: List[RuleRecord] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "verdict": self.verdict,
            "severity": self.severity,
            "rules": [r.to_dict() for r in self.rules],
        }


_SEVERITY_RANK = {
    Severity.ERROR.value: 3,
    Severity.WARNING.value: 2,
    Severity.INFO.value: 1,
}


def _worst_severity(rules: List[RuleRecord]) -> str:
    worst = "none"
    worst_rank = 0
    for r in rules:
        rank = _SEVERITY_RANK.get(r.severity, 0)
        if rank > worst_rank:
            worst_rank = rank
            worst = r.severity
    return worst


# --------------------------------------------------------------------------- #
# The certificate
# --------------------------------------------------------------------------- #
@dataclass
class ConformanceReport:
    """A structured, content-hashed conformance certificate.

    Build it with :meth:`from_verifiers`; render it with :meth:`to_dict` /
    :meth:`to_json` / :meth:`to_markdown`. The ``signature`` is a SHA-256 hash of
    the report body (everything except the signature itself), so it is
    reproducible and tamper-evident without any key material.
    """

    title: str = "HarnessCAD Conformance Certificate"
    checks: List[CheckRecord] = field(default_factory=list)
    measurements: dict = field(default_factory=dict)
    provenance: dict = field(default_factory=dict)

    # -- construction ------------------------------------------------------- #
    @classmethod
    def from_verifiers(cls, backend, opdag, verifiers,
                       title: str = "HarnessCAD Conformance Certificate"
                       ) -> "ConformanceReport":
        """Run every verifier once against ``(backend, opdag)`` and aggregate."""
        checks: List[CheckRecord] = []
        for v in verifiers:
            checks.append(_run_one(v, backend, opdag))

        measurements = _measurements(backend)
        provenance = {
            "opdag_hash": _opdag_hash(opdag),
            "model_digest": _model_digest(backend),
        }
        return cls(title=title, checks=checks,
                   measurements=measurements, provenance=provenance)

    # -- roll-ups ----------------------------------------------------------- #
    @property
    def verdict(self) -> str:
        """Overall verdict: 'pass' only if every check passed."""
        return "fail" if any(c.verdict == "fail" for c in self.checks) else "pass"

    @property
    def ok(self) -> bool:
        return self.verdict == "pass"

    def counts(self) -> dict:
        passed = sum(1 for c in self.checks if c.verdict == "pass")
        failed = sum(1 for c in self.checks if c.verdict == "fail")
        return {"total": len(self.checks), "passed": passed, "failed": failed}

    # -- serialisation ------------------------------------------------------ #
    def _body(self) -> dict:
        """The report body that the signature is computed over (no signature)."""
        return {
            "title": self.title,
            "verdict": self.verdict,
            "summary": self.counts(),
            "checks": [c.to_dict() for c in self.checks],
            "measurements": self.measurements,
            "provenance": self.provenance,
        }

    def signature(self) -> str:
        """SHA-256 content hash of the report body = the certificate signature."""
        blob = json.dumps(self._body(), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(blob.encode("utf-8")).hexdigest()

    def to_dict(self) -> dict:
        body = self._body()
        body["signature"] = self.signature()
        return body

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), sort_keys=True, indent=indent)

    def to_markdown(self) -> str:
        counts = self.counts()
        verdict = self.verdict.upper()
        lines: List[str] = []
        lines.append(f"# {self.title}")
        lines.append("")
        lines.append(f"**Verdict: {verdict}** "
                     f"({counts['passed']}/{counts['total']} checks passed, "
                     f"{counts['failed']} failed)")
        lines.append("")

        # Model digest / provenance.
        lines.append("## Model")
        digest = self.provenance.get("model_digest") or "(unavailable)"
        ohash = self.provenance.get("opdag_hash") or "(unavailable)"
        lines.append(f"- Model digest: `{digest}`")
        lines.append(f"- Op-DAG provenance hash: `{ohash}`")
        lines.append("")

        # Measurements.
        if self.measurements:
            lines.append("## Measurements")
            for key in sorted(self.measurements):
                lines.append(f"- {key}: {self.measurements[key]}")
            lines.append("")

        # Per-check verdicts.
        lines.append("## Checks")
        for c in self.checks:
            mark = "PASS" if c.verdict == "pass" else "FAIL"
            lines.append(f"### [{mark}] {c.name}")
            if not c.rules:
                lines.append("- (no diagnostics)")
            for r in c.rules:
                where = f" @ {r.where}" if r.where else ""
                lines.append(
                    f"- **{r.severity}** `{r.rule}`{where}: {r.message}")
            lines.append("")

        lines.append("## Signature")
        lines.append(f"- SHA-256 content hash: `{self.signature()}`")
        lines.append("")
        return "\n".join(lines)


# --------------------------------------------------------------------------- #
# Internals
# --------------------------------------------------------------------------- #
def _run_one(verifier, backend, opdag) -> CheckRecord:
    """Run one verifier and roll its diagnostics into a :class:`CheckRecord`.

    A verifier that itself raises is recorded as a failing check rather than
    crashing the whole certificate (graceful degrade)."""
    name = getattr(verifier, "name", type(verifier).__name__)
    try:
        report: VerifyReport = verifier.check(backend, opdag)
        diagnostics = list(report.diagnostics)
        ok = report.ok
    except Exception as exc:  # noqa: BLE001 - one bad verifier must not kill the run
        rule = RuleRecord(
            rule="verifier-error",
            severity=Severity.ERROR.value,
            message=f"verifier '{name}' raised {type(exc).__name__}: {exc}",
            where=name,
            verdict="fail",
        )
        return CheckRecord(name=name, verdict="fail",
                           severity=Severity.ERROR.value, rules=[rule])

    rules: List[RuleRecord] = []
    for d in diagnostics:
        sev = d.severity.value if hasattr(d.severity, "value") else str(d.severity)
        rules.append(RuleRecord(
            rule=d.code,
            severity=sev,
            message=d.message,
            where=d.where,
            verdict=_verdict_for(sev),
        ))
    return CheckRecord(
        name=name,
        verdict="pass" if ok else "fail",
        severity=_worst_severity(rules),
        rules=rules,
    )


def _measurements(backend) -> dict:
    """The ``query('metrics')`` measurements (falling back to 'measure' then
    'summary'), as a plain JSON-safe dict. Empty when nothing is measurable."""
    for q in ("metrics", "measure", "summary"):
        m = _safe_query(backend, q)
        if m:
            return _json_safe(m)
    return {}


def _opdag_hash(opdag) -> str:
    """A deterministic provenance hash of the op stream behind ``opdag``."""
    ops = _iter_ops(opdag)
    try:
        payload = [o.to_dict() if hasattr(o, "to_dict") else o for o in ops]
        blob = json.dumps(payload, sort_keys=True, separators=(",", ":"),
                          default=str)
    except Exception:  # noqa: BLE001 - never let provenance crash the export
        blob = repr(ops)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def _model_digest(backend) -> Optional[str]:
    """The backend's own state digest when it exposes one, else None."""
    fn = getattr(backend, "state_digest", None)
    if callable(fn):
        try:
            return str(fn())
        except Exception:  # noqa: BLE001 - a backend without a stable digest degrades
            return None
    return None


def _iter_ops(opdag) -> list:
    if opdag is None:
        return []
    ops_attr = getattr(opdag, "ops", None)
    if callable(ops_attr):
        try:
            return list(ops_attr())
        except Exception:  # noqa: BLE001
            return []
    if isinstance(opdag, (list, tuple)):
        return list(opdag)
    return []


def _safe_query(backend, q: str) -> dict:
    try:
        result = backend.query(q)
    except Exception:  # noqa: BLE001 - an unsupported query must degrade, not crash
        return {}
    return result if isinstance(result, dict) else {}


def _json_safe(value):
    """Coerce a query result into a JSON-serialisable structure (deterministic)."""
    try:
        json.dumps(value)
        return value
    except (TypeError, ValueError):
        if isinstance(value, dict):
            return {str(k): _json_safe(v) for k, v in value.items()}
        if isinstance(value, (list, tuple)):
            return [_json_safe(v) for v in value]
        return str(value)
