"""checks_requirements — assert a built model contains what the brief asked for.

The verification half of the NL front-of-pipeline: given the typed
:class:`spec.formalize.RequirementSet` extracted from a brief, check the *built*
model against each countable / parametric ask — "does it actually have the 4
holes?", "is it 100 mm long within tolerance?".

:class:`RequirementsCheck` is a standalone :class:`verify.Verifier`
(``name='requirements'``) a caller adds explicitly (it is not in
``verify.default_verifiers()``). It reads the backend's read-only queries
(``'summary'`` / ``'metrics'`` / ``'measure'``) and, like
:class:`contract.ContractCheck`, degrades gracefully: an unmet, *measurable*
requirement is an ERROR; a requirement the backend cannot measure is an INFO
skip, never an ERROR. So the same requirement set runs against the
dependency-free stub and a real geometry kernel alike.
"""

from __future__ import annotations

from typing import List, Optional

from verify import Diagnostic, Severity, VerifyReport
from spec.formalize import RequirementSet, Requirement

# canonical dimension label -> bounding-box axis index (measure['bbox'] order)
_AXIS_INDEX = {
    "length": 0, "long": 0,
    "width": 1, "wide": 1,
    "height": 2, "tall": 2, "high": 2,
    "depth": 2, "deep": 2,
    "thickness": 2, "thick": 2,
}

_EPS = 1e-9


class RequirementsCheck:
    """Verify each requirement against the built model's numeric queries."""

    name = "requirements"

    def __init__(self, reqset: RequirementSet) -> None:
        self.reqset = reqset

    def check(self, backend, opdag=None) -> VerifyReport:
        diags: List[Diagnostic] = []

        summary = _query(backend, "summary")
        metrics = _query(backend, "metrics")
        measure = _query(backend, "measure")

        default_tol = self._default_tol()

        for r in self.reqset.requirements:
            if r.kind == "count":
                self._check_count(r, summary, metrics, diags)
            elif r.kind in ("dimension", "envelope"):
                self._check_dimension(r, measure, metrics, default_tol, diags)
            else:
                # material / tolerance / feature: not numerically measurable
                # from the backend queries -> INFO skip, never ERROR.
                diags.append(_info(
                    "req-unmeasurable",
                    f"{r.kind} requirement "
                    f"'{r.source_phrase or r.target}' not machine-measurable; "
                    "skipped", r.label))

        return VerifyReport(diags)

    # -- counts ------------------------------------------------------------- #
    def _check_count(self, r: Requirement, summary, metrics,
                     diags: List[Diagnostic]) -> None:
        actual = _count_lookup(r, summary, metrics)
        target = int(r.target) if r.target is not None else 0
        if actual is None:
            diags.append(_info(
                "req-skipped",
                f"count of '{r.label or 'features'}' "
                f"(want {target}) not reported by backend; skipped", r.label))
            return
        if actual < target:
            diags.append(_err(
                "count-unmet",
                f"model reports {actual} '{r.label or 'feature'}'"
                f"(s) but the brief requires {target} "
                f"('{r.source_phrase or ''}')", r.label))

    # -- dimensions --------------------------------------------------------- #
    def _check_dimension(self, r: Requirement, measure, metrics,
                         default_tol: float, diags: List[Diagnostic]) -> None:
        actual = _dim_lookup(r, measure, metrics)
        if r.target is None:
            return
        target = float(r.target)
        if actual is None:
            diags.append(_info(
                "req-skipped",
                f"dimension '{r.label or '?'}'={target:g} not measurable by "
                "backend; skipped", r.label))
            return
        tol = r.tolerance if r.tolerance is not None else default_tol
        if abs(float(actual) - target) > tol + _EPS:
            diags.append(_err(
                "dimension-unmet",
                f"dimension '{r.label or '?'}'={float(actual):.4g} out of "
                f"tolerance {target:g} +/- {tol:g} "
                f"('{r.source_phrase or ''}')", r.label))

    # -- helpers ------------------------------------------------------------ #
    def _default_tol(self) -> float:
        tols = self.reqset.by_kind("tolerance")
        if tols and tols[0].target is not None:
            return float(tols[0].target)
        return 0.0


def with_requirements(verifiers, reqset: RequirementSet) -> list:
    """Return ``verifiers`` with a :class:`RequirementsCheck` appended."""
    return list(verifiers) + [RequirementsCheck(reqset)]


# --------------------------------------------------------------------------- #
# Lookups
# --------------------------------------------------------------------------- #
def _count_lookup(r: Requirement, summary, metrics) -> Optional[float]:
    """Resolve the count for requirement ``r``: prefer a label-specific key in
    'metrics'/'summary' (e.g. 'hole_count', 'holes'), else 'feature_count'."""
    label = (r.label or "").lower()
    keys: List[str] = []
    if label:
        keys += [f"{label}_count", f"{label}s", label]
    keys.append("feature_count")
    for key in keys:
        for src in (metrics, summary):
            if src and isinstance(src.get(key), (int, float)):
                return src[key]
    return None


def _dim_lookup(r: Requirement, measure, metrics) -> Optional[float]:
    """Resolve the built dimension for requirement ``r``: a named key in
    'metrics', else the matching axis of measure['bbox']."""
    label = (r.label or "").lower()
    if metrics and isinstance(metrics.get(label), (int, float)):
        return metrics[label]
    if measure:
        bbox = measure.get("bbox")
        idx = _AXIS_INDEX.get(label)
        if idx is not None and isinstance(bbox, (list, tuple)) and idx < len(bbox):
            v = bbox[idx]
            if isinstance(v, (int, float)):
                return v
    return None


def _query(backend, q: str) -> Optional[dict]:
    """Read a backend query, returning None when unanswered (backends return {}
    for unknown queries) so callers can INFO-skip."""
    try:
        result = backend.query(q)
    except Exception:  # noqa: BLE001 - an unsupported query must degrade
        return None
    return result or None


def _err(code: str, msg: str, where: Optional[str] = None) -> Diagnostic:
    return Diagnostic(Severity.ERROR, code, msg, where)


def _info(code: str, msg: str, where: Optional[str] = None) -> Diagnostic:
    return Diagnostic(Severity.INFO, code, msg, where)
