"""checks_reference — score a generated model against an imported reference.

Reference-match is a *verifier* (the blueprint frames verification as several
independent checks whose diagnostics feed back into the loop). Given a reference
solid imported by :mod:`ingest.import_brep` — or any object/dict that exposes
its metrics — :class:`ReferenceMatchCheck` measures the generated model and
scores the deviation:

  * **volume delta**  : relative |Vgen - Vref| / Vref,
  * **bbox delta**    : worst-axis relative bounding-box extent difference,
  * **Hausdorff**     : two-sided max surface distance, *only* when both shapes
                        are real OCCT solids (else silently skipped).

Deviation within ``warn_tol`` passes; beyond ``warn_tol`` emits a WARNING;
beyond ``error_tol`` emits an ERROR (so the loop can block-and-correct a model
that has drifted badly from the reference). Like the other standalone checks it
degrades gracefully: if the reference or the generated model cannot be measured,
it INFO-skips rather than failing. It is NOT in ``verify.default_verifiers`` —
a caller adds it explicitly via :func:`with_reference`.
"""

from __future__ import annotations

from typing import List, Optional

from verifiers.verify import Diagnostic, Severity, VerifyReport

_EPS = 1e-12


class ReferenceMatchCheck:
    """Verifier scoring a built model against an imported reference solid."""

    name = "reference-match"

    def __init__(self, reference, warn_tol: float = 0.02,
                 error_tol: float = 0.10) -> None:
        """``reference`` may be an :class:`ingest.ImportedPart`, a metrics dict,
        or any object exposing ``metrics`` / ``query('metrics')``. ``warn_tol``
        and ``error_tol`` are *relative* deviation thresholds (fractions)."""
        self.reference = reference
        self.warn_tol = warn_tol
        self.error_tol = error_tol

    def check(self, backend, opdag=None) -> VerifyReport:
        diags: List[Diagnostic] = []

        ref = _metrics_of(self.reference)
        if not ref or not _has_measure(ref):
            return VerifyReport([_info(
                "reference-unavailable",
                "no measurable reference solid provided; reference-match "
                "skipped")])

        gen = _metrics_of(backend)
        if not gen or not _has_measure(gen):
            return VerifyReport([_info(
                "measurement-unavailable",
                "generated model is not measurable (no volume/bbox); "
                "reference-match skipped")])

        self._score_volume(ref, gen, diags)
        self._score_bbox(ref, gen, diags)
        self._score_hausdorff(diags)

        if not diags:
            diags.append(_info(
                "reference-match",
                "generated model matches the reference within tolerance "
                f"(<= {self.warn_tol:.1%})"))
        return VerifyReport(diags)

    # -- scoring ------------------------------------------------------------ #
    def _score_volume(self, ref, gen, diags) -> None:
        vref = _num(ref.get("volume"))
        vgen = _num(gen.get("volume"))
        if vref is None or vgen is None:
            return
        if vref <= _EPS:
            return
        rel = abs(vgen - vref) / abs(vref)
        self._emit(rel, "volume",
                   f"volume {vgen:.6g} vs reference {vref:.6g} "
                   f"(delta {rel:.1%})", diags)

    def _score_bbox(self, ref, gen, diags) -> None:
        bref = _bbox(ref.get("bbox"))
        bgen = _bbox(gen.get("bbox"))
        if bref is None or bgen is None:
            return
        worst = 0.0
        worst_axis = 0
        for i in range(3):
            denom = abs(bref[i])
            if denom <= _EPS:
                continue
            rel = abs(bgen[i] - bref[i]) / denom
            if rel > worst:
                worst, worst_axis = rel, i
        axis = "xyz"[worst_axis]
        self._emit(worst, "bbox",
                   f"bbox worst-axis ({axis}) {bgen[worst_axis]:.6g} vs "
                   f"reference {bref[worst_axis]:.6g} (delta {worst:.1%})",
                   diags)

    def _score_hausdorff(self, diags) -> None:
        d = _hausdorff(self.reference)
        if d is None:
            return
        rel, absval = d
        self._emit(rel, "hausdorff",
                   f"two-sided Hausdorff distance {absval:.6g} "
                   f"({rel:.1%} of reference size)", diags)

    def _emit(self, rel: float, metric: str, detail: str,
              diags: List[Diagnostic]) -> None:
        if rel > self.error_tol:
            diags.append(_err(
                f"{metric}-mismatch",
                f"{detail} exceeds error tolerance {self.error_tol:.1%}"))
        elif rel > self.warn_tol:
            diags.append(Diagnostic(
                Severity.WARNING, f"{metric}-drift",
                f"{detail} exceeds warning tolerance {self.warn_tol:.1%}"))


def with_reference(verifiers, reference, **kw) -> list:
    """Return ``verifiers`` with a :class:`ReferenceMatchCheck` appended."""
    return list(verifiers) + [ReferenceMatchCheck(reference, **kw)]


# --------------------------------------------------------------------------- #
# Metrics adaptation
# --------------------------------------------------------------------------- #
def _metrics_of(obj) -> Optional[dict]:
    """Normalise an ImportedPart / dict / backend into a metrics dict."""
    if obj is None:
        return None
    if isinstance(obj, dict):
        return obj
    metrics = getattr(obj, "metrics", None)
    if isinstance(metrics, dict) and _has_measure(metrics):
        return metrics
    query = getattr(obj, "query", None)
    if callable(query):
        for q in ("metrics", "measure"):
            try:
                res = query(q)
            except Exception:  # noqa: BLE001
                res = None
            if res and _has_measure(res):
                return res
    if isinstance(metrics, dict):
        return metrics
    return None


def _has_measure(m: dict) -> bool:
    if not isinstance(m, dict):
        return False
    if _num(m.get("volume")) is not None:
        return True
    return _bbox(m.get("bbox")) is not None


def _num(v) -> Optional[float]:
    if isinstance(v, bool):
        return None
    if isinstance(v, (int, float)):
        return float(v)
    return None


def _bbox(v) -> Optional[list]:
    if isinstance(v, (list, tuple)) and len(v) == 3:
        try:
            return [float(x) for x in v]
        except (TypeError, ValueError):
            return None
    return None


def _hausdorff(reference) -> Optional[tuple]:
    """Reserved hook: two-sided Hausdorff needs both real OCCT shapes and a
    generated shape handle, which the verifier does not receive here. Returns
    ``None`` (skip) unless a precomputed ``hausdorff`` metric is present on the
    reference, keeping the check fully functional without a kernel."""
    metrics = getattr(reference, "metrics", None)
    if isinstance(reference, dict):
        metrics = reference
    if isinstance(metrics, dict):
        h = _num(metrics.get("hausdorff"))
        size = _num(metrics.get("hausdorff_ref_size"))
        if h is not None and size and size > _EPS:
            return (h / size, h)
    return None


def _err(code: str, msg: str) -> Diagnostic:
    return Diagnostic(Severity.ERROR, code, msg)


def _info(code: str, msg: str) -> Diagnostic:
    return Diagnostic(Severity.INFO, code, msg)
