"""Compliance critic — a standalone advisory verifier for manufacturing / export
concerns.

Beyond geometric validity and design-for-manufacturing economics, a part can trip
*non-geometric* gates: export-control review thresholds, region-specific
manufacturing rules, and additive-manufacturing printability limits (unsupported
overhang, minimum feature size). This verifier surfaces those as advisories so a
human can decide, early, whether a design needs review before it is quoted or
shipped.

IMPORTANT — the export-control / ITAR portion is a **coarse, configurable
heuristic, NOT an authoritative determination**. Real export-control
classification (ITAR / EAR, USML categories, ECCNs) is a legal question about the
part's end-use, technology, and jurisdiction that no geometry check can decide.
All this verifier does is raise a "you may need a compliance review" flag when a
configured size/feature threshold is crossed — a prompt for a human expert, never
a ruling. Thresholds are fully configurable and default to deliberately generic
values.

Like :class:`checks_dfm.DFMCheck` and :class:`checks_standards.StandardsCheck`,
every finding is advisory: WARNING for a tripped threshold, INFO for a note or an
unmeasurable skip. This verifier **never** emits an ERROR and can never flip a
:class:`verify.VerifyReport` to ``ok == False``; it never *blocks*.

Standalone by design (not wired into :func:`verify.default_verifiers`); a caller
adds it via :func:`with_compliance`.

What is inspectable today: overall size / volume from ``query('measure')`` and —
when a backend exposes it — additive metrics (max unsupported overhang angle, min
feature size) from an optional ``query('metrics')``. When the backing query is
unavailable (e.g. the stub answers only 'summary'), the dependent check INFO-skips
rather than crashing or erroring.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

from harnesscad.eval.verifiers.verify import Diagnostic, Severity, VerifyReport


# --------------------------------------------------------------------------- #
# Rules (configurable compliance heuristics + limits)
# --------------------------------------------------------------------------- #
@dataclass
class ComplianceRules:
    """Configurable compliance heuristics and additive-manufacturing limits.

    All thresholds are advisory and configurable. The export-control fields are a
    heuristic prompt for human review (see the module docstring), NOT an
    authoritative ITAR/EAR determination.
    """

    # -- export-control heuristic (NOT authoritative) -----------------------
    export_control_enabled: bool = True
    # Overall largest-dimension (mm) / volume (mm^3) above which a review note
    # fires. Deliberately generic defaults; tune per program.
    itar_size_threshold: float = 600.0        # mm (largest bbox edge)
    itar_volume_threshold: float = 5.0e7      # mm^3

    # -- region-specific manufacturing toggles ------------------------------
    # A region label plus its per-region overall-size ceiling (mm). Left None to
    # disable the regional check. Callers can extend regional_limits per market.
    region: Optional[str] = None
    regional_limits: Dict[str, float] = field(default_factory=lambda: {
        "EU": 800.0,
        "US": 1000.0,
        "APAC": 900.0,
    })

    # -- additive-manufacturing limits --------------------------------------
    am_enabled: bool = True
    # Max unsupported overhang angle (degrees from vertical) that prints without
    # support; steeper faces need support / may fail. 45 deg is the common rule.
    max_overhang_angle: float = 45.0
    min_feature_size: float = 0.4             # mm; below this, features may not print

    def to_dict(self) -> dict:
        return {
            "export_control_enabled": self.export_control_enabled,
            "itar_size_threshold": self.itar_size_threshold,
            "itar_volume_threshold": self.itar_volume_threshold,
            "region": self.region,
            "regional_limits": dict(self.regional_limits),
            "am_enabled": self.am_enabled,
            "max_overhang_angle": self.max_overhang_angle,
            "min_feature_size": self.min_feature_size,
        }

    @classmethod
    def from_dict(cls, d: Optional[dict]) -> "ComplianceRules":
        d = d or {}
        defaults = cls()
        regional = d.get("regional_limits")
        if regional is None:
            regional = dict(defaults.regional_limits)
        else:
            regional = {str(k): float(v) for k, v in regional.items()}
        return cls(
            export_control_enabled=bool(d.get(
                "export_control_enabled", defaults.export_control_enabled)),
            itar_size_threshold=float(d.get(
                "itar_size_threshold", defaults.itar_size_threshold)),
            itar_volume_threshold=float(d.get(
                "itar_volume_threshold", defaults.itar_volume_threshold)),
            region=d.get("region", defaults.region),
            regional_limits=regional,
            am_enabled=bool(d.get("am_enabled", defaults.am_enabled)),
            max_overhang_angle=float(d.get(
                "max_overhang_angle", defaults.max_overhang_angle)),
            min_feature_size=float(d.get(
                "min_feature_size", defaults.min_feature_size)),
        )


# --------------------------------------------------------------------------- #
# The verifier
# --------------------------------------------------------------------------- #
class ComplianceCheck:
    """A :class:`verify.Verifier` that flags compliance concerns (advisory only).

    ``check(backend, opdag)`` reads ``query('measure')`` (overall size/volume) and
    an optional ``query('metrics')`` (additive overhang / min-feature) and returns
    a :class:`verify.VerifyReport`. A tripped threshold is a WARNING carrying a
    clear rationale; a note or unmeasurable case is an INFO. This verifier NEVER
    emits an ERROR and never blocks.

    Codes emitted:
      * WARNING ``export-control-review``   — size/volume over a heuristic
                                              (non-authoritative) ITAR/EAR limit.
      * WARNING ``regional-limit``          — over the configured region ceiling.
      * WARNING ``unsupported-overhang``    — overhang steeper than the AM max.
      * WARNING ``below-min-feature``       — a reported feature below the AM floor.
      * INFO    ``*-skipped``               — a backing query was unavailable.
    """

    name = "compliance"

    def __init__(self, rules: Optional[ComplianceRules] = None) -> None:
        self.rules = rules or ComplianceRules()

    def check(self, backend, opdag) -> VerifyReport:
        diags: List[Diagnostic] = []
        self._check_export_control(backend, diags)
        self._check_additive(backend, diags)
        return VerifyReport(diags)

    # -- export-control + regional (size/volume from 'measure') -------------
    def _check_export_control(self, backend, diags: List[Diagnostic]) -> None:
        r = self.rules
        do_export = r.export_control_enabled
        do_region = r.region is not None and r.region in r.regional_limits
        if not (do_export or do_region):
            return

        measure = _query(backend, "measure")
        if measure is None:
            diags.append(_info(
                "export-control-skipped",
                "export-control / regional size checks skipped: backend has no "
                "'measure' query (only a real geometry kernel exposes bbox)."))
            return
        bbox = measure.get("bbox")
        dims = [float(v) for v in bbox[:3]] if bbox and len(bbox) >= 3 else []
        positive = [d for d in dims if d > 0.0]
        volume = float(measure.get("volume", 0.0) or 0.0)
        if not positive and volume <= 0.0:
            diags.append(_info(
                "export-control-skipped",
                "export-control / regional size checks skipped: 'measure' "
                "returned no positive bbox/volume (no solid yet)."))
            return

        largest = max(positive) if positive else 0.0

        if do_export:
            if largest > r.itar_size_threshold:
                diags.append(_warn(
                    "export-control-review",
                    f"overall size {largest:g} mm exceeds the heuristic "
                    f"export-control review threshold {r.itar_size_threshold:g} "
                    f"mm. HEURISTIC ONLY, not an ITAR/EAR determination: flag for "
                    f"a human compliance review.",
                    where="size"))
            if volume > r.itar_volume_threshold:
                diags.append(_warn(
                    "export-control-review",
                    f"overall volume {volume:g} mm^3 exceeds the heuristic "
                    f"export-control review threshold {r.itar_volume_threshold:g} "
                    f"mm^3. HEURISTIC ONLY, not an ITAR/EAR determination: flag "
                    f"for a human compliance review.",
                    where="volume"))

        if do_region and positive:
            ceiling = r.regional_limits[r.region]
            if largest > ceiling:
                diags.append(_warn(
                    "regional-limit",
                    f"overall size {largest:g} mm exceeds the configured "
                    f"'{r.region}' regional manufacturing ceiling {ceiling:g} mm.",
                    where=r.region))

    # -- additive manufacturing (overhang / min feature from 'metrics') -----
    def _check_additive(self, backend, diags: List[Diagnostic]) -> None:
        r = self.rules
        if not r.am_enabled:
            return
        metrics = _query(backend, "metrics")
        if metrics is None:
            diags.append(_info(
                "overhang-skipped",
                "additive overhang / min-feature checks skipped: backend has no "
                "'metrics' query (needs a kernel that reports face angles)."))
            return

        # Overhang: accept either a single max value or a list of angles.
        max_over = metrics.get("max_overhang_deg")
        angles = metrics.get("overhang_angles")
        if max_over is None and angles:
            try:
                max_over = max(float(a) for a in angles)
            except (TypeError, ValueError):
                max_over = None
        if max_over is None:
            diags.append(_info(
                "overhang-skipped",
                "additive overhang check skipped: 'metrics' reported no "
                "'max_overhang_deg' / 'overhang_angles'."))
        else:
            max_over = float(max_over)
            if max_over > r.max_overhang_angle:
                diags.append(_warn(
                    "unsupported-overhang",
                    f"max unsupported overhang {max_over:g} deg exceeds the "
                    f"printable limit {r.max_overhang_angle:g} deg (from "
                    f"vertical); the part needs support or reorientation.",
                    where="overhang"))

        # Minimum feature size (optional).
        min_feat = metrics.get("min_feature_mm")
        if min_feat is not None:
            try:
                min_feat = float(min_feat)
            except (TypeError, ValueError):
                min_feat = None
            if min_feat is not None and 0.0 < min_feat < r.min_feature_size:
                diags.append(_warn(
                    "below-min-feature",
                    f"smallest feature {min_feat:g} mm is below the additive "
                    f"minimum feature size {r.min_feature_size:g} mm; it may not "
                    f"reproduce.",
                    where="min-feature"))


# --------------------------------------------------------------------------- #
# Wiring helper
# --------------------------------------------------------------------------- #
def with_compliance(verifiers, rules: Optional[ComplianceRules] = None) -> List:
    """Return a new verifier list with a :class:`ComplianceCheck` appended.

    Mirrors :func:`checks_dfm.with_dfm`::

        from harnesscad.eval.verifiers.verify import default_verifiers
        from harnesscad.eval.verifiers.compliance import with_compliance
        verifiers = with_compliance(default_verifiers())
    """
    return list(verifiers) + [ComplianceCheck(rules)]


# --------------------------------------------------------------------------- #
# Helpers (mirror checks_dfm's graceful-degradation conventions)
# --------------------------------------------------------------------------- #
def _query(backend, q: str) -> Optional[dict]:
    """Read a backend query, returning None when the backend does not answer it
    (backends return {} for unknown queries) so callers can INFO-skip."""
    try:
        result = backend.query(q)
    except Exception:  # noqa: BLE001 - an unsupported query must degrade, not crash
        return None
    return result or None


def _warn(code: str, msg: str, where: Optional[str] = None) -> Diagnostic:
    return Diagnostic(Severity.WARNING, code, msg, where)


def _info(code: str, msg: str, where: Optional[str] = None) -> Diagnostic:
    return Diagnostic(Severity.INFO, code, msg, where)
