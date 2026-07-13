"""annomap_gdt — GD&T feature-control-frame representation + validity checker.

Khan et al. stress "strict GD&T disambiguation" and note that "visually similar
annotations may impose fundamentally different constraints depending on symbol
type, modifiers, and datum references". The paper's dataset is dimension-centric
so its GD&T heuristics are "not fully exercised", but the *rules* a GD&T callout
must obey to be a valid manufacturing constraint are deterministic and drawn
straight from ASME Y14.5 / ISO 1101. This module encodes:

  * the 14 geometric characteristics grouped into form / orientation / location /
    runout / profile;
  * which characteristics are **datum-less** (form controls: flatness,
    straightness, circularity, cylindricity) versus **datum-referenced**
    (orientation / location / runout) with a required datum count;
  * which characteristics may carry a **diameter (Ø) tolerance zone** (position,
    concentricity, coaxial), and which forbid material modifiers;
  * material-condition modifier legality (MMC/LMC applicable only to features of
    size, never to a plane/flatness control);
  * datum-precedence sanity (primary/secondary/tertiary, no repeats).

Given a parsed GD&T frame (from :func:`annomap_parser.parse_gdt_frame`) it returns
a list of :class:`Finding` records (ERROR / WARNING / OK). Pure stdlib; advisory
and reproducible.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

# Characteristic -> category.
FORM = "form"
ORIENTATION = "orientation"
LOCATION = "location"
RUNOUT = "runout"
PROFILE = "profile"

CHARACTERISTICS: Dict[str, str] = {
    "flatness": FORM,
    "straightness": FORM,
    "circularity": FORM,
    "cylindricity": FORM,
    "angularity": ORIENTATION,
    "perpendicularity": ORIENTATION,
    "parallelism": ORIENTATION,
    "position": LOCATION,
    "concentricity": LOCATION,
    "symmetry": LOCATION,
    "circular_runout": RUNOUT,
    "total_runout": RUNOUT,
    "runout": RUNOUT,
    "profile_of_a_line": PROFILE,
    "profile_of_a_surface": PROFILE,
}

# Datum requirement: minimum number of datum references. Form controls take none.
_MIN_DATUMS: Dict[str, int] = {
    FORM: 0,
    ORIENTATION: 1,
    LOCATION: 1,      # position may in principle be datum-less for a pattern-to-
                      # -itself, but Y14.5 location controls normally reference one
    RUNOUT: 1,
    PROFILE: 0,       # profile may be applied with or without datums
}

# Characteristics for which a diameter (Ø) tolerance zone is meaningful.
_DIAMETRAL_OK = frozenset({"position", "concentricity", "circular_runout",
                           "total_runout"})

# Characteristics that are ALWAYS datum-less; a datum reference is an error.
_DATUMLESS = frozenset({"flatness", "straightness", "circularity",
                        "cylindricity"})

# Characteristics that accept material-condition modifiers (features of size).
_MODIFIER_OK = frozenset({"position", "concentricity", "symmetry",
                          "perpendicularity", "parallelism", "angularity"})

_SEVERITY_ORDER = {"OK": 0, "WARNING": 1, "ERROR": 2}


@dataclass
class Finding:
    severity: str          # "OK" | "WARNING" | "ERROR"
    code: str
    message: str

    def to_dict(self) -> dict:
        return {"severity": self.severity, "code": self.code,
                "message": self.message}


@dataclass
class GDTValidation:
    symbol: Optional[str]
    ok: bool
    findings: List[Finding] = field(default_factory=list)

    @property
    def errors(self) -> List[Finding]:
        return [f for f in self.findings if f.severity == "ERROR"]

    @property
    def warnings(self) -> List[Finding]:
        return [f for f in self.findings if f.severity == "WARNING"]

    def to_dict(self) -> dict:
        return {"symbol": self.symbol, "ok": self.ok,
                "findings": [f.to_dict() for f in self.findings]}


def category_of(symbol: str) -> Optional[str]:
    return CHARACTERISTICS.get((symbol or "").lower())


def validate_frame(frame: Dict[str, object]) -> GDTValidation:
    """Validate a parsed GD&T feature-control frame.

    ``frame`` is the dict returned by :func:`annomap_parser.parse_gdt_frame`:
    ``{symbol, tolerance, diametral_zone, modifier, datums}``. Returns a
    :class:`GDTValidation`; ``ok`` is True when there are no ERROR findings.
    """
    findings: List[Finding] = []
    symbol = frame.get("symbol") if frame else None
    if not symbol:
        return GDTValidation(None, False, [Finding(
            "ERROR", "no_symbol", "frame carries no recognised characteristic")])
    symbol = str(symbol).lower()
    category = CHARACTERISTICS.get(symbol)
    if category is None:
        return GDTValidation(symbol, False, [Finding(
            "ERROR", "unknown_symbol",
            "unknown characteristic '%s'" % symbol)])

    tolerance = frame.get("tolerance")
    diametral = bool(frame.get("diametral_zone"))
    modifier = frame.get("modifier")
    datums = list(frame.get("datums") or [])

    # 1) Tolerance value must be a positive finite number.
    if tolerance is None:
        findings.append(Finding("ERROR", "missing_tolerance",
                                "no tolerance value in control frame"))
    else:
        try:
            tv = float(tolerance)
            if tv <= 0.0:
                findings.append(Finding("ERROR", "nonpositive_tolerance",
                                        "tolerance must be > 0 (got %g)" % tv))
        except (TypeError, ValueError):
            findings.append(Finding("ERROR", "bad_tolerance",
                                    "tolerance is not numeric"))

    # 2) Datum-reference legality.
    min_datums = _MIN_DATUMS.get(category, 0)
    if symbol in _DATUMLESS and datums:
        findings.append(Finding(
            "ERROR", "datum_on_form_control",
            "form control '%s' must not reference datums %s" % (symbol, datums)))
    elif len(datums) < min_datums:
        findings.append(Finding(
            "ERROR", "missing_datum",
            "'%s' (%s) requires >=%d datum reference(s), got %d"
            % (symbol, category, min_datums, len(datums))))

    # 3) Datum-precedence sanity: no repeated datum letters.
    if len(set(datums)) != len(datums):
        findings.append(Finding("ERROR", "duplicate_datum",
                                "datum letters repeat: %s" % datums))
    if len(datums) > 3:
        findings.append(Finding("WARNING", "too_many_datums",
                                "more than 3 datum references is unusual"))

    # 4) Diameter zone legality.
    if diametral and symbol not in _DIAMETRAL_OK:
        findings.append(Finding(
            "WARNING", "unexpected_diametral_zone",
            "Ø tolerance zone not conventional for '%s'" % symbol))

    # 5) Material-condition modifier legality.
    if modifier in ("MMC", "LMC") and symbol not in _MODIFIER_OK:
        findings.append(Finding(
            "ERROR", "modifier_not_applicable",
            "material modifier %s not applicable to '%s' (not a feature of size)"
            % (modifier, symbol)))

    if not any(f.severity == "ERROR" for f in findings):
        findings.append(Finding("OK", "valid",
                                "GD&T frame '%s' is well-formed" % symbol))
    ok = not any(f.severity == "ERROR" for f in findings)
    return GDTValidation(symbol, ok, findings)


def validate_frames(frames: Sequence[Dict[str, object]]) -> List[GDTValidation]:
    return [validate_frame(f) for f in frames]


def worst_severity(validations: Sequence[GDTValidation]) -> str:
    worst = "OK"
    for v in validations:
        for f in v.findings:
            if _SEVERITY_ORDER[f.severity] > _SEVERITY_ORDER[worst]:
                worst = f.severity
    return worst
