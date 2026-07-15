"""Natural-language CAD brief IR with deterministic default resolution.

Ported from **text-to-cad** (the improved ``.agents/skills/cad`` skill), whose
workflow converts a user's prose into an *internal CAD brief* -- a fixed
note-taking scaffold (``assets/design-brief-template.md``) with a defined set of
fields (Model, Task type, Units, Coordinate convention, Overall dimensions,
Functional features, Manufacturing assumptions, Positioning/mating, STEP paths,
Validation targets, Assumptions) -- and then proceeds under an explicit set of
*default assumptions* (units = mm, origin = part centre, base plane = XY, up =
+Z, closed positive-volume solids, standard hole clearances, cosmetic fillet
sizes) rather than interrogating the user.

This module makes that scheme a concrete, checkable IR:

* :class:`CADBrief` -- the typed brief with the template's fields;
* :func:`parse_brief` -- parse the ``- Key: value`` scaffold into a brief;
* :func:`resolve_defaults` -- fill unstated fields from the skill's documented
  defaults, recording per-field *provenance* (stated vs inferred vs missing);
* :func:`clearance_for` -- the skill's M3/M4/M5 normal-clearance table
  (3.4/4.5/5.5 mm) as a deterministic lookup;
* :func:`completeness` -- a report of which critical fields are stated, inferred
  or still missing -- the checkable "is this brief buildable?" gate.

It is a distinct sibling of :mod:`harnesscad.domain.spec.intent_categories`
(which *classifies* an utterance): this one *structures* a full design brief and
resolves its defaults so downstream planning is deterministic.

Pure stdlib, deterministic.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

__all__ = [
    "BRIEF_FIELDS",
    "CRITICAL_FIELDS",
    "DEFAULTS",
    "CLEARANCE_TABLE",
    "CADBrief",
    "FieldProvenance",
    "parse_brief",
    "resolve_defaults",
    "clearance_for",
    "completeness",
]

# The template fields (order preserved for stable reporting).
BRIEF_FIELDS: Tuple[str, ...] = (
    "model",
    "task_type",
    "units",
    "coordinate_convention",
    "overall_dimensions",
    "functional_features",
    "manufacturing_assumptions",
    "positioning_requirements",
    "step_source_path",
    "step_target_path",
    "secondary_outputs",
    "validation_targets",
    "assumptions",
)

# Fields a buildable brief must ultimately carry (some via defaults).
CRITICAL_FIELDS: Tuple[str, ...] = (
    "model",
    "units",
    "coordinate_convention",
    "overall_dimensions",
)

# The skill's documented default assumptions.
DEFAULTS: Dict[str, str] = {
    "units": "millimeters",
    "coordinate_convention": "origin at part centre; base plane XY; up +Z",
    "task_type": "create",
    "manufacturing_assumptions": "closed positive-volume solids; wall 2.0-3.0 mm; cosmetic fillet 1.0-3.0 mm",
    "secondary_outputs": "STEP only",
    "validation_targets": "closed solid; positive volume; sane bounding box",
}

# Normal clearance-hole diameters (mm) for common metric fasteners.
CLEARANCE_TABLE: Dict[str, float] = {
    "M3": 3.4,
    "M4": 4.5,
    "M5": 5.5,
}

# Map template label text -> field key.
_LABEL_TO_FIELD: Dict[str, str] = {
    "model": "model",
    "task type": "task_type",
    "units": "units",
    "coordinate convention": "coordinate_convention",
    "overall dimensions": "overall_dimensions",
    "functional features": "functional_features",
    "manufacturing assumptions": "manufacturing_assumptions",
    "positioning/mating requirements": "positioning_requirements",
    "positioning requirements": "positioning_requirements",
    "step source path": "step_source_path",
    "step target path": "step_target_path",
    "secondary outputs": "secondary_outputs",
    "validation targets": "validation_targets",
    "explorer link target": "explorer_link_target",
    "assumptions": "assumptions",
}


@dataclass
class FieldProvenance:
    """Where a resolved field value came from."""

    field_name: str
    status: str  # "stated" | "inferred" | "missing"
    value: Optional[str] = None


@dataclass
class CADBrief:
    """A structured CAD brief (text-to-cad's internal scaffold)."""

    model: Optional[str] = None
    task_type: Optional[str] = None
    units: Optional[str] = None
    coordinate_convention: Optional[str] = None
    overall_dimensions: Optional[str] = None
    functional_features: Optional[str] = None
    manufacturing_assumptions: Optional[str] = None
    positioning_requirements: Optional[str] = None
    step_source_path: Optional[str] = None
    step_target_path: Optional[str] = None
    secondary_outputs: Optional[str] = None
    validation_targets: Optional[str] = None
    assumptions: Optional[str] = None
    explorer_link_target: Optional[str] = None

    def get(self, name: str) -> Optional[str]:
        return getattr(self, name, None)

    def stated_fields(self) -> List[str]:
        return [f for f in BRIEF_FIELDS if _nonempty(self.get(f))]


def _nonempty(value: Optional[str]) -> bool:
    return value is not None and str(value).strip() not in ("", "-", "TBD", "N/A")


def parse_brief(text: str) -> CADBrief:
    """Parse the ``- Key: value`` brief scaffold into a :class:`CADBrief`.

    Unrecognised labels are ignored; blank/placeholder values become ``None``.
    Nested sub-bullets (e.g. under Positioning) are appended to their parent.
    """
    brief = CADBrief()
    last_field: Optional[str] = None
    for raw in text.splitlines():
        line = raw.strip()
        if not line:
            continue
        stripped = line.lstrip("-* \t")
        if ":" not in stripped:
            # A continuation sub-bullet for the last field.
            if last_field is not None and stripped:
                prev = brief.get(last_field) or ""
                setattr(brief, last_field, (prev + "; " + stripped).strip("; "))
            continue
        label, _, value = stripped.partition(":")
        key = _LABEL_TO_FIELD.get(label.strip().lower())
        if key is None:
            last_field = None
            continue
        value = value.strip()
        if _nonempty(value):
            setattr(brief, key, value)
        last_field = key
    return brief


def resolve_defaults(brief: CADBrief) -> Tuple[CADBrief, List[FieldProvenance]]:
    """Return a copy of ``brief`` with defaults filled + per-field provenance."""
    resolved = CADBrief(**{f: brief.get(f) for f in _all_fields(brief)})
    provenance: List[FieldProvenance] = []
    for name in BRIEF_FIELDS:
        stated = _nonempty(brief.get(name))
        if stated:
            provenance.append(FieldProvenance(name, "stated", brief.get(name)))
            continue
        if name in DEFAULTS:
            setattr(resolved, name, DEFAULTS[name])
            provenance.append(FieldProvenance(name, "inferred", DEFAULTS[name]))
        else:
            provenance.append(FieldProvenance(name, "missing", None))
    return resolved, provenance


def _all_fields(brief: CADBrief) -> List[str]:
    return list(BRIEF_FIELDS) + ["explorer_link_target"]


def clearance_for(fastener: str) -> float:
    """Normal metric clearance-hole diameter (mm) for ``fastener`` (e.g. 'M3')."""
    key = fastener.strip().upper()
    if key not in CLEARANCE_TABLE:
        raise KeyError(f"no clearance entry for {fastener!r} (known: {sorted(CLEARANCE_TABLE)})")
    return CLEARANCE_TABLE[key]


@dataclass
class CompletenessReport:
    """Which critical fields are stated / inferred / still missing."""

    buildable: bool
    stated: List[str] = field(default_factory=list)
    inferred: List[str] = field(default_factory=list)
    missing: List[str] = field(default_factory=list)


def completeness(brief: CADBrief) -> CompletenessReport:
    """Assess whether a brief is buildable after default resolution.

    A brief is *buildable* when every critical field is either stated by the
    user or supplied by a documented default. Fields that are neither are
    reported as ``missing`` and block the build.
    """
    resolved, provenance = resolve_defaults(brief)
    by_name = {p.field_name: p.status for p in provenance}
    stated = [f for f in CRITICAL_FIELDS if by_name.get(f) == "stated"]
    inferred = [f for f in CRITICAL_FIELDS if by_name.get(f) == "inferred"]
    missing = [f for f in CRITICAL_FIELDS if by_name.get(f) == "missing"]
    return CompletenessReport(
        buildable=not missing,
        stated=stated,
        inferred=inferred,
        missing=missing,
    )
