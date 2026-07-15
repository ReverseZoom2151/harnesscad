"""Provenance evidence bundle for a design spec's standards data (Anvilate).

**Anvilate**'s promise is "geometry an engineer can trust *with the evidence
attached*". Its ``evidence`` module walks a typed Design Spec and rolls every
standards record the part leans on into an auditable provenance bundle -- one
:class:`SourceRecord` per source (the material, each standard component
interface, the general-tolerance class that always applies, the fit on a
toleranced bore, the geometric-tolerance standard for a call-out), each naming
the standard or dataset behind it. "Nothing is asserted without a citation; this
is what a design review reads."

This module reimplements that collector over a lightweight, stdlib spec (a plain
mapping or the provided dataclasses -- no OCCT, no external units library). It
adds a **deterministic bundle hash** so two identical specs produce byte-identical
provenance, and a :meth:`EvidenceBundle.missing_citations` gate that flags any
referenced material/component the databases cannot vouch for -- the export gate
Anvilate wants ("nothing unvalidated leaves the tool").

It is distinct from :mod:`harnesscad.domain.standards.ingest` (which parses a
standard's clause text): this *aggregates* the citations a specific part relies
on into a review-ready, hashable bundle.

Pure stdlib, deterministic.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Dict, List, Mapping, Optional, Sequence

__all__ = [
    "GENERAL_TOLERANCE_STANDARD",
    "SourceRecord",
    "MaterialRecord",
    "ComponentRecord",
    "EvidenceBundle",
    "default_materials_db",
    "default_components_db",
    "collect_provenance",
]

# The general-tolerance class that always applies to a machined part.
GENERAL_TOLERANCE_STANDARD = "ISO 2768"
_FIT_STANDARD = "ISO 286"
_GDT_STANDARD = "ISO 1101"


@dataclass(frozen=True)
class SourceRecord:
    """One cited source in the provenance bundle."""

    kind: str  # "material" | "component" | "general_tolerance" | "fit" | "gdt"
    ref: str
    name: str
    sources: Sequence[str]

    def as_dict(self) -> Dict[str, object]:
        return {
            "kind": self.kind,
            "ref": self.ref,
            "name": self.name,
            "sources": list(self.sources),
        }


@dataclass(frozen=True)
class MaterialRecord:
    """A material the spec may reference."""

    ref: str
    name: str
    standard: str


@dataclass(frozen=True)
class ComponentRecord:
    """A standard component (motor, bearing, ...) the spec may interface to."""

    ref: str
    name: str
    standard: str


@dataclass
class EvidenceBundle:
    """The rolled-up provenance for one spec."""

    records: List[SourceRecord] = field(default_factory=list)
    unresolved: List[str] = field(default_factory=list)

    def kinds(self) -> List[str]:
        return [r.kind for r in self.records]

    def refs(self) -> List[str]:
        return [r.ref for r in self.records]

    def missing_citations(self) -> List[str]:
        """Refs that could not be cited (not found in the databases)."""
        return list(self.unresolved)

    def is_fully_cited(self) -> bool:
        return not self.unresolved

    def digest(self) -> str:
        """A deterministic SHA-256 over the sorted canonical records.

        Identical specs -> identical digest, regardless of record insertion
        order -- the stable audit fingerprint a design review can quote.
        """
        canonical = sorted(
            (json.dumps(r.as_dict(), sort_keys=True, ensure_ascii=True) for r in self.records)
        )
        payload = json.dumps(
            {"records": canonical, "unresolved": sorted(self.unresolved)},
            sort_keys=True,
            ensure_ascii=True,
        )
        return hashlib.sha256(payload.encode("ascii")).hexdigest()


def default_materials_db() -> Dict[str, MaterialRecord]:
    """A small stand-in materials database."""
    return {
        "AA-6061-T6": MaterialRecord("AA-6061-T6", "Aluminium 6061-T6", "ASM / AA standards"),
        "SS-304": MaterialRecord("SS-304", "Stainless steel 304", "ASTM A240"),
        "S355": MaterialRecord("S355", "Structural steel S355", "EN 10025"),
    }


def default_components_db() -> Dict[str, ComponentRecord]:
    """A small stand-in standard-components database."""
    return {
        "NEMA23": ComponentRecord("NEMA23", "NEMA 23 stepper flange", "NEMA ICS 16-2001"),
        "6204": ComponentRecord("6204", "6204 deep-groove ball bearing", "ISO 15:2017"),
        "M6": ComponentRecord("M6", "M6 hex bolt", "ISO 4014"),
    }


def _spec_get(spec: Mapping[str, object], key: str, default=None):
    return spec.get(key, default) if isinstance(spec, Mapping) else getattr(spec, key, default)


def collect_provenance(
    spec: Mapping[str, object],
    *,
    materials: Optional[Mapping[str, MaterialRecord]] = None,
    components: Optional[Mapping[str, ComponentRecord]] = None,
) -> EvidenceBundle:
    """Walk ``spec`` and roll its referenced standards data into a bundle.

    ``spec`` is a mapping (or attribute object) with optional keys:

    * ``material`` -- a material ref string;
    * ``interfaces`` -- a sequence of standard-component ref strings;
    * ``dimensions`` -- a sequence of ``{"tag", "fit"}`` mappings (an ISO 286
      fit designation on a toleranced dimension);
    * ``geometric_tolerances`` -- a sequence of ``{"feature", "characteristic"}``
      mappings (an ISO 1101 geometric call-out).

    The general-tolerance standard (ISO 2768) is always added -- it applies to
    every machined part whether or not the spec names it.
    """
    materials = materials if materials is not None else default_materials_db()
    components = components if components is not None else default_components_db()

    records: List[SourceRecord] = []
    unresolved: List[str] = []

    # Material.
    material_ref = _spec_get(spec, "material")
    if material_ref:
        mat = materials.get(str(material_ref))
        if mat is not None:
            records.append(SourceRecord("material", mat.ref, mat.name, [mat.standard]))
        else:
            unresolved.append(f"material:{material_ref}")

    # Standard components.
    for iface in _spec_get(spec, "interfaces", []) or []:
        ref = str(iface)
        comp = components.get(ref)
        if comp is not None:
            records.append(SourceRecord("component", comp.ref, comp.name, [comp.standard]))
        else:
            unresolved.append(f"component:{ref}")

    # ISO 2768 general tolerances always apply.
    records.append(
        SourceRecord(
            "general_tolerance",
            GENERAL_TOLERANCE_STANDARD,
            "General geometric tolerances",
            [GENERAL_TOLERANCE_STANDARD],
        )
    )

    # ISO 286 fits on toleranced dimensions.
    for dim in _spec_get(spec, "dimensions", []) or []:
        tag = str(_spec_get(dim, "tag", "dimension"))
        fit = _spec_get(dim, "fit")
        if fit:
            records.append(
                SourceRecord("fit", tag, f"{fit} fit on {tag}", [f"{_FIT_STANDARD} ({fit})"])
            )

    # ISO 1101 geometric tolerances.
    for gt in _spec_get(spec, "geometric_tolerances", []) or []:
        feature = str(_spec_get(gt, "feature", "feature"))
        char = str(_spec_get(gt, "characteristic", "geometric"))
        records.append(
            SourceRecord("gdt", feature, f"{char} on {feature}", [_GDT_STANDARD])
        )

    return EvidenceBundle(records=records, unresolved=unresolved)
