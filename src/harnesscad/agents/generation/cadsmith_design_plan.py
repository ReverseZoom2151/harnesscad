"""CADSmith Planner design-plan schema (CADSmith sec. III-B).

The Planner agent in CADSmith converts a raw natural-language prompt into a
*structured design plan* — a JSON object with a component list, target bounding
box dimensions in millimetres, geometric constraints (hole counts, hole
diameters, symmetry), and notes for downstream agents. The Planner emits **no**
CadQuery code; its sole job is to turn ambiguous prose into an unambiguous
specification the Coder can implement, and against which the Validator can later
measure discrepancies.

This module is the deterministic, stdlib-only realisation of that schema:

  * typed dataclasses (:class:`Component`, :class:`GeometricConstraints`,
    :class:`DesignPlan`) with millimetre semantics,
  * strict JSON round-trip (``to_json`` / ``from_json``) so the handoff is a
    stable contract between agents,
  * structural validation (``validate``) that catches malformed plans before
    they reach the Coder,
  * a prompt-convention checker (``check_prompt_conventions``) enforcing the
    benchmark's stated conventions: explicit millimetre dimensions, axis
    annotation (XY base plane, Z-up), and origin-centred geometry.

No LLM, no wall clock, no randomness — the *schema and its invariants*, which
are the deterministic, reusable half of the Planner.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field, asdict
from typing import List, Optional, Tuple


# --------------------------------------------------------------------------- #
# Component
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Component:
    """A single sub-part of the design.

    ``z_range`` (when known) records the axial span in millimetres, matching the
    benchmark convention of Z-up, origin-centred parts described by explicit
    Z-ranges (e.g. "bottom flange Z=0 to Z=10").
    """

    name: str
    description: str = ""
    z_range: Optional[Tuple[float, float]] = None

    def to_dict(self) -> dict:
        d = {"name": self.name, "description": self.description}
        if self.z_range is not None:
            d["z_range"] = [float(self.z_range[0]), float(self.z_range[1])]
        return d

    @staticmethod
    def from_dict(d: dict) -> "Component":
        zr = d.get("z_range")
        z_range = (float(zr[0]), float(zr[1])) if zr is not None else None
        return Component(
            name=str(d["name"]),
            description=str(d.get("description", "")),
            z_range=z_range,
        )


# --------------------------------------------------------------------------- #
# Geometric constraints
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class GeometricConstraints:
    """Quantitative constraints the Validator will later check against kernel
    measurements: how many holes, at what diameters, and which symmetry the part
    should exhibit."""

    hole_count: int = 0
    hole_diameters_mm: Tuple[float, ...] = ()
    symmetry: Tuple[str, ...] = ()          # e.g. ("axial-z", "mirror-x")
    notes: str = ""

    def to_dict(self) -> dict:
        return {
            "hole_count": int(self.hole_count),
            "hole_diameters_mm": [float(x) for x in self.hole_diameters_mm],
            "symmetry": list(self.symmetry),
            "notes": self.notes,
        }

    @staticmethod
    def from_dict(d: dict) -> "GeometricConstraints":
        return GeometricConstraints(
            hole_count=int(d.get("hole_count", 0)),
            hole_diameters_mm=tuple(float(x) for x in d.get("hole_diameters_mm", ())),
            symmetry=tuple(str(s) for s in d.get("symmetry", ())),
            notes=str(d.get("notes", "")),
        )


# --------------------------------------------------------------------------- #
# Design plan
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class DesignPlan:
    """The Planner's full output. ``target_bbox_mm`` is the (x, y, z) bounding
    box extent in millimetres — the primary dimensional anchor for validation."""

    components: Tuple[Component, ...]
    target_bbox_mm: Tuple[float, float, float]
    constraints: GeometricConstraints = field(default_factory=GeometricConstraints)
    notes: str = ""

    # -- serialisation ----------------------------------------------------- #
    def to_dict(self) -> dict:
        return {
            "components": [c.to_dict() for c in self.components],
            "target_bbox_mm": [float(v) for v in self.target_bbox_mm],
            "constraints": self.constraints.to_dict(),
            "notes": self.notes,
        }

    def to_json(self, *, indent: Optional[int] = None) -> str:
        # sort_keys keeps the handoff byte-stable for a given plan.
        return json.dumps(self.to_dict(), indent=indent, sort_keys=True)

    @staticmethod
    def from_dict(d: dict) -> "DesignPlan":
        bbox = d["target_bbox_mm"]
        return DesignPlan(
            components=tuple(Component.from_dict(c) for c in d.get("components", ())),
            target_bbox_mm=(float(bbox[0]), float(bbox[1]), float(bbox[2])),
            constraints=GeometricConstraints.from_dict(d.get("constraints", {})),
            notes=str(d.get("notes", "")),
        )

    @staticmethod
    def from_json(text: str) -> "DesignPlan":
        return DesignPlan.from_dict(json.loads(text))


# --------------------------------------------------------------------------- #
# Validation
# --------------------------------------------------------------------------- #
def validate(plan: DesignPlan) -> Tuple[str, ...]:
    """Return a tuple of structural problems with the plan (empty == valid).

    Deterministic checks only — these are the invariants that must hold before a
    plan is worth handing to the Coder.
    """
    issues: List[str] = []
    if not plan.components:
        issues.append("no-components")
    seen = set()
    for c in plan.components:
        if not c.name:
            issues.append("component-missing-name")
        if c.name in seen:
            issues.append(f"duplicate-component:{c.name}")
        seen.add(c.name)
        if c.z_range is not None and c.z_range[0] > c.z_range[1]:
            issues.append(f"inverted-z-range:{c.name}")

    if len(plan.target_bbox_mm) != 3:
        issues.append("bbox-not-3d")
    if any(v <= 0 for v in plan.target_bbox_mm):
        issues.append("bbox-non-positive")

    ct = plan.constraints
    if ct.hole_count < 0:
        issues.append("negative-hole-count")
    if any(dia <= 0 for dia in ct.hole_diameters_mm):
        issues.append("non-positive-hole-diameter")
    # If diameters are enumerated per hole, their count should not exceed the
    # declared hole count (a diameter with no hole is inconsistent).
    if ct.hole_diameters_mm and ct.hole_count and \
            len(ct.hole_diameters_mm) > ct.hole_count:
        issues.append("more-diameters-than-holes")
    return tuple(issues)


def is_valid(plan: DesignPlan) -> bool:
    return not validate(plan)


# --------------------------------------------------------------------------- #
# Prompt convention checker
# --------------------------------------------------------------------------- #
# Benchmark convention: explicit millimetre dimensions, axis-annotated
# orientation (XY base plane, Z-up), origin-centred geometry (sec. III-A).
_MM = re.compile(r"\b\d+(?:\.\d+)?\s*mm\b", re.IGNORECASE)
_AXIS = re.compile(r"\b(?:z[\s-]?up|z\s*axis|xy\s*(?:base\s*)?plane|"
                   r"[xyz]\s*=\s*-?\d)", re.IGNORECASE)
_ORIGIN = re.compile(r"\b(?:origin[\s-]?center|center(?:ed)?\s+at\s+the\s+origin|"
                     r"origin[\s-]?centred)\b", re.IGNORECASE)


def check_prompt_conventions(prompt: str) -> Tuple[str, ...]:
    """Return convention violations for a benchmark prompt (empty == compliant).

    Flags prompts that omit explicit millimetre dimensions, lack axis
    annotation, or do not state origin-centring — the ambiguities the Planner is
    meant to eliminate.
    """
    problems: List[str] = []
    if not _MM.search(prompt):
        problems.append("no-explicit-mm-dimensions")
    if not _AXIS.search(prompt):
        problems.append("no-axis-annotation")
    if not _ORIGIN.search(prompt):
        problems.append("no-origin-centering")
    return tuple(problems)
