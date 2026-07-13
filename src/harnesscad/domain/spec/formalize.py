"""formalize — NL brief -> typed, machine-checkable :class:`RequirementSet`.

The blueprint's front-of-pipeline: "translate an NL problem definition into a
formal spec" and "extract the countable / parametric asks from the prompt". A
:class:`Requirement` is one such ask, typed by ``kind``:

  ================  =========================================================
  kind              example phrase              target / unit
  ================  =========================================================
  ``count``         "4 mounting holes"          target=4,  label='hole'
  ``dimension``     "100 mm long"               target=100, unit='mm', label='length'
  ``envelope``      "100mm x 50mm x 8mm"        three dimension reqs
  ``material``      "aluminium"                 target='aluminium'
  ``tolerance``     "+/- 0.1 mm"                target=0.1, unit='mm'
  ``feature``       "a fillet"                  label='fillet'
  ================  =========================================================

:func:`formalize` runs an injected :class:`llm.base.LLM` for structured
extraction when one is supplied, and *always* falls back to a deterministic
regex parser (no network, stdlib only) so tests need nothing installed.

:func:`to_contract` seeds a :class:`contract.Contract` dict (feed it straight
to :meth:`contract.Contract.from_dict`) so the same formal spec drives the
machine-verifiable acceptance check.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

# --------------------------------------------------------------------------- #
# Requirement / RequirementSet
# --------------------------------------------------------------------------- #
_KINDS = ("count", "dimension", "material", "tolerance", "feature", "envelope")


@dataclass
class Requirement:
    """One extracted ask from a brief.

    ``target`` is an int for counts, a float for dimensions/tolerances, and a
    string for materials/features. ``label`` names *what* the target describes
    (the feature noun for a count, the canonical axis word for a dimension).
    ``source_phrase`` keeps the span of the brief it came from, for provenance.
    """

    kind: str
    target: Any = None
    tolerance: Optional[float] = None
    unit: Optional[str] = None
    label: Optional[str] = None
    source_phrase: str = ""

    def to_dict(self) -> dict:
        d: dict = {"kind": self.kind, "target": self.target}
        if self.tolerance is not None:
            d["tolerance"] = self.tolerance
        if self.unit is not None:
            d["unit"] = self.unit
        if self.label is not None:
            d["label"] = self.label
        if self.source_phrase:
            d["source_phrase"] = self.source_phrase
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "Requirement":
        return cls(
            kind=d.get("kind", "feature"),
            target=d.get("target"),
            tolerance=(float(d["tolerance"]) if d.get("tolerance") is not None
                       else None),
            unit=d.get("unit"),
            label=d.get("label"),
            source_phrase=d.get("source_phrase", ""),
        )


@dataclass
class RequirementSet:
    """An ordered list of :class:`Requirement` plus optional part metadata."""

    requirements: List[Requirement] = field(default_factory=list)
    name: str = ""
    description: str = ""

    def __iter__(self):
        return iter(self.requirements)

    def __len__(self) -> int:
        return len(self.requirements)

    def add(self, req: Requirement) -> None:
        self.requirements.append(req)

    def by_kind(self, kind: str) -> List[Requirement]:
        return [r for r in self.requirements if r.kind == kind]

    def has_kind(self, kind: str) -> bool:
        return any(r.kind == kind for r in self.requirements)

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "description": self.description,
            "requirements": [r.to_dict() for r in self.requirements],
        }

    @classmethod
    def from_dict(cls, d: dict) -> "RequirementSet":
        return cls(
            requirements=[Requirement.from_dict(r)
                          for r in (d.get("requirements") or [])],
            name=d.get("name", ""),
            description=d.get("description", ""),
        )


# --------------------------------------------------------------------------- #
# Heuristic vocabulary
# --------------------------------------------------------------------------- #
_FEATURE_NOUNS = (
    "hole", "boss", "slot", "rib", "pocket", "tab", "fillet", "chamfer",
    "cutout", "mount", "standoff", "thread", "counterbore", "countersink",
)

_MATERIALS = (
    "stainless steel", "mild steel", "aluminium", "aluminum", "steel", "brass",
    "titanium", "copper", "bronze", "abs", "pla", "petg", "nylon", "delrin",
    "acrylic", "polycarbonate", "plywood", "plastic", "wood",
)

# canonical dimension label -> bounding-box axis
_AXIS_OF_LABEL = {
    "length": "x", "long": "x",
    "width": "y", "wide": "y",
    "height": "z", "tall": "z", "high": "z",
    "depth": "z", "deep": "z",
    "thickness": "z", "thick": "z",
}

# named-dimension descriptor -> canonical label
_DIM_WORD = {
    "long": "length", "length": "length",
    "wide": "width", "width": "width",
    "tall": "height", "high": "height", "height": "height",
    "deep": "depth", "depth": "depth",
    "thick": "thickness", "thickness": "thickness",
    "diameter": "diameter", "dia": "diameter", "radius": "radius",
}

_UNIT = r"(mm|cm|millimet(?:er|re)s?|centimet(?:er|re)s?|m|in|inch|inches|\")"
_NUM = r"(\d+(?:\.\d+)?)"

_COUNT_RE = re.compile(
    r"\b(\d+)\s+((?:mounting\s+|through\s+)?(?:" +
    "|".join(_FEATURE_NOUNS) + r")s?)\b",
    re.IGNORECASE,
)
_DIM_NAMED_RE = re.compile(
    r"\b" + _NUM + r"\s*" + _UNIT + r"?\s*(" + "|".join(_DIM_WORD) + r")\b",
    re.IGNORECASE,
)
_ENVELOPE_RE = re.compile(
    _NUM + r"\s*" + _UNIT + r"?\s*[x×]\s*" +
    _NUM + r"\s*" + _UNIT + r"?" +
    r"(?:\s*[x×]\s*" + _NUM + r"\s*" + _UNIT + r"?)?",
    re.IGNORECASE,
)
_TOL_RE = re.compile(
    r"(?:\+/-|\+/\-|\+-|±|plus or minus)\s*" + _NUM + r"\s*" + _UNIT + r"?",
    re.IGNORECASE,
)


def _norm_unit(u: Optional[str]) -> Optional[str]:
    if not u:
        return None
    u = u.lower()
    if u.startswith("milli") or u == "mm":
        return "mm"
    if u.startswith("centi") or u == "cm":
        return "cm"
    if u in ('"', "in", "inch", "inches"):
        return "in"
    if u == "m":
        return "m"
    return u


# --------------------------------------------------------------------------- #
# Heuristic parser
# --------------------------------------------------------------------------- #
def _heuristic(brief: str) -> RequirementSet:
    reqs: List[Requirement] = []
    seen_spans: List[tuple] = []

    # 1. envelope "A x B x C" (consumes those numbers so they are not re-read).
    envelope_labels = ("length", "width", "height")
    for m in _ENVELOPE_RE.finditer(brief):
        nums = [m.group(1), m.group(3), m.group(5)]
        units = [m.group(2), m.group(4), m.group(6)]
        chosen_unit = _norm_unit(next((u for u in units if u), None)) or "mm"
        for i, raw in enumerate(nums):
            if raw is None:
                continue
            reqs.append(Requirement(
                kind="dimension", target=float(raw),
                unit=_norm_unit(units[i]) or chosen_unit,
                label=envelope_labels[i], source_phrase=m.group(0).strip()))
        seen_spans.append(m.span())

    def _overlaps(span):
        return any(span[0] < e and s < span[1] for s, e in seen_spans)

    # 2. named single dimensions ("100 mm long", "8mm thick", "20mm diameter").
    for m in _DIM_NAMED_RE.finditer(brief):
        if _overlaps(m.span()):
            continue
        label = _DIM_WORD[m.group(3).lower()]
        reqs.append(Requirement(
            kind="dimension", target=float(m.group(1)),
            unit=_norm_unit(m.group(2)) or "mm",
            label=label, source_phrase=m.group(0).strip()))
        seen_spans.append(m.span())

    # 3. countable features ("4 holes", "2 mounting bosses").
    for m in _COUNT_RE.finditer(brief):
        noun = re.sub(r"^(?:mounting|through)\s+", "", m.group(2).lower())
        noun = noun[:-1] if noun.endswith("s") else noun
        reqs.append(Requirement(
            kind="count", target=int(m.group(1)), label=noun,
            source_phrase=m.group(0).strip()))

    # 4. tolerance ("+/- 0.1 mm").
    for m in _TOL_RE.finditer(brief):
        reqs.append(Requirement(
            kind="tolerance", target=float(m.group(1)),
            unit=_norm_unit(m.group(2)), source_phrase=m.group(0).strip()))

    # 5. material (first match wins).
    low = brief.lower()
    for mat in _MATERIALS:
        idx = low.find(mat)
        if idx != -1:
            reqs.append(Requirement(
                kind="material", target=mat, label="material",
                source_phrase=brief[idx:idx + len(mat)]))
            break

    return RequirementSet(requirements=reqs, description=brief.strip())


# --------------------------------------------------------------------------- #
# Public entry point
# --------------------------------------------------------------------------- #
def formalize(brief: str, llm=None) -> RequirementSet:
    """Turn a natural-language ``brief`` into a :class:`RequirementSet`.

    With an ``llm`` (any :class:`llm.base.LLM`), ask for structured output
    against :func:`requirement_schema` and parse it; if the model is
    unavailable or returns nothing usable, fall back to the deterministic
    :func:`_heuristic` parser. Without an ``llm``, use the heuristic directly.
    """
    if not brief or not brief.strip():
        return RequirementSet()

    if llm is not None:
        parsed = _formalize_with_llm(brief, llm)
        if parsed is not None and parsed.requirements:
            if not parsed.description:
                parsed.description = brief.strip()
            return parsed
        # else: degrade gracefully to the heuristic below.

    return _heuristic(brief)


def _formalize_with_llm(brief: str, llm) -> Optional[RequirementSet]:
    from harnesscad.agents.llm.base import Message  # local import: llm layer is optional

    schema = requirement_schema()
    messages = [
        Message("system",
                "You extract countable and parametric requirements from a CAD "
                "brief. Return ONLY JSON matching the given schema: a list of "
                "typed requirements (count, dimension, material, tolerance, "
                "feature, envelope). Extract exactly what the brief asks for."),
        Message("user", brief),
    ]
    try:
        result = llm.complete(messages, response_schema=schema)
    except Exception:  # noqa: BLE001 - a dead provider must not crash formalize
        return None
    text = getattr(result, "text", "") or ""
    if not text.strip():
        return None
    try:
        data = json.loads(text)
    except (json.JSONDecodeError, TypeError):
        return None
    if isinstance(data, list):
        data = {"requirements": data}
    if not isinstance(data, dict):
        return None
    try:
        return RequirementSet.from_dict(data)
    except Exception:  # noqa: BLE001 - malformed structured output -> fall back
        return None


# --------------------------------------------------------------------------- #
# Contract seeding
# --------------------------------------------------------------------------- #
def to_contract(reqset: RequirementSet) -> dict:
    """Seed a :class:`contract.Contract` dict from a ``RequirementSet``.

    The returned dict is accepted verbatim by :meth:`contract.Contract.from_dict`.
    Dimension requirements become bbox tolerances (dimension tolerance, else a
    single ``tolerance`` requirement's value, else 0). Count requirements become
    ``min_features`` (total) and, for holes specifically, ``hole_count``.
    """
    d: dict = {"name": reqset.name or "part"}
    desc = reqset.description or ""

    default_tol = None
    tols = reqset.by_kind("tolerance")
    if tols and tols[0].target is not None:
        default_tol = float(tols[0].target)

    bbox: Dict[str, dict] = {}
    for r in reqset.by_kind("dimension"):
        axis = _AXIS_OF_LABEL.get((r.label or "").lower())
        if axis is None or r.target is None or axis in bbox:
            continue
        tol = r.tolerance if r.tolerance is not None else (default_tol or 0.0)
        bbox[axis] = {"target": float(r.target), "tol": float(tol)}
    if bbox:
        d["bbox"] = bbox

    counts = reqset.by_kind("count")
    total = sum(int(r.target) for r in counts if r.target is not None)
    if total:
        d["min_features"] = total
    holes = sum(int(r.target) for r in counts
                if (r.label or "").lower() == "hole" and r.target is not None)
    if holes:
        d["hole_count"] = holes

    materials = reqset.by_kind("material")
    if materials:
        mat = str(materials[0].target)
        desc = (desc + f" [material: {mat}]").strip()
    if desc:
        d["description"] = desc

    return d


# --------------------------------------------------------------------------- #
# JSON schema (for LLM structured output)
# --------------------------------------------------------------------------- #
def requirement_schema() -> dict:
    """JSON schema for a :class:`RequirementSet` (for LLM structured output).

    This does not call an LLM; it only returns the shape to ask the model for.
    """
    requirement = {
        "type": "object",
        "properties": {
            "kind": {"type": "string", "enum": list(_KINDS),
                     "description": "requirement type"},
            "target": {
                "description": "int for count, number for dimension/tolerance, "
                               "string for material/feature",
                "type": ["number", "string"],
            },
            "tolerance": {"type": "number", "minimum": 0,
                          "description": "symmetric +/- band for a dimension"},
            "unit": {"type": "string", "description": "e.g. 'mm', 'in'"},
            "label": {"type": "string",
                      "description": "feature noun (count) or axis word "
                                     "(dimension: length/width/height/diameter)"},
            "source_phrase": {"type": "string",
                              "description": "span of the brief it came from"},
        },
        "required": ["kind", "target"],
        "additionalProperties": False,
    }
    return {
        "$schema": "http://json-schema.org/draft-07/schema#",
        "title": "RequirementSet",
        "type": "object",
        "properties": {
            "name": {"type": "string"},
            "description": {"type": "string"},
            "requirements": {"type": "array", "items": requirement},
        },
        "required": ["requirements"],
        "additionalProperties": False,
    }
