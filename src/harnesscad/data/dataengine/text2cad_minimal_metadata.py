"""Text2CAD Minimal Metadata Generator (Khan et al., NeurIPS 2024, Sec. 3 / 11).

Before Text2CAD asks an LLM to write natural-language instructions, it preprocesses
the raw DeepCAD construction JSON with a **Minimal Metadata Generator** (paper
Sec. 3, Fig. 8). The raw JSON has two properties that make an LLM hallucinate:

  1. *Random, meaningless keys* -- DeepCAD keys entities by opaque uuids such as
     ``"FI4bCL9y0XvsF52"``. If these are passed through, the LLM starts referring to
     curves/sketches by the random key. The generator replaces them with meaningful,
     positional names (``part_1``, ``loop_1``, ``face_1``, ``curve_1``, ...).
  2. *Redundant design information* -- e.g. ``{"type": "ModelParameter", "role":
     "AgainstDistance"}`` carries nothing the NL instruction needs, so it is dropped.

The output is a condensed, human-readable representation of the shapes and their
relational attributes, optionally augmented with the VLM shape descriptions of each
part and the final model. This module reproduces that deterministic restructuring
(no LLM, no randomness): stable positional renaming grouped by entity type, and a
configurable redundant-key / redundant-type filter.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Mapping


class MinimalMetadataError(ValueError):
    """Raised for malformed raw metadata."""


# Keys whose values are pure bookkeeping and never help an NL instruction.
DEFAULT_REDUNDANT_KEYS: frozenset[str] = frozenset({
    "uuid", "guid", "id", "role", "reference", "curveType",
    "isConstruction", "constructionGeom", "fixed", "visible",
    "colour", "color", "revision", "documentId", "timestamp",
})

# Entity ``type`` values that are pure bookkeeping records and are dropped wholesale.
DEFAULT_REDUNDANT_TYPES: frozenset[str] = frozenset({
    "ModelParameter", "Reference", "Annotation",
})

# Canonical positional prefix per entity type. Anything else falls back to "entity".
DEFAULT_TYPE_PREFIX: Mapping[str, str] = {
    "Part": "part",
    "Body": "part",
    "Sketch": "sketch",
    "Profile": "face",
    "Face": "face",
    "Loop": "loop",
    "Curve": "curve",
    "Line": "curve",
    "Arc": "curve",
    "Circle": "curve",
    "Extrude": "extrude",
    "ExtrudeFeature": "extrude",
    "CoordSystem": "coordinate_system",
}

# A DeepCAD-style random key: mixed-case alnum, no vowel-word structure, >= 10 chars.
_RANDOM_KEY_RE = re.compile(r"^[A-Za-z0-9_]{10,}$")
_HAS_LETTER = re.compile(r"[A-Za-z]")
_HAS_DIGIT = re.compile(r"[0-9]")


def is_random_key(key: str, *, whitelist: frozenset[str] = frozenset()) -> bool:
    """Heuristically decide whether ``key`` is a meaningless DeepCAD-style uuid.

    A key is "random" when it is long (>= 10 chars), contains *both* letters and
    digits, and is not an explicitly meaningful whitelisted word. Human-authored
    keys such as ``"part_1"`` or ``"extrude"`` are never treated as random.
    """
    if key in whitelist:
        return False
    if not _RANDOM_KEY_RE.match(key):
        return False
    # Meaningful snake_case names (part_1) have letters+digits but a separating '_'
    # immediately before the trailing digit run; treat those as meaningful.
    if re.match(r"^[a-z]+(_[a-z0-9]+)*_\d+$", key):
        return False
    return bool(_HAS_LETTER.search(key) and _HAS_DIGIT.search(key))


def _prefix_for(entity: Any, type_prefix: Mapping[str, str]) -> str:
    """Positional-name prefix for an entity based on its declared ``type``."""
    if isinstance(entity, Mapping):
        etype = entity.get("type")
        if isinstance(etype, str) and etype in type_prefix:
            return type_prefix[etype]
    return "entity"


@dataclass(frozen=True)
class MinimalMetadata:
    """Result of restructuring a raw CAD-construction JSON."""

    entities: dict[str, Any]
    key_map: dict[str, str]                 # original key -> meaningful key
    dropped_keys: tuple[str, ...]           # redundant keys removed
    dropped_entities: tuple[str, ...]       # redundant-typed entities removed
    shape_descriptions: dict[str, str] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        """Flat serialisable view (deterministic key order)."""
        out: dict[str, Any] = {"entities": self.entities}
        if self.shape_descriptions:
            out["shape_descriptions"] = dict(self.shape_descriptions)
        return out


def _clean_value(
    value: Any,
    redundant_keys: frozenset[str],
    redundant_types: frozenset[str],
    type_prefix: Mapping[str, str],
    dropped_keys: list[str],
    dropped_entities: list[str],
    key_map: dict[str, str],
) -> Any:
    """Recursively strip redundant keys and rename random keys within ``value``."""
    if isinstance(value, Mapping):
        return _clean_mapping(
            value, redundant_keys, redundant_types, type_prefix,
            dropped_keys, dropped_entities, key_map,
        )
    if isinstance(value, (list, tuple)):
        return [
            _clean_value(item, redundant_keys, redundant_types, type_prefix,
                         dropped_keys, dropped_entities, key_map)
            for item in value
        ]
    return value


def _clean_mapping(
    mapping: Mapping[str, Any],
    redundant_keys: frozenset[str],
    redundant_types: frozenset[str],
    type_prefix: Mapping[str, str],
    dropped_keys: list[str],
    dropped_entities: list[str],
    key_map: dict[str, str],
) -> dict[str, Any]:
    """Restructure a single mapping: drop redundant keys, rename random keys."""
    counters: dict[str, int] = {}
    result: dict[str, Any] = {}
    for key, value in mapping.items():
        # Drop redundant bookkeeping keys outright.
        if key in redundant_keys:
            dropped_keys.append(key)
            continue
        # Drop whole entities whose declared type is pure bookkeeping.
        if isinstance(value, Mapping) and value.get("type") in redundant_types:
            dropped_entities.append(key)
            continue
        cleaned = _clean_value(
            value, redundant_keys, redundant_types, type_prefix,
            dropped_keys, dropped_entities, key_map,
        )
        if is_random_key(key):
            prefix = _prefix_for(value, type_prefix)
            counters[prefix] = counters.get(prefix, 0) + 1
            new_key = f"{prefix}_{counters[prefix]}"
            key_map[key] = new_key
            result[new_key] = cleaned
        else:
            result[key] = cleaned
    return result


def generate_minimal_metadata(
    raw: Mapping[str, Any],
    *,
    shape_descriptions: Mapping[str, str] | None = None,
    redundant_keys: frozenset[str] = DEFAULT_REDUNDANT_KEYS,
    redundant_types: frozenset[str] = DEFAULT_REDUNDANT_TYPES,
    type_prefix: Mapping[str, str] = DEFAULT_TYPE_PREFIX,
) -> MinimalMetadata:
    """Restructure raw DeepCAD-style JSON into minimal, human-readable metadata.

    Random uuid keys are replaced by positional names grouped per entity type
    (``part_1``, ``loop_1``, ...); redundant keys and redundant-typed entities are
    removed; optional VLM shape descriptions are attached. Deterministic: identical
    input yields identical output, with positional numbering following the raw
    mapping's iteration order.
    """
    if not isinstance(raw, Mapping):
        raise MinimalMetadataError("raw metadata must be a mapping")
    dropped_keys: list[str] = []
    dropped_entities: list[str] = []
    key_map: dict[str, str] = {}
    entities = _clean_mapping(
        raw, redundant_keys, redundant_types, type_prefix,
        dropped_keys, dropped_entities, key_map,
    )
    return MinimalMetadata(
        entities=entities,
        key_map=key_map,
        dropped_keys=tuple(dropped_keys),
        dropped_entities=tuple(dropped_entities),
        shape_descriptions=dict(shape_descriptions or {}),
    )
