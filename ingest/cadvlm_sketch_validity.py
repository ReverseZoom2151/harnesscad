"""Whole-sketch primitive/constraint validity checks for CadVLM outputs.

CadVLM generates a sketch as a primitive sequence ``S`` plus a constraint sequence
``C`` (Sec 3). A *decoded* sketch is only usable in CAD software if it is
structurally well-formed, and the paper is explicit about what well-formed means:

* every entity is a **line** (start/end), **arc** (start/mid/end) or **circle**
  (four circumference points), i.e. a fixed point-count per type;
* all coordinate tokens live in the quantised ``[1, 64]`` range (Sec 3, Fig 1); and
* every constraint (Appendix A / Table 6) is one of the 13 reserved value tokens
  (65..77), references entity indices that actually exist in ``S``, and supplies at
  least the minimum number of references its type needs.

``ingest.cadvlm_codec`` validates one entity or one constraint tuple in isolation;
this module composes those into a single, deterministic *whole-sketch* validator that
returns every problem it finds (empty tuple == valid), which is exactly the check a
generator or an autocomplete loop needs before handing a sketch downstream.
"""

from __future__ import annotations

from dataclasses import dataclass

from dataengine.sketch_constraint_ontology import BY_TOKEN, resolve
from ingest.cadvlm_codec import HIGH, LOW


# Fixed coordinate-point counts per entity type (Sec 3).
_ENTITY_POINTS = {"line": 2, "arc": 3, "circle": 4}


def _coord_ok(value) -> bool:
    return isinstance(value, int) and LOW <= value <= HIGH


def entity_issues(entity_tokens) -> tuple:
    """Problems with one entity token tuple ``(kind, *coords)``; empty if valid."""
    tokens = tuple(entity_tokens)
    if not tokens:
        return ("empty-entity",)
    kind = tokens[0]
    if kind not in _ENTITY_POINTS:
        return (f"unknown-entity:{kind}",)
    expected = 1 + 2 * _ENTITY_POINTS[kind]
    problems = []
    if len(tokens) != expected:
        problems.append(f"{kind}:bad-token-count:{len(tokens)}!={expected}")
    for coord in tokens[1:]:
        if not _coord_ok(coord):
            problems.append(f"{kind}:coord-out-of-range:{coord}")
    return tuple(problems)


def constraint_issues(constraint_tokens, n_entities: int) -> tuple:
    """Problems with one constraint tuple ``(type_token, *entity_indices)``."""
    tokens = tuple(constraint_tokens)
    if not tokens:
        return ("empty-constraint",)
    type_token = tokens[0]
    if type_token not in BY_TOKEN:
        return (f"unknown-constraint-token:{type_token}",)
    kind = BY_TOKEN[type_token]
    refs = tokens[1:]
    problems = []
    if len(refs) < kind.minimum_references:
        problems.append(
            f"{kind.name}:insufficient-references:{len(refs)}<{kind.minimum_references}")
    for ref in refs:
        if not (isinstance(ref, int) and 0 <= ref < n_entities):
            problems.append(f"{kind.name}:reference-out-of-range:{ref}")
    return tuple(problems)


@dataclass(frozen=True)
class ValidityReport:
    """Result of :func:`check_sketch`."""

    valid: bool
    entity_issues: tuple           # (entity_index, issue) pairs
    constraint_issues: tuple       # (constraint_index, issue) pairs

    @property
    def all_issues(self) -> tuple:
        return self.entity_issues + self.constraint_issues


def check_sketch(entities, constraints=()) -> ValidityReport:
    """Validate a decoded sketch's primitives and constraint references.

    ``entities`` is a sequence of entity token tuples (``ingest.cadvlm_codec``
    output form); ``constraints`` a sequence of ``(type_token, *entity_indices)``
    tuples. Returns a :class:`ValidityReport` collecting every problem found.
    """
    ent_list = tuple(entities)
    e_problems = tuple(
        (i, msg) for i, ent in enumerate(ent_list) for msg in entity_issues(ent)
    )
    c_problems = tuple(
        (j, msg)
        for j, con in enumerate(constraints)
        for msg in constraint_issues(con, len(ent_list))
    )
    return ValidityReport(
        valid=not (e_problems or c_problems),
        entity_issues=e_problems,
        constraint_issues=c_problems,
    )


def constraint_name(type_token: int) -> str:
    """Human-readable name for a reserved constraint value token (65..77)."""
    return resolve(type_token).name
