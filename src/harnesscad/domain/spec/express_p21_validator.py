"""Validate a part-21 DATA section against a parsed EXPRESS schema.

This is the capability that only becomes possible once both halves exist: the
EXPRESS *schema* model (:mod:`spec.express_schema_parser` +
:mod:`spec.express_inheritance`) and the part-21 *data* model
(:mod:`formats.stepllm_parser`).  ruststep's runtime does exactly this pairing
-- an ``.exp`` schema compiled by espr defines the entities, and a ``.step``
file's records are then deserialised against those definitions.

The harness previously validated data only against a small hand-written entity
table (:mod:`formats.stepllm_schema`).  Here validation is driven by an
*arbitrary* parsed EXPRESS schema, and crucially it accounts for **inheritance**:
a part-21 record ``#N = FOO(a, b, c)`` must supply the flattened attribute list
of ``FOO`` (supertype attributes first), so arity is checked against
:func:`~spec.express_inheritance.flatten_attributes`, not the entity's own
attribute count.

Checks performed per instance:

  * the entity type is declared in the schema (unknown type -> issue);
  * the supplied attribute count equals the flattened (inherited + own) arity;
  * each attribute value's shape is compatible with its declared type kind
    (aggregate attribute <-> list value; ``$`` allowed only where OPTIONAL);
  * complex instances (multiple simultaneous types) have every part declared.

Value-vs-type checking is deliberately structural (kind-level), mirroring the
part-21 side's untyped value model; WHERE-rule *expressions* are reported as
present but not evaluated.  Pure/deterministic, stdlib only.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from harnesscad.io.formats.step import DERIVED, Enum, Real, Ref, StepFile, Typed, UNSET
from harnesscad.domain.spec.express_inheritance import InheritanceGraph, flatten_attributes
from harnesscad.domain.spec.express_schema_parser import Schema, TypeRef


@dataclass(frozen=True)
class Issue:
    entity_id: int
    entity_type: str
    detail: str

    def __str__(self) -> str:
        return f"#{self.entity_id} {self.entity_type}: {self.detail}"


@dataclass
class ValidationReport:
    issues: list = field(default_factory=list)
    checked: int = 0
    skipped: list = field(default_factory=list)   # ids of complex/unknown skips

    @property
    def ok(self) -> bool:
        return not self.issues

    def summary(self) -> str:
        return (f"ok={self.ok}, checked={self.checked}, "
                f"issues={len(self.issues)}, skipped={len(self.skipped)}")


def _value_matches(kind: str, value) -> bool:
    """Structural compatibility of a part-21 value with an EXPRESS type kind."""

    if value is UNSET or value is DERIVED:
        # Optionality is handled by the caller; a $/* is always shape-tolerated.
        return True
    if kind == "simple_real":
        return isinstance(value, (Real, int)) and not isinstance(value, bool)
    if kind == "simple_int":
        return isinstance(value, int) and not isinstance(value, bool)
    if kind == "simple_str":
        return isinstance(value, str)
    if kind == "simple_bool":
        return isinstance(value, Enum) and value.name in ("T", "F", "U")
    if kind == "simple_logical":
        return isinstance(value, Enum)
    if kind == "aggregate":
        return isinstance(value, (list, tuple))
    if kind == "named":
        # An entity ref, an enum literal (defined enum type) or a typed value.
        return isinstance(value, (Ref, Enum, Typed, str, Real, int))
    return True


def _type_kind(ty: TypeRef) -> str:
    if ty.is_aggregate():
        return "aggregate"
    if ty.kind == "simple":
        name = ty.name.upper()
        if name == "REAL" or name == "NUMBER":
            return "simple_real"
        if name == "INTEGER":
            return "simple_int"
        if name in ("STRING", "BINARY"):
            return "simple_str"
        if name == "BOOLEAN":
            return "simple_bool"
        if name == "LOGICAL":
            return "simple_logical"
    return "named"


def _check_simple(ent_id: int, keyword: str, params, attrs) -> list:
    """Check one simple record's params against a flattened attribute list."""

    issues: list = []
    if len(params) != len(attrs):
        issues.append(Issue(
            ent_id, keyword,
            f"expected {len(attrs)} attributes (with inheritance), "
            f"got {len(params)}"))
        return issues
    for attr, value in zip(attrs, params):
        if value is UNSET and not attr.optional:
            issues.append(Issue(
                ent_id, keyword,
                f"attribute {attr.name!r} is $ but not OPTIONAL"))
            continue
        kind = _type_kind(attr.type_ref)
        if not _value_matches(kind, value):
            issues.append(Issue(
                ent_id, keyword,
                f"attribute {attr.name!r}: value {type(value).__name__} "
                f"incompatible with {attr.type_ref.kind}"
                f"{'/' + attr.type_ref.name if attr.type_ref.name else ''}"))
    return issues


def validate_data(step: StepFile, schema: Schema,
                  graph: InheritanceGraph | None = None) -> ValidationReport:
    """Validate every DATA instance in ``step`` against ``schema``.

    ``graph`` (an :class:`InheritanceGraph`) is used to flatten inherited
    attributes; if omitted it is built from ``schema``.
    """

    if graph is None:
        from harnesscad.domain.spec.express_inheritance import build_inheritance
        graph = build_inheritance(schema)

    # EXPRESS is case-insensitive; part-21 keywords are conventionally upper
    # case while schema declarations are lower case. Resolve via a folded map.
    by_fold = {name.upper(): name for name in schema.entities}

    report = ValidationReport()
    for ent_id in step.order:
        entity = step.entities[ent_id]
        report.checked += 1

        if entity.keyword is None:
            # Complex instance: validate each simultaneous type part.
            for part in entity.params:
                if not isinstance(part, Typed):
                    continue
                if part.keyword.upper() not in by_fold:
                    report.issues.append(Issue(
                        ent_id, part.keyword,
                        "unknown entity type in complex instance"))
            continue

        resolved = by_fold.get(entity.keyword.upper())
        if resolved is None:
            report.issues.append(Issue(
                ent_id, entity.keyword, "unknown entity type"))
            report.skipped.append(ent_id)
            continue

        attrs = flatten_attributes(graph, resolved)
        report.issues.extend(
            _check_simple(ent_id, entity.keyword, entity.params, attrs))

    return report
