"""Operation-gating contract layer: validate a CISP op stream BEFORE execution.

The problem this solves: a generator proposes a stream of CISP operations, and
some of those operations must never run -- they would delete a mounting
interface, re-cut a certified housing, or drive a parameter outside the range
its designer signed off on. Discovering that *after* execution is too late, so
this module judges the whole stream up front against a declarative contract and
refuses the offending ops with typed, machine-readable reasons.

The contract has four parts:

* an **allowed-operations catalog** -- per feature, which operation types are
  ``allowed`` / ``forbidden`` / ``conditional``; a conditional rule carries the
  ``preconditions`` that must hold and the ``blocked_by_constraints`` that must
  not be active;
* **protected regions** -- per-feature whitelists (``allowed_operations`` /
  ``forbidden_operations``) that *override* the catalog, so a feature can be
  frozen without editing the catalog it appears in;
* a **parameter-edit preflight** -- a ``modify_parameter`` op is additionally
  checked against declared bounds (min / max / discrete values) and against the
  hard rule that a parameter edit must never change topology;
* a **report vocabulary** -- the whole-stream verdict maps onto the patch
  lifecycle's status enum (``ready_for_validation`` /
  ``violates_protected_target`` / ``blocked``) and tracks which protected
  features were touched and which were left intact.

The JSON Schema shape checks are implemented here in pure Python (the two
gating schemas are embedded as data below) so a catalog dict can be
structurally validated without any third-party jsonschema dependency.

The gate consumes the CISP op stream defined in
:mod:`harnesscad.core.cisp.ops` (frozen dataclasses with a stable ``OP`` tag,
or their ``to_dict()`` forms). Each CISP op is mapped onto the catalog's
``operation_type`` vocabulary (every geometry-creating op is an
``add_feature``; :class:`~harnesscad.core.cisp.ops.SetParam` is a
``modify_parameter``), resolved to the feature it touches, and judged against
the catalog. :func:`gate` returns a per-op decision list; every refusal is a
TYPED diagnostic (:class:`GateDiagnostic`) with the op index, op name,
feature, a machine reason code (``forbidden`` / ``unknown_op`` /
``unknown_feature`` / ``precondition_unmet`` / ``blocked_by_constraint`` /
``protected_region`` / ``value_out_of_bounds`` / ``topology_changing``) and a
human message.

Preconditions are free strings; the evaluable subset understood here is
``"key"`` (truthy lookup in the caller's context dict) and
``"key <cmp> value"`` with ``<cmp>`` one of ``== != >= <= > <``. A
precondition that cannot be evaluated is conservatively UNMET (default-deny).
``blocked_by_constraints`` names are matched against the context's
``active_constraints`` collection.

Pure stdlib, deterministic, no model or kernel calls.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from typing import (Callable, Dict, Iterator, List, Mapping, Optional, Sequence,
                    Tuple, Union)

from harnesscad.core.cisp import ops

__all__ = [
    "OPERATION_TYPES",
    "OPERATION_STATUSES",
    "REASON_CODES",
    "OperationRule",
    "FeatureOperations",
    "AllowedOperationsCatalog",
    "ProtectedRegion",
    "GateDiagnostic",
    "GateDecision",
    "GateReport",
    "validate_catalog",
    "validate_protected_regions",
    "catalog_from_dict",
    "protected_regions_from_dict",
    "op_operation_type",
    "op_feature_ref",
    "evaluate_precondition",
    "gate",
    "main",
]

# ---------------------------------------------------------------------------
# Vocabulary (the catalog and patch-proposal operation_type / status enums)
# ---------------------------------------------------------------------------

#: The catalog's (and patch proposal's) operation_type enum.
OPERATION_TYPES: Tuple[str, ...] = (
    "modify_parameter",
    "add_feature",
    "remove_feature",
    "protect_feature",
    "assign_material",
    "assign_boundary_condition",
    "assign_load",
)

#: Per-operation admissibility status enum.
OPERATION_STATUSES: Tuple[str, ...] = ("allowed", "forbidden", "conditional")

#: Machine reason codes a refusal diagnostic may carry.
REASON_CODES: Tuple[str, ...] = (
    "forbidden",
    "unknown_op",
    "unknown_feature",
    "precondition_unmet",
    "blocked_by_constraint",
    "protected_region",
    "value_out_of_bounds",
    "topology_changing",
)

#: A catalog entry whose feature_id (or feature_type) is this wildcard matches
#: any op that names no known feature -- the catch-all row.
WILDCARD_FEATURE = "*"

# CISP op tag -> catalog operation_type. Every registered mutating CISP op
# creates or places geometry (add_feature) except the editability primitive
# SetParam, which is exactly the catalog's modify_parameter. Built from the
# ops registry so the mapping can never name an op CISP does not have.
_OP_TYPE_OVERRIDES: Dict[str, str] = {
    "set_param": "modify_parameter",
}
_OP_TYPE_MAP: Dict[str, str] = {
    tag: _OP_TYPE_OVERRIDES.get(tag, "add_feature") for tag in ops._REGISTRY
}

# Op fields consulted (in order) to resolve which feature an op touches.
_FEATURE_REF_FIELDS: Tuple[str, ...] = (
    "feature",
    "feature_or_body",
    "sketch",
    "face_or_sketch",
    "target",
    "tool",
    "part",
    "a",
)


# ---------------------------------------------------------------------------
# Embedded schema structure (data, not executable jsonschema)
#
# The two gating documents -- the allowed-operations catalog and the protected
# regions list -- are described here as plain dicts in the JSON Schema subset
# the checker below understands. Repeated shapes are factored into the small
# builders first so the documents read as their own field lists.
# ---------------------------------------------------------------------------

_NON_EMPTY_STRING: Dict[str, object] = {"type": "string", "minLength": 1}
_ANY_STRING: Dict[str, object] = {"type": "string"}
_BOOLEAN: Dict[str, object] = {"type": "boolean"}


def _array_of(item_spec: Mapping[str, object],
              min_items: Optional[int] = None) -> Dict[str, object]:
    """An ``array`` spec over ``item_spec``, optionally requiring a minimum."""
    spec: Dict[str, object] = {"type": "array", "items": dict(item_spec)}
    if min_items is not None:
        spec["minItems"] = min_items
    return spec


def _closed_object(required: Sequence[str],
                   properties: Mapping[str, object]) -> Dict[str, object]:
    """An object that must carry exactly ``required`` and nothing unexpected."""
    return {
        "type": "object",
        "required": list(required),
        "additionalProperties": False,
        "properties": dict(properties),
    }


def _enum_of(values: Sequence[str]) -> Dict[str, object]:
    return {"type": "string", "enum": list(values)}


_FORMAT_VERSION: Dict[str, object] = {"type": "string", "const": "0.1.0"}

_OPERATION_RULE_SCHEMA: Dict[str, object] = _closed_object(
    required=("operation_type", "status", "reason", "preconditions",
              "blocked_by_constraints"),
    properties={
        "operation_type": _enum_of(OPERATION_TYPES),
        "status": _enum_of(OPERATION_STATUSES),
        "reason": _NON_EMPTY_STRING,
        "preconditions": _array_of(_NON_EMPTY_STRING),
        "blocked_by_constraints": _array_of(_NON_EMPTY_STRING),
    },
)

_FEATURE_ENTRY_SCHEMA: Dict[str, object] = _closed_object(
    required=("feature_id", "feature_type", "protected", "interface_roles",
              "operations"),
    properties={
        "feature_id": _NON_EMPTY_STRING,
        "feature_type": _NON_EMPTY_STRING,
        "protected": _BOOLEAN,
        "interface_roles": _array_of(_NON_EMPTY_STRING),
        "operations": _array_of(_OPERATION_RULE_SCHEMA, min_items=1),
    },
)

ALLOWED_OPERATIONS_CATALOG_SCHEMA: Dict[str, object] = _closed_object(
    required=("format_version", "catalog_id", "generated_by", "generated_at_utc",
              "source_files", "feature_operations", "notes"),
    properties={
        "format_version": _FORMAT_VERSION,
        "catalog_id": _NON_EMPTY_STRING,
        "generated_by": _NON_EMPTY_STRING,
        "generated_at_utc": _NON_EMPTY_STRING,
        "source_files": _array_of(_NON_EMPTY_STRING),
        "feature_operations": _array_of(_FEATURE_ENTRY_SCHEMA),
        "notes": _array_of(_NON_EMPTY_STRING, min_items=1),
    },
)

_PROTECTED_REGION_SCHEMA: Dict[str, object] = _closed_object(
    required=("feature_id", "reason", "allowed_operations",
              "forbidden_operations"),
    properties={
        "feature_id": _NON_EMPTY_STRING,
        "reason": _NON_EMPTY_STRING,
        "allowed_operations": _array_of(_ANY_STRING),
        "forbidden_operations": _array_of(_ANY_STRING),
    },
)

PROTECTED_REGIONS_SCHEMA: Dict[str, object] = _closed_object(
    required=("format_version", "protected_regions"),
    properties={
        "format_version": _FORMAT_VERSION,
        "protected_regions": _array_of(_PROTECTED_REGION_SCHEMA),
    },
)


# ---------------------------------------------------------------------------
# Stdlib structural (schema-shape) validation
#
# Each JSON Schema keyword the two documents use becomes one small generator of
# complaint strings. ``_shape_errors`` runs the applicable ones and recurses;
# adding a keyword means adding a function, not another branch in a monolith.
# ---------------------------------------------------------------------------

_PY_TYPES: Dict[str, object] = {
    "object": dict,
    "array": list,
    "string": str,
    "boolean": bool,
    "integer": int,
    "number": (int, float),
    "null": type(None),
}

#: JSON types for which a Python ``bool`` must NOT be accepted, even though
#: ``bool`` subclasses ``int``.
_BOOL_HOSTILE_TYPES = ("integer", "number")

_Complaints = Iterator[str]


def _check_type(value: object, spec: Mapping[str, object], path: str) -> _Complaints:
    """``type``: the value's Python type must match the declared JSON type."""
    declared = spec.get("type")
    if declared is None:
        return
    python_type = _PY_TYPES.get(str(declared))
    if python_type is None:
        return
    if isinstance(value, bool) and declared in _BOOL_HOSTILE_TYPES:
        yield "%s: expected %s, got boolean" % (path, declared)
    elif not isinstance(value, python_type):  # type: ignore[arg-type]
        yield "%s: expected %s, got %s" % (path, declared, type(value).__name__)


def _check_const(value: object, spec: Mapping[str, object], path: str) -> _Complaints:
    if "const" in spec and value != spec["const"]:
        yield "%s: must equal %r" % (path, spec["const"])


def _check_enum(value: object, spec: Mapping[str, object], path: str) -> _Complaints:
    permitted = spec.get("enum")
    if isinstance(permitted, list) and value not in permitted:
        yield "%s: %r not one of %s" % (path, value, ", ".join(map(str, permitted)))


def _check_min_length(value: object, spec: Mapping[str, object],
                      path: str) -> _Complaints:
    limit = spec.get("minLength")
    if isinstance(value, str) and isinstance(limit, int) and len(value) < limit:
        yield "%s: string shorter than minLength %d" % (path, limit)


def _check_min_items(value: object, spec: Mapping[str, object],
                     path: str) -> _Complaints:
    limit = spec.get("minItems")
    if isinstance(value, list) and isinstance(limit, int) and len(value) < limit:
        yield "%s: array shorter than minItems %d" % (path, limit)


def _check_required(value: object, spec: Mapping[str, object],
                    path: str) -> _Complaints:
    required = spec.get("required")
    if isinstance(value, dict) and isinstance(required, list):
        for key in required:
            if key not in value:
                yield "%s: missing required field %r" % (path, key)


def _check_no_extras(value: object, spec: Mapping[str, object],
                     path: str) -> _Complaints:
    """``additionalProperties: false``: an unlisted key is a contract drift."""
    if not isinstance(value, dict) or spec.get("additionalProperties") is not False:
        return
    declared = spec.get("properties")
    declared = declared if isinstance(declared, Mapping) else {}
    for key in sorted(value):
        if key not in declared:
            yield "%s: unexpected field %r" % (path, key)


#: Every non-recursive keyword check, run in this order against every node.
_KEYWORD_CHECKS: Tuple[Callable[[object, Mapping[str, object], str], _Complaints], ...] = (
    _check_const,
    _check_enum,
    _check_min_length,
    _check_min_items,
    _check_required,
    _check_no_extras,
)


def _shape_errors(value: object, spec: Mapping[str, object], path: str) -> List[str]:
    """Every way ``value`` fails ``spec``, as dotted-path complaint strings.

    A type mismatch short-circuits: once a node is the wrong kind of thing, the
    keyword checks below it would only produce noise about a value that was
    never going to be inspected.
    """
    type_errors = list(_check_type(value, spec, path))
    if type_errors:
        return type_errors

    errors: List[str] = []
    for check in _KEYWORD_CHECKS:
        errors.extend(check(value, spec, path))

    if isinstance(value, list):
        item_spec = spec.get("items")
        if isinstance(item_spec, Mapping):
            for position, item in enumerate(value):
                errors.extend(
                    _shape_errors(item, item_spec, "%s[%d]" % (path, position)))
    elif isinstance(value, dict):
        declared = spec.get("properties")
        if isinstance(declared, Mapping):
            for key in sorted(declared):
                if key in value:
                    errors.extend(_shape_errors(
                        value[key], declared[key], "%s.%s" % (path, key)))
    return errors


def validate_catalog(catalog: Mapping[str, object]) -> List[str]:
    """Structural errors of a catalog dict against the embedded catalog schema.

    Returns an empty list when the dict is shape-valid.
    """
    return _shape_errors(catalog, ALLOWED_OPERATIONS_CATALOG_SCHEMA, "catalog")


def validate_protected_regions(doc: Mapping[str, object]) -> List[str]:
    """Structural errors of a protected-regions dict against its schema."""
    return _shape_errors(doc, PROTECTED_REGIONS_SCHEMA, "protected_regions")


# ---------------------------------------------------------------------------
# Dataclasses mirroring the catalog / protected-regions schemas
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class OperationRule:
    """One admissibility rule for one operation_type on one feature.

    A conditional rule carries the ``preconditions`` that must hold and the
    ``blocked_by_constraints`` that must NOT be active for the operation to
    proceed; for an ``allowed`` or ``forbidden`` rule both are ignored.
    """

    operation_type: str
    status: str
    reason: str
    preconditions: Tuple[str, ...] = ()
    blocked_by_constraints: Tuple[str, ...] = ()


@dataclass(frozen=True)
class FeatureOperations:
    """Per-feature catalog entry: ``feature_operations[]`` in the schema."""

    feature_id: str
    feature_type: str
    protected: bool
    interface_roles: Tuple[str, ...] = ()
    operations: Tuple[OperationRule, ...] = ()

    def rule_for(self, operation_type: str) -> Optional[OperationRule]:
        """The entry's rule for ``operation_type``, or None when unlisted."""
        for rule in self.operations:
            if rule.operation_type == operation_type:
                return rule
        return None


@dataclass(frozen=True)
class AllowedOperationsCatalog:
    """The whole allowed-operations catalog document."""

    format_version: str
    catalog_id: str
    generated_by: str
    generated_at_utc: str
    source_files: Tuple[str, ...] = ()
    feature_operations: Tuple[FeatureOperations, ...] = ()
    notes: Tuple[str, ...] = ()

    def entry_for(self, feature_ref: str) -> Optional[FeatureOperations]:
        """Resolve a feature reference: id match, then type match, then '*'.

        Three passes rather than one, because a catalog that names a feature
        explicitly must win over one that only matches its type, and both must
        win over the catch-all row -- regardless of the order the rows happen to
        be written in.
        """
        if feature_ref:
            for select in (lambda e: e.feature_id, lambda e: e.feature_type):
                for entry in self.feature_operations:
                    if select(entry) == feature_ref:
                        return entry
        for entry in self.feature_operations:
            if WILDCARD_FEATURE in (entry.feature_id, entry.feature_type):
                return entry
        return None


@dataclass(frozen=True)
class ProtectedRegion:
    """One ``protected_regions[]`` entry: a whitelist override per feature.

    An op touching this feature is refused when its operation_type is in
    ``forbidden_operations``, or when ``allowed_operations`` is non-empty and
    the operation_type is not in it.
    """

    feature_id: str
    reason: str
    allowed_operations: Tuple[str, ...] = ()
    forbidden_operations: Tuple[str, ...] = ()

    def refuses(self, operation_type: str) -> bool:
        if operation_type in self.forbidden_operations:
            return True
        # An empty whitelist means "nothing is singled out", not "nothing is
        # permitted" -- otherwise a region listing only forbidden ops would
        # refuse everything.
        return bool(self.allowed_operations) and \
            operation_type not in self.allowed_operations


def _string_tuple(raw: object) -> Tuple[str, ...]:
    """A tuple of strings from a schema array field that may be absent."""
    if isinstance(raw, (list, tuple)):
        return tuple(str(item) for item in raw)
    return ()


def _rule_from_dict(raw: Mapping[str, object]) -> OperationRule:
    return OperationRule(
        operation_type=str(raw["operation_type"]),
        status=str(raw["status"]),
        reason=str(raw["reason"]),
        preconditions=_string_tuple(raw.get("preconditions")),
        blocked_by_constraints=_string_tuple(raw.get("blocked_by_constraints")),
    )


def _entry_from_dict(raw: Mapping[str, object]) -> FeatureOperations:
    return FeatureOperations(
        feature_id=str(raw["feature_id"]),
        feature_type=str(raw["feature_type"]),
        protected=bool(raw["protected"]),
        interface_roles=_string_tuple(raw.get("interface_roles")),
        operations=tuple(_rule_from_dict(rule) for rule in raw["operations"]),
    )


def catalog_from_dict(d: Mapping[str, object]) -> AllowedOperationsCatalog:
    """Build the typed catalog from a shape-valid dict (validate first)."""
    entries = d.get("feature_operations")
    return AllowedOperationsCatalog(
        format_version=str(d.get("format_version", "")),
        catalog_id=str(d.get("catalog_id", "")),
        generated_by=str(d.get("generated_by", "")),
        generated_at_utc=str(d.get("generated_at_utc", "")),
        source_files=_string_tuple(d.get("source_files")),
        feature_operations=tuple(
            _entry_from_dict(entry)
            for entry in (entries if isinstance(entries, (list, tuple)) else ())
        ),
        notes=_string_tuple(d.get("notes")),
    )


def protected_regions_from_dict(
        d: Mapping[str, object]) -> Tuple[ProtectedRegion, ...]:
    """Build typed protected regions from a shape-valid dict."""
    raw_regions = d.get("protected_regions")
    return tuple(
        ProtectedRegion(
            feature_id=str(region["feature_id"]),
            reason=str(region["reason"]),
            allowed_operations=_string_tuple(region.get("allowed_operations")),
            forbidden_operations=_string_tuple(region.get("forbidden_operations")),
        )
        for region in (raw_regions if isinstance(raw_regions, (list, tuple)) else ())
    )


# ---------------------------------------------------------------------------
# Gate decisions
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class GateDiagnostic:
    """Typed refusal: why one op in the stream was refused."""

    op_index: int
    op_name: str
    feature: str
    reason_code: str        # one of REASON_CODES
    message: str


@dataclass(frozen=True)
class GateDecision:
    """Per-op verdict. ``allowed`` is False iff ``diagnostic`` is set."""

    op_index: int
    op_name: str
    feature: str
    operation_type: str
    allowed: bool
    rule_status: Optional[str] = None   # the matched rule's status, if any
    diagnostic: Optional[GateDiagnostic] = None


@dataclass(frozen=True)
class GateReport:
    """Whole-stream result, mirroring patch-proposal bookkeeping.

    ``protected_targets_checked`` lists every protected feature any op
    touched; ``protected_targets_avoided`` the subset no op violated. The
    ``patch_status`` maps the outcome onto the patch-proposal status enum.
    """

    decisions: Tuple[GateDecision, ...] = ()
    protected_targets_checked: Tuple[str, ...] = ()
    protected_targets_avoided: Tuple[str, ...] = ()

    @property
    def ok(self) -> bool:
        return all(d.allowed for d in self.decisions)

    @property
    def refusals(self) -> Tuple[GateDiagnostic, ...]:
        return tuple(d.diagnostic for d in self.decisions
                     if d.diagnostic is not None)

    @property
    def patch_status(self) -> str:
        """The patch-proposal status for this gate outcome.

        Violating a protected target is called out separately from an ordinary
        block, because it is the one failure a caller must never retry blindly.
        """
        if self.ok:
            return "ready_for_validation"
        if any(r.reason_code == "protected_region" for r in self.refusals):
            return "violates_protected_target"
        return "blocked"


# ---------------------------------------------------------------------------
# Op-stream introspection
# ---------------------------------------------------------------------------

def _as_dict(op: Union[ops.Op, Mapping[str, object]]) -> Dict[str, object]:
    if isinstance(op, ops.Op):
        return op.to_dict()
    return dict(op)


def op_operation_type(op: Union[ops.Op, Mapping[str, object]]) -> Optional[str]:
    """The catalog operation_type for a CISP op, or None for an unknown tag."""
    return _OP_TYPE_MAP.get(str(_as_dict(op).get("op", "")))


def op_feature_ref(op: Union[ops.Op, Mapping[str, object]],
                   oplog: Sequence[Union[ops.Op, Mapping[str, object]]] = (),
                   ) -> str:
    """The feature a CISP op touches (first non-empty reference field).

    A ``set_param`` op targets a prior op by index; when ``oplog`` is given
    the reference is resolved through the targeted op, so an edit to a hole's
    diameter is gated against the hole's feature.
    """
    fields = _as_dict(op)

    if fields.get("op") == "set_param" and oplog:
        try:
            target = int(fields.get("target", -1))  # type: ignore[arg-type]
        except (TypeError, ValueError):
            target = -1
        if 0 <= target < len(oplog):
            return op_feature_ref(oplog[target])

    for name in _FEATURE_REF_FIELDS:
        value = fields.get(name)
        if isinstance(value, str) and value:
            return value
    return ""


# ---------------------------------------------------------------------------
# Precondition / constraint evaluation
# ---------------------------------------------------------------------------

#: Comparators in match precedence: the two-character forms must be tried
#: before their single-character prefixes or ">=" would be read as ">".
_COMPARATORS: Tuple[str, ...] = ("==", "!=", ">=", "<=", ">", "<")

#: Comparators usable on any pair of values, and those needing two real numbers.
_EQUALITY_TESTS: Dict[str, Callable[[object, object], bool]] = {
    "==": lambda have, want: bool(have == want),
    "!=": lambda have, want: bool(have != want),
}
_ORDERED_TESTS: Dict[str, Callable[[float, float], bool]] = {
    ">=": lambda have, want: have >= want,
    "<=": lambda have, want: have <= want,
    ">": lambda have, want: have > want,
    "<": lambda have, want: have < want,
}


def _parse_literal(text: str) -> object:
    """The right-hand side of a precondition, typed as narrowly as it reads."""
    text = text.strip()
    if len(text) >= 2 and text[0] == text[-1] and text[0] in ("'", '"'):
        return text[1:-1]
    folded = text.lower()
    if folded == "true":
        return True
    if folded == "false":
        return False
    for converter in (int, float):
        try:
            return converter(text)
        except ValueError:
            continue
    return text


def _is_real_number(value: object) -> bool:
    """True for an int/float that is not a bool -- ``bool`` subclasses ``int``."""
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _split_comparison(text: str) -> Optional[Tuple[str, str, str]]:
    """Split ``"a >= 3"`` into ``("a", ">=", "3")``, or None if there is no
    comparator anywhere in the string."""
    for comparator in _COMPARATORS:
        if comparator in text:
            key, _, literal = text.partition(comparator)
            return key.strip(), comparator, literal
    return None


def evaluate_precondition(precondition: str,
                          context: Mapping[str, object]) -> Tuple[bool, str]:
    """Evaluate one precondition string against the context dict.

    Grammar: ``"key"`` (truthy) or ``"key <cmp> value"`` with ``<cmp>`` in
    ``== != >= <= > <``. Returns ``(met, detail)``; anything unparseable or a
    missing key is conservatively UNMET (default-deny), because a contract we
    cannot read is not a contract we may assume is satisfied.
    """
    text = precondition.strip()
    comparison = _split_comparison(text)

    if comparison is None:
        if not text:
            return False, "empty precondition"
        if text not in context:
            return False, "context has no value for %r" % text
        return bool(context[text]), "%s is %r" % (text, context[text])

    key, comparator, literal = comparison
    if not key:
        return False, "unparseable precondition %r" % precondition
    if key not in context:
        return False, "context has no value for %r" % key

    have = context[key]
    want = _parse_literal(literal)

    equality = _EQUALITY_TESTS.get(comparator)
    if equality is not None:
        return (equality(have, want),
                "%s %s %r (actual %r)" % (key, comparator, want, have))

    if not _is_real_number(have) or not _is_real_number(want):
        return False, "non-numeric ordered comparison in %r" % precondition
    met = _ORDERED_TESTS[comparator](float(have), float(want))  # type: ignore[arg-type]
    return met, "%s %s %r (actual %r)" % (key, comparator, want, have)


def _active_constraints(context: Mapping[str, object]) -> Tuple[str, ...]:
    """The context's ``active_constraints``, normalised to a sorted tuple."""
    raw = context.get("active_constraints", ())
    if isinstance(raw, str):
        return (raw,)
    if isinstance(raw, (list, tuple, set, frozenset)):
        return tuple(sorted(str(name) for name in raw))
    return ()


# ---------------------------------------------------------------------------
# The gate
# ---------------------------------------------------------------------------

class _OpJudge:
    """Judges ops one at a time and remembers what protected features it saw.

    The judgement is a chain of narrowing questions; the first that produces a
    diagnostic ends the op. Splitting it this way means each rule is one short
    method that can be read -- and argued about -- on its own.
    """

    def __init__(self, catalog: AllowedOperationsCatalog,
                 regions: Mapping[str, ProtectedRegion],
                 context: Mapping[str, object]) -> None:
        self._catalog = catalog
        self._regions = regions
        self._context = context
        self._active = _active_constraints(context)
        self.touched: List[str] = []
        self.violated: List[str] = []

    # -- bookkeeping ------------------------------------------------------
    @staticmethod
    def _remember(bucket: List[str], feature: str) -> None:
        if feature not in bucket:
            bucket.append(feature)

    def avoided(self) -> Tuple[str, ...]:
        return tuple(f for f in self.touched if f not in self.violated)

    # -- individual rules -------------------------------------------------
    def _protected_region_refusal(self, feature: str,
                                  op_type: str) -> Optional[Tuple[str, str]]:
        """``(code, message)`` when a protected region forbids this op."""
        region = self._regions.get(feature)
        if region is None:
            return None
        self._remember(self.touched, feature)
        if not region.refuses(op_type):
            return None
        self._remember(self.violated, feature)
        return ("protected_region",
                "feature %r is a protected region (%s); operation %r is not "
                "admitted there" % (feature, region.reason, op_type))

    def _parameter_refusal(self, fields: Mapping[str, object]
                           ) -> Optional[Tuple[str, str]]:
        """Preflight a ``modify_parameter`` op before the catalog sees it.

        Two rules travel with a parameter edit: the new value must sit inside
        the parameter's declared bounds, and the parameter must not be one whose
        edit would change topology -- a parameter edit that re-topologises the
        model is a different operation wearing a parameter edit's clothes.
        """
        param = str(fields.get("param", ""))

        topology_changing = self._context.get("topology_changing_parameters", ())
        if isinstance(topology_changing, (list, tuple, set, frozenset)) \
                and param in topology_changing:
            return ("topology_changing",
                    "parameter %r is topology-changing; parameter edits must "
                    "not change topology (parameter_edit preflight)" % param)

        declared = self._context.get("parameter_bounds")
        if not isinstance(declared, Mapping):
            return None
        bounds = declared.get(param)
        if not isinstance(bounds, Mapping):
            return None

        value = fields.get("value")
        discrete = bounds.get("discrete_values")
        if isinstance(discrete, (list, tuple)) and discrete:
            if value in discrete:
                return None
            return ("value_out_of_bounds",
                    "value %r for %r is not one of the declared discrete_values "
                    "%r" % (value, param, list(discrete)))

        if not _is_real_number(value):
            return None
        low = bounds.get("min")
        high = bounds.get("max")
        if _is_real_number(low) and value < low:  # type: ignore[operator]
            return ("value_out_of_bounds",
                    "value %r for %r is below the declared min %r"
                    % (value, param, low))
        if _is_real_number(high) and value > high:  # type: ignore[operator]
            return ("value_out_of_bounds",
                    "value %r for %r is above the declared max %r"
                    % (value, param, high))
        return None

    def _rule_refusal(self, rule: OperationRule, entry: FeatureOperations,
                      op_type: str) -> Optional[Tuple[str, str]]:
        """Apply a matched catalog rule: forbidden, or a conditional's terms."""
        if rule.status == "forbidden":
            return ("forbidden",
                    "operation %r on feature %r is forbidden: %s"
                    % (op_type, entry.feature_id, rule.reason))
        if rule.status != "conditional":
            return None

        blocking = [name for name in rule.blocked_by_constraints
                    if name in self._active]
        if blocking:
            return ("blocked_by_constraint",
                    "operation %r on feature %r is blocked by active "
                    "constraint(s): %s"
                    % (op_type, entry.feature_id, ", ".join(blocking)))

        for precondition in rule.preconditions:
            met, detail = evaluate_precondition(precondition, self._context)
            if not met:
                return ("precondition_unmet",
                        "precondition %r unmet for operation %r on feature %r: "
                        "%s" % (precondition, op_type, entry.feature_id, detail))
        return None

    # -- the chain --------------------------------------------------------
    def judge(self, index: int,
              op: Union[ops.Op, Mapping[str, object]],
              op_stream: Sequence[Union[ops.Op, Mapping[str, object]]],
              ) -> GateDecision:
        fields = _as_dict(op)
        op_name = str(fields.get("op", ""))
        feature = op_feature_ref(op, op_stream)

        def refuse(op_type: str, status: Optional[str],
                   refusal: Tuple[str, str]) -> GateDecision:
            code, message = refusal
            return GateDecision(
                op_index=index, op_name=op_name, feature=feature,
                operation_type=op_type, allowed=False, rule_status=status,
                diagnostic=GateDiagnostic(
                    op_index=index, op_name=op_name, feature=feature,
                    reason_code=code, message=message))

        op_type = op_operation_type(op)
        if op_type is None:
            return refuse("", None, (
                "unknown_op",
                "op tag %r is not a registered CISP operation" % op_name))

        region_refusal = self._protected_region_refusal(feature, op_type)
        if region_refusal is not None:
            return refuse(op_type, None, region_refusal)

        if op_type == "modify_parameter":
            parameter_refusal = self._parameter_refusal(fields)
            if parameter_refusal is not None:
                return refuse(op_type, None, parameter_refusal)

        # The catalog is a whitelist: an unmatched feature, or a feature with no
        # rule for this operation, is denied rather than waved through.
        entry = self._catalog.entry_for(feature)
        if entry is None:
            return refuse(op_type, None, (
                "unknown_feature",
                "no catalog entry admits feature %r (and no wildcard entry "
                "exists); default is deny" % feature))

        rule = entry.rule_for(op_type)
        if rule is None:
            return refuse(op_type, None, (
                "unknown_op",
                "catalog entry for feature %r lists no rule for operation %r; "
                "unlisted operations are denied" % (entry.feature_id, op_type)))

        rule_refusal = self._rule_refusal(rule, entry, op_type)
        if rule_refusal is not None:
            return refuse(op_type, rule.status, rule_refusal)

        return GateDecision(
            op_index=index, op_name=op_name, feature=feature,
            operation_type=op_type, allowed=True, rule_status=rule.status)


def gate(op_stream: Sequence[Union[ops.Op, Mapping[str, object]]],
         catalog: Union[AllowedOperationsCatalog, Mapping[str, object]],
         context: Optional[Mapping[str, object]] = None,
         protected_regions: Sequence[ProtectedRegion] = (),
         ) -> GateReport:
    """Judge every op in a CISP op stream against the catalog BEFORE execution.

    ``op_stream`` is a sequence of :class:`harnesscad.core.cisp.ops.Op`
    instances or their dict forms. ``catalog`` is a typed
    :class:`AllowedOperationsCatalog` or a raw catalog dict (validated
    structurally first; a shape-invalid dict raises ``ValueError`` so a
    malformed contract can never silently admit an op). ``context`` supplies
    the values preconditions are evaluated against, the
    ``active_constraints`` collection, and the optional parameter-edit
    preflight inputs (``parameter_bounds``, ``topology_changing_parameters``).
    ``protected_regions`` override the catalog for the features they name.

    Decision order per op: unknown op tag -> protected region ->
    parameter-edit preflight (modify_parameter only) -> catalog rule
    (allowed / forbidden / conditional with preconditions +
    blocked_by_constraints). An op whose feature matches no catalog entry
    (and no wildcard entry) is refused ``unknown_feature`` -- the catalog is
    a whitelist, so the default is deny.
    """
    if not isinstance(catalog, AllowedOperationsCatalog):
        errors = validate_catalog(catalog)
        if errors:
            raise ValueError(
                "catalog fails allowed_operations_catalog schema shape: "
                + "; ".join(errors))
        catalog = catalog_from_dict(catalog)

    judge = _OpJudge(
        catalog=catalog,
        regions={region.feature_id: region for region in protected_regions},
        context=context or {},
    )
    decisions = tuple(judge.judge(index, op, op_stream)
                      for index, op in enumerate(op_stream))

    return GateReport(
        decisions=decisions,
        protected_targets_checked=tuple(judge.touched),
        protected_targets_avoided=judge.avoided(),
    )


# ---------------------------------------------------------------------------
# Selfcheck
# ---------------------------------------------------------------------------

def _selfcheck() -> int:
    catalog_dict: Dict[str, object] = {
        "format_version": "0.1.0",
        "catalog_id": "op_gate_selfcheck",
        "generated_by": "harnesscad.core.cisp.op_gate --selfcheck",
        "generated_at_utc": "1970-01-01T00:00:00Z",
        "source_files": ["<selfcheck>"],
        "feature_operations": [
            {
                "feature_id": "base_plate",
                "feature_type": "plate",
                "protected": False,
                "interface_roles": ["load_interface"],
                "operations": [
                    {
                        "operation_type": "add_feature",
                        "status": "allowed",
                        "reason": "additive geometry on the base plate is safe",
                        "preconditions": [],
                        "blocked_by_constraints": [],
                    },
                    {
                        "operation_type": "modify_parameter",
                        "status": "conditional",
                        "reason": "dimensional edits allowed above the CNC "
                                  "minimum wall",
                        "preconditions": ["wall_thickness >= 3"],
                        "blocked_by_constraints": ["locked_for_release"],
                    },
                    {
                        "operation_type": "remove_feature",
                        "status": "forbidden",
                        "reason": "the base plate carries the mounting "
                                  "interface",
                        "preconditions": [],
                        "blocked_by_constraints": [],
                    },
                ],
            },
            {
                "feature_id": "sealed_housing",
                "feature_type": "housing",
                "protected": False,
                "interface_roles": [],
                "operations": [
                    {
                        "operation_type": "add_feature",
                        "status": "forbidden",
                        "reason": "the housing envelope is certified; no new "
                                  "geometry may touch it",
                        "preconditions": [],
                        "blocked_by_constraints": [],
                    },
                ],
            },
            {
                "feature_id": "mounting_hole_pattern",
                "feature_type": "mounting_hole_pattern",
                "protected": True,
                "interface_roles": ["bolted_interface"],
                "operations": [
                    {
                        "operation_type": "add_feature",
                        "status": "allowed",
                        "reason": "catalog would allow it, but the protected "
                                  "region overrides",
                        "preconditions": [],
                        "blocked_by_constraints": [],
                    },
                ],
            },
        ],
        "notes": ["deterministic selfcheck catalog"],
    }
    assert validate_catalog(catalog_dict) == [], "selfcheck catalog invalid"

    broken = dict(catalog_dict)
    del broken["catalog_id"]
    broken["extra"] = 1
    errs = validate_catalog(broken)
    assert any("catalog_id" in e for e in errs), errs
    assert any("extra" in e for e in errs), errs

    regions_dict: Dict[str, object] = {
        "format_version": "0.1.0",
        "protected_regions": [
            {
                "feature_id": "mounting_hole_pattern",
                "reason": "bolted interface; do not modify casually",
                "allowed_operations": ["protect_feature"],
                "forbidden_operations": ["remove_feature", "modify_parameter"],
            },
        ],
    }
    assert validate_protected_regions(regions_dict) == []
    regions = protected_regions_from_dict(regions_dict)

    # Case 1 + 2: allowed add_feature, and a conditional modify_parameter that
    # PASSES (precondition met, no active blocking constraint).
    stream = [
        ops.Extrude(sketch="base_plate", distance=8.0),
        ops.SetParam(target=0, param="distance", value=6.0),
    ]
    report = gate(stream, catalog_dict, context={"wall_thickness": 4.0})
    assert report.ok, report.refusals
    assert [d.operation_type for d in report.decisions] == \
        ["add_feature", "modify_parameter"]
    assert report.patch_status == "ready_for_validation"
    print("case allowed + conditional-pass: ok "
          "(patch_status=%s)" % report.patch_status)

    # Case 3: conditional FAIL, precondition unmet.
    report = gate(stream, catalog_dict, context={"wall_thickness": 2.0})
    assert not report.decisions[1].allowed
    assert report.refusals[0].reason_code == "precondition_unmet"
    assert report.patch_status == "blocked"
    print("case conditional-fail (precondition_unmet): %s"
          % report.refusals[0].message)

    # Case 4: conditional FAIL, blocked by an active constraint.
    report = gate(stream, catalog_dict, context={
        "wall_thickness": 4.0,
        "active_constraints": ["locked_for_release"],
    })
    assert report.refusals[0].reason_code == "blocked_by_constraint"
    print("case conditional-fail (blocked_by_constraint): %s"
          % report.refusals[0].message)

    # Case 5: forbidden operation (any add_feature touching the housing).
    report = gate([ops.Extrude(sketch="sealed_housing", distance=2.0)],
                  catalog_dict, context={})
    assert report.refusals[0].reason_code == "forbidden"
    print("case forbidden: %s" % report.refusals[0].message)

    # Case 6: protected region refusal (parameter edit on a bolted interface).
    stream = [
        ops.Hole(face_or_sketch="mounting_hole_pattern", diameter=5.0),
        ops.SetParam(target=0, param="diameter", value=6.0),
    ]
    report = gate(stream, catalog_dict, context={"wall_thickness": 4.0},
                  protected_regions=regions)
    diags = report.refusals
    assert all(d.reason_code == "protected_region" for d in diags), diags
    assert len(diags) == 2  # add_feature AND modify_parameter both refused
    assert report.protected_targets_checked == ("mounting_hole_pattern",)
    assert report.protected_targets_avoided == ()
    assert report.patch_status == "violates_protected_target"
    print("case protected-region: %s (patch_status=%s)"
          % (diags[0].message, report.patch_status))

    # Case 7: unknown op tag and unknown feature (default deny).
    report = gate([{"op": "warp_drive"},
                   ops.Extrude(sketch="ghost_feature", distance=1.0)],
                  catalog_dict, context={})
    assert report.refusals[0].reason_code == "unknown_op"
    assert report.refusals[1].reason_code == "unknown_feature"
    print("case unknown_op + unknown_feature: refused as expected")

    # Case 8: parameter-edit preflight -- bounds and topology-changing.
    stream = [
        ops.Extrude(sketch="base_plate", distance=8.0),
        ops.SetParam(target=0, param="distance", value=50.0),
    ]
    report = gate(stream, catalog_dict, context={
        "wall_thickness": 4.0,
        "parameter_bounds": {"distance": {"min": 2.0, "max": 10.0}},
    })
    assert report.refusals[0].reason_code == "value_out_of_bounds"
    print("case parameter-edit bounds: %s" % report.refusals[0].message)
    report = gate(stream, catalog_dict, context={
        "wall_thickness": 4.0,
        "topology_changing_parameters": ["distance"],
    })
    assert report.refusals[0].reason_code == "topology_changing"
    print("case parameter-edit topology-changing: %s"
          % report.refusals[0].message)

    # Round trip: typed catalog behaves identically to the raw dict.
    typed = catalog_from_dict(catalog_dict)
    r1 = gate(stream, typed, context={"wall_thickness": 4.0})
    r2 = gate(stream, catalog_dict, context={"wall_thickness": 4.0})
    assert r1 == r2

    print("op_gate selfcheck passed")
    return 0


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        prog="op_gate",
        description="Gate a CISP op stream against a per-feature "
                    "allowed-operations catalog.")
    parser.add_argument("--selfcheck", action="store_true",
                        help="run the built-in gate scenarios and exit 0")
    parser.add_argument("--catalog", type=str, default=None,
                        help="path to a catalog JSON to shape-validate")
    args = parser.parse_args(argv)
    if args.catalog:
        with open(args.catalog, "r", encoding="utf-8") as fh:
            doc = json.load(fh)
        errors = validate_catalog(doc)
        if errors:
            for err in errors:
                print(err)
            return 1
        print("catalog is shape-valid (%s)" % doc.get("catalog_id", "?"))
        return 0
    if args.selfcheck:
        return _selfcheck()
    parser.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
