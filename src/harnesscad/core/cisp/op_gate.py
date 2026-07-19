"""Operation-gating contract layer: validate a CISP op stream BEFORE execution.

Derived from cad-cae-copilot (MIT License, Copyright (c) 2026 armpro24-blip).

Implements an operation-admissibility contract built from four schema
contracts:

* an allowed-operations catalog -- the per-feature catalog of
  allowed / forbidden / conditional operations, where a conditional operation
  carries ``preconditions`` and ``blocked_by_constraints``;
* protected regions -- per-feature protected regions with
  ``allowed_operations`` / ``forbidden_operations`` whitelists that override
  the catalog (an op touching a protected region is refused unless the region
  explicitly admits it);
* a parameter-edit preflight contract:
  value-in-bounds checks against declared ``bounds`` (min / max /
  discrete_values) and the hard rule that "topology-changing parameters must
  be refused";
* a patch-proposal lifecycle whose
  ``protected_targets_checked`` / ``protected_targets_avoided`` bookkeeping and
  status vocabulary (``ready_for_validation`` / ``violates_protected_target``
  / ``blocked``) this gate's report mirrors.

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

Preconditions are the schema's free strings; the evaluable subset understood
here is ``"key"`` (truthy lookup in the caller's context dict) and
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
from typing import Dict, List, Mapping, Optional, Sequence, Tuple, Union

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
# These two dicts are the gating-relevant JSON Schemas (the allowed-operations
# catalog and the protected-regions schema), embedded as data minus the
# $-metadata keys. They drive the stdlib structural checker below.
# ---------------------------------------------------------------------------

_OPERATION_RULE_SCHEMA: Dict[str, object] = {
    "type": "object",
    "required": [
        "operation_type",
        "status",
        "reason",
        "preconditions",
        "blocked_by_constraints",
    ],
    "additionalProperties": False,
    "properties": {
        "operation_type": {"type": "string", "enum": list(OPERATION_TYPES)},
        "status": {"type": "string", "enum": list(OPERATION_STATUSES)},
        "reason": {"type": "string", "minLength": 1},
        "preconditions": {
            "type": "array",
            "items": {"type": "string", "minLength": 1},
        },
        "blocked_by_constraints": {
            "type": "array",
            "items": {"type": "string", "minLength": 1},
        },
    },
}

ALLOWED_OPERATIONS_CATALOG_SCHEMA: Dict[str, object] = {
    "type": "object",
    "required": [
        "format_version",
        "catalog_id",
        "generated_by",
        "generated_at_utc",
        "source_files",
        "feature_operations",
        "notes",
    ],
    "additionalProperties": False,
    "properties": {
        "format_version": {"type": "string", "const": "0.1.0"},
        "catalog_id": {"type": "string", "minLength": 1},
        "generated_by": {"type": "string", "minLength": 1},
        "generated_at_utc": {"type": "string", "minLength": 1},
        "source_files": {
            "type": "array",
            "items": {"type": "string", "minLength": 1},
        },
        "feature_operations": {
            "type": "array",
            "items": {
                "type": "object",
                "required": [
                    "feature_id",
                    "feature_type",
                    "protected",
                    "interface_roles",
                    "operations",
                ],
                "additionalProperties": False,
                "properties": {
                    "feature_id": {"type": "string", "minLength": 1},
                    "feature_type": {"type": "string", "minLength": 1},
                    "protected": {"type": "boolean"},
                    "interface_roles": {
                        "type": "array",
                        "items": {"type": "string", "minLength": 1},
                    },
                    "operations": {
                        "type": "array",
                        "minItems": 1,
                        "items": _OPERATION_RULE_SCHEMA,
                    },
                },
            },
        },
        "notes": {
            "type": "array",
            "minItems": 1,
            "items": {"type": "string", "minLength": 1},
        },
    },
}

PROTECTED_REGIONS_SCHEMA: Dict[str, object] = {
    "type": "object",
    "required": ["format_version", "protected_regions"],
    "additionalProperties": False,
    "properties": {
        "format_version": {"type": "string", "const": "0.1.0"},
        "protected_regions": {
            "type": "array",
            "items": {
                "type": "object",
                "required": [
                    "feature_id",
                    "reason",
                    "allowed_operations",
                    "forbidden_operations",
                ],
                "additionalProperties": False,
                "properties": {
                    "feature_id": {"type": "string", "minLength": 1},
                    "reason": {"type": "string", "minLength": 1},
                    "allowed_operations": {
                        "type": "array",
                        "items": {"type": "string"},
                    },
                    "forbidden_operations": {
                        "type": "array",
                        "items": {"type": "string"},
                    },
                },
            },
        },
    },
}


# ---------------------------------------------------------------------------
# Stdlib structural (schema-shape) validation
# ---------------------------------------------------------------------------

_TYPE_MAP = {
    "object": dict,
    "array": list,
    "string": str,
    "boolean": bool,
    "integer": int,
    "number": (int, float),
    "null": type(None),
}


def _shape_errors(value: object, spec: Mapping[str, object],
                  path: str, errors: List[str]) -> None:
    """Recursive shape check of ``value`` against an embedded schema dict.

    Implements the subset of JSON Schema the two gating schemas use: type,
    const, enum, minLength, minItems, required, properties,
    additionalProperties: false, items. Appends dotted-path messages.
    """
    expected = spec.get("type")
    if expected is not None:
        py = _TYPE_MAP.get(str(expected))
        if py is not None:
            # bool is an int subclass; keep integer/number honest.
            if isinstance(value, bool) and expected in ("integer", "number"):
                errors.append("%s: expected %s, got boolean" % (path, expected))
                return
            if not isinstance(value, py):
                errors.append(
                    "%s: expected %s, got %s"
                    % (path, expected, type(value).__name__))
                return
    if "const" in spec and value != spec["const"]:
        errors.append("%s: must equal %r" % (path, spec["const"]))
    if "enum" in spec and value not in spec["enum"]:  # type: ignore[operator]
        errors.append(
            "%s: %r not one of %s"
            % (path, value, ", ".join(map(str, spec["enum"]))))  # type: ignore[arg-type]
    if isinstance(value, str):
        min_len = spec.get("minLength")
        if isinstance(min_len, int) and len(value) < min_len:
            errors.append("%s: string shorter than minLength %d" % (path, min_len))
    if isinstance(value, list):
        min_items = spec.get("minItems")
        if isinstance(min_items, int) and len(value) < min_items:
            errors.append("%s: array shorter than minItems %d" % (path, min_items))
        item_spec = spec.get("items")
        if isinstance(item_spec, Mapping):
            for i, item in enumerate(value):
                _shape_errors(item, item_spec, "%s[%d]" % (path, i), errors)
    if isinstance(value, dict):
        required = spec.get("required")
        if isinstance(required, list):
            for key in required:
                if key not in value:
                    errors.append("%s: missing required field %r" % (path, key))
        props = spec.get("properties")
        props = props if isinstance(props, Mapping) else {}
        if spec.get("additionalProperties") is False:
            for key in sorted(value):
                if key not in props:
                    errors.append("%s: unexpected field %r" % (path, key))
        for key in sorted(props):
            if key in value:
                _shape_errors(value[key], props[key],  # type: ignore[index]
                              "%s.%s" % (path, key), errors)


def validate_catalog(catalog: Mapping[str, object]) -> List[str]:
    """Structural errors of a catalog dict against the embedded catalog schema.

    Returns an empty list when the dict is shape-valid.
    """
    errors: List[str] = []
    _shape_errors(catalog, ALLOWED_OPERATIONS_CATALOG_SCHEMA, "catalog", errors)
    return errors


def validate_protected_regions(doc: Mapping[str, object]) -> List[str]:
    """Structural errors of a protected-regions dict against its schema."""
    errors: List[str] = []
    _shape_errors(doc, PROTECTED_REGIONS_SCHEMA, "protected_regions", errors)
    return errors


# ---------------------------------------------------------------------------
# Dataclasses mirroring the catalog / protected-regions schemas
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class OperationRule:
    """One admissibility rule for one operation_type on one feature.

    Field names and semantics are exactly the catalog schema's
    ``feature_operations[].operations[]`` item: a conditional rule carries the
    ``preconditions`` that must hold and the ``blocked_by_constraints`` that
    must NOT be active for the operation to proceed.
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
        """Resolve a feature reference: id match, then type match, then '*'."""
        if feature_ref:
            for entry in self.feature_operations:
                if entry.feature_id == feature_ref:
                    return entry
            for entry in self.feature_operations:
                if entry.feature_type == feature_ref:
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
        return bool(self.allowed_operations) and \
            operation_type not in self.allowed_operations


def catalog_from_dict(d: Mapping[str, object]) -> AllowedOperationsCatalog:
    """Build the typed catalog from a shape-valid dict (validate first)."""
    entries = []
    for fe in d.get("feature_operations", ()):  # type: ignore[union-attr]
        rules = tuple(
            OperationRule(
                operation_type=str(op["operation_type"]),
                status=str(op["status"]),
                reason=str(op["reason"]),
                preconditions=tuple(op.get("preconditions", ())),
                blocked_by_constraints=tuple(op.get("blocked_by_constraints", ())),
            )
            for op in fe["operations"]
        )
        entries.append(FeatureOperations(
            feature_id=str(fe["feature_id"]),
            feature_type=str(fe["feature_type"]),
            protected=bool(fe["protected"]),
            interface_roles=tuple(fe.get("interface_roles", ())),
            operations=rules,
        ))
    return AllowedOperationsCatalog(
        format_version=str(d.get("format_version", "")),
        catalog_id=str(d.get("catalog_id", "")),
        generated_by=str(d.get("generated_by", "")),
        generated_at_utc=str(d.get("generated_at_utc", "")),
        source_files=tuple(d.get("source_files", ())),  # type: ignore[arg-type]
        feature_operations=tuple(entries),
        notes=tuple(d.get("notes", ())),  # type: ignore[arg-type]
    )


def protected_regions_from_dict(
        d: Mapping[str, object]) -> Tuple[ProtectedRegion, ...]:
    """Build typed protected regions from a shape-valid dict."""
    return tuple(
        ProtectedRegion(
            feature_id=str(r["feature_id"]),
            reason=str(r["reason"]),
            allowed_operations=tuple(r.get("allowed_operations", ())),
            forbidden_operations=tuple(r.get("forbidden_operations", ())),
        )
        for r in d.get("protected_regions", ())  # type: ignore[union-attr]
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
    """Whole-stream result, mirroring patch_proposal bookkeeping.

    ``protected_targets_checked`` lists every protected feature any op
    touched; ``protected_targets_avoided`` the subset no op violated. The
    ``patch_status`` maps the outcome onto the patch_proposal status enum.
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
        """patch_proposal.schema.json status for this gate outcome."""
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
    tag = str(_as_dict(op).get("op", ""))
    return _OP_TYPE_MAP.get(tag)


def op_feature_ref(op: Union[ops.Op, Mapping[str, object]],
                   oplog: Sequence[Union[ops.Op, Mapping[str, object]]] = (),
                   ) -> str:
    """The feature a CISP op touches (first non-empty reference field).

    A ``set_param`` op targets a prior op by index; when ``oplog`` is given
    the reference is resolved through the targeted op, so an edit to a hole's
    diameter is gated against the hole's feature.
    """
    d = _as_dict(op)
    if d.get("op") == "set_param" and oplog:
        try:
            idx = int(d.get("target", -1))  # type: ignore[arg-type]
        except (TypeError, ValueError):
            idx = -1
        if 0 <= idx < len(oplog):
            return op_feature_ref(oplog[idx])
    for name in _FEATURE_REF_FIELDS:
        value = d.get(name)
        if isinstance(value, str) and value:
            return value
    return ""


# ---------------------------------------------------------------------------
# Precondition / constraint evaluation
# ---------------------------------------------------------------------------

_COMPARATORS = ("==", "!=", ">=", "<=", ">", "<")


def _parse_literal(text: str) -> object:
    text = text.strip()
    if len(text) >= 2 and text[0] == text[-1] and text[0] in ("'", '"'):
        return text[1:-1]
    lowered = text.lower()
    if lowered == "true":
        return True
    if lowered == "false":
        return False
    try:
        return int(text)
    except ValueError:
        pass
    try:
        return float(text)
    except ValueError:
        pass
    return text


def evaluate_precondition(precondition: str,
                          context: Mapping[str, object]) -> Tuple[bool, str]:
    """Evaluate one precondition string against the context dict.

    Grammar: ``"key"`` (truthy) or ``"key <cmp> value"`` with ``<cmp>`` in
    ``== != >= <= > <``. Returns ``(met, detail)``; anything unparseable or a
    missing key is conservatively UNMET (default-deny).
    """
    text = precondition.strip()
    for cmp_op in _COMPARATORS:
        if cmp_op in text:
            key, _, raw = text.partition(cmp_op)
            key = key.strip()
            if not key:
                return False, "unparseable precondition %r" % precondition
            if key not in context:
                return False, "context has no value for %r" % key
            have = context[key]
            want = _parse_literal(raw)
            if cmp_op == "==":
                return (have == want,
                        "%s == %r (actual %r)" % (key, want, have))
            if cmp_op == "!=":
                return (have != want,
                        "%s != %r (actual %r)" % (key, want, have))
            # Ordered comparison requires two real numbers.
            if not isinstance(have, (int, float)) or isinstance(have, bool) \
                    or not isinstance(want, (int, float)):
                return False, ("non-numeric ordered comparison in %r"
                               % precondition)
            met = {
                ">=": have >= want,
                "<=": have <= want,
                ">": have > want,
                "<": have < want,
            }[cmp_op]
            return met, "%s %s %r (actual %r)" % (key, cmp_op, want, have)
    if not text:
        return False, "empty precondition"
    if text not in context:
        return False, "context has no value for %r" % text
    return bool(context[text]), "%s is %r" % (text, context[text])


def _active_constraints(context: Mapping[str, object]) -> Tuple[str, ...]:
    raw = context.get("active_constraints", ())
    if isinstance(raw, str):
        return (raw,)
    if isinstance(raw, (list, tuple, set, frozenset)):
        return tuple(sorted(str(c) for c in raw))
    return ()


# ---------------------------------------------------------------------------
# Parameter-edit preflight (parameter_edit.schema.json semantics)
# ---------------------------------------------------------------------------

def _parameter_edit_preflight(index: int, d: Mapping[str, object],
                              feature: str,
                              context: Mapping[str, object],
                              ) -> Optional[GateDiagnostic]:
    """Preflight a modify_parameter op the way parameter_edit's adapter does.

    Two checks travel with the port: ``value_in_bounds`` against a declared
    ``bounds`` block (min / max / discrete_values, read from the context's
    ``parameter_bounds`` mapping keyed by parameter name), and the hard rule
    that topology-changing parameters must be refused (context's
    ``topology_changing_parameters`` collection).
    """
    param = str(d.get("param", ""))
    op_name = str(d.get("op", ""))
    topo = context.get("topology_changing_parameters", ())
    if isinstance(topo, (list, tuple, set, frozenset)) and param in topo:
        return GateDiagnostic(
            op_index=index, op_name=op_name, feature=feature,
            reason_code="topology_changing",
            message="parameter %r is topology-changing; parameter edits must "
                    "not change topology (parameter_edit preflight)" % param)
    bounds_map = context.get("parameter_bounds")
    if not isinstance(bounds_map, Mapping):
        return None
    bounds = bounds_map.get(param)
    if not isinstance(bounds, Mapping):
        return None
    value = d.get("value")
    discrete = bounds.get("discrete_values")
    if isinstance(discrete, (list, tuple)) and discrete:
        if value not in discrete:
            return GateDiagnostic(
                op_index=index, op_name=op_name, feature=feature,
                reason_code="value_out_of_bounds",
                message="value %r for %r is not one of the declared "
                        "discrete_values %r" % (value, param, list(discrete)))
        return None
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        return None
    lo = bounds.get("min")
    hi = bounds.get("max")
    if isinstance(lo, (int, float)) and value < lo:
        return GateDiagnostic(
            op_index=index, op_name=op_name, feature=feature,
            reason_code="value_out_of_bounds",
            message="value %r for %r is below the declared min %r"
                    % (value, param, lo))
    if isinstance(hi, (int, float)) and value > hi:
        return GateDiagnostic(
            op_index=index, op_name=op_name, feature=feature,
            reason_code="value_out_of_bounds",
            message="value %r for %r is above the declared max %r"
                    % (value, param, hi))
    return None


# ---------------------------------------------------------------------------
# The gate
# ---------------------------------------------------------------------------

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
    ctx: Mapping[str, object] = context or {}
    if not isinstance(catalog, AllowedOperationsCatalog):
        errors = validate_catalog(catalog)
        if errors:
            raise ValueError(
                "catalog fails allowed_operations_catalog schema shape: "
                + "; ".join(errors))
        catalog = catalog_from_dict(catalog)

    protected_by_feature: Dict[str, ProtectedRegion] = {
        r.feature_id: r for r in protected_regions
    }
    active = _active_constraints(ctx)
    decisions: List[GateDecision] = []
    checked: List[str] = []
    violated: List[str] = []

    def refuse(index: int, op_name: str, feature: str, op_type: str,
               status: Optional[str], code: str, message: str) -> None:
        decisions.append(GateDecision(
            op_index=index, op_name=op_name, feature=feature,
            operation_type=op_type, allowed=False, rule_status=status,
            diagnostic=GateDiagnostic(
                op_index=index, op_name=op_name, feature=feature,
                reason_code=code, message=message)))

    for index, op in enumerate(op_stream):
        d = _as_dict(op)
        op_name = str(d.get("op", ""))
        feature = op_feature_ref(op, op_stream)
        op_type = op_operation_type(op)

        if op_type is None:
            refuse(index, op_name, feature, "", None, "unknown_op",
                   "op tag %r is not a registered CISP operation" % op_name)
            continue

        region = protected_by_feature.get(feature)
        if region is not None:
            if feature not in checked:
                checked.append(feature)
            if region.refuses(op_type):
                if feature not in violated:
                    violated.append(feature)
                refuse(index, op_name, feature, op_type, None,
                       "protected_region",
                       "feature %r is a protected region (%s); operation %r "
                       "is not admitted there"
                       % (feature, region.reason, op_type))
                continue

        if op_type == "modify_parameter":
            diag = _parameter_edit_preflight(index, d, feature, ctx)
            if diag is not None:
                decisions.append(GateDecision(
                    op_index=index, op_name=op_name, feature=feature,
                    operation_type=op_type, allowed=False,
                    rule_status=None, diagnostic=diag))
                continue

        entry = catalog.entry_for(feature)
        if entry is None:
            refuse(index, op_name, feature, op_type, None, "unknown_feature",
                   "no catalog entry admits feature %r (and no wildcard "
                   "entry exists); default is deny" % feature)
            continue
        rule = entry.rule_for(op_type)
        if rule is None:
            refuse(index, op_name, feature, op_type, None, "unknown_op",
                   "catalog entry for feature %r lists no rule for operation "
                   "%r; unlisted operations are denied"
                   % (entry.feature_id, op_type))
            continue

        if rule.status == "forbidden":
            refuse(index, op_name, feature, op_type, rule.status, "forbidden",
                   "operation %r on feature %r is forbidden: %s"
                   % (op_type, entry.feature_id, rule.reason))
            continue

        if rule.status == "conditional":
            blocking = [c for c in rule.blocked_by_constraints if c in active]
            if blocking:
                refuse(index, op_name, feature, op_type, rule.status,
                       "blocked_by_constraint",
                       "operation %r on feature %r is blocked by active "
                       "constraint(s): %s"
                       % (op_type, entry.feature_id, ", ".join(blocking)))
                continue
            unmet = None
            for pre in rule.preconditions:
                met, detail = evaluate_precondition(pre, ctx)
                if not met:
                    unmet = (pre, detail)
                    break
            if unmet is not None:
                refuse(index, op_name, feature, op_type, rule.status,
                       "precondition_unmet",
                       "precondition %r unmet for operation %r on feature "
                       "%r: %s" % (unmet[0], op_type, entry.feature_id,
                                   unmet[1]))
                continue

        # status == "allowed", or a conditional that passed every check.
        decisions.append(GateDecision(
            op_index=index, op_name=op_name, feature=feature,
            operation_type=op_type, allowed=True, rule_status=rule.status))

    avoided = tuple(f for f in checked if f not in violated)
    return GateReport(
        decisions=tuple(decisions),
        protected_targets_checked=tuple(checked),
        protected_targets_avoided=avoided,
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

    # Case 8: parameter_edit preflight -- bounds and topology-changing.
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
