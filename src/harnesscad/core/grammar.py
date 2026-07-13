"""Grammar-constrained op-decoding layer (HARNESS_BLUEPRINT.md sec.9, sec.4/11).

Blueprint sec.9 asks for **grammar-constrained decoding** (XGrammar / Outlines /
llama.cpp GBNF) so op emission is *syntactically guaranteed* — a context-free
grammar for the CAD command language — and sec.4/sec.11 pin **typed op emission
at temp 0**. This module derives both artefacts *from the op registry itself*
(``cisp.ops._REGISTRY``) so they can never drift from the real op set:

- :func:`op_json_schema` — a single JSON Schema (a discriminated union over op
  tags, each with its typed, enum-constrained fields) that a structured-output
  API or a constrained decoder can enforce at decode time.
- :func:`op_grammar` — a GBNF/EBNF context-free grammar (JSON specialised to the
  op set) suitable to hand to XGrammar / Outlines / llama.cpp ``--grammar``.
- :class:`GrammarConstraint` — a stdlib-only validator that, *lacking* a real
  constrained decoder, enforces the same guarantees **post-hoc**: it validates a
  candidate op-JSON string against the derived schema and returns typed errors,
  composing with :func:`llm.structured` / :func:`cisp.ops.parse_op`.

**Upgrade path.** No XGrammar/Outlines dependency is taken here on purpose — the
two artefacts are the drop-in inputs a real backend consumes. With XGrammar the
grammar is *enforced at decode* (every sampled token is masked to the grammar, so
malformed ops are impossible); :class:`GrammarConstraint` then degrades from a
safety net to a redundant assertion. To adopt it: feed :func:`op_grammar` to
``xgrammar.Grammar.from_ebnf`` (or :func:`op_json_schema` to
``xgrammar.Grammar.from_json_schema`` / Outlines / an OpenAI ``response_format``)
and wire the resulting logit processor into the sampler at temp 0.

Absolute imports; stdlib only.
"""

from __future__ import annotations

import dataclasses
import json
import typing
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Union

from harnesscad.core.cisp.ops import CONSTRAINT_DOF, Op, _REGISTRY, parse_op

# --- enum tables -----------------------------------------------------------
# Enum-valued fields, keyed by (op_tag, field_name). Constraint kinds come
# straight from cisp.ops.CONSTRAINT_DOF so they stay in sync; the plane and
# boolean-kind vocabularies have no registry table, so they live here (the one
# spot the grammar layer owns). See blueprint sec.9: "enum-heavy, flat schemas".
_PLANES = ("XY", "YZ", "XZ")
_BOOLEAN_KINDS = ("union", "cut", "intersect")


def _enum_values(tag: str, field_name: str) -> Optional[List[str]]:
    """Return the allowed enum literals for a field, or None if it is free-form."""
    if (tag, field_name) == ("new_sketch", "plane"):
        return list(_PLANES)
    if (tag, field_name) == ("boolean", "kind"):
        return list(_BOOLEAN_KINDS)
    if (tag, field_name) == ("constrain", "kind"):
        return sorted(CONSTRAINT_DOF)
    return None


# --- field introspection ---------------------------------------------------
@dataclass(frozen=True)
class FieldSpec:
    """A single typed field of an op, resolved from its dataclass annotation."""

    name: str
    base: str  # one of: string | number | integer | array | boolean
    optional: bool  # Optional[...] annotation -> may be null / omitted
    enum: Optional[List[str]] = None
    item_base: str = "string"  # element type when base == "array"

    @property
    def json_type(self) -> Any:
        """JSON Schema ``type`` for this field (a list when nullable)."""
        if self.optional:
            return [self.base, "null"]
        return self.base


# Map a (possibly stringised, thanks to ``from __future__ import annotations``)
# annotation to a JSON-flavoured base type.
_PY_TO_BASE = {
    "str": "string",
    "float": "number",
    "int": "integer",
    "bool": "boolean",
    "tuple": "array",
}


def _resolve_annotation(annotation: Any) -> tuple[str, bool]:
    """Return ``(base, optional)`` for a dataclass field annotation.

    Handles both real typing objects (e.g. ``Optional[float]``, whether the
    module used ``from __future__ import annotations`` or not) and stringised
    annotations (e.g. ``"Optional[float]"`` / ``"typing.Optional[float]"``).
    """
    # Real typing object (e.g. Optional[str] == Union[str, None]).
    origin = typing.get_origin(annotation)
    if origin is Union:
        args = [a for a in typing.get_args(annotation) if a is not type(None)]
        optional = len(args) < len(typing.get_args(annotation))
        inner = args[0] if args else str
        base = _PY_TO_BASE.get(getattr(inner, "__name__", str(inner)), "string")
        return base, optional
    if isinstance(annotation, type):
        return _PY_TO_BASE.get(annotation.__name__, "string"), False

    # Stringised annotation (from __future__ import annotations).
    text = str(annotation).strip()
    text = text.replace("typing.", "")
    optional = False
    if text.startswith("Optional[") and text.endswith("]"):
        optional = True
        text = text[len("Optional["):-1].strip()
    elif text.startswith("Union[") and text.endswith("]") and "None" in text:
        optional = True
        inner = [p.strip() for p in text[len("Union["):-1].split(",")]
        inner = [p for p in inner if p not in ("None", "NoneType")]
        text = inner[0] if inner else "str"
    base = _PY_TO_BASE.get(text, "string")
    return base, optional


def op_field_specs(cls: type) -> List[FieldSpec]:
    """Resolve the typed fields (excluding the ``OP`` tag) of an op dataclass."""
    specs: List[FieldSpec] = []
    for f in dataclasses.fields(cls):
        if f.name == "OP":
            continue
        base, optional = _resolve_annotation(f.type)
        enum = _enum_values(cls.OP, f.name)
        specs.append(FieldSpec(name=f.name, base=base, optional=optional, enum=enum))
    return specs


def _registry(allowed: Optional[Sequence[str]] = None) -> Dict[str, type]:
    """The op registry, optionally narrowed to ``allowed`` tags (state hook)."""
    if allowed is None:
        return dict(_REGISTRY)
    allow = set(allowed)
    return {tag: cls for tag, cls in _REGISTRY.items() if tag in allow}


# --- JSON Schema -----------------------------------------------------------
def _op_branch_schema(tag: str, cls: type) -> dict:
    """JSON Schema object for one op tag: ``op`` const + typed, enum'd fields."""
    properties: Dict[str, Any] = {
        "op": {"const": tag, "type": "string"},
    }
    required: List[str] = ["op"]
    for spec in op_field_specs(cls):
        prop: Dict[str, Any] = {"type": spec.json_type}
        if spec.enum is not None:
            prop["enum"] = list(spec.enum)
        if spec.base == "array":
            prop["items"] = {"type": spec.item_base}
        properties[spec.name] = prop
        if not spec.optional:
            required.append(spec.name)
    return {
        "type": "object",
        "properties": properties,
        "required": required,
        "additionalProperties": False,
    }


def op_json_schema(allowed: Optional[Sequence[str]] = None) -> dict:
    """A discriminated-union JSON Schema over every registered op.

    Derived live from ``cisp.ops._REGISTRY`` so a newly registered op is covered
    automatically. Each ``oneOf`` branch pins ``{"op": {"const": <tag>}}`` (the
    discriminator) plus that op's typed fields, with enum constraints on
    ``plane`` (XY/YZ/XZ), boolean ``kind`` (union/cut/intersect) and constraint
    ``kind`` (the CONSTRAINT_DOF keys). ``additionalProperties: false`` makes
    emission strict. This is the object a constrained decoder / structured-output
    API enforces to make a single op *syntactically guaranteed*.

    Pass ``allowed`` to restrict the union to a subset of op tags (see
    :func:`grammar_for_state`).
    """
    reg = _registry(allowed)
    branches = [_op_branch_schema(tag, cls) for tag, cls in reg.items()]
    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "title": "CISP op",
        "description": "A single grammar-constrained CISP CAD op (see grammar.py).",
        "type": "object",
        "required": ["op"],
        "discriminator": {"propertyName": "op"},
        "oneOf": branches,
    }


# --- GBNF grammar ----------------------------------------------------------
_GBNF_PRELUDE = """\
# CISP op command-language grammar (GBNF/EBNF), derived from cisp.ops._REGISTRY.
# Suitable for XGrammar (Grammar.from_ebnf), Outlines, or llama.cpp --grammar.
# A JSON grammar specialised to the op set: every alternative fixes its "op" tag
# and its typed, enum-constrained fields, so a decoder masked by this grammar can
# only emit a syntactically valid op.
ws     ::= [ \\t\\n\\r]*
sign   ::= "-"?
digits ::= [0-9]+
number ::= sign digits ("." digits)?
char   ::= [^"\\\\] | "\\\\" ["\\\\/bfnrt]
string ::= "\\"" char* "\\""
strarray ::= "[" ws ( string ( ws "," ws string )* )? ws "]"
"""


def _gbnf_rule_name(tag: str) -> str:
    """A safe GBNF rule identifier for an op tag (``add_point`` -> ``op_add_point``)."""
    return "op_" + "".join(ch if (ch.isalnum() or ch == "_") else "_" for ch in tag)


def _gbnf_field_value(spec: FieldSpec) -> str:
    """The right-hand side matching a field's value."""
    if spec.enum is not None:
        return "( " + " | ".join('"\\"%s\\""' % v for v in spec.enum) + " )"
    if spec.base == "array":
        return "strarray"
    if spec.base in ("number", "integer"):
        return "number"
    if spec.base == "boolean":
        return '( "true" | "false" )'
    return "string"


def _gbnf_field(spec: FieldSpec) -> str:
    """A ``"name" : value`` member fragment (the leading comma is added by caller)."""
    key = '"\\"%s\\""' % spec.name
    return '%s ws ":" ws %s' % (key, _gbnf_field_value(spec))


def _gbnf_op_rule(tag: str, cls: type) -> str:
    """One GBNF alternative: an object fixing "op":"<tag>" then its fields.

    Required fields are emitted in declaration order; Optional[...] fields become
    ``( "," ws member )?`` so they may be omitted. JSON member order is fixed by
    the grammar to one canonical ordering (fine for a constrained decoder).
    """
    parts: List[str] = ['"{" ws "\\"op\\"" ws ":" ws "\\"%s\\""' % tag]
    for spec in op_field_specs(cls):
        member = _gbnf_field(spec)
        if spec.optional:
            parts.append('( ws "," ws %s )?' % member)
        else:
            parts.append('ws "," ws %s' % member)
    parts.append('ws "}"')
    return "%s ::= %s" % (_gbnf_rule_name(tag), " ".join(parts))


def op_grammar(allowed: Optional[Sequence[str]] = None) -> str:
    """Emit a GBNF/EBNF context-free grammar for the op command language.

    Derived live from ``cisp.ops._REGISTRY`` so it stays in sync automatically:
    the ``root`` rule is the alternation of one rule per op tag, and each op rule
    encodes that op's typed fields (enum literals inlined). Hand this to a
    grammar-constrained decoder (XGrammar/Outlines/llama.cpp GBNF) to make op
    emission syntactically guaranteed at decode time.

    Pass ``allowed`` to restrict ``root`` to a subset of op tags (state hook).
    """
    reg = _registry(allowed)
    lines: List[str] = [_GBNF_PRELUDE.rstrip()]
    rule_names = [_gbnf_rule_name(tag) for tag in reg]
    lines.append("root ::= " + " | ".join(rule_names))
    for tag, cls in reg.items():
        lines.append(_gbnf_op_rule(tag, cls))
    return "\n".join(lines) + "\n"


# --- post-hoc validator (the safety net) -----------------------------------
class GrammarError(ValueError):
    """A typed grammar/schema violation.

    ``kind`` classifies the failure so callers can react programmatically:
    ``json`` (unparseable), ``structure`` (not an op object), ``unknown_tag``
    (op tag not in the registry), ``required`` (missing field), ``type``
    (wrong JSON type), ``enum`` (value outside the allowed set), ``additional``
    (unexpected field), ``not_allowed`` (op forbidden in the current state),
    ``parse`` (rejected by cisp.ops.parse_op).
    """

    def __init__(self, message: str, *, kind: str, path: str = "") -> None:
        super().__init__(message)
        self.kind = kind
        self.path = path

    def __repr__(self) -> str:  # pragma: no cover - debug aid
        loc = f" at {self.path}" if self.path else ""
        return f"GrammarError[{self.kind}]{loc}: {self.args[0]}"


def _type_ok(base: str, value: Any) -> bool:
    """Does ``value`` satisfy JSON Schema base type ``base``? (bool is not a number)."""
    if isinstance(value, bool):
        return base == "boolean"
    if base == "string":
        return isinstance(value, str)
    if base == "number":
        return isinstance(value, (int, float))
    if base == "integer":
        return isinstance(value, int)
    if base == "array":
        return isinstance(value, list)
    if base == "boolean":
        return isinstance(value, bool)
    return False


@dataclass
class GrammarConstraint:
    """A post-hoc enforcer of the op grammar/schema (the safety net).

    Lacking a real constrained decoder, this validates a candidate op-JSON
    (string or already-decoded dict) against the schema derived from
    ``cisp.ops._REGISTRY`` and returns *typed* :class:`GrammarError`s, then
    composes with :func:`cisp.ops.parse_op` for the final structural check —
    exactly the guarantees XGrammar would give, only enforced after generation
    instead of during it. **With XGrammar this becomes enforced-at-decode** and
    :meth:`validate` degrades to a redundant assertion.

    ``allowed`` optionally narrows the accepted op tags (see
    :func:`grammar_for_state`).
    """

    allowed: Optional[Sequence[str]] = None
    _specs: Dict[str, List[FieldSpec]] = field(default_factory=dict, init=False, repr=False)

    def __post_init__(self) -> None:
        self._reg = _registry(self.allowed)
        self._specs = {tag: op_field_specs(cls) for tag, cls in self._reg.items()}

    # -- artefacts (kept in sync with the same narrowing) --
    def schema(self) -> dict:
        return op_json_schema(self.allowed)

    def grammar(self) -> str:
        return op_grammar(self.allowed)

    # -- validation --
    def validate(self, candidate: Union[str, dict]) -> List[GrammarError]:
        """Return a list of typed errors (empty == the candidate is valid)."""
        errors: List[GrammarError] = []
        if isinstance(candidate, str):
            if not candidate.strip():
                return [GrammarError("empty candidate; expected an op-JSON object",
                                     kind="json")]
            try:
                obj = json.loads(candidate)
            except json.JSONDecodeError as e:
                return [GrammarError(f"not valid JSON: {e}", kind="json")]
        else:
            obj = candidate

        if not isinstance(obj, dict):
            return [GrammarError(
                f"expected a JSON object, got {type(obj).__name__}", kind="structure")]
        if "op" not in obj:
            return [GrammarError("missing required 'op' tag", kind="required",
                                 path="op")]
        tag = obj["op"]
        if not isinstance(tag, str) or tag not in _REGISTRY:
            valid = ", ".join(sorted(_REGISTRY))
            return [GrammarError(f"unknown op '{tag}'; valid ops: {valid}",
                                 kind="unknown_tag", path="op")]
        if tag not in self._reg:
            allowed = ", ".join(sorted(self._reg))
            return [GrammarError(
                f"op '{tag}' is not allowed in this state; allowed: {allowed}",
                kind="not_allowed", path="op")]

        specs = self._specs[tag]
        known = {"op"} | {s.name for s in specs}
        # Unexpected fields (additionalProperties: false).
        for key in obj:
            if key not in known:
                errors.append(GrammarError(
                    f"op '{tag}' has no field '{key}'", kind="additional", path=key))
        # Per-field type / enum / required checks.
        for spec in specs:
            if spec.name not in obj:
                if not spec.optional:
                    errors.append(GrammarError(
                        f"op '{tag}' is missing required field '{spec.name}'",
                        kind="required", path=spec.name))
                continue
            value = obj[spec.name]
            if value is None:
                if not spec.optional:
                    errors.append(GrammarError(
                        f"field '{spec.name}' must not be null", kind="type",
                        path=spec.name))
                continue
            if not _type_ok(spec.base, value):
                errors.append(GrammarError(
                    f"field '{spec.name}' must be {spec.base}, got "
                    f"{type(value).__name__}", kind="type", path=spec.name))
                continue
            if spec.base == "array":
                for i, el in enumerate(value):
                    if not _type_ok(spec.item_base, el):
                        errors.append(GrammarError(
                            f"field '{spec.name}[{i}]' must be {spec.item_base}",
                            kind="type", path=f"{spec.name}[{i}]"))
            if spec.enum is not None and value not in spec.enum:
                errors.append(GrammarError(
                    f"field '{spec.name}'='{value}' not in enum "
                    f"{{{', '.join(spec.enum)}}}", kind="enum", path=spec.name))

        if errors:
            return errors
        # Final compose with the real parser — the structural ground truth.
        try:
            parse_op(dict(obj))
        except Exception as e:  # TypeError/KeyError from parse_op
            errors.append(GrammarError(
                f"rejected by parse_op: {e}", kind="parse", path="op"))
        return errors

    def accepts(self, candidate: Union[str, dict]) -> bool:
        """True iff the candidate passes every check."""
        return not self.validate(candidate)

    def check(self, candidate: Union[str, dict]) -> Op:
        """Validate and return the parsed :class:`Op`, or raise the first error."""
        errors = self.validate(candidate)
        if errors:
            raise errors[0]
        return parse_op(dict(candidate) if isinstance(candidate, dict)
                        else json.loads(candidate))


# --- state hook ------------------------------------------------------------
def allowed_ops_for_state(has_sketch: bool = False, has_solid: bool = False) -> List[str]:
    """Which op tags are legal given coarse model state (documented hook).

    Encodes the blueprint's sequencing invariant (sec.9/sec.18): you cannot run a
    feature before a sketch exists, and a boolean needs two solids. This is a
    *coarse* gate meant to narrow the grammar for constrained decoding; the
    kernel's ``before_tool_gate`` (guardrails.py) remains the authoritative check.
    """
    allowed: List[str] = []
    for tag in _REGISTRY:
        if tag in ("new_sketch",):
            allowed.append(tag)  # always legal to start a sketch
        elif tag in ("add_point", "add_line", "add_circle", "add_rectangle",
                     "constrain"):
            if has_sketch:
                allowed.append(tag)  # need a sketch to add geometry / constrain
        elif tag in ("extrude", "fillet"):
            if has_sketch:
                allowed.append(tag)  # cannot extrude before a sketch exists
        elif tag == "boolean":
            if has_solid:
                allowed.append(tag)  # boolean needs existing solids
        else:  # unknown future op: allow by default (registry-derived, no drift)
            allowed.append(tag)
    return allowed


def grammar_for_state(has_sketch: bool = False, has_solid: bool = False,
                      as_schema: bool = False) -> Union[str, dict]:
    """State-narrowed grammar (or schema) — e.g. no extrude before a sketch.

    A documented hook for progressively constraining decoding as the model is
    built up. Returns a GBNF string by default, or the JSON Schema when
    ``as_schema=True``. See :func:`allowed_ops_for_state`.
    """
    allowed = allowed_ops_for_state(has_sketch=has_sketch, has_solid=has_solid)
    return op_json_schema(allowed) if as_schema else op_grammar(allowed)


def constraint_for_state(has_sketch: bool = False,
                         has_solid: bool = False) -> GrammarConstraint:
    """A :class:`GrammarConstraint` narrowed to the ops legal in the given state."""
    return GrammarConstraint(
        allowed=allowed_ops_for_state(has_sketch=has_sketch, has_solid=has_solid))
