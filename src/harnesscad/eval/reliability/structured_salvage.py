"""Truncation salvage for structured model output -- budget, gate, one retry.

The failure this module exists for: a model asked for a large JSON record hits
its token budget mid-string, the completion is unparseable, and the whole
generation is thrown away. That is not a model quality problem, it is a budget
problem, and it is fixable without another sampling round.

ATTRIBUTION
-----------
Reimplemented (no copied text) from the facts documented in
``resources/cad_repos/Forma-OSS-main/Forma-OSS-main/blueprint_core/llm_providers.py``
-- specifically ``_structured_max_tokens``, ``_validate_structured_json``,
``_salvage_json_text``, ``_prune_truncated_tail`` and the two-attempt loop in
``generate_structured``. Forma-OSS is Mozilla Public License 2.0, a file-level
copyleft this repository's vendoring policy does not admit, so nothing is
copied: what is reused are the facts (which numbers, in which order, gated by
what) expressed in original code. The numeric constants below are Forma's,
measured against its own fine-tuned model; they are documented as such and are
overridable.

THE FOUR LAYERS, and who owns each here
---------------------------------------
A. **Token-budget floor per schema size** -- :func:`structured_max_tokens`. A
   budget is ALWAYS computed for a structured call; a large schema whose
   configured budget sits under the floor is raised to the floor. NEW here.

B. **Close-and-prune salvage GATED BY FULL SCHEMA VALIDATION** --
   :func:`validate_structured_json`. The close-and-prune mechanics already exist
   in :mod:`harnesscad.agents.llm.json_salvage` (itself ported from the same
   Forma functions) and are imported, not reimplemented. What is NEW here is the
   GATE, and the gate is the whole point: **salvage cannot invent content.**
   Closing brackets and dropping a half-written trailing item can only ever
   recover a record the model actually finished writing and got cut off; it can
   never fabricate a field. Full schema validation of the salvaged object is
   what makes that invariant enforceable rather than aspirational -- a salvage
   that produced a semantically wrong record still FAILS, exactly as an
   unsalvaged one would. Never relax this into a partial or best-effort check.

C. **Exactly ONE bounded budget-escalation retry** -- :func:`generate_structured`.
   Two attempts, never three: the first at the computed budget, and on a
   VALIDATION failure one retry at a doubled budget clamped to the ceiling.
   Transport errors are not retried here (the caller's retry/backoff owns those)
   and a second validation failure is reported, not re-escalated. NEW here.

D. **json_schema response_format** -- ALREADY COVERED, not rebuilt.
   :meth:`harnesscad.agents.llm.litellm_backend.LiteLLMBackend._build_kwargs`
   already wraps a bare JSON Schema into an OpenAI ``json_schema``
   response_format (and passes a ready one through). :func:`response_format` is
   a thin deterministic helper for callers that are not on that backend; it adds
   only Forma's two details -- a sanitised schema name and ``strict: False``.

Deterministic and stdlib-only: no wall clock, no randomness, no provider SDK, no
jsonschema/pydantic dependency. The module never calls an LLM itself -- it takes
a callable and drives the budget/gate/retry policy around it -- so it is fully
testable offline.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, field
from typing import Any, Callable, List, Mapping, Optional, Sequence, Tuple, Union

from harnesscad.agents.llm.json_salvage import loads_with_salvage

__all__ = [
    "DEFAULT_STRUCTURED_MAX_TOKENS",
    "LARGE_SCHEMA_CHAR_THRESHOLD",
    "STRUCTURED_MAX_TOKENS_CEILING",
    "STRUCTURED_MAX_TOKENS_FLOOR",
    "SalvageOutcome",
    "escalate_budget",
    "generate_structured",
    "response_format",
    "schema_errors",
    "schema_name",
    "schema_size",
    "structured_max_tokens",
    "validate_structured_json",
]


# --------------------------------------------------------------------------- #
# Layer A: token budgets (constants are Forma-OSS's, measured on its own model)
# --------------------------------------------------------------------------- #
#: Budget sent when the caller configured none at all.
DEFAULT_STRUCTURED_MAX_TOKENS = 8192
#: Floor for a large schema: below this, big records truncate mid-record.
STRUCTURED_MAX_TOKENS_FLOOR = 6000
#: Hard cap on the single escalation retry.
STRUCTURED_MAX_TOKENS_CEILING = 16384
#: A schema whose JSON serialises to at least this many characters is "large".
LARGE_SCHEMA_CHAR_THRESHOLD = 2000


def schema_size(schema: Mapping[str, Any]) -> int:
    """Characters in the schema's canonical JSON form.

    Deterministic (``sort_keys``), so the same schema always sizes the same and
    a budget decision is reproducible.
    """
    try:
        return len(json.dumps(schema, sort_keys=True))
    except (TypeError, ValueError):
        return 0


def schema_name(schema: Mapping[str, Any], default: str = "StructuredResponse") -> str:
    """A provider-safe name for a schema: alphanumerics, ``_`` and ``-`` only.

    Mirrors Forma's ``_schema_name`` rule (other characters become ``_``),
    reading the name from the schema's ``title``.
    """
    raw = schema.get("title") if isinstance(schema, Mapping) else None
    raw = str(raw) if raw else default
    cleaned = "".join(c if (c.isalnum() or c in {"_", "-"}) else "_" for c in raw)
    return cleaned or default


def structured_max_tokens(schema: Mapping[str, Any],
                          configured: Optional[int] = None) -> int:
    """The budget to send for a structured call against ``schema``.

    ``configured`` is the caller's own max_tokens, or None when unset. Rules
    (Forma's, in order):

      * unset -> :data:`DEFAULT_STRUCTURED_MAX_TOKENS`; a budget is ALWAYS sent,
        because an unset budget is what truncates large records in the first
        place;
      * a LARGE schema (>= :data:`LARGE_SCHEMA_CHAR_THRESHOLD` chars) whose
        configured budget is under :data:`STRUCTURED_MAX_TOKENS_FLOOR` -> raised
        to the floor;
      * anything else -> the configured budget, untouched. A small schema keeps
        a deliberately small budget.
    """
    budget = configured if configured and configured > 0 else DEFAULT_STRUCTURED_MAX_TOKENS
    if schema_size(schema) >= LARGE_SCHEMA_CHAR_THRESHOLD and budget < STRUCTURED_MAX_TOKENS_FLOOR:
        return STRUCTURED_MAX_TOKENS_FLOOR
    return budget


def escalate_budget(budget: int) -> int:
    """The retry budget: double it, never below the floor, never above the ceiling.

    Bounded by construction -- :func:`generate_structured` calls this at most
    once, and the ceiling caps it even if a caller loops.
    """
    return min(max(int(budget) * 2, STRUCTURED_MAX_TOKENS_FLOOR),
               STRUCTURED_MAX_TOKENS_CEILING)


# --------------------------------------------------------------------------- #
# Layer D: response_format (thin helper; litellm_backend already covers this)
# --------------------------------------------------------------------------- #
def response_format(schema: Mapping[str, Any], strict: bool = False) -> dict:
    """An OpenAI ``json_schema`` response_format for ``schema``.

    Only for callers not going through
    :class:`harnesscad.agents.llm.litellm_backend.LiteLLMBackend`, which already
    builds this. ``strict=False`` is Forma's default: strict mode rejects
    schemas with optional fields on several providers.
    """
    return {
        "type": "json_schema",
        "json_schema": {
            "name": schema_name(schema),
            "schema": dict(schema),
            "strict": bool(strict),
        },
    }


# --------------------------------------------------------------------------- #
# The validation gate's default validator (stdlib subset of JSON Schema)
# --------------------------------------------------------------------------- #
_TYPE_MAP = {
    "object": dict,
    "array": list,
    "string": str,
    "boolean": bool,
    "integer": int,
    "number": (int, float),
    "null": type(None),
}


def schema_errors(value: Any, schema: Mapping[str, Any],
                  path: str = "$") -> List[str]:
    """Validate ``value`` against ``schema``; return dotted-path error strings.

    A deliberately small, dependency-free subset of JSON Schema -- type, enum,
    const, required, properties, additionalProperties: false, items, minItems,
    minLength, minimum, maximum -- which is what a structured-output contract
    needs. An empty list means valid.

    This is the FULL check the salvage gate runs; it must stay strict about
    ``required``, since a missing required field is precisely how a truncated
    record differs from a complete one. (:mod:`harnesscad.core.cisp.op_gate`
    carries a sibling shape-checker bound to its own embedded gating schemas;
    this one is standalone so the gate has no cross-layer dependency.)
    """
    errors: List[str] = []
    if not isinstance(schema, Mapping):
        return errors

    expected = schema.get("type")
    if expected is not None:
        py = _TYPE_MAP.get(str(expected))
        if py is not None:
            # bool is an int subclass; keep integer/number honest.
            if isinstance(value, bool) and expected in ("integer", "number"):
                return ["%s: expected %s, got boolean" % (path, expected)]
            if not isinstance(value, py):
                return ["%s: expected %s, got %s"
                        % (path, expected, type(value).__name__)]

    if "const" in schema and value != schema["const"]:
        errors.append("%s: must equal %r" % (path, schema["const"]))
    enum = schema.get("enum")
    if isinstance(enum, list) and value not in enum:
        errors.append("%s: %r is not one of the %d allowed values"
                      % (path, value, len(enum)))

    if isinstance(value, str):
        min_len = schema.get("minLength")
        if isinstance(min_len, int) and len(value) < min_len:
            errors.append("%s: string shorter than minLength %d" % (path, min_len))

    if isinstance(value, (int, float)) and not isinstance(value, bool):
        minimum = schema.get("minimum")
        if isinstance(minimum, (int, float)) and value < minimum:
            errors.append("%s: %r below minimum %r" % (path, value, minimum))
        maximum = schema.get("maximum")
        if isinstance(maximum, (int, float)) and value > maximum:
            errors.append("%s: %r above maximum %r" % (path, value, maximum))

    if isinstance(value, list):
        min_items = schema.get("minItems")
        if isinstance(min_items, int) and len(value) < min_items:
            errors.append("%s: array shorter than minItems %d" % (path, min_items))
        item_schema = schema.get("items")
        if isinstance(item_schema, Mapping):
            for index, item in enumerate(value):
                errors.extend(schema_errors(item, item_schema,
                                            "%s[%d]" % (path, index)))

    if isinstance(value, dict):
        required = schema.get("required")
        if isinstance(required, list):
            for key in required:
                if key not in value:
                    errors.append("%s: missing required field %r" % (path, key))
        props = schema.get("properties")
        props = props if isinstance(props, Mapping) else {}
        if schema.get("additionalProperties") is False:
            for key in sorted(value):
                if key not in props:
                    errors.append("%s: unexpected field %r" % (path, key))
        for key in sorted(props):
            if key in value:
                sub = props[key]
                if isinstance(sub, Mapping):
                    errors.extend(schema_errors(value[key], sub,
                                                "%s.%s" % (path, key)))
    return errors


# --------------------------------------------------------------------------- #
# Layer B: the gate
# --------------------------------------------------------------------------- #
@dataclass
class SalvageOutcome:
    """The result of a gated structured decode.

    ``ok`` is True only when an object both decoded AND passed full schema
    validation. ``value`` is that object (None when ``ok`` is False -- an
    invalid record is never handed back, so a caller cannot accidentally use a
    half-written one). ``salvaged`` says whether repair was needed to get there;
    ``notes`` traces what the salvage ladder did; ``errors`` carries the schema
    violations that closed the gate; ``attempts`` and ``budget`` record the
    retry policy's decisions; ``reason`` is a short machine-stable code.
    """

    ok: bool
    value: Any = None
    salvaged: bool = False
    notes: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
    attempts: int = 0
    budget: int = 0
    reason: str = ""

    def to_dict(self) -> dict:
        return {
            "ok": self.ok,
            "value": self.value,
            "salvaged": self.salvaged,
            "notes": list(self.notes),
            "errors": list(self.errors),
            "attempts": self.attempts,
            "budget": self.budget,
            "reason": self.reason,
        }


#: A validator takes the decoded object and returns schema violations. Returning
#: an empty list means valid. Supply one to plug in pydantic/jsonschema.
Validator = Callable[[Any], List[str]]


def validate_structured_json(text: str,
                             schema: Optional[Mapping[str, Any]] = None,
                             validator: Optional[Validator] = None) -> SalvageOutcome:
    """Decode ``text`` through the salvage ladder, GATED by full validation.

    The decode ladder (direct parse -> fence strip -> embedded document ->
    close-and-prune salvage) is
    :func:`harnesscad.agents.llm.json_salvage.loads_with_salvage`. This function
    adds the gate: whatever comes out is validated against ``schema`` (or by
    ``validator``, which wins when both are given), and a record that fails is
    REJECTED regardless of how much salvage went into producing it.

    That rejection is the invariant. Salvage only closes structure and drops a
    half-written tail -- it cannot invent a field -- so if the validated result
    is wrong, it was wrong in the model's output, and passing it downstream
    would be laundering a truncation into a false success. With neither
    ``schema`` nor ``validator`` there is no gate to run, and a decoded object is
    returned with ``reason='ungated'`` so the caller can see the gate was absent.

    Never raises.
    """
    obj, notes = loads_with_salvage(text if isinstance(text, str) else "")
    salvaged = bool(notes)

    if obj is None:
        return SalvageOutcome(ok=False, salvaged=salvaged, notes=notes,
                              reason="undecodable")

    if validator is None and schema is None:
        return SalvageOutcome(ok=True, value=obj, salvaged=salvaged, notes=notes,
                              reason="ungated")

    if validator is not None:
        try:
            errors = list(validator(obj) or [])
        except Exception as exc:  # noqa: BLE001 - a raising validator means invalid
            errors = ["validator raised: %s" % exc]
    else:
        errors = schema_errors(obj, schema or {})

    if errors:
        # The gate closed. Report, never return the object.
        return SalvageOutcome(ok=False, salvaged=salvaged, notes=notes,
                              errors=errors,
                              reason="salvaged-but-invalid" if salvaged else "invalid")

    return SalvageOutcome(ok=True, value=obj, salvaged=salvaged, notes=notes,
                          reason="salvaged" if salvaged else "clean")


# --------------------------------------------------------------------------- #
# Layer C: exactly one bounded budget-escalation retry
# --------------------------------------------------------------------------- #
#: A structured call takes a token budget and returns the completion text, or
#: ``(text, finish_reason)`` when the provider reports one.
StructuredCall = Callable[[int], Union[str, Tuple[str, Optional[str]]]]


def _normalise_call_result(result: Any) -> Tuple[str, Optional[str]]:
    if isinstance(result, tuple) and len(result) == 2:
        text, finish = result
        return (text if isinstance(text, str) else ""), finish
    return (result if isinstance(result, str) else ""), None


def generate_structured(call: StructuredCall,
                        schema: Optional[Mapping[str, Any]] = None,
                        validator: Optional[Validator] = None,
                        configured_max_tokens: Optional[int] = None) -> SalvageOutcome:
    """Drive a structured call through the budget / gate / one-retry policy.

    ``call(max_tokens)`` returns the completion text (or ``(text,
    finish_reason)``). Exactly TWO attempts, never three:

      1. attempt 1 at :func:`structured_max_tokens` for the schema;
      2. on a VALIDATION failure only, attempt 2 at :func:`escalate_budget`.

    A second failure is returned, not escalated again -- the retry is bounded by
    the code path, not just by the ceiling. Exceptions raised by ``call`` are
    transport failures and propagate immediately: retry/backoff for those belongs
    to the caller, and retrying them here would silently double every request.

    The returned outcome records ``attempts`` and the final ``budget``.
    """
    budget = structured_max_tokens(schema or {}, configured_max_tokens)
    outcome = SalvageOutcome(ok=False, reason="no-attempt")

    for attempt in (1, 2):
        text, finish_reason = _normalise_call_result(call(budget))
        outcome = validate_structured_json(text, schema, validator)
        outcome.attempts = attempt
        outcome.budget = budget
        if finish_reason:
            outcome.notes = outcome.notes + ["finish_reason=%s" % finish_reason]
        if outcome.ok:
            return outcome
        if attempt == 1:
            budget = escalate_budget(budget)

    outcome.reason = outcome.reason or "invalid"
    outcome.notes = outcome.notes + ["exhausted the single budget-escalation retry"]
    return outcome


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def main(argv: Optional[Sequence[str]] = None) -> int:
    """CLI entry point. ``--selfcheck`` proves the real invariants: the budget
    floor fires only for large schemas, the retry happens exactly once, and --
    the load-bearing one -- salvage cannot invent content."""
    parser = argparse.ArgumentParser(
        prog="python -m harnesscad.eval.reliability.structured_salvage",
        description="Structured-output truncation salvage: token-budget floor, "
                    "schema-gated salvage, one bounded escalation retry.",
    )
    parser.add_argument(
        "--selfcheck", action="store_true",
        help="run deterministic checks on synthetic truncated payloads; exit 0 on success.")
    args = parser.parse_args(list(argv) if argv is not None else None)

    if not args.selfcheck:
        parser.print_help()
        return 0

    failures: List[str] = []
    checks = 0

    def check(label: str, condition: bool) -> None:
        nonlocal checks
        checks += 1
        if not condition:
            failures.append(label)

    # A small schema and a genuinely large one (padded past the threshold).
    small = {"title": "Part", "type": "object",
             "required": ["id"], "properties": {"id": {"type": "string"}}}
    large = {"title": "Notes", "type": "object", "required": ["items"],
             "properties": {"items": {"type": "array", "items": {
                 "type": "object", "required": ["k", "v"],
                 "properties": {"k": {"type": "string"},
                                "v": {"type": "integer"},
                                "note": {"type": "string",
                                         "description": "x" * 2200}}}}}}
    check("large schema is over the threshold",
          schema_size(large) >= LARGE_SCHEMA_CHAR_THRESHOLD)
    check("small schema is under the threshold",
          schema_size(small) < LARGE_SCHEMA_CHAR_THRESHOLD)

    # --- Layer A ---
    check("unset budget gets the default",
          structured_max_tokens(small, None) == DEFAULT_STRUCTURED_MAX_TOKENS)
    check("small schema keeps a small configured budget",
          structured_max_tokens(small, 256) == 256)
    check("large schema is raised to the floor",
          structured_max_tokens(large, 256) == STRUCTURED_MAX_TOKENS_FLOOR)
    check("large schema keeps an already-generous budget",
          structured_max_tokens(large, 12000) == 12000)
    check("budget sizing is deterministic",
          structured_max_tokens(large, 256) == structured_max_tokens(large, 256))

    # --- escalation is bounded ---
    check("escalation doubles", escalate_budget(4000) == 8000)
    check("escalation respects the floor",
          escalate_budget(10) == STRUCTURED_MAX_TOKENS_FLOOR)
    check("escalation is capped at the ceiling",
          escalate_budget(999999) == STRUCTURED_MAX_TOKENS_CEILING)

    # --- Layer B: THE INVARIANT ------------------------------------------
    # A record truncated mid-way through a COMPLETE item: salvage closes it and
    # the result validates, because every field it keeps was really written.
    truncated_ok = '{"items": [{"k": "a", "v": 1}, {"k": "b", "v": 2}, {"k": "c'
    out = validate_structured_json(truncated_ok, large)
    check("recoverable truncation is salvaged", out.ok and out.salvaged)
    check("salvage kept only the complete items",
          out.ok and out.value == {"items": [{"k": "a", "v": 1}, {"k": "b", "v": 2}]})

    # THE INVARIANT. Here the model was cut off just as it began the required
    # 'id' field. Salvage recovers everything that WAS written ({"note": "x"})
    # and -- having no 'id' to recover -- must not manufacture one. So the
    # record is structurally fine, decodes cleanly, and the gate still closes.
    out = validate_structured_json('{"note": "x", "id', small)
    check("salvage cannot invent a required field", not out.ok)
    check("invalid salvage withholds the value", out.value is None)
    check("invalid salvage names the missing field",
          any("'id'" in e for e in out.errors))
    check("invalid salvage is labelled honestly",
          out.reason == "salvaged-but-invalid")

    # Salvage that cannot even produce a decodable object is reported as such,
    # never as a success.
    out = validate_structured_json('{"nam', small)
    check("undecodable salvage fails closed",
          not out.ok and out.value is None and out.reason == "undecodable")

    # A wrong-typed record fails the gate even though it parses perfectly.
    out = validate_structured_json('{"id": 17}', small)
    check("gate rejects a well-formed but wrong record", not out.ok)
    check("clean-but-invalid is labelled 'invalid'", out.reason == "invalid")

    # Clean input passes untouched and is not marked salvaged.
    out = validate_structured_json('{"id": "p1"}', small)
    check("clean input passes the gate",
          out.ok and not out.salvaged and out.value == {"id": "p1"})

    # Undecodable input never raises.
    out = validate_structured_json("not json at all", small)
    check("undecodable input is reported, not raised",
          not out.ok and out.reason == "undecodable")

    # A custom validator wins over the schema, and one that raises means invalid.
    out = validate_structured_json('{"id": "p1"}', small,
                                   validator=lambda obj: ["nope"])
    check("custom validator overrides the schema", not out.ok)
    out = validate_structured_json('{"id": "p1"}', small,
                                   validator=lambda obj: (_ for _ in ()).throw(ValueError("boom")))
    check("raising validator means invalid, not a crash",
          not out.ok and any("boom" in e for e in out.errors))

    # --- Layer C: exactly one retry --------------------------------------
    budgets: List[int] = []

    def always_truncated(max_tokens: int) -> str:
        budgets.append(max_tokens)
        return '{"nam'

    out = generate_structured(always_truncated, small, configured_max_tokens=256)
    check("failure retries exactly once", len(budgets) == 2)
    check("attempts is reported as 2", out.attempts == 2)
    check("the retry escalated the budget", budgets[1] > budgets[0])
    check("the retry budget is the escalation of the first",
          budgets[1] == escalate_budget(budgets[0]))
    check("exhausted retry is labelled", not out.ok
          and any("exhausted" in n for n in out.notes))

    budgets.clear()

    def truncated_then_complete(max_tokens: int) -> str:
        budgets.append(max_tokens)
        return '{"id": "p1"}' if len(budgets) > 1 else '{"id'

    out = generate_structured(truncated_then_complete, small,
                              configured_max_tokens=256)
    check("a successful retry returns the record",
          out.ok and out.value == {"id": "p1"})
    check("success stops after two attempts", out.attempts == 2 and len(budgets) == 2)

    budgets.clear()

    def clean_first(max_tokens: int) -> Tuple[str, Optional[str]]:
        budgets.append(max_tokens)
        return '{"id": "p1"}', "stop"

    out = generate_structured(clean_first, small, configured_max_tokens=256)
    check("a first-attempt success never retries",
          out.ok and out.attempts == 1 and len(budgets) == 1)
    check("finish_reason is threaded through",
          any("finish_reason=stop" in n for n in out.notes))

    # Transport errors are the caller's problem: they must propagate, not double
    # the request count.
    def boom(max_tokens: int) -> str:
        budgets.append(max_tokens)
        raise ConnectionError("transport down")

    budgets.clear()
    try:
        generate_structured(boom, small)
        check("transport error propagates", False)
    except ConnectionError:
        check("transport error propagates", True)
    check("transport error is not retried", len(budgets) == 1)

    # --- Layer D helper ---
    fmt = response_format(small)
    check("response_format is a json_schema envelope",
          fmt["type"] == "json_schema" and fmt["json_schema"]["strict"] is False
          and fmt["json_schema"]["name"] == "Part")
    check("schema name is sanitised",
          schema_name({"title": "Mechanical Notes/v2"}) == "Mechanical_Notes_v2")

    if failures:
        print("SELFCHECK FAILED: %s" % ", ".join(failures), file=sys.stderr)
        return 1
    print("PASS: structured_salvage selfcheck (%d checks; budget floor, "
          "salvage-cannot-invent-content gate, exactly-one bounded retry)" % checks)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
