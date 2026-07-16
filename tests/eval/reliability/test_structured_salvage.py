"""Tests for the structured-output truncation salvage layer.

Three layers, matching structured_salvage.py:
  * structured_max_tokens / escalate_budget -- the token-budget floor per schema
    size, and the bounded escalation arithmetic;
  * validate_structured_json -- the schema-validation GATE around salvage, whose
    invariant ("salvage cannot invent content") is the reason the module exists;
  * generate_structured -- exactly one bounded retry, and no retry at all for
    transport errors.

All offline: the "provider" is a local callable, so every test is deterministic.
"""

import unittest

from harnesscad.eval.reliability.structured_salvage import (
    DEFAULT_STRUCTURED_MAX_TOKENS,
    LARGE_SCHEMA_CHAR_THRESHOLD,
    STRUCTURED_MAX_TOKENS_CEILING,
    STRUCTURED_MAX_TOKENS_FLOOR,
    SalvageOutcome,
    escalate_budget,
    generate_structured,
    response_format,
    schema_errors,
    schema_name,
    schema_size,
    structured_max_tokens,
    validate_structured_json,
)

SMALL_SCHEMA = {
    "title": "Part",
    "type": "object",
    "required": ["id"],
    "properties": {"id": {"type": "string"}, "qty": {"type": "integer"}},
}

LARGE_SCHEMA = {
    "title": "Notes",
    "type": "object",
    "required": ["items"],
    "properties": {
        "items": {
            "type": "array",
            "items": {
                "type": "object",
                "required": ["k", "v"],
                "properties": {
                    "k": {"type": "string"},
                    "v": {"type": "integer"},
                    # padding so the schema really is "large" (Forma's
                    # MechanicalNotes is 6656 chars; the threshold is 2000)
                    "note": {"type": "string", "description": "x" * 2200},
                },
            },
        },
    },
}


class TestSchemaSizing(unittest.TestCase):
    def test_the_two_fixtures_straddle_the_threshold(self):
        self.assertGreaterEqual(schema_size(LARGE_SCHEMA), LARGE_SCHEMA_CHAR_THRESHOLD)
        self.assertLess(schema_size(SMALL_SCHEMA), LARGE_SCHEMA_CHAR_THRESHOLD)

    def test_schema_size_is_deterministic(self):
        self.assertEqual(schema_size(LARGE_SCHEMA), schema_size(dict(LARGE_SCHEMA)))

    def test_schema_size_survives_unserialisable_schemas(self):
        self.assertEqual(schema_size({"bad": object()}), 0)

    def test_schema_name_sanitises_and_defaults(self):
        self.assertEqual(schema_name({"title": "Mechanical Notes/v2"}),
                         "Mechanical_Notes_v2")
        self.assertEqual(schema_name({}), "StructuredResponse")


class TestTokenBudgetFloor(unittest.TestCase):
    """Layer A: a budget is always computed; large schemas get a floor."""

    def test_unset_budget_gets_the_default(self):
        self.assertEqual(structured_max_tokens(SMALL_SCHEMA, None),
                         DEFAULT_STRUCTURED_MAX_TOKENS)
        self.assertEqual(structured_max_tokens(SMALL_SCHEMA, 0),
                         DEFAULT_STRUCTURED_MAX_TOKENS)

    def test_small_schema_keeps_a_small_budget(self):
        # The floor must NOT fire for small schemas, or every trivial call gets
        # an 6000-token budget it cannot use.
        self.assertEqual(structured_max_tokens(SMALL_SCHEMA, 256), 256)

    def test_large_schema_under_the_floor_is_raised(self):
        self.assertEqual(structured_max_tokens(LARGE_SCHEMA, 256),
                         STRUCTURED_MAX_TOKENS_FLOOR)

    def test_large_schema_above_the_floor_is_untouched(self):
        self.assertEqual(structured_max_tokens(LARGE_SCHEMA, 12000), 12000)

    def test_budget_is_deterministic(self):
        self.assertEqual(structured_max_tokens(LARGE_SCHEMA, 256),
                         structured_max_tokens(LARGE_SCHEMA, 256))


class TestEscalationIsBounded(unittest.TestCase):
    def test_escalation_doubles(self):
        self.assertEqual(escalate_budget(4000), 8000)

    def test_escalation_respects_the_floor(self):
        self.assertEqual(escalate_budget(10), STRUCTURED_MAX_TOKENS_FLOOR)

    def test_escalation_is_capped(self):
        self.assertEqual(escalate_budget(999999), STRUCTURED_MAX_TOKENS_CEILING)
        # ...and is idempotent at the ceiling, so it can never run away.
        self.assertEqual(escalate_budget(STRUCTURED_MAX_TOKENS_CEILING),
                         STRUCTURED_MAX_TOKENS_CEILING)


class TestSalvageGate(unittest.TestCase):
    """Layer B: the gate. Salvage cannot invent content."""

    def test_clean_input_passes_unsalvaged(self):
        out = validate_structured_json('{"id": "p1"}', SMALL_SCHEMA)
        self.assertIsInstance(out, SalvageOutcome)
        self.assertTrue(out.ok)
        self.assertFalse(out.salvaged)
        self.assertEqual(out.value, {"id": "p1"})
        self.assertEqual(out.reason, "clean")

    def test_recoverable_truncation_is_salvaged_and_validates(self):
        # Cut off partway through the third item: the first two were really
        # written, so recovering them invents nothing.
        text = '{"items": [{"k": "a", "v": 1}, {"k": "b", "v": 2}, {"k": "c'
        out = validate_structured_json(text, LARGE_SCHEMA)
        self.assertTrue(out.ok)
        self.assertTrue(out.salvaged)
        self.assertEqual(out.value,
                         {"items": [{"k": "a", "v": 1}, {"k": "b", "v": 2}]})
        self.assertEqual(out.reason, "salvaged")

    def test_salvage_cannot_invent_a_required_field(self):
        # THE INVARIANT. The model was cut off just as it started 'id'. Salvage
        # recovers what was written and must not manufacture the missing field,
        # so a structurally perfect salvage still fails the gate.
        out = validate_structured_json('{"qty": 3, "id', SMALL_SCHEMA)
        self.assertFalse(out.ok)
        self.assertTrue(out.salvaged)
        self.assertEqual(out.reason, "salvaged-but-invalid")
        self.assertTrue(any("'id'" in e for e in out.errors))

    def test_an_invalid_record_is_never_handed_back(self):
        # A caller must not be able to reach into a rejected outcome and use the
        # half-written record anyway.
        out = validate_structured_json('{"qty": 3, "id', SMALL_SCHEMA)
        self.assertIsNone(out.value)

    def test_gate_rejects_a_well_formed_but_wrong_record(self):
        out = validate_structured_json('{"id": 17}', SMALL_SCHEMA)
        self.assertFalse(out.ok)
        self.assertEqual(out.reason, "invalid")
        self.assertFalse(out.salvaged)

    def test_undecodable_input_fails_closed_without_raising(self):
        out = validate_structured_json("no json here", SMALL_SCHEMA)
        self.assertFalse(out.ok)
        self.assertIsNone(out.value)
        self.assertEqual(out.reason, "undecodable")

    def test_non_string_input_fails_closed(self):
        out = validate_structured_json(None, SMALL_SCHEMA)  # type: ignore[arg-type]
        self.assertFalse(out.ok)
        self.assertEqual(out.reason, "undecodable")

    def test_without_a_schema_the_absence_of_a_gate_is_reported(self):
        out = validate_structured_json('{"anything": 1}')
        self.assertTrue(out.ok)
        self.assertEqual(out.reason, "ungated")

    def test_custom_validator_overrides_the_schema(self):
        out = validate_structured_json('{"id": "p1"}', SMALL_SCHEMA,
                                       validator=lambda obj: ["rejected"])
        self.assertFalse(out.ok)
        self.assertEqual(out.errors, ["rejected"])

    def test_a_raising_validator_means_invalid_not_a_crash(self):
        def boom(obj):
            raise ValueError("kaboom")

        out = validate_structured_json('{"id": "p1"}', SMALL_SCHEMA, validator=boom)
        self.assertFalse(out.ok)
        self.assertTrue(any("kaboom" in e for e in out.errors))

    def test_markdown_fenced_output_is_recovered(self):
        out = validate_structured_json('```json\n{"id": "p1"}\n```', SMALL_SCHEMA)
        self.assertTrue(out.ok)
        self.assertEqual(out.value, {"id": "p1"})

    def test_outcome_to_dict_is_complete(self):
        d = validate_structured_json('{"id": "p1"}', SMALL_SCHEMA).to_dict()
        self.assertEqual(set(d), {"ok", "value", "salvaged", "notes", "errors",
                                  "attempts", "budget", "reason"})


class TestSchemaErrors(unittest.TestCase):
    """The default stdlib validator: strict enough to be a real gate."""

    def test_valid_object_has_no_errors(self):
        self.assertEqual(schema_errors({"id": "x", "qty": 2}, SMALL_SCHEMA), [])

    def test_missing_required_is_an_error(self):
        self.assertTrue(schema_errors({"qty": 2}, SMALL_SCHEMA))

    def test_booleans_are_not_integers(self):
        self.assertTrue(schema_errors({"id": "x", "qty": True}, SMALL_SCHEMA))

    def test_nested_items_are_checked(self):
        errs = schema_errors({"items": [{"k": "a", "v": 1}, {"k": "b"}]},
                             LARGE_SCHEMA)
        self.assertTrue(any("items[1]" in e for e in errs))

    def test_additional_properties_false_is_enforced(self):
        schema = {"type": "object", "properties": {"a": {"type": "string"}},
                  "additionalProperties": False}
        self.assertTrue(schema_errors({"a": "x", "b": 1}, schema))
        self.assertEqual(schema_errors({"a": "x"}, schema), [])

    def test_enum_and_bounds(self):
        schema = {"type": "object", "properties": {
            "mode": {"enum": ["cut", "add"]},
            "n": {"type": "integer", "minimum": 1, "maximum": 5}}}
        self.assertTrue(schema_errors({"mode": "zap"}, schema))
        self.assertTrue(schema_errors({"n": 9}, schema))
        self.assertEqual(schema_errors({"mode": "cut", "n": 3}, schema), [])


class TestOneBoundedRetry(unittest.TestCase):
    """Layer C: exactly two attempts, never three."""

    def setUp(self):
        self.budgets = []

    def test_failure_retries_exactly_once(self):
        def always_bad(max_tokens):
            self.budgets.append(max_tokens)
            return '{"qty": 3, "id'

        out = generate_structured(always_bad, SMALL_SCHEMA,
                                  configured_max_tokens=256)
        self.assertFalse(out.ok)
        self.assertEqual(len(self.budgets), 2)
        self.assertEqual(out.attempts, 2)
        self.assertTrue(any("exhausted" in n for n in out.notes))

    def test_the_retry_escalates_the_budget(self):
        def always_bad(max_tokens):
            self.budgets.append(max_tokens)
            return "garbage"

        generate_structured(always_bad, SMALL_SCHEMA, configured_max_tokens=256)
        self.assertEqual(self.budgets[0], 256)
        self.assertEqual(self.budgets[1], escalate_budget(256))
        self.assertGreater(self.budgets[1], self.budgets[0])

    def test_a_successful_retry_returns_the_record(self):
        def truncated_then_complete(max_tokens):
            self.budgets.append(max_tokens)
            return '{"id": "p1"}' if len(self.budgets) > 1 else '{"id'

        out = generate_structured(truncated_then_complete, SMALL_SCHEMA,
                                  configured_max_tokens=256)
        self.assertTrue(out.ok)
        self.assertEqual(out.value, {"id": "p1"})
        self.assertEqual(out.attempts, 2)
        self.assertEqual(out.budget, escalate_budget(256))

    def test_a_first_attempt_success_never_retries(self):
        def clean(max_tokens):
            self.budgets.append(max_tokens)
            return '{"id": "p1"}'

        out = generate_structured(clean, SMALL_SCHEMA, configured_max_tokens=256)
        self.assertTrue(out.ok)
        self.assertEqual(out.attempts, 1)
        self.assertEqual(len(self.budgets), 1)

    def test_first_attempt_uses_the_schema_sized_budget(self):
        def clean(max_tokens):
            self.budgets.append(max_tokens)
            return '{"items": []}'

        generate_structured(clean, LARGE_SCHEMA, configured_max_tokens=256)
        # The large schema's first attempt starts at the floor, not at 256.
        self.assertEqual(self.budgets[0], STRUCTURED_MAX_TOKENS_FLOOR)

    def test_finish_reason_is_threaded_through(self):
        def with_reason(max_tokens):
            return '{"id": "p1"}', "length"

        out = generate_structured(with_reason, SMALL_SCHEMA)
        self.assertTrue(out.ok)
        self.assertTrue(any("finish_reason=length" in n for n in out.notes))

    def test_transport_errors_propagate_and_are_not_retried(self):
        # Retrying a transport error here would silently double every request;
        # retry/backoff for those belongs to the caller.
        def boom(max_tokens):
            self.budgets.append(max_tokens)
            raise ConnectionError("down")

        with self.assertRaises(ConnectionError):
            generate_structured(boom, SMALL_SCHEMA)
        self.assertEqual(len(self.budgets), 1)


class TestResponseFormat(unittest.TestCase):
    def test_json_schema_envelope(self):
        fmt = response_format(SMALL_SCHEMA)
        self.assertEqual(fmt["type"], "json_schema")
        self.assertEqual(fmt["json_schema"]["name"], "Part")
        self.assertIs(fmt["json_schema"]["strict"], False)
        self.assertEqual(fmt["json_schema"]["schema"], SMALL_SCHEMA)

    def test_strict_is_opt_in(self):
        self.assertIs(response_format(SMALL_SCHEMA, strict=True)["json_schema"]["strict"],
                      True)


if __name__ == "__main__":
    unittest.main()
