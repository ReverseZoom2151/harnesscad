"""Tests for programs.cadhub_param_schema (unified code-CAD parameter schema)."""

import unittest

from harnesscad.domain.programs.params.cadhub_param_schema import (
    INPUT_BOOLEAN,
    INPUT_CHOICE_NUMBER,
    INPUT_CHOICE_STRING,
    INPUT_NUMBER,
    INPUT_STRING,
    TYPE_BOOLEAN,
    TYPE_NUMBER,
    TYPE_STRING,
    UnknownParamLanguage,
    defaults,
    from_cadquery,
    from_jscad,
    from_openscad,
    normalize,
    openscad_parameter_set,
    schema_digest,
    validate_values,
)


def by_name(params):
    return {p.name: p for p in params}


class TestOpenScad(unittest.TestCase):
    def test_scalar_types(self):
        params = from_openscad(
            [
                {"name": "w", "type": "number", "initial": 10, "min": 1, "max": 50, "step": 1},
                {"name": "on", "type": "boolean", "initial": True, "caption": "On?"},
                {"name": "label", "type": "string", "initial": "hi", "maxLength": 5},
            ]
        )
        p = by_name(params)
        self.assertEqual(p["w"].input, INPUT_NUMBER)
        self.assertEqual((p["w"].min, p["w"].max, p["w"].step), (1, 50, 1))
        self.assertEqual(p["on"].input, INPUT_BOOLEAN)
        self.assertEqual(p["on"].caption, "On?")
        self.assertEqual(p["label"].max_length, 5)

    def test_choices(self):
        params = from_openscad(
            [
                {
                    "name": "colour",
                    "type": "string",
                    "initial": "red",
                    "options": [{"name": "Red", "value": "red"}, {"name": "Blue", "value": "blue"}],
                },
                {
                    "name": "size",
                    "type": "number",
                    "initial": 2,
                    "options": [{"name": "S", "value": 1}, {"name": "L", "value": 2}],
                },
            ]
        )
        p = by_name(params)
        self.assertEqual(p["colour"].input, INPUT_CHOICE_STRING)
        self.assertEqual(p["colour"].option_values(), ["red", "blue"])
        self.assertEqual(p["size"].input, INPUT_CHOICE_NUMBER)
        self.assertEqual(p["size"].option_values(), [1, 2])

    def test_vector_param_skipped(self):
        params = from_openscad([{"name": "v", "type": "number", "initial": [1, 2, 3]}])
        self.assertEqual(params, [])


class TestJsCad(unittest.TestCase):
    def test_numeric_family_and_decimal(self):
        params = from_jscad(
            [
                {"name": "grp", "type": "group", "caption": "G"},
                {"name": "n", "type": "slider", "initial": 5, "min": 0, "max": 10, "step": 1},
                {"name": "f", "type": "float", "initial": 1.5, "step": 0.1},
            ]
        )
        p = by_name(params)
        self.assertNotIn("grp", p)  # groups are layout-only
        self.assertEqual(p["n"].decimal, 0)
        self.assertEqual(p["f"].decimal, 2)
        self.assertEqual(p["n"].type, TYPE_NUMBER)

    def test_textual_and_checkbox(self):
        params = from_jscad(
            [
                {"name": "t", "type": "text", "initial": "x", "placeholder": "type", "maxLength": 3},
                {"name": "c", "type": "color", "initial": "#fff", "placeholder": "nope"},
                {"name": "b", "type": "checkbox", "checked": True},
            ]
        )
        p = by_name(params)
        self.assertEqual(p["t"].placeholder, "type")
        self.assertEqual(p["t"].max_length, 3)
        self.assertEqual(p["c"].placeholder, "")  # placeholder only for text/date/url
        self.assertIsNone(p["c"].max_length)
        self.assertEqual(p["b"].type, TYPE_BOOLEAN)
        self.assertTrue(p["b"].initial)

    def test_choice_numeric_vs_string(self):
        params = from_jscad(
            [
                {"name": "n", "type": "choice", "initial": 2, "values": [1, 2], "captions": ["one", "two"]},
                {"name": "s", "type": "radio", "initial": "a", "values": ["a", "b"]},
            ]
        )
        p = by_name(params)
        self.assertEqual(p["n"].input, INPUT_CHOICE_NUMBER)
        self.assertEqual([o.name for o in p["n"].options], ["one", "two"])
        self.assertEqual(p["s"].input, INPUT_CHOICE_STRING)
        self.assertEqual([o.name for o in p["s"].options], ["a", "b"])  # captions default to values


class TestCadQuery(unittest.TestCase):
    def test_declared_and_inferred_types(self):
        params = from_cadquery(
            [
                {"name": "w", "type": "number", "initial": 3},
                {"name": "s", "initial": "abc"},
                {"name": "b", "initial": True},
                {"name": "z", "type": "number", "initial": None},
            ]
        )
        p = by_name(params)
        self.assertEqual(p["w"].initial, 3)
        self.assertEqual(p["s"].type, TYPE_STRING)
        self.assertEqual(p["b"].type, TYPE_BOOLEAN)
        self.assertEqual(p["z"].initial, 0)
        self.assertEqual(p["s"].input, INPUT_STRING)


class TestNormalize(unittest.TestCase):
    def test_dispatch(self):
        self.assertEqual(len(normalize("openscad", [{"name": "a", "type": "number", "initial": 1}])), 1)
        with self.assertRaises(UnknownParamLanguage):
            normalize("curv", [])


class TestValidateValues(unittest.TestCase):
    def setUp(self):
        self.params = from_openscad(
            [
                {"name": "w", "type": "number", "initial": 10, "min": 1, "max": 20},
                {"name": "label", "type": "string", "initial": "ab", "maxLength": 3},
                {"name": "on", "type": "boolean", "initial": False},
                {
                    "name": "colour",
                    "type": "string",
                    "initial": "red",
                    "options": [{"name": "Red", "value": "red"}, {"name": "Blue", "value": "blue"}],
                },
            ]
        )

    def test_defaults_used_for_missing(self):
        clean, issues = validate_values(self.params, {})
        self.assertEqual(clean, defaults(self.params))
        self.assertEqual(issues, [])

    def test_clamping(self):
        clean, issues = validate_values(self.params, {"w": 99})
        self.assertEqual(clean["w"], 20)
        self.assertEqual(issues[0].kind, "range")
        clean, _ = validate_values(self.params, {"w": -5})
        self.assertEqual(clean["w"], 1)

    def test_truncation_and_option_reset(self):
        clean, issues = validate_values(self.params, {"label": "toolong", "colour": "green"})
        self.assertEqual(clean["label"], "too")
        self.assertEqual(clean["colour"], "red")
        kinds = sorted(i.kind for i in issues)
        self.assertEqual(kinds, ["length", "option"])

    def test_unknown_key_reported_and_dropped(self):
        clean, issues = validate_values(self.params, {"nope": 1})
        self.assertNotIn("nope", clean)
        self.assertEqual(issues[0].kind, "unknown")

    def test_type_coercion(self):
        clean, issues = validate_values(self.params, {"w": "12", "on": 1})
        self.assertEqual(clean["w"], 12)
        self.assertTrue(clean["on"])
        self.assertTrue(any(i.kind == "type" for i in issues))

    def test_bad_number(self):
        clean, issues = validate_values(self.params, {"w": "abc"})
        self.assertEqual(clean["w"], 10)
        self.assertEqual(issues[0].kind, "type")


class TestOpenScadParameterSet(unittest.TestCase):
    def test_payload_shape(self):
        payload = openscad_parameter_set({"w": 10.0, "on": True, "label": "hi"})
        self.assertEqual(payload["fileFormatVersion"], "1")
        self.assertEqual(
            payload["parameterSets"]["default"],
            {"label": "hi", "on": "true", "w": "10"},
        )

    def test_named_set(self):
        payload = openscad_parameter_set({"a": 1}, set_name="small")
        self.assertIn("small", payload["parameterSets"])


class TestDigest(unittest.TestCase):
    def test_stable_and_sensitive(self):
        a = from_openscad([{"name": "w", "type": "number", "initial": 1, "max": 5}])
        b = from_openscad([{"name": "w", "type": "number", "initial": 1, "max": 6}])
        self.assertEqual(schema_digest(a), schema_digest(a))
        self.assertNotEqual(schema_digest(a), schema_digest(b))


if __name__ == "__main__":
    unittest.main()
