import itertools
import json
import random
import unittest

from harnesscad.domain.drawings.annotation_schema import (
    ANNOTATION_COLORS,
    VALID_ANNOTATION_TYPES,
    VALID_DIMENSION_TYPES,
    VALID_GEOMETRIC_CHARACTERISTICS,
    VALID_MATERIAL_CONDITIONS,
    BoundingBox,
    DatumAnnotation,
    DimensionAnnotation,
    FcfAnnotation,
    NoteAnnotation,
    SurfaceFinishAnnotation,
    clamp,
    extract_json_blob,
    main,
    parse_annotation,
    parse_annotation_response,
    validate_bounding_box,
    validate_confidence,
)

SEED = 20260719


def _box(**over):
    d = {"x": 10, "y": 20, "width": 15, "height": 8, "color": "green"}
    d.update(over)
    return d


def _base_raw(ann_type, **over):
    d = {
        "id": "ann_x",
        "type": ann_type,
        "label": "L",
        "value": "V",
        "view": "Front View",
        "boundingBox": _box(),
        "confidence": 0.9,
    }
    d.update(over)
    return d


class ClampTests(unittest.TestCase):
    def test_clamp_inside_range(self):
        self.assertEqual(clamp(0.5, 0.0, 1.0), 0.5)

    def test_clamp_below(self):
        self.assertEqual(clamp(-3.0, 0.0, 1.0), 0.0)

    def test_clamp_above(self):
        self.assertEqual(clamp(4.0, 0.0, 1.0), 1.0)

    def test_clamp_at_bounds(self):
        self.assertEqual(clamp(0.0, 0.0, 1.0), 0.0)
        self.assertEqual(clamp(1.0, 0.0, 1.0), 1.0)


class ValidateConfidenceTests(unittest.TestCase):
    def test_clamps_above_one(self):
        self.assertEqual(validate_confidence(1.7), 1.0)

    def test_clamps_below_zero(self):
        self.assertEqual(validate_confidence(-0.4), 0.0)

    def test_default_for_missing(self):
        self.assertEqual(validate_confidence(None), 0.5)

    def test_default_for_string(self):
        # known-bad vector: model emits "high" instead of a number
        self.assertEqual(validate_confidence("high"), 0.5)

    def test_default_for_nan(self):
        self.assertEqual(validate_confidence(float("nan")), 0.5)

    def test_bool_rejected(self):
        # divergence-worthy detail: Python bools are ints, but True is not a
        # confidence, so _num rejects them and the 0.5 default applies.
        self.assertEqual(validate_confidence(True), 0.5)

    def test_int_accepted(self):
        self.assertEqual(validate_confidence(1), 1.0)


class ValidateBoundingBoxTests(unittest.TestCase):
    def test_valid_box(self):
        bb = validate_bounding_box(_box(), "purple")
        self.assertEqual(bb.x, 10.0)
        self.assertEqual(bb.y, 20.0)
        self.assertEqual(bb.width, 15.0)
        self.assertEqual(bb.height, 8.0)
        self.assertEqual(bb.color, "green")

    def test_non_dict_rejected(self):
        self.assertIsNone(validate_bounding_box(None, "green"))
        self.assertIsNone(validate_bounding_box("nope", "green"))
        self.assertIsNone(validate_bounding_box([1, 2, 3, 4], "green"))

    def test_missing_coordinate_rejected(self):
        raw = _box()
        del raw["x"]
        self.assertIsNone(validate_bounding_box(raw, "green"))

    def test_non_numeric_coordinate_rejected(self):
        self.assertIsNone(validate_bounding_box(_box(x="10"), "green"))

    def test_nan_coordinate_rejected(self):
        self.assertIsNone(validate_bounding_box(_box(y=float("nan")), "green"))

    def test_zero_width_rejected(self):
        self.assertIsNone(validate_bounding_box(_box(width=0), "green"))

    def test_zero_height_rejected(self):
        self.assertIsNone(validate_bounding_box(_box(height=0), "green"))

    def test_negative_width_rejected(self):
        self.assertIsNone(validate_bounding_box(_box(width=-5), "green"))

    def test_clamps_oversized_and_negative(self):
        bb = validate_bounding_box(
            _box(x=-20, y=500, width=200, height=300), "green"
        )
        self.assertEqual(bb.x, 0.0)
        self.assertEqual(bb.y, 100.0)
        self.assertEqual(bb.width, 100.0)
        self.assertEqual(bb.height, 100.0)

    def test_clamps_tiny_width_up_to_floor(self):
        bb = validate_bounding_box(_box(width=0.001, height=0.02), "green")
        self.assertEqual(bb.width, 0.1)
        self.assertEqual(bb.height, 0.1)

    def test_fallback_color_when_absent(self):
        raw = _box()
        del raw["color"]
        self.assertEqual(validate_bounding_box(raw, "cyan").color, "cyan")

    def test_fallback_color_when_non_string(self):
        self.assertEqual(validate_bounding_box(_box(color=7), "cyan").color, "cyan")

    def test_to_dict_keys(self):
        bb = validate_bounding_box(_box(), "green")
        self.assertEqual(
            bb.to_dict(),
            {"x": 10.0, "y": 20.0, "width": 15.0, "height": 8.0, "color": "green"},
        )


class ExtractJsonBlobTests(unittest.TestCase):
    def test_plain_object(self):
        self.assertEqual(extract_json_blob('{"a": 1}'), {"a": 1})

    def test_markdown_fenced(self):
        # known-bad vector from the TS suite: model wraps JSON in ```json fences
        text = 'Here you go:\n```json\n{"a": 1}\n```'
        self.assertEqual(extract_json_blob(text), {"a": 1})

    def test_no_braces(self):
        self.assertIsNone(extract_json_blob("no json here"))

    def test_empty_string(self):
        self.assertIsNone(extract_json_blob(""))

    def test_malformed_json(self):
        self.assertIsNone(extract_json_blob("{not valid json"))
        self.assertIsNone(extract_json_blob("{'a': 1,}"))

    def test_top_level_array_rejected(self):
        # a bare JSON array has no {...} span at all
        self.assertIsNone(extract_json_blob("[1, 2, 3]"))

    def test_greedy_match_spans_to_last_brace(self):
        # harness uses a greedy \{[\s\S]*\} so nested/trailing braces are
        # included; two sibling objects therefore fail to parse.
        self.assertIsNone(extract_json_blob('{"a": 1} and {"b": 2}'))


class ParseAnnotationTypeGateTests(unittest.TestCase):
    def test_non_dict_returns_none(self):
        self.assertIsNone(parse_annotation(None, 0))
        self.assertIsNone(parse_annotation("dimension", 0))
        self.assertIsNone(parse_annotation(42, 0))

    def test_missing_type_returns_none(self):
        raw = _base_raw("note")
        del raw["type"]
        self.assertIsNone(parse_annotation(raw, 0))

    def test_invalid_type_returns_none(self):
        # known-bad vector: invalid annotation category
        for bad in ("bogus", "DIMENSION", "", "fcf ", "notes", 3):
            self.assertIsNone(parse_annotation(_base_raw(bad), 0), bad)

    def test_all_valid_types_survive_with_required_fields(self):
        extras = {
            "dimension": {"dimensionType": "linear", "nominalValue": 1.0},
            "fcf": {"geometricCharacteristic": "position", "toleranceValue": 0.05},
            "datum": {"datumLetter": "A"},
            "surface_finish": {"roughnessValue": 1.6},
            "note": {},
        }
        for t in VALID_ANNOTATION_TYPES:
            ann = parse_annotation(_base_raw(t, **extras[t]), 0)
            self.assertIsNotNone(ann, t)
            self.assertEqual(ann.type, t)

    def test_missing_bounding_box_drops_annotation(self):
        raw = _base_raw("note")
        del raw["boundingBox"]
        self.assertIsNone(parse_annotation(raw, 0))

    def test_zero_width_bounding_box_drops_annotation(self):
        self.assertIsNone(
            parse_annotation(_base_raw("note", boundingBox=_box(width=0)), 0)
        )

    def test_zero_height_bounding_box_drops_annotation(self):
        self.assertIsNone(
            parse_annotation(_base_raw("note", boundingBox=_box(height=0)), 0)
        )


class ParseAnnotationFallbackTests(unittest.TestCase):
    def test_id_fallback_by_index(self):
        raw = _base_raw("note")
        del raw["id"]
        self.assertEqual(parse_annotation(raw, 0).id, "ann_1")
        self.assertEqual(parse_annotation(raw, 4).id, "ann_5")

    def test_empty_id_falls_back(self):
        self.assertEqual(parse_annotation(_base_raw("note", id=""), 2).id, "ann_3")

    def test_non_string_id_falls_back(self):
        self.assertEqual(parse_annotation(_base_raw("note", id=9), 0).id, "ann_1")

    def test_color_palette_cycles_by_index(self):
        raw = _base_raw("note", boundingBox=_box(color=None))
        for i in range(len(ANNOTATION_COLORS) * 2 + 1):
            ann = parse_annotation(raw, i)
            self.assertEqual(
                ann.bounding_box.color, ANNOTATION_COLORS[i % len(ANNOTATION_COLORS)]
            )

    def test_label_value_view_defaults(self):
        raw = {"type": "note", "boundingBox": _box()}
        ann = parse_annotation(raw, 0)
        self.assertEqual(ann.label, "")
        self.assertEqual(ann.value, "")
        self.assertEqual(ann.view, "View 1")
        self.assertIsNone(ann.description)
        self.assertEqual(ann.confidence, 0.5)
        self.assertEqual(ann.needs_review, False)

    def test_non_string_label_value_view_defaulted(self):
        ann = parse_annotation(_base_raw("note", label=1, value=[], view={}), 0)
        self.assertEqual(ann.label, "")
        self.assertEqual(ann.value, "")
        self.assertEqual(ann.view, "View 1")

    def test_description_only_when_string(self):
        self.assertEqual(
            parse_annotation(_base_raw("note", description="hi"), 0).description, "hi"
        )
        self.assertIsNone(
            parse_annotation(_base_raw("note", description=5), 0).description
        )

    def test_needs_review_camel_and_snake(self):
        self.assertTrue(parse_annotation(_base_raw("note", needsReview=True), 0).needs_review)
        self.assertTrue(
            parse_annotation(_base_raw("note", needs_review=True), 0).needs_review
        )

    def test_needs_review_non_bool_defaults_false(self):
        self.assertFalse(
            parse_annotation(_base_raw("note", needsReview="yes"), 0).needs_review
        )

    def test_camel_case_wins_over_snake_case(self):
        # harness divergence from the TS original: the harness accepts BOTH
        # camelCase (LLM/TS spelling) and snake_case (harness-native dicts),
        # with camelCase taking precedence.
        raw = _base_raw("datum", datumLetter="A", datum_letter="B")
        self.assertEqual(parse_annotation(raw, 0).datum_letter, "A")

    def test_snake_case_accepted_alone(self):
        raw = _base_raw("dimension", dimension_type="radius", nominal_value=3.2)
        del raw["boundingBox"]
        raw["bounding_box"] = _box()
        ann = parse_annotation(raw, 0)
        self.assertEqual(ann.dimension_type, "radius")
        self.assertEqual(ann.nominal_value, 3.2)


class ParseDimensionTests(unittest.TestCase):
    def test_valid_dimension(self):
        ann = parse_annotation(
            _base_raw(
                "dimension",
                dimensionType="linear",
                nominalValue=40.2,
                plusTolerance=0.1,
                minusTolerance=-0.1,
                unit="mm",
            ),
            0,
        )
        self.assertIsInstance(ann, DimensionAnnotation)
        self.assertEqual(ann.dimension_type, "linear")
        self.assertEqual(ann.nominal_value, 40.2)
        self.assertEqual(ann.plus_tolerance, 0.1)
        self.assertEqual(ann.minus_tolerance, -0.1)
        self.assertEqual(ann.unit, "mm")

    def test_all_valid_dimension_types(self):
        for dt in VALID_DIMENSION_TYPES:
            ann = parse_annotation(
                _base_raw("dimension", dimensionType=dt, nominalValue=1.0), 0
            )
            self.assertEqual(ann.dimension_type, dt)

    def test_invalid_dimension_type_dropped(self):
        # known-bad vector from the TS suite
        for bad in ("Linear", "bogus", "", 5, None):
            self.assertIsNone(
                parse_annotation(
                    _base_raw("dimension", dimensionType=bad, nominalValue=1.0), 0
                ),
                bad,
            )

    def test_missing_nominal_value_dropped(self):
        self.assertIsNone(
            parse_annotation(_base_raw("dimension", dimensionType="linear"), 0)
        )

    def test_non_numeric_nominal_value_dropped(self):
        self.assertIsNone(
            parse_annotation(
                _base_raw("dimension", dimensionType="linear", nominalValue="40.2"), 0
            )
        )

    def test_optional_tolerances_default_none(self):
        ann = parse_annotation(
            _base_raw("dimension", dimensionType="angular", nominalValue=90), 0
        )
        self.assertIsNone(ann.plus_tolerance)
        self.assertIsNone(ann.minus_tolerance)
        self.assertIsNone(ann.unit)

    def test_non_string_unit_dropped_to_none(self):
        ann = parse_annotation(
            _base_raw("dimension", dimensionType="linear", nominalValue=1, unit=3), 0
        )
        self.assertIsNone(ann.unit)

    def test_to_dict_omits_absent_optionals(self):
        ann = parse_annotation(
            _base_raw("dimension", dimensionType="linear", nominalValue=1), 0
        )
        d = ann.to_dict()
        self.assertEqual(d["type"], "dimension")
        self.assertEqual(d["dimension_type"], "linear")
        self.assertNotIn("plus_tolerance", d)
        self.assertNotIn("unit", d)
        self.assertNotIn("description", d)


class ParseFcfTests(unittest.TestCase):
    def test_valid_fcf(self):
        ann = parse_annotation(
            _base_raw(
                "fcf",
                geometricCharacteristic="position",
                toleranceValue=0.05,
                materialCondition="MMC",
                datumReferences=["A", "B", "C"],
            ),
            0,
        )
        self.assertIsInstance(ann, FcfAnnotation)
        self.assertEqual(ann.geometric_characteristic, "position")
        self.assertEqual(ann.tolerance_value, 0.05)
        self.assertEqual(ann.material_condition, "MMC")
        self.assertEqual(ann.datum_references, ("A", "B", "C"))

    def test_all_14_characteristics_accepted(self):
        self.assertEqual(len(VALID_GEOMETRIC_CHARACTERISTICS), 14)
        for gc in VALID_GEOMETRIC_CHARACTERISTICS:
            ann = parse_annotation(
                _base_raw("fcf", geometricCharacteristic=gc, toleranceValue=0.1), 0
            )
            self.assertEqual(ann.geometric_characteristic, gc)

    def test_invalid_characteristic_dropped(self):
        # known-bad vector from the TS suite
        for bad in ("Position", "roundness", "", None, 1):
            self.assertIsNone(
                parse_annotation(
                    _base_raw("fcf", geometricCharacteristic=bad, toleranceValue=0.1), 0
                ),
                bad,
            )

    def test_missing_tolerance_value_dropped(self):
        self.assertIsNone(
            parse_annotation(_base_raw("fcf", geometricCharacteristic="flatness"), 0)
        )

    def test_invalid_material_condition_becomes_none(self):
        # known-bad vector: TS sets materialCondition to null for bad enums
        for bad in ("mmc", "BOGUS", 7, None):
            ann = parse_annotation(
                _base_raw(
                    "fcf",
                    geometricCharacteristic="position",
                    toleranceValue=0.1,
                    materialCondition=bad,
                ),
                0,
            )
            self.assertIsNone(ann.material_condition, bad)

    def test_all_material_conditions_accepted(self):
        for mc in VALID_MATERIAL_CONDITIONS:
            ann = parse_annotation(
                _base_raw(
                    "fcf",
                    geometricCharacteristic="position",
                    toleranceValue=0.1,
                    materialCondition=mc,
                ),
                0,
            )
            self.assertEqual(ann.material_condition, mc)

    def test_datum_references_filtered(self):
        ann = parse_annotation(
            _base_raw(
                "fcf",
                geometricCharacteristic="position",
                toleranceValue=0.1,
                datumReferences=["A", "b", "CD", "", 5, None, "E"],
            ),
            0,
        )
        self.assertEqual(ann.datum_references, ("A", "E"))

    def test_datum_references_truncated_to_three(self):
        ann = parse_annotation(
            _base_raw(
                "fcf",
                geometricCharacteristic="position",
                toleranceValue=0.1,
                datumReferences=["A", "B", "C", "D", "E"],
            ),
            0,
        )
        self.assertEqual(ann.datum_references, ("A", "B", "C"))

    def test_non_list_datum_references_becomes_empty(self):
        ann = parse_annotation(
            _base_raw(
                "fcf",
                geometricCharacteristic="position",
                toleranceValue=0.1,
                datumReferences="ABC",
            ),
            0,
        )
        self.assertEqual(ann.datum_references, ())

    def test_to_dict_always_carries_fcf_fields(self):
        ann = parse_annotation(
            _base_raw("fcf", geometricCharacteristic="flatness", toleranceValue=0.05), 0
        )
        d = ann.to_dict()
        self.assertIsNone(d["material_condition"])
        self.assertEqual(d["datum_references"], [])


class ParseDatumTests(unittest.TestCase):
    def test_valid_datum(self):
        ann = parse_annotation(_base_raw("datum", datumLetter="A"), 0)
        self.assertIsInstance(ann, DatumAnnotation)
        self.assertEqual(ann.datum_letter, "A")

    def test_every_uppercase_letter_accepted(self):
        for ch in "ABCDEFGHIJKLMNOPQRSTUVWXYZ":
            ann = parse_annotation(_base_raw("datum", datumLetter=ch), 0)
            self.assertEqual(ann.datum_letter, ch)

    def test_invalid_datum_letters_dropped(self):
        # known-bad vectors from the TS suite: lowercase, multi-char, digits
        for bad in ("a", "abc", "AB", "", "1", "?", 5, None):
            self.assertIsNone(
                parse_annotation(_base_raw("datum", datumLetter=bad), 0), bad
            )

    # BUG: _DATUM_LETTER_RE = re.compile(r"^[A-Z]$") -- in Python "$" also
    # matches immediately before a trailing newline, so a datum letter of
    # "A\n" (perfectly representable in JSON) is wrongly accepted. Exact
    # reproducing input: parse_annotation({"type": "datum", "datumLetter":
    # "A\n", "boundingBox": {...}}, 0) returns a DatumAnnotation instead of
    # None. Fix would be r"\A[A-Z]\Z". Same flaw affects FCF datumReferences.
    @unittest.expectedFailure
    def test_datum_letter_with_trailing_newline_should_be_rejected(self):
        self.assertIsNone(parse_annotation(_base_raw("datum", datumLetter="A\n"), 0))

    @unittest.expectedFailure
    def test_datum_reference_with_trailing_newline_should_be_rejected(self):
        ann = parse_annotation(
            _base_raw(
                "fcf",
                geometricCharacteristic="position",
                toleranceValue=0.1,
                datumReferences=["A\n"],
            ),
            0,
        )
        self.assertEqual(ann.datum_references, ())


class ParseSurfaceFinishTests(unittest.TestCase):
    def test_valid_surface_finish(self):
        ann = parse_annotation(
            _base_raw("surface_finish", roughnessValue=1.6, processNote="Ground"), 0
        )
        self.assertIsInstance(ann, SurfaceFinishAnnotation)
        self.assertEqual(ann.roughness_value, 1.6)
        self.assertEqual(ann.process_note, "Ground")

    def test_missing_roughness_dropped(self):
        self.assertIsNone(parse_annotation(_base_raw("surface_finish"), 0))

    def test_non_numeric_roughness_dropped(self):
        self.assertIsNone(
            parse_annotation(_base_raw("surface_finish", roughnessValue="1.6"), 0)
        )

    def test_non_string_process_note_becomes_none(self):
        ann = parse_annotation(
            _base_raw("surface_finish", roughnessValue=0.8, processNote=12), 0
        )
        self.assertIsNone(ann.process_note)
        self.assertNotIn("process_note", ann.to_dict())


class ParseNoteTests(unittest.TestCase):
    def test_valid_note(self):
        ann = parse_annotation(
            _base_raw("note", label="GENERAL", value="MM", description="d"), 0
        )
        self.assertIsInstance(ann, NoteAnnotation)
        d = ann.to_dict()
        self.assertEqual(d["type"], "note")
        self.assertEqual(d["description"], "d")
        # note carries no type-specific fields
        self.assertEqual(
            sorted(d),
            ["bounding_box", "confidence", "description", "id", "label",
             "needs_review", "type", "value", "view"],
        )


class ParseAnnotationResponseTests(unittest.TestCase):
    def test_unparseable_returns_defaults(self):
        anns, views, desc = parse_annotation_response("no json here")
        self.assertEqual(anns, [])
        self.assertEqual(views, ["View 1"])
        self.assertIsNone(desc)

    def test_empty_string_returns_defaults(self):
        self.assertEqual(parse_annotation_response(""), ([], ["View 1"], None))

    def test_markdown_fenced_response(self):
        payload = {
            "annotations": [_base_raw("datum", datumLetter="A")],
            "views": ["Front View"],
            "description": "a bracket",
        }
        text = "Sure!\n```json\n%s\n```" % json.dumps(payload)
        anns, views, desc = parse_annotation_response(text)
        self.assertEqual(len(anns), 1)
        self.assertEqual(views, ["Front View"])
        self.assertEqual(desc, "a bracket")

    def test_non_array_annotations_treated_as_empty(self):
        # known-bad vector: model returns an object/string for "annotations"
        for bad in ('{"annotations": {"a": 1}}', '{"annotations": "none"}',
                    '{"annotations": null}', "{}"):
            anns, views, desc = parse_annotation_response(bad)
            self.assertEqual(anns, [], bad)
            self.assertEqual(views, ["View 1"], bad)

    def test_views_defaults_when_missing(self):
        self.assertEqual(parse_annotation_response('{"annotations": []}')[1], ["View 1"])

    def test_views_defaults_when_all_entries_invalid(self):
        out = parse_annotation_response('{"views": ["", 1, null]}')
        self.assertEqual(out[1], ["View 1"])

    def test_views_filters_non_strings(self):
        out = parse_annotation_response('{"views": ["A", 1, "", "B"]}')
        self.assertEqual(out[1], ["A", "B"])

    def test_views_non_list_defaults(self):
        self.assertEqual(parse_annotation_response('{"views": "Front"}')[1], ["View 1"])

    def test_description_only_when_string(self):
        self.assertIsNone(parse_annotation_response('{"description": 5}')[2])
        self.assertEqual(parse_annotation_response('{"description": "x"}')[2], "x")

    def test_malformed_annotations_are_dropped_but_indices_advance(self):
        # index-based id/color fallbacks use the RAW index, so a dropped
        # entry still consumes an index.
        payload = {
            "annotations": [
                {"type": "bogus"},
                {"type": "note", "boundingBox": _box(color=None)},
            ]
        }
        anns, _, _ = parse_annotation_response(json.dumps(payload))
        self.assertEqual(len(anns), 1)
        self.assertEqual(anns[0].id, "ann_2")
        self.assertEqual(anns[0].bounding_box.color, ANNOTATION_COLORS[1])

    def test_mixed_response_all_five_types(self):
        payload = {
            "annotations": [
                _base_raw("dimension", dimensionType="linear", nominalValue=1),
                _base_raw("fcf", geometricCharacteristic="position",
                          toleranceValue=0.05),
                _base_raw("datum", datumLetter="A"),
                _base_raw("surface_finish", roughnessValue=1.6),
                _base_raw("note"),
            ],
            "views": ["V"],
        }
        anns, views, _ = parse_annotation_response(json.dumps(payload))
        self.assertEqual([a.type for a in anns], list(VALID_ANNOTATION_TYPES))
        self.assertEqual(views, ["V"])

    def test_top_level_json_array_returns_defaults(self):
        self.assertEqual(parse_annotation_response("[1, 2]"), ([], ["View 1"], None))


class RoundTripPropertyTests(unittest.TestCase):
    # Substitute for the fast-check property tests in the TS suite: hypothesis
    # is unavailable, so these enumerate small domains exhaustively with
    # itertools and draw the rest from random.Random(FIXED SEED).

    def _make_raw(self, rng, ann_type, index):
        raw = {
            "id": "id_%d" % index,
            "type": ann_type,
            "label": "label_%d" % index,
            "value": "value_%d" % index,
            "view": "View %d" % (index % 3),
            "boundingBox": {
                "x": round(rng.uniform(0, 100), 3),
                "y": round(rng.uniform(0, 100), 3),
                "width": round(rng.uniform(0.1, 100), 3),
                "height": round(rng.uniform(0.1, 100), 3),
                "color": rng.choice(ANNOTATION_COLORS),
            },
            "confidence": round(rng.uniform(0, 1), 4),
            "needsReview": rng.choice([True, False]),
        }
        if rng.random() < 0.5:
            raw["description"] = "desc_%d" % index
        if ann_type == "dimension":
            raw["dimensionType"] = rng.choice(VALID_DIMENSION_TYPES)
            raw["nominalValue"] = round(rng.uniform(-1000, 1000), 4)
            if rng.random() < 0.5:
                raw["plusTolerance"] = round(rng.uniform(0, 1), 4)
            if rng.random() < 0.5:
                raw["minusTolerance"] = round(rng.uniform(-1, 0), 4)
            if rng.random() < 0.5:
                raw["unit"] = rng.choice(["mm", "in", "deg"])
        elif ann_type == "fcf":
            raw["geometricCharacteristic"] = rng.choice(
                VALID_GEOMETRIC_CHARACTERISTICS
            )
            raw["toleranceValue"] = round(rng.uniform(0, 10), 4)
            raw["materialCondition"] = rng.choice(
                list(VALID_MATERIAL_CONDITIONS) + [None]
            )
            raw["datumReferences"] = rng.sample("ABCDEFG", rng.randint(0, 3))
        elif ann_type == "datum":
            raw["datumLetter"] = rng.choice("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
        elif ann_type == "surface_finish":
            raw["roughnessValue"] = round(rng.uniform(0, 100), 4)
            if rng.random() < 0.5:
                raw["processNote"] = "Ground"
        return raw

    def test_parse_to_dict_reparse_is_a_fixed_point(self):
        rng = random.Random(SEED)
        for index, ann_type in enumerate(
            itertools.chain.from_iterable(
                itertools.repeat(VALID_ANNOTATION_TYPES, 20)
            )
        ):
            raw = self._make_raw(rng, ann_type, index)
            first = parse_annotation(raw, index)
            self.assertIsNotNone(first, raw)
            # to_dict emits snake_case, which _get also accepts -> reparsing
            # the serialised form must reproduce the identical object.
            second = parse_annotation(first.to_dict(), index)
            self.assertEqual(first, second, raw)

    def test_json_round_trip_is_stable(self):
        rng = random.Random(SEED + 1)
        for index, ann_type in enumerate(
            itertools.chain.from_iterable(itertools.repeat(VALID_ANNOTATION_TYPES, 10))
        ):
            raw = self._make_raw(rng, ann_type, index)
            ann = parse_annotation(raw, index)
            revived = parse_annotation(json.loads(json.dumps(ann.to_dict())), index)
            self.assertEqual(ann, revived)

    def test_every_type_x_bad_type_string_is_rejected(self):
        bad_types = ["", " ", "Dimension", "FCF", "datum_", "surfacefinish", "notes"]
        for ann_type, bad in itertools.product(VALID_ANNOTATION_TYPES, bad_types):
            raw = self._make_raw(random.Random(SEED), ann_type, 0)
            raw["type"] = bad
            self.assertIsNone(parse_annotation(raw, 0), (ann_type, bad))

    def test_confidence_always_within_unit_interval(self):
        rng = random.Random(SEED + 2)
        candidates = [-1e6, -0.01, 0.0, 0.5, 1.0, 1.01, 1e6, "high", None, True]
        candidates += [rng.uniform(-5, 5) for _ in range(40)]
        for c in candidates:
            ann = parse_annotation(_base_raw("note", confidence=c), 0)
            self.assertTrue(0.0 <= ann.confidence <= 1.0, c)

    def test_bounding_box_always_within_declared_ranges(self):
        rng = random.Random(SEED + 3)
        for _ in range(80):
            raw = _base_raw(
                "note",
                boundingBox={
                    "x": rng.uniform(-500, 500),
                    "y": rng.uniform(-500, 500),
                    "width": rng.uniform(0.0001, 500),
                    "height": rng.uniform(0.0001, 500),
                },
            )
            bb = parse_annotation(raw, 0).bounding_box
            self.assertTrue(0.0 <= bb.x <= 100.0)
            self.assertTrue(0.0 <= bb.y <= 100.0)
            self.assertTrue(0.1 <= bb.width <= 100.0)
            self.assertTrue(0.1 <= bb.height <= 100.0)


class DataclassTests(unittest.TestCase):
    def test_bounding_box_is_frozen(self):
        bb = BoundingBox(x=1, y=2, width=3, height=4, color="red")
        with self.assertRaises(Exception):
            bb.x = 9

    def test_annotation_is_frozen(self):
        ann = parse_annotation(_base_raw("note"), 0)
        with self.assertRaises(Exception):
            ann.label = "other"

    def test_default_type_strings(self):
        bb = BoundingBox(x=1, y=2, width=3, height=4, color="red")
        common = dict(id="i", label="l", value="v", view="w", bounding_box=bb,
                      confidence=0.5)
        self.assertEqual(NoteAnnotation(**common).type, "note")
        self.assertEqual(DatumAnnotation(datum_letter="A", **common).type, "datum")
        self.assertEqual(
            SurfaceFinishAnnotation(roughness_value=1.0, **common).type,
            "surface_finish",
        )


class MainTests(unittest.TestCase):
    def test_selfcheck_passes(self):
        self.assertEqual(main(["--selfcheck"]), 0)

    def test_selfcheck_json_passes(self):
        self.assertEqual(main(["--selfcheck", "--json"]), 0)

    def test_no_args_prints_help(self):
        self.assertEqual(main([]), 0)

    def test_synthetic_response_invariants(self):
        from harnesscad.domain.drawings.annotation_schema import _SYNTHETIC_RESPONSE

        anns, views, desc = parse_annotation_response(_SYNTHETIC_RESPONSE)
        self.assertEqual(len(anns), 5)
        fcf = next(a for a in anns if a.type == "fcf")
        self.assertEqual(fcf.id, "ann_2")
        self.assertEqual(fcf.confidence, 1.0)
        self.assertEqual(fcf.bounding_box.width, 100.0)
        self.assertEqual(fcf.bounding_box.color, "blue")
        self.assertEqual(fcf.datum_references, ("A", "B", "C"))
        note = next(a for a in anns if a.type == "note")
        self.assertEqual(note.confidence, 0.5)
        self.assertEqual(views, ["Front View", "Side View", "Title Block"])
        self.assertEqual(desc, "Synthetic bracket drawing")


if __name__ == "__main__":
    unittest.main()
