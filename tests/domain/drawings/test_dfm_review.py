# Tests for harnesscad.domain.drawings.dfm_review.
#
# Known-bad vectors ported from the CAD-Annotator reference repo
# (artifacts/api-server/src/lib/dfm-reviewer.test.ts), Apache-2.0.
# Credit: CAD-Annotator contributors.
#
# DIVERGENCE FROM UPSTREAM (deliberate, per the harness module docstring):
# the TS dfm-reviewer ran exactly ONE deterministic pre-check (datum scheme
# completeness) and delegated the rest to an LLM. The harness port EXPANDS the
# deterministic side to FOUR checks: datum_scheme_completeness (ported),
# missing_tolerance, surface_finish_consistency and over_tolerancing (new).
# All four are exercised below; the three new ones have no upstream vectors.
#
# Second divergence: upstream drops relatedAnnotationIds entirely when every
# reference is invalid (field becomes undefined). The harness keeps a DfmFinding
# with an EMPTY TUPLE and omits the key only in to_dict(). Both are asserted.
#
# Third divergence: upstream's LLM seam is a module-level mocked OpenAI client;
# the harness injects an optional ``llm`` callable (prompt -> text). The
# upstream "empty choices array" vector is modelled as an llm callable that
# raises IndexError, which is what ``choices[0]`` would do.

import itertools
import random
import unittest
from unittest import mock

from harnesscad.domain.drawings.annotation_schema import (
    BoundingBox,
    DatumAnnotation,
    DimensionAnnotation,
    FcfAnnotation,
    NoteAnnotation,
    SurfaceFinishAnnotation,
)
from harnesscad.domain.drawings.dfm_review import (
    MACHINING_CAPABILITY_FLOOR_MM,
    SURFACE_FINISH_TABLE,
    VALID_CATEGORIES,
    VALID_SEVERITIES,
    DfmFinding,
    build_dfm_prompt,
    check_datum_scheme_completeness,
    check_missing_tolerance,
    check_over_tolerancing,
    check_surface_finish_consistency,
    deterministic_dfm_findings,
    parse_dfm_response,
    parse_single_finding,
    required_max_roughness,
    review_dfm,
)

# hypothesis is NOT installed in this repo and must not be added, so the
# upstream fast-check property tests are replaced by table-driven exhaustive
# enumeration over small domains (itertools) plus a fixed-seed random sweep.
SEED = 20260719

BBOX = BoundingBox(x=10.0, y=20.0, width=15.0, height=8.0, color="green")


def _base(ann_id, label="L", value="V", confidence=0.9):
    return {
        "id": ann_id,
        "label": label,
        "value": value,
        "view": "Front View",
        "bounding_box": BBOX,
        "confidence": confidence,
    }


def make_datum(letter, ann_id=None):
    return DatumAnnotation(
        datum_letter=letter,
        **_base(ann_id or ("datum_%s" % letter), "Datum %s" % letter, letter),
    )


def make_dimension(**kw):
    fields = {
        "dimension_type": "linear",
        "nominal_value": 40.2,
        "plus_tolerance": 0.1,
        "minus_tolerance": -0.1,
        "unit": "mm",
    }
    ann_id = kw.pop("id", "dim_1")
    fields.update(kw)
    return DimensionAnnotation(**fields, **_base(ann_id, "40.2 +/-0.1", "40.2"))


def make_fcf(**kw):
    fields = {
        "geometric_characteristic": "position",
        "tolerance_value": 0.05,
        "material_condition": "MMC",
        "datum_references": ("A", "B"),
    }
    ann_id = kw.pop("id", "fcf_1")
    fields.update(kw)
    return FcfAnnotation(**fields, **_base(ann_id, "Position 0.05 MMC A B", "0.05"))


def make_surface_finish(**kw):
    fields = {"roughness_value": 1.6, "process_note": None}
    ann_id = kw.pop("id", "sf_1")
    fields.update(kw)
    return SurfaceFinishAnnotation(**fields, **_base(ann_id, "Ra 1.6", "1.6"))


def make_note(ann_id="note_1"):
    return NoteAnnotation(**_base(ann_id, "General note", "All dimensions in mm"))


# --------------------------------------------------------------------------- #
# Check 1 of 4: datum scheme completeness (ported verbatim from the TS)
# --------------------------------------------------------------------------- #


class DatumSchemeCompletenessTests(unittest.TestCase):
    def test_zero_datums_warns(self):
        result = check_datum_scheme_completeness([make_dimension(), make_fcf()])
        self.assertIsNotNone(result)
        self.assertEqual(result.category, "datum_scheme_completeness")
        self.assertEqual(result.severity, "warning")
        self.assertIn("No datums detected", result.description)
        self.assertTrue(result.recommendation)
        self.assertEqual(result.id, "dfm_datum_scheme_completeness")

    def test_one_unique_datum_warns_and_lists_letter(self):
        result = check_datum_scheme_completeness([make_datum("A")])
        self.assertIsNotNone(result)
        self.assertIn("1 unique datum", result.description)
        self.assertIn("A", result.description)
        self.assertIn("datum_A", result.related_annotation_ids)

    def test_two_unique_datums_warn(self):
        result = check_datum_scheme_completeness([make_datum("A"), make_datum("B")])
        self.assertIsNotNone(result)
        self.assertIn("2 unique datum", result.description)
        self.assertIn("A", result.description)
        self.assertIn("B", result.description)

    def test_exactly_three_unique_datums_is_clean(self):
        anns = [make_datum("A"), make_datum("B"), make_datum("C")]
        self.assertIsNone(check_datum_scheme_completeness(anns))

    def test_more_than_three_unique_datums_is_clean(self):
        anns = [make_datum(c) for c in "ABCD"]
        self.assertIsNone(check_datum_scheme_completeness(anns))

    def test_duplicate_letters_count_once_but_all_ids_related(self):
        anns = [
            make_datum("A", "datum_A_1"),
            make_datum("A", "datum_A_2"),
            make_datum("B", "datum_B_1"),
        ]
        result = check_datum_scheme_completeness(anns)
        self.assertIsNotNone(result)
        self.assertIn("2 unique datum", result.description)
        self.assertEqual(len(result.related_annotation_ids), 3)

    def test_non_datum_annotations_are_ignored(self):
        anns = [make_dimension(), make_fcf(), make_surface_finish(), make_note()]
        result = check_datum_scheme_completeness(anns)
        self.assertIsNotNone(result)
        self.assertIn("No datums detected", result.description)
        self.assertEqual(result.related_annotation_ids, ())

    def test_empty_annotation_list_warns(self):
        result = check_datum_scheme_completeness([])
        self.assertIsNotNone(result)
        self.assertIn("No datums detected", result.description)

    def test_exhaustive_datum_letter_subsets_up_to_four(self):
        # Substitutes for a fast-check property (hypothesis unavailable):
        # exhaustively enumerate every multiset of datum letters drawn from
        # {A,B,C,D} of length 0..4 and assert the <3-unique rule holds.
        letters = "ABCD"
        for n in range(5):
            for combo in itertools.product(letters, repeat=n):
                anns = [
                    make_datum(c, "datum_%s_%d" % (c, i)) for i, c in enumerate(combo)
                ]
                result = check_datum_scheme_completeness(anns)
                unique = len(set(combo))
                if unique >= 3:
                    self.assertIsNone(result, combo)
                else:
                    self.assertIsNotNone(result, combo)
                    self.assertEqual(result.severity, "warning")
                    self.assertEqual(len(result.related_annotation_ids), n)


# --------------------------------------------------------------------------- #
# Check 2 of 4: missing tolerance (HARNESS-ONLY, no upstream vector)
# --------------------------------------------------------------------------- #


class MissingToleranceTests(unittest.TestCase):
    def test_fully_toleranced_dimension_is_clean(self):
        self.assertEqual(check_missing_tolerance([make_dimension()]), [])

    def test_only_plus_tolerance_is_clean(self):
        ann = make_dimension(plus_tolerance=0.1, minus_tolerance=None)
        self.assertEqual(check_missing_tolerance([ann]), [])

    def test_only_minus_tolerance_is_clean(self):
        ann = make_dimension(plus_tolerance=None, minus_tolerance=-0.1)
        self.assertEqual(check_missing_tolerance([ann]), [])

    def test_zero_tolerance_counts_as_present(self):
        # 0.0 is not None, so the check does NOT fire. Documents the exact
        # boundary of the "is not None" guard.
        ann = make_dimension(plus_tolerance=0.0, minus_tolerance=None)
        self.assertEqual(check_missing_tolerance([ann]), [])

    def test_untoleranced_linear_is_a_warning(self):
        ann = make_dimension(plus_tolerance=None, minus_tolerance=None)
        findings = check_missing_tolerance([ann])
        self.assertEqual(len(findings), 1)
        self.assertEqual(findings[0].severity, "warning")
        self.assertEqual(findings[0].category, "missing_tolerance")
        self.assertEqual(findings[0].id, "dfm_missing_tolerance_dim_1")
        self.assertEqual(findings[0].related_annotation_ids, ("dim_1",))

    def test_severity_by_dimension_type_table(self):
        # Table-driven over the whole small domain of dimension types.
        expected = {
            "linear": "warning",
            "diameter": "warning",
            "angular": "info",
            "radius": "info",
        }
        for dim_type, severity in expected.items():
            with self.subTest(dim_type=dim_type):
                ann = make_dimension(
                    dimension_type=dim_type,
                    plus_tolerance=None,
                    minus_tolerance=None,
                )
                findings = check_missing_tolerance([ann])
                self.assertEqual(len(findings), 1)
                self.assertEqual(findings[0].severity, severity)

    def test_non_dimension_annotations_never_flagged(self):
        anns = [make_datum("A"), make_fcf(), make_surface_finish(), make_note()]
        self.assertEqual(check_missing_tolerance(anns), [])

    def test_one_finding_per_untoleranced_dimension(self):
        anns = [
            make_dimension(id="d%d" % i, plus_tolerance=None, minus_tolerance=None)
            for i in range(3)
        ]
        findings = check_missing_tolerance(anns)
        self.assertEqual(len(findings), 3)
        self.assertEqual(
            sorted(f.id for f in findings),
            ["dfm_missing_tolerance_d0", "dfm_missing_tolerance_d1",
             "dfm_missing_tolerance_d2"],
        )


# --------------------------------------------------------------------------- #
# Check 3 of 4: surface finish consistency (HARNESS-ONLY)
# --------------------------------------------------------------------------- #


class RequiredMaxRoughnessTests(unittest.TestCase):
    def test_table_boundaries(self):
        cases = [
            (0.0, 0.4),
            (0.02, 0.4),
            (0.025, 0.8),   # boundary: strict < means 0.025 falls to next row
            (0.05, 0.8),
            (0.1, 1.6),     # boundary
            (0.2, 1.6),
            (0.25, None),   # boundary: no constraint at/above the last row
            (0.5, None),
        ]
        for band, expected in cases:
            with self.subTest(band=band):
                self.assertEqual(required_max_roughness(band), expected)

    def test_monotone_non_decreasing_over_random_bands(self):
        # Substitutes for a fast-check property (hypothesis unavailable).
        # Seed: 20260719. Tighter bands must never permit a rougher finish.
        rng = random.Random(SEED)
        bands = sorted(rng.uniform(0.0, 0.4) for _ in range(500))
        INF = float("inf")
        prev = 0.0
        for band in bands:
            ra = required_max_roughness(band)
            current = INF if ra is None else ra
            self.assertGreaterEqual(current, prev)
            prev = current

    def test_table_shape_is_the_documented_rule_of_thumb(self):
        self.assertEqual(SURFACE_FINISH_TABLE, ((0.025, 0.4), (0.1, 0.8), (0.25, 1.6)))


class SurfaceFinishConsistencyTests(unittest.TestCase):
    def test_no_tolerance_band_means_no_findings(self):
        anns = [make_surface_finish(roughness_value=25.0), make_datum("A")]
        self.assertEqual(check_surface_finish_consistency(anns), [])

    def test_loose_band_imposes_no_constraint(self):
        # band 1.0 mm >= 0.25 -> required_max_roughness is None
        anns = [
            make_dimension(plus_tolerance=0.5, minus_tolerance=-0.5),
            make_surface_finish(roughness_value=25.0),
        ]
        self.assertEqual(check_surface_finish_consistency(anns), [])

    def test_tight_band_with_rough_finish_warns(self):
        # tightest band = 0.02 mm -> requires Ra <= 0.4; Ra 3.2 is rougher.
        anns = [
            make_dimension(plus_tolerance=0.01, minus_tolerance=0.01),
            make_surface_finish(roughness_value=3.2),
        ]
        findings = check_surface_finish_consistency(anns)
        self.assertEqual(len(findings), 1)
        self.assertEqual(findings[0].category, "surface_finish_consistency")
        self.assertEqual(findings[0].severity, "warning")
        self.assertEqual(findings[0].id, "dfm_surface_finish_sf_1")
        # relates the finish AND the annotation carrying the tightest band
        self.assertEqual(findings[0].related_annotation_ids, ("sf_1", "dim_1"))

    def test_finish_at_the_limit_is_clean(self):
        anns = [
            make_dimension(plus_tolerance=0.01, minus_tolerance=0.01),
            make_surface_finish(roughness_value=0.4),
        ]
        self.assertEqual(check_surface_finish_consistency(anns), [])

    def test_fcf_tolerance_can_be_the_tightest_band(self):
        anns = [
            make_dimension(plus_tolerance=1.0, minus_tolerance=-1.0),
            make_fcf(id="fcf_tight", tolerance_value=0.02),
            make_surface_finish(roughness_value=3.2),
        ]
        findings = check_surface_finish_consistency(anns)
        self.assertEqual(len(findings), 1)
        self.assertEqual(findings[0].related_annotation_ids, ("sf_1", "fcf_tight"))

    def test_nonpositive_fcf_tolerance_ignored_as_band(self):
        anns = [make_fcf(tolerance_value=0.0), make_surface_finish(roughness_value=3.2)]
        self.assertEqual(check_surface_finish_consistency(anns), [])

    def test_symmetric_tolerance_band_sums_absolute_values(self):
        # +0.02 / -0.02 -> band 0.04 (< 0.1) -> Ra <= 0.8, so Ra 1.6 warns
        # but Ra 0.8 does not. Proves the band is the SUM of |plus| + |minus|.
        rough = [
            make_dimension(plus_tolerance=0.02, minus_tolerance=-0.02),
            make_surface_finish(roughness_value=1.6),
        ]
        self.assertEqual(len(check_surface_finish_consistency(rough)), 1)
        fine = [
            make_dimension(plus_tolerance=0.02, minus_tolerance=-0.02),
            make_surface_finish(roughness_value=0.8),
        ]
        self.assertEqual(check_surface_finish_consistency(fine), [])

    def test_all_rough_finishes_flagged(self):
        anns = [
            make_dimension(plus_tolerance=0.005, minus_tolerance=-0.005),
            make_surface_finish(id="sf_a", roughness_value=3.2),
            make_surface_finish(id="sf_b", roughness_value=6.3),
            make_surface_finish(id="sf_ok", roughness_value=0.2),
        ]
        findings = check_surface_finish_consistency(anns)
        self.assertEqual(sorted(f.related_annotation_ids[0] for f in findings),
                         ["sf_a", "sf_b"])


# --------------------------------------------------------------------------- #
# Check 4 of 4: over-tolerancing (HARNESS-ONLY)
# --------------------------------------------------------------------------- #


class OverTolerancingTests(unittest.TestCase):
    def test_sub_capability_fcf_warns(self):
        findings = check_over_tolerancing([make_fcf(tolerance_value=0.005)])
        self.assertEqual(len(findings), 1)
        self.assertEqual(findings[0].category, "over_tolerancing")
        self.assertEqual(findings[0].severity, "warning")
        self.assertEqual(findings[0].id, "dfm_over_tolerancing_fcf_1")
        self.assertEqual(findings[0].related_annotation_ids, ("fcf_1",))
        # description names the geometric characteristic and the floor
        self.assertIn("position", findings[0].description)
        self.assertIn("0.01", findings[0].description)

    def test_boundary_table(self):
        # Guard is 0 < tol < FLOOR (strict on both ends).
        cases = [
            (-0.001, 0),
            (0.0, 0),
            (0.0001, 1),
            (0.009999, 1),
            (MACHINING_CAPABILITY_FLOOR_MM, 0),   # exactly at the floor: clean
            (0.05, 0),
            (1.0, 0),
        ]
        for tol, expected in cases:
            with self.subTest(tolerance=tol):
                findings = check_over_tolerancing([make_fcf(tolerance_value=tol)])
                self.assertEqual(len(findings), expected)

    def test_non_fcf_annotations_never_flagged(self):
        anns = [
            make_dimension(plus_tolerance=0.0001, minus_tolerance=-0.0001),
            make_datum("A"),
            make_surface_finish(),
            make_note(),
        ]
        self.assertEqual(check_over_tolerancing(anns), [])

    def test_floor_constant_is_ten_microns(self):
        self.assertEqual(MACHINING_CAPABILITY_FLOOR_MM, 0.01)


# --------------------------------------------------------------------------- #
# All four checks together
# --------------------------------------------------------------------------- #


class DeterministicFindingsTests(unittest.TestCase):
    def _kitchen_sink(self):
        return [
            make_datum("A", "ann_1"),
            make_dimension(id="ann_2", plus_tolerance=0.01, minus_tolerance=0.01),
            make_dimension(id="ann_3", dimension_type="diameter",
                           plus_tolerance=None, minus_tolerance=None),
            make_surface_finish(id="ann_4", roughness_value=3.2),
            make_fcf(id="ann_5", geometric_characteristic="flatness",
                     tolerance_value=0.005, material_condition=None,
                     datum_references=()),
        ]

    def test_all_four_deterministic_checks_fire(self):
        # Proves the harness's expansion from 1 upstream check to 4.
        findings = deterministic_dfm_findings(self._kitchen_sink())
        self.assertEqual(
            sorted(f.category for f in findings),
            [
                "datum_scheme_completeness",
                "missing_tolerance",
                "over_tolerancing",
                "surface_finish_consistency",
            ],
        )

    def test_fixed_ordering(self):
        findings = deterministic_dfm_findings(self._kitchen_sink())
        self.assertEqual(
            [f.category for f in findings],
            [
                "datum_scheme_completeness",
                "missing_tolerance",
                "surface_finish_consistency",
                "over_tolerancing",
            ],
        )

    def test_clean_drawing_produces_no_findings(self):
        anns = [
            make_datum("A", "d1"),
            make_datum("B", "d2"),
            make_datum("C", "d3"),
            make_dimension(plus_tolerance=0.5, minus_tolerance=-0.5),
            # NOTE: the FCF tolerance participates in the tightest-band
            # calculation, so it must be >= 0.25 mm for the drawing to be
            # surface-finish-clean at Ra 1.6.
            make_fcf(tolerance_value=0.3),
            make_surface_finish(roughness_value=1.6),
        ]
        self.assertEqual(deterministic_dfm_findings(anns), [])

    def test_empty_input_yields_only_the_datum_finding(self):
        findings = deterministic_dfm_findings([])
        self.assertEqual(len(findings), 1)
        self.assertEqual(findings[0].category, "datum_scheme_completeness")

    def test_every_emitted_category_and_severity_is_in_the_vocabulary(self):
        for f in deterministic_dfm_findings(self._kitchen_sink()):
            self.assertIn(f.category, VALID_CATEGORIES)
            self.assertIn(f.severity, VALID_SEVERITIES)
            self.assertTrue(f.description.strip())
            self.assertTrue(f.recommendation.strip())


# --------------------------------------------------------------------------- #
# Prompt construction (upstream vectors)
# --------------------------------------------------------------------------- #


class BuildDfmPromptTests(unittest.TestCase):
    def test_includes_annotation_data_and_category_definitions(self):
        prompt = build_dfm_prompt([make_datum("A"), make_fcf(), make_dimension()])
        for token in (
            "datum",
            "position",
            "over_tolerancing",
            "missing_tolerance",
            "datum_scheme_completeness",
            "surface_finish_consistency",
        ):
            self.assertIn(token, prompt)

    def test_includes_type_specific_fields(self):
        anns = [
            make_dimension(nominal_value=42.5, unit="mm"),
            make_fcf(geometric_characteristic="flatness", tolerance_value=0.01),
            make_datum("B"),
            make_surface_finish(roughness_value=3.2, process_note="Milled"),
            make_note(),
        ]
        prompt = build_dfm_prompt(anns)
        for token in ("42.5", "flatness", "3.2", "Milled", "datumLetter"):
            self.assertIn(token, prompt)

    def test_empty_annotations_still_yields_a_prompt_with_empty_array(self):
        prompt = build_dfm_prompt([])
        self.assertIsInstance(prompt, str)
        self.assertGreater(len(prompt), 0)
        self.assertIn("[]", prompt)

    def test_deterministic(self):
        anns = [make_datum("A"), make_fcf()]
        self.assertEqual(build_dfm_prompt(anns), build_dfm_prompt(anns))


# --------------------------------------------------------------------------- #
# Response parsing: KNOWN-BAD VECTORS ported from dfm-reviewer.test.ts
# --------------------------------------------------------------------------- #

VALID_IDS = {"ann_1", "ann_2", "ann_3"}


def _resp(findings):
    import json

    return json.dumps({"findings": findings})


class ParseDfmResponseTests(unittest.TestCase):
    def test_parses_multiple_valid_findings(self):
        content = _resp(
            [
                {
                    "id": "dfm_1",
                    "category": "over_tolerancing",
                    "severity": "warning",
                    "description": "Multiple tight tolerances detected",
                    "recommendation": "Consider relaxing non-critical tolerances",
                    "relatedAnnotationIds": ["ann_1", "ann_2"],
                },
                {
                    "id": "dfm_2",
                    "category": "missing_tolerance",
                    "severity": "error",
                    "description": "Critical feature lacks tolerance",
                    "recommendation": "Add position tolerance to feature",
                    "relatedAnnotationIds": ["ann_3"],
                },
            ]
        )
        findings = parse_dfm_response(content, VALID_IDS)
        self.assertEqual(len(findings), 2)
        self.assertEqual(findings[0].category, "over_tolerancing")
        self.assertEqual(findings[0].related_annotation_ids, ("ann_1", "ann_2"))
        self.assertEqual(findings[1].category, "missing_tolerance")

    def test_markdown_fenced_json(self):
        content = (
            "```json\n"
            '{\n  "findings": [\n    {\n'
            '      "id": "dfm_1",\n'
            '      "category": "general",\n'
            '      "severity": "info",\n'
            '      "description": "Drawing looks good",\n'
            '      "recommendation": "No changes needed"\n'
            "    }\n  ]\n}\n"
            "```"
        )
        findings = parse_dfm_response(content, VALID_IDS)
        self.assertEqual(len(findings), 1)
        self.assertEqual(findings[0].category, "general")

    def test_prose_wrapped_json(self):
        content = "Sure! Here you go:\n" + _resp(
            [{"id": "x", "category": "general", "severity": "info",
              "description": "d", "recommendation": "r"}]
        ) + "\nHope that helps."
        self.assertEqual(len(parse_dfm_response(content, VALID_IDS)), 1)

    def test_invalid_category_dropped(self):
        content = _resp(
            [
                {"id": "dfm_1", "category": "invalid_category",
                 "severity": "warning", "description": "Some issue",
                 "recommendation": "Fix it"},
                {"id": "dfm_2", "category": "over_tolerancing",
                 "severity": "warning", "description": "Valid issue",
                 "recommendation": "Fix it properly"},
            ]
        )
        findings = parse_dfm_response(content, VALID_IDS)
        self.assertEqual(len(findings), 1)
        self.assertEqual(findings[0].id, "dfm_2")

    def test_invalid_severity_dropped(self):
        content = _resp(
            [{"id": "dfm_1", "category": "over_tolerancing",
              "severity": "critical", "description": "Some issue",
              "recommendation": "Fix it"}]
        )
        self.assertEqual(parse_dfm_response(content, VALID_IDS), [])

    def test_empty_description_dropped(self):
        content = _resp(
            [{"id": "dfm_1", "category": "over_tolerancing", "severity": "warning",
              "description": "", "recommendation": "Fix it"}]
        )
        self.assertEqual(parse_dfm_response(content, VALID_IDS), [])

    def test_whitespace_only_description_dropped(self):
        content = _resp(
            [{"id": "dfm_1", "category": "over_tolerancing", "severity": "warning",
              "description": "   \t\n ", "recommendation": "Fix it"}]
        )
        self.assertEqual(parse_dfm_response(content, VALID_IDS), [])

    def test_empty_recommendation_dropped(self):
        content = _resp(
            [{"id": "dfm_1", "category": "over_tolerancing", "severity": "warning",
              "description": "Some issue", "recommendation": ""}]
        )
        self.assertEqual(parse_dfm_response(content, VALID_IDS), [])

    def test_fallback_id_from_index(self):
        content = _resp(
            [{"category": "general", "severity": "info",
              "description": "No id provided", "recommendation": "Add an id"}]
        )
        findings = parse_dfm_response(content, VALID_IDS)
        self.assertEqual(len(findings), 1)
        self.assertEqual(findings[0].id, "dfm_1")

    def test_fallback_id_uses_position_in_the_raw_list_not_the_output_list(self):
        content = _resp(
            [
                {"category": "bogus", "severity": "info", "description": "d",
                 "recommendation": "r"},
                {"category": "general", "severity": "info", "description": "d",
                 "recommendation": "r"},
            ]
        )
        findings = parse_dfm_response(content, VALID_IDS)
        self.assertEqual(len(findings), 1)
        self.assertEqual(findings[0].id, "dfm_2")

    def test_related_ids_filtered_to_the_valid_set(self):
        content = _resp(
            [{"id": "dfm_1", "category": "over_tolerancing", "severity": "warning",
              "description": "Issue found", "recommendation": "Fix it",
              "relatedAnnotationIds": ["ann_1", "invalid_id", "ann_3"]}]
        )
        findings = parse_dfm_response(content, VALID_IDS)
        self.assertEqual(findings[0].related_annotation_ids, ("ann_1", "ann_3"))

    def test_all_related_ids_invalid_yields_empty_tuple(self):
        # DIVERGENCE: upstream leaves relatedAnnotationIds undefined; the
        # harness keeps an empty tuple and omits the key from to_dict().
        content = _resp(
            [{"id": "dfm_1", "category": "over_tolerancing", "severity": "warning",
              "description": "Issue found", "recommendation": "Fix it",
              "relatedAnnotationIds": ["invalid_1", "invalid_2"]}]
        )
        findings = parse_dfm_response(content, VALID_IDS)
        self.assertEqual(len(findings), 1)
        self.assertEqual(findings[0].related_annotation_ids, ())
        self.assertNotIn("related_annotation_ids", findings[0].to_dict())

    def test_completely_invalid_json(self):
        self.assertEqual(parse_dfm_response("not json at all", VALID_IDS), [])

    def test_empty_string(self):
        self.assertEqual(parse_dfm_response("", VALID_IDS), [])

    def test_truncated_json(self):
        self.assertEqual(
            parse_dfm_response('{"findings": [{"category": "gen', VALID_IDS), []
        )

    def test_findings_not_an_array(self):
        for bad in ('{"findings": "not an array"}', '{"findings": 5}',
                    '{"findings": {"a": 1}}', '{"findings": null}', '{}'):
            with self.subTest(payload=bad):
                self.assertEqual(parse_dfm_response(bad, VALID_IDS), [])

    def test_trims_description_and_recommendation(self):
        content = _resp(
            [{"id": "dfm_1", "category": "general", "severity": "info",
              "description": "  padded description  ",
              "recommendation": "  padded recommendation  "}]
        )
        findings = parse_dfm_response(content, VALID_IDS)
        self.assertEqual(findings[0].description, "padded description")
        self.assertEqual(findings[0].recommendation, "padded recommendation")


class ParseSingleFindingTests(unittest.TestCase):
    IDS = {"ann_1"}

    def test_none_input(self):
        self.assertIsNone(parse_single_finding(None, 0, self.IDS))

    def test_non_dict_inputs(self):
        for bad in ("string", 5, [], 3.5, True):
            with self.subTest(raw=bad):
                self.assertIsNone(parse_single_finding(bad, 0, self.IDS))

    def test_missing_category(self):
        raw = {"severity": "warning", "description": "d", "recommendation": "r"}
        self.assertIsNone(parse_single_finding(raw, 0, self.IDS))

    def test_missing_severity(self):
        raw = {"category": "general", "description": "d", "recommendation": "r"}
        self.assertIsNone(parse_single_finding(raw, 0, self.IDS))

    def test_non_string_category_or_severity(self):
        self.assertIsNone(parse_single_finding(
            {"category": 5, "severity": "info", "description": "d",
             "recommendation": "r"}, 0, self.IDS))
        self.assertIsNone(parse_single_finding(
            {"category": "general", "severity": ["info"], "description": "d",
             "recommendation": "r"}, 0, self.IDS))

    def test_non_string_description_or_recommendation(self):
        self.assertIsNone(parse_single_finding(
            {"category": "general", "severity": "info", "description": 42,
             "recommendation": "r"}, 0, self.IDS))
        self.assertIsNone(parse_single_finding(
            {"category": "general", "severity": "info", "description": "d",
             "recommendation": None}, 0, self.IDS))

    def test_accepts_the_full_category_severity_cross_product(self):
        # Exhaustive over the small domain (5 x 3 = 15) -- substitutes for a
        # fast-check property since hypothesis is unavailable.
        for category, severity in itertools.product(
            sorted(VALID_CATEGORIES), sorted(VALID_SEVERITIES)
        ):
            with self.subTest(category=category, severity=severity):
                raw = {"category": category, "severity": severity,
                       "description": "desc", "recommendation": "rec"}
                result = parse_single_finding(raw, 0, self.IDS)
                self.assertIsNotNone(result)
                self.assertEqual(result.category, category)
                self.assertEqual(result.severity, severity)

    def test_rejects_near_miss_vocabulary(self):
        for category in ("Over_Tolerancing", "over-tolerancing", "", "GENERAL"):
            with self.subTest(category=category):
                self.assertIsNone(parse_single_finding(
                    {"category": category, "severity": "info",
                     "description": "d", "recommendation": "r"}, 0, self.IDS))
        for severity in ("Warning", "critical", "", "WARN"):
            with self.subTest(severity=severity):
                self.assertIsNone(parse_single_finding(
                    {"category": "general", "severity": severity,
                     "description": "d", "recommendation": "r"}, 0, self.IDS))

    def test_snake_case_related_ids_accepted(self):
        # HARNESS EXTENSION: the TS only reads relatedAnnotationIds.
        raw = {"category": "general", "severity": "info", "description": "d",
               "recommendation": "r", "related_annotation_ids": ["ann_1", "nope"]}
        result = parse_single_finding(raw, 0, self.IDS)
        self.assertEqual(result.related_annotation_ids, ("ann_1",))

    def test_camel_case_wins_over_snake_case(self):
        raw = {"category": "general", "severity": "info", "description": "d",
               "recommendation": "r", "relatedAnnotationIds": ["ann_1"],
               "related_annotation_ids": ["nope"]}
        result = parse_single_finding(raw, 0, self.IDS)
        self.assertEqual(result.related_annotation_ids, ("ann_1",))

    def test_non_list_related_ids_ignored(self):
        raw = {"category": "general", "severity": "info", "description": "d",
               "recommendation": "r", "relatedAnnotationIds": "ann_1"}
        result = parse_single_finding(raw, 0, self.IDS)
        self.assertEqual(result.related_annotation_ids, ())

    def test_non_string_members_of_related_ids_dropped(self):
        raw = {"category": "general", "severity": "info", "description": "d",
               "recommendation": "r",
               "relatedAnnotationIds": ["ann_1", 5, None, {"a": 1}]}
        result = parse_single_finding(raw, 0, self.IDS)
        self.assertEqual(result.related_annotation_ids, ("ann_1",))

    def test_empty_string_id_falls_back_to_index(self):
        raw = {"id": "", "category": "general", "severity": "info",
               "description": "d", "recommendation": "r"}
        self.assertEqual(parse_single_finding(raw, 6, self.IDS).id, "dfm_7")

    def test_non_string_id_falls_back_to_index(self):
        raw = {"id": 12, "category": "general", "severity": "info",
               "description": "d", "recommendation": "r"}
        self.assertEqual(parse_single_finding(raw, 0, self.IDS).id, "dfm_1")


# --------------------------------------------------------------------------- #
# review_dfm: LLM integration via the injected callable seam (unittest.mock)
# --------------------------------------------------------------------------- #


class ReviewDfmTests(unittest.TestCase):
    def test_no_llm_returns_deterministic_only(self):
        anns = [make_datum("A"), make_fcf()]
        self.assertEqual(review_dfm(anns), deterministic_dfm_findings(anns))

    def test_llm_is_called_exactly_once_with_the_built_prompt(self):
        anns = [make_datum("A")]
        llm = mock.MagicMock(return_value='{"findings": []}')
        review_dfm(anns, llm=llm)
        llm.assert_called_once_with(build_dfm_prompt(anns))

    def test_deterministic_datum_finding_present_below_three_datums(self):
        anns = [make_datum("A"), make_datum("B"), make_fcf()]
        llm = mock.MagicMock(return_value='{"findings": []}')
        findings = review_dfm(anns, llm=llm)
        datum = [f for f in findings if f.category == "datum_scheme_completeness"]
        self.assertEqual(len(datum), 1)
        self.assertEqual(datum[0].severity, "warning")
        self.assertEqual(datum[0].id, "dfm_datum_scheme_completeness")

    def test_no_datum_finding_at_three_or_more_datums(self):
        anns = [make_datum("A"), make_datum("B"), make_datum("C"), make_fcf()]
        llm = mock.MagicMock(return_value='{"findings": []}')
        findings = review_dfm(anns, llm=llm)
        self.assertEqual(
            [f for f in findings
             if f.id == "dfm_datum_scheme_completeness"], []
        )

    def test_merges_llm_findings_with_deterministic_ones(self):
        anns = [make_datum("A"), make_fcf()]
        llm = mock.MagicMock(return_value=_resp(
            [{"id": "dfm_llm_1", "category": "over_tolerancing",
              "severity": "warning", "description": "Tight tolerances detected",
              "recommendation": "Relax tolerances",
              "relatedAnnotationIds": ["fcf_1"]}]
        ))
        findings = review_dfm(anns, llm=llm)
        self.assertGreaterEqual(len(findings), 2)
        cats = {f.category for f in findings}
        self.assertIn("datum_scheme_completeness", cats)
        self.assertIn("over_tolerancing", cats)
        llm_finding = next(f for f in findings if f.id == "dfm_llm_1")
        self.assertEqual(llm_finding.related_annotation_ids, ("fcf_1",))

    def test_llm_datum_duplicate_is_dropped_in_favour_of_deterministic(self):
        anns = [make_datum("A")]
        llm = mock.MagicMock(return_value=_resp(
            [{"id": "dfm_llm_datum", "category": "datum_scheme_completeness",
              "severity": "error", "description": "LLM also found datum issue",
              "recommendation": "Add more datums"},
             {"id": "dfm_llm_other", "category": "general", "severity": "info",
              "description": "General observation",
              "recommendation": "No action needed"}]
        ))
        findings = review_dfm(anns, llm=llm)
        datum = [f for f in findings if f.category == "datum_scheme_completeness"]
        self.assertEqual(len(datum), 1)
        self.assertEqual(datum[0].id, "dfm_datum_scheme_completeness")
        self.assertTrue(any(f.category == "general" for f in findings))

    def test_llm_datum_finding_kept_when_no_deterministic_one(self):
        # Three datums -> no deterministic datum finding -> LLM's survives.
        anns = [make_datum("A"), make_datum("B"), make_datum("C"),
                make_dimension(plus_tolerance=0.5, minus_tolerance=-0.5),
                make_fcf(tolerance_value=0.05)]
        llm = mock.MagicMock(return_value=_resp(
            [{"id": "dfm_llm_datum", "category": "datum_scheme_completeness",
              "severity": "error", "description": "Datums are mis-ordered",
              "recommendation": "Re-order the DRF"}]
        ))
        findings = review_dfm(anns, llm=llm)
        self.assertEqual([f.id for f in findings], ["dfm_llm_datum"])

    def test_llm_exception_is_swallowed(self):
        anns = [make_datum("A"), make_fcf()]
        llm = mock.MagicMock(side_effect=RuntimeError("API error"))
        findings = review_dfm(anns, llm=llm)
        self.assertEqual(findings, deterministic_dfm_findings(anns))
        self.assertEqual(len(findings), 1)
        self.assertEqual(findings[0].category, "datum_scheme_completeness")

    def test_llm_empty_choices_equivalent_is_swallowed(self):
        # Upstream vector: OpenAI returns {choices: []} and choices[0] throws.
        # The harness seam is a callable, so an IndexError models it exactly.
        anns = [make_datum("A")]

        def empty_choices(prompt):
            choices = []
            return choices[0]["message"]["content"]

        findings = review_dfm(anns, llm=empty_choices)
        self.assertTrue(
            any(f.category == "datum_scheme_completeness" for f in findings)
        )
        self.assertEqual(findings, deterministic_dfm_findings(anns))

    def test_llm_returning_non_string_is_swallowed(self):
        anns = [make_datum("A")]
        findings = review_dfm(anns, llm=mock.MagicMock(return_value=None))
        self.assertEqual(findings, deterministic_dfm_findings(anns))

    def test_llm_garbage_yields_no_extra_findings(self):
        anns = [make_datum("A")]
        for junk in ("", "I'm sorry, I can't help with that.",
                     "```json\n{oops}\n```", '{"findings": "nope"}'):
            with self.subTest(junk=junk):
                self.assertEqual(
                    review_dfm(anns, llm=mock.MagicMock(return_value=junk)),
                    deterministic_dfm_findings(anns),
                )

    def test_llm_findings_appended_after_deterministic_ones(self):
        anns = [make_datum("A")]
        llm = mock.MagicMock(return_value=_resp(
            [{"id": "z", "category": "general", "severity": "info",
              "description": "d", "recommendation": "r"}]
        ))
        findings = review_dfm(anns, llm=llm)
        self.assertEqual(findings[-1].id, "z")
        self.assertEqual(findings[0].id, "dfm_datum_scheme_completeness")


class DfmFindingTests(unittest.TestCase):
    def test_to_dict_omits_empty_related_ids(self):
        f = DfmFinding(id="i", category="general", severity="info",
                       description="d", recommendation="r")
        self.assertEqual(
            f.to_dict(),
            {"id": "i", "category": "general", "severity": "info",
             "description": "d", "recommendation": "r"},
        )

    def test_to_dict_includes_related_ids_as_list(self):
        f = DfmFinding(id="i", category="general", severity="info",
                       description="d", recommendation="r",
                       related_annotation_ids=("a", "b"))
        self.assertEqual(f.to_dict()["related_annotation_ids"], ["a", "b"])

    def test_findings_are_frozen_and_hashable(self):
        f = DfmFinding(id="i", category="general", severity="info",
                       description="d", recommendation="r")
        with self.assertRaises(Exception):
            f.id = "other"
        self.assertIsInstance(hash(f), int)


if __name__ == "__main__":
    unittest.main()
