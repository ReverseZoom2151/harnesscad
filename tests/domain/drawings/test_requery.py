# Tests for harnesscad.domain.drawings.requery.
#
# Known-bad vectors and invariants ported from the CAD-Annotator reference repo
# (artifacts/api-server/src/lib/requery-service.test.ts), Apache-2.0.
# Credit: CAD-Annotator contributors.
#
# DIVERGENCES FROM UPSTREAM (per the harness module docstring):
#  1. Upstream owns image cropping (sharp) and calls OpenAI directly. The
#     harness injects the whole impure step as an optional ``vision`` callable
#     ``(CropRegion, prompt) -> raw text``; with vision=None, low-confidence
#     annotations are merely flagged needs_review (was_requeried False). That
#     no-vision path has no upstream counterpart.
#  2. Upstream's property tests use fast-check. hypothesis is NOT installed
#     here and must not be added, so those properties are re-expressed as
#     table-driven exhaustive sweeps over small domains (itertools) plus a
#     fixed-seed random sweep with random.Random(SEED). SEED = 20260719.
#  3. Merge is snake_case-base + camelCase-override (upstream is a plain TS
#     object spread), and the harness additionally forces the original id and
#     bounding box back on and re-validates through parse_annotation, so an
#     unparseable merge degrades to "original flagged needs_review".

import itertools
import json
import math
import random
import unittest
from unittest import mock

from harnesscad.domain.drawings.annotation_schema import (
    BoundingBox,
    DatumAnnotation,
    DimensionAnnotation,
    NoteAnnotation,
)
from harnesscad.domain.drawings.gdt_prompts import build_focused_requery_prompt
from harnesscad.domain.drawings.requery import (
    CROP_PADDING,
    REQUERY_CONFIDENCE_THRESHOLD,
    CropRegion,
    ReQueryResult,
    apply_requery_decision,
    compute_crop_region,
    requery_low_confidence,
)

SEED = 20260719


def make_note(ann_id="ann-1", confidence=0.5, needs_review=False, bbox=None):
    return NoteAnnotation(
        id=ann_id,
        label="Test Note",
        value="Some note text",
        view="front",
        bounding_box=bbox or BoundingBox(x=10, y=10, width=5, height=5,
                                         color="#00FF00"),
        confidence=confidence,
        needs_review=needs_review,
    )


def make_datum(ann_id="ann_2", letter="Z", confidence=0.4):
    return DatumAnnotation(
        id=ann_id,
        label="Datum ?",
        value="?",
        view="Front View",
        bounding_box=BoundingBox(x=20, y=30, width=10, height=10, color="red"),
        confidence=confidence,
        datum_letter=letter,
    )


# --------------------------------------------------------------------------- #
# Crop geometry (upstream Property 4)
# --------------------------------------------------------------------------- #


class ComputeCropRegionTests(unittest.TestCase):
    def test_known_vector_from_the_module_selfcheck(self):
        # bbox 95,95,10,10 percent on a 1000x800 image, spilling past 100 pct.
        crop = compute_crop_region(
            BoundingBox(x=95, y=95, width=10, height=10, color="red"), 1000, 800
        )
        self.assertEqual(crop.left, 935)
        self.assertEqual(crop.top, 748)
        self.assertLessEqual(crop.left + crop.width, 1000)
        self.assertLessEqual(crop.top + crop.height, 800)

    def test_padding_expands_by_fifteen_percent_of_the_box(self):
        # 10,10,20,20 pct on 1000x1000 -> box 100..300 px, pad 30 px each side.
        crop = compute_crop_region(
            BoundingBox(x=10, y=10, width=20, height=20, color="g"), 1000, 1000
        )
        self.assertEqual(crop, CropRegion(left=70, top=70, width=260, height=260))

    def test_zero_padding_is_the_unpadded_floor_ceil_box(self):
        crop = compute_crop_region(
            BoundingBox(x=10, y=10, width=20, height=20, color="g"),
            1000, 1000, padding=0.0,
        )
        self.assertEqual(crop, CropRegion(left=100, top=100, width=200, height=200))

    def test_left_and_top_clamped_at_zero(self):
        crop = compute_crop_region(
            BoundingBox(x=0, y=0, width=5, height=5, color="g"), 1000, 800
        )
        self.assertEqual(crop.left, 0)
        self.assertEqual(crop.top, 0)

    def test_minimum_one_pixel_in_each_dimension(self):
        crop = compute_crop_region(
            BoundingBox(x=0.0, y=0.0, width=0.1, height=0.1, color="g"), 1, 1
        )
        self.assertGreaterEqual(crop.width, 1)
        self.assertGreaterEqual(crop.height, 1)

    def test_to_dict(self):
        crop = CropRegion(left=1, top=2, width=3, height=4)
        self.assertEqual(crop.to_dict(),
                         {"left": 1, "top": 2, "width": 3, "height": 4})

    def test_all_fields_are_integers(self):
        crop = compute_crop_region(
            BoundingBox(x=33.3, y=66.6, width=7.7, height=1.1, color="g"), 999, 777
        )
        for v in (crop.left, crop.top, crop.width, crop.height):
            self.assertIsInstance(v, int)

    def _assert_crop_properties(self, bbox, img_w, img_h):
        crop = compute_crop_region(bbox, img_w, img_h, CROP_PADDING)

        bbox_left_px = (bbox.x / 100.0) * img_w
        bbox_top_px = (bbox.y / 100.0) * img_h
        bbox_right_px = bbox_left_px + (bbox.width / 100.0) * img_w
        bbox_bottom_px = bbox_top_px + (bbox.height / 100.0) * img_h

        ctx = (bbox, img_w, img_h, crop)
        # (a) the crop contains the original bounding box (clipped to image)
        self.assertLessEqual(crop.left, math.floor(bbox_left_px), ctx)
        self.assertLessEqual(crop.top, math.floor(bbox_top_px), ctx)
        self.assertGreaterEqual(
            crop.left + crop.width, min(math.ceil(bbox_right_px), img_w), ctx
        )
        self.assertGreaterEqual(
            crop.top + crop.height, min(math.ceil(bbox_bottom_px), img_h), ctx
        )
        # (b) clamped within image bounds
        self.assertGreaterEqual(crop.left, 0, ctx)
        self.assertGreaterEqual(crop.top, 0, ctx)
        self.assertLessEqual(crop.left + crop.width, img_w, ctx)
        self.assertLessEqual(crop.top + crop.height, img_h, ctx)
        # (c) positive dimensions
        self.assertGreater(crop.width, 0, ctx)
        self.assertGreater(crop.height, 0, ctx)

    def test_crop_containment_over_an_exhaustive_small_grid(self):
        # Upstream Property 4, re-expressed as an exhaustive sweep over a small
        # domain because hypothesis is unavailable. 4*4*3*3*4 = 576 cases.
        for x, y, w, h, (iw, ih) in itertools.product(
            (0.0, 0.1, 50.0, 99.0),
            (0.0, 0.1, 50.0, 99.0),
            (0.1, 20.0, 100.0),
            (0.1, 20.0, 100.0),
            ((1, 1), (1, 10000), (10000, 1), (1024, 768)),
        ):
            with self.subTest(x=x, y=y, w=w, h=h, img=(iw, ih)):
                self._assert_crop_properties(
                    BoundingBox(x=x, y=y, width=w, height=h, color="g"), iw, ih
                )

    def test_crop_containment_over_a_fixed_seed_random_sweep(self):
        # Substitutes for fast-check's 500 random runs. Seed: 20260719.
        rng = random.Random(SEED)
        for _ in range(500):
            bbox = BoundingBox(
                x=rng.uniform(0, 99),
                y=rng.uniform(0, 99),
                width=rng.uniform(0.1, 100),
                height=rng.uniform(0.1, 100),
                color="g",
            )
            self._assert_crop_properties(
                bbox, rng.randint(1, 10000), rng.randint(1, 10000)
            )

    def test_crop_padding_constant(self):
        self.assertEqual(CROP_PADDING, 0.15)


# --------------------------------------------------------------------------- #
# Decision rule (upstream Property 9)
# --------------------------------------------------------------------------- #


class ApplyRequeryDecisionTests(unittest.TestCase):
    def test_threshold_constant(self):
        self.assertEqual(REQUERY_CONFIDENCE_THRESHOLD, 0.6)

    def test_confident_requery_replaces_original_verbatim(self):
        original = make_note("orig-1", confidence=0.2)
        requery = make_note("requery-1", confidence=0.9)
        result = apply_requery_decision(original, requery)
        self.assertIs(result, requery)
        self.assertFalse(result.needs_review)

    def test_requery_exactly_at_threshold_replaces(self):
        original = make_note("orig-1", confidence=0.55)
        requery = make_note("requery-1", confidence=0.6)
        result = apply_requery_decision(original, requery)
        self.assertEqual(result.id, "requery-1")
        self.assertFalse(result.needs_review)

    def test_weak_requery_with_higher_confidence_wins_and_flags(self):
        original = make_note("orig-1", confidence=0.3)
        requery = make_note("requery-1", confidence=0.5)
        result = apply_requery_decision(original, requery)
        self.assertEqual(result.id, "requery-1")
        self.assertEqual(result.confidence, 0.5)
        self.assertTrue(result.needs_review)

    def test_weak_requery_with_lower_confidence_keeps_original_flagged(self):
        original = make_note("orig-1", confidence=0.5)
        requery = make_note("requery-1", confidence=0.2)
        result = apply_requery_decision(original, requery)
        self.assertEqual(result.id, "orig-1")
        self.assertEqual(result.confidence, 0.5)
        self.assertTrue(result.needs_review)

    def test_tie_below_threshold_favours_the_requery(self):
        # Documented tie-break, matching the TS >= comparison.
        original = make_note("orig-1", confidence=0.4)
        requery = make_note("requery-1", confidence=0.4)
        result = apply_requery_decision(original, requery)
        self.assertEqual(result.id, "requery-1")
        self.assertTrue(result.needs_review)

    def test_decision_over_an_exhaustive_confidence_grid(self):
        # Upstream Property 9 (a)+(b)+(c) as an exhaustive sweep; hypothesis
        # is unavailable so the domain is a fixed 11x11 confidence grid.
        grid = [i / 10.0 for i in range(11)]
        for orig_conf, req_conf in itertools.product(grid, grid):
            with self.subTest(orig=orig_conf, req=req_conf):
                original = make_note("orig-1", confidence=orig_conf)
                requery = make_note("requery-1", confidence=req_conf)
                result = apply_requery_decision(original, requery)

                if req_conf >= REQUERY_CONFIDENCE_THRESHOLD:
                    self.assertEqual(result.confidence, req_conf)
                    self.assertEqual(result.id, "requery-1")
                    self.assertFalse(result.needs_review)
                else:
                    self.assertEqual(result.confidence,
                                     max(orig_conf, req_conf))
                    self.assertTrue(result.needs_review)
                    self.assertEqual(
                        result.id,
                        "requery-1" if req_conf >= orig_conf else "orig-1",
                    )
                # (c) output confidence never below the minimum input
                self.assertGreaterEqual(result.confidence,
                                        min(orig_conf, req_conf))

    def test_decision_over_a_fixed_seed_random_sweep(self):
        # Substitutes for fast-check's 500 random runs. Seed: 20260719.
        rng = random.Random(SEED)
        for _ in range(500):
            orig_conf = rng.uniform(0.0, 1.0)
            req_conf = rng.uniform(0.0, 1.0)
            result = apply_requery_decision(
                make_note("orig-1", confidence=orig_conf),
                make_note("requery-1", confidence=req_conf),
            )
            self.assertGreaterEqual(result.confidence,
                                    min(orig_conf, req_conf))
            if req_conf >= REQUERY_CONFIDENCE_THRESHOLD:
                self.assertEqual(result.confidence, req_conf)
            else:
                self.assertEqual(result.confidence, max(orig_conf, req_conf))
                self.assertTrue(result.needs_review)

    def test_original_is_not_mutated(self):
        original = make_note("orig-1", confidence=0.5)
        apply_requery_decision(original, make_note("requery-1", confidence=0.1))
        self.assertFalse(original.needs_review)


# --------------------------------------------------------------------------- #
# requery_low_confidence: no-vision path (harness-only)
# --------------------------------------------------------------------------- #


class RequeryWithoutVisionTests(unittest.TestCase):
    def test_high_confidence_untouched(self):
        ann = make_note("hi", confidence=0.95)
        results = requery_low_confidence([ann], 1000, 800)
        self.assertEqual(len(results), 1)
        self.assertIs(results[0].annotation, ann)
        self.assertFalse(results[0].was_requeried)
        self.assertFalse(results[0].annotation.needs_review)

    def test_annotation_exactly_at_threshold_is_not_requeried(self):
        ann = make_note("edge", confidence=REQUERY_CONFIDENCE_THRESHOLD)
        results = requery_low_confidence([ann], 100, 100)
        self.assertIs(results[0].annotation, ann)
        self.assertFalse(results[0].was_requeried)

    def test_low_confidence_flagged_but_not_marked_requeried(self):
        ann = make_note("lo", confidence=0.4)
        results = requery_low_confidence([ann], 1000, 800)
        self.assertTrue(results[0].annotation.needs_review)
        self.assertFalse(results[0].was_requeried)
        self.assertEqual(results[0].annotation.confidence, 0.4)
        self.assertEqual(results[0].annotation.id, "lo")

    def test_empty_input(self):
        self.assertEqual(requery_low_confidence([], 100, 100), [])

    def test_order_and_length_preserved(self):
        anns = [make_note("a", 0.9), make_note("b", 0.1), make_note("c", 0.7)]
        results = requery_low_confidence(anns, 500, 500)
        self.assertEqual([r.annotation.id for r in results], ["a", "b", "c"])


# --------------------------------------------------------------------------- #
# requery_low_confidence: vision path (mocked; never hits a network)
# --------------------------------------------------------------------------- #


class RequeryWithVisionTests(unittest.TestCase):
    def test_vision_not_called_for_high_confidence(self):
        vision = mock.MagicMock(return_value="{}")
        requery_low_confidence([make_note("hi", 0.95)], 100, 100, vision=vision)
        vision.assert_not_called()

    def test_vision_called_once_with_crop_and_focused_prompt(self):
        ann = make_datum("ann_2", "Z", confidence=0.4)
        vision = mock.MagicMock(return_value=json.dumps(
            {"type": "datum", "label": "Datum B", "value": "B",
             "confidence": 0.92, "datumLetter": "B"}
        ))
        requery_low_confidence([ann], 1000, 800, vision=vision)
        vision.assert_called_once()
        crop_arg, prompt_arg = vision.call_args[0]
        self.assertEqual(
            crop_arg, compute_crop_region(ann.bounding_box, 1000, 800, CROP_PADDING)
        )
        self.assertEqual(
            prompt_arg,
            build_focused_requery_prompt(ann.type, ann.label, ann.value),
        )

    def test_confident_requery_replaces_but_preserves_id_and_bbox(self):
        ann = make_datum("ann_2", "Z", confidence=0.4)
        vision = mock.MagicMock(return_value=(
            "```json\n" + json.dumps(
                {"type": "datum", "label": "Datum B", "value": "B",
                 "confidence": 0.92, "datumLetter": "B",
                 "id": "should_be_ignored",
                 "boundingBox": {"x": 1, "y": 1, "width": 1, "height": 1}}
            ) + "\n```"
        ))
        result = requery_low_confidence([ann], 1000, 800, vision=vision)[0]
        self.assertTrue(result.was_requeried)
        self.assertEqual(result.annotation.id, "ann_2")
        self.assertEqual(result.annotation.value, "B")
        self.assertEqual(result.annotation.datum_letter, "B")
        self.assertEqual(result.annotation.confidence, 0.92)
        self.assertEqual(result.annotation.bounding_box, ann.bounding_box)
        self.assertFalse(result.annotation.needs_review)

    def test_weak_requery_keeps_higher_confidence_reading_flagged(self):
        ann = make_datum("ann_3", "Y", confidence=0.3)
        vision = mock.MagicMock(return_value=json.dumps(
            {"type": "datum", "label": "Datum ?", "value": "?",
             "confidence": 0.35, "datumLetter": "Y"}
        ))
        result = requery_low_confidence([ann], 1000, 800, vision=vision)[0]
        self.assertTrue(result.was_requeried)
        self.assertTrue(result.annotation.needs_review)
        self.assertEqual(result.annotation.confidence, 0.35)
        self.assertEqual(result.annotation.id, "ann_3")

    def test_weak_requery_below_the_original_keeps_the_original(self):
        ann = make_datum("ann_4", "Y", confidence=0.55)
        vision = mock.MagicMock(return_value=json.dumps(
            {"type": "datum", "label": "junk", "value": "j",
             "confidence": 0.05, "datumLetter": "Q"}
        ))
        result = requery_low_confidence([ann], 1000, 800, vision=vision)[0]
        self.assertTrue(result.was_requeried)
        self.assertTrue(result.annotation.needs_review)
        self.assertEqual(result.annotation.datum_letter, "Y")
        self.assertEqual(result.annotation.confidence, 0.55)

    # -- KNOWN-BAD VECTORS ------------------------------------------------- #

    def test_unparseable_responses_degrade_to_flagged_original(self):
        bad_vectors = [
            "",                                   # empty response
            "I'm sorry, I cannot read this.",     # refusal prose, no JSON
            "```json\n{not valid json}\n```",     # fenced but broken
            '{"type": "datum", "confidence": 0.9',  # truncated
            '["not", "an", "object"]',            # JSON array, not object
            '{"type": "datum", "datumLetter": "lowercase"}',  # fails validation
            '{"type": "bogus_type", "confidence": 0.9}',      # unknown type
        ]
        for content in bad_vectors:
            with self.subTest(content=content):
                ann = make_datum("ann_2", "Z", confidence=0.4)
                result = requery_low_confidence(
                    [ann], 1000, 800, vision=mock.MagicMock(return_value=content)
                )[0]
                self.assertTrue(result.was_requeried)
                self.assertTrue(result.annotation.needs_review)
                self.assertEqual(result.annotation.id, "ann_2")
                self.assertEqual(result.annotation.datum_letter, "Z")
                self.assertEqual(result.annotation.confidence, 0.4)

    def test_vision_exception_degrades_to_flagged_original(self):
        ann = make_datum("ann_2", "Z", confidence=0.4)
        vision = mock.MagicMock(side_effect=RuntimeError("vision API down"))
        result = requery_low_confidence([ann], 1000, 800, vision=vision)[0]
        self.assertTrue(result.was_requeried)
        self.assertTrue(result.annotation.needs_review)
        self.assertEqual(result.annotation.datum_letter, "Z")

    def test_vision_returning_non_string_degrades(self):
        # extract_json_blob would raise on a non-str; the loop swallows it.
        ann = make_datum("ann_2", "Z", confidence=0.4)
        result = requery_low_confidence(
            [ann], 1000, 800, vision=mock.MagicMock(return_value=12345)
        )[0]
        self.assertTrue(result.was_requeried)
        self.assertTrue(result.annotation.needs_review)

    def test_empty_choices_equivalent_degrades(self):
        # Upstream vector: OpenAI returns {choices: []}; choices[0] throws.
        ann = make_datum("ann_2", "Z", confidence=0.4)

        def empty_choices(crop, prompt):
            choices = []
            return choices[0]["message"]["content"]

        result = requery_low_confidence([ann], 1000, 800, vision=empty_choices)[0]
        self.assertTrue(result.was_requeried)
        self.assertTrue(result.annotation.needs_review)

    def test_missing_confidence_falls_back_to_the_original(self):
        ann = make_datum("ann_2", "Z", confidence=0.4)
        vision = mock.MagicMock(return_value=json.dumps(
            {"type": "datum", "label": "Datum B", "value": "B",
             "datumLetter": "B"}
        ))
        result = requery_low_confidence([ann], 1000, 800, vision=vision)[0]
        self.assertEqual(result.annotation.confidence, 0.4)
        self.assertTrue(result.annotation.needs_review)
        self.assertEqual(result.annotation.datum_letter, "B")

    def test_non_numeric_confidence_falls_back_to_the_original(self):
        for bad_conf in ("high", None, True, [0.9], {"v": 0.9}):
            with self.subTest(confidence=bad_conf):
                ann = make_datum("ann_2", "Z", confidence=0.4)
                vision = mock.MagicMock(return_value=json.dumps(
                    {"type": "datum", "label": "Datum B", "value": "B",
                     "confidence": bad_conf, "datumLetter": "B"}
                ))
                result = requery_low_confidence(
                    [ann], 1000, 800, vision=vision
                )[0]
                self.assertEqual(result.annotation.confidence, 0.4)

    def test_out_of_range_confidence_is_clamped_by_the_tolerant_parser(self):
        ann = make_datum("ann_2", "Z", confidence=0.4)
        vision = mock.MagicMock(return_value=json.dumps(
            {"type": "datum", "label": "Datum B", "value": "B",
             "confidence": 7.5, "datumLetter": "B"}
        ))
        result = requery_low_confidence([ann], 1000, 800, vision=vision)[0]
        self.assertEqual(result.annotation.confidence, 1.0)

    def test_bad_bounding_box_in_the_response_is_ignored(self):
        ann = make_datum("ann_2", "Z", confidence=0.4)
        vision = mock.MagicMock(return_value=json.dumps(
            {"type": "datum", "label": "Datum B", "value": "B",
             "confidence": 0.9, "datumLetter": "B",
             "boundingBox": {"x": "nope", "width": -5}}
        ))
        result = requery_low_confidence([ann], 1000, 800, vision=vision)[0]
        self.assertEqual(result.annotation.bounding_box, ann.bounding_box)
        self.assertEqual(result.annotation.confidence, 0.9)

    def test_non_string_type_in_the_response_keeps_the_original_type(self):
        ann = make_datum("ann_2", "Z", confidence=0.4)
        vision = mock.MagicMock(return_value=json.dumps(
            {"type": 42, "label": "Datum B", "value": "B",
             "confidence": 0.9, "datumLetter": "B"}
        ))
        result = requery_low_confidence([ann], 1000, 800, vision=vision)[0]
        self.assertEqual(result.annotation.type, "datum")
        self.assertEqual(result.annotation.value, "B")

    def test_type_change_is_honoured_when_the_new_type_validates(self):
        ann = make_datum("ann_2", "Z", confidence=0.4)
        vision = mock.MagicMock(return_value=json.dumps(
            {"type": "dimension", "label": "40.2", "value": "40.2",
             "confidence": 0.9, "dimensionType": "linear", "nominalValue": 40.2}
        ))
        result = requery_low_confidence([ann], 1000, 800, vision=vision)[0]
        self.assertIsInstance(result.annotation, DimensionAnnotation)
        self.assertEqual(result.annotation.id, "ann_2")
        self.assertEqual(result.annotation.bounding_box, ann.bounding_box)

    def test_empty_original_id_is_restored_after_reparse(self):
        # The merged dict is re-validated through parse_annotation, which
        # substitutes "ann_1" for a falsy id. The final dc_replace must put
        # the (empty) original id back, so id is preserved verbatim.
        ann = DatumAnnotation(
            id="", label="Datum ?", value="?", view="Front View",
            bounding_box=BoundingBox(x=20, y=30, width=10, height=10,
                                     color="red"),
            confidence=0.4, datum_letter="Z",
        )
        vision = mock.MagicMock(return_value=json.dumps(
            {"type": "datum", "label": "Datum B", "value": "B",
             "confidence": 0.92, "datumLetter": "B"}
        ))
        result = requery_low_confidence([ann], 1000, 800, vision=vision)[0]
        self.assertEqual(result.annotation.id, "")
        self.assertEqual(result.annotation.value, "B")

    def test_out_of_range_original_bbox_is_restored_verbatim(self):
        # validate_bounding_box clamps x/y to 0-100 and width/height to
        # 0.1-100 during the re-parse; the final dc_replace must restore the
        # ORIGINAL (unclamped) bounding box object, not the clamped one.
        odd_bbox = BoundingBox(x=150.0, y=-5.0, width=0.01, height=250.0,
                               color="red")
        ann = DatumAnnotation(
            id="ann_2", label="Datum ?", value="?", view="Front View",
            bounding_box=odd_bbox, confidence=0.4, datum_letter="Z",
        )
        vision = mock.MagicMock(return_value=json.dumps(
            {"type": "datum", "label": "Datum B", "value": "B",
             "confidence": 0.92, "datumLetter": "B"}
        ))
        result = requery_low_confidence([ann], 1000, 800, vision=vision)[0]
        self.assertEqual(result.annotation.bounding_box, odd_bbox)
        self.assertEqual(result.annotation.bounding_box.x, 150.0)
        self.assertEqual(result.annotation.bounding_box.width, 0.01)

    def test_one_call_per_low_confidence_annotation_only(self):
        anns = [make_note("a", 0.9), make_datum("b", "B", 0.1),
                make_note("c", 0.2), make_note("d", 0.7)]
        vision = mock.MagicMock(return_value="no json")
        results = requery_low_confidence(anns, 500, 500, vision=vision)
        self.assertEqual(vision.call_count, 2)
        self.assertEqual([r.was_requeried for r in results],
                         [False, True, True, False])

    def test_no_retry_on_failure(self):
        # At most ONE attempt per annotation, even when the call blows up.
        vision = mock.MagicMock(side_effect=RuntimeError("boom"))
        requery_low_confidence([make_datum("x", "A", 0.1)], 100, 100,
                               vision=vision)
        self.assertEqual(vision.call_count, 1)


class ReQueryResultTests(unittest.TestCase):
    def test_to_dict(self):
        ann = make_note("a", 0.9)
        d = ReQueryResult(annotation=ann, was_requeried=False).to_dict()
        self.assertEqual(d["was_requeried"], False)
        self.assertEqual(d["annotation"]["id"], "a")
        self.assertEqual(d["annotation"]["type"], "note")


if __name__ == "__main__":
    unittest.main()
