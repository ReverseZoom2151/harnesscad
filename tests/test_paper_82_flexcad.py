"""Tests for paper 82 FlexCAD deterministic modules."""

from __future__ import annotations

import random
import unittest

from reconstruction.flexcad_text import (
    CADModel,
    LEVEL_CAD,
    LEVEL_SE,
    LEVEL_SKETCH,
    LEVEL_EXTRUSION,
    LEVEL_FACE,
    LEVEL_LOOP,
    LEVEL_CURVE,
    MaskTarget,
    ParseError,
    ADD,
    CUT,
    SE_MASK,
    LOOP_MASK,
    CURVE_MASK,
    curve,
    loop,
    face,
    sketch,
    extrusion,
    se,
    model,
    serialize,
    parse,
    tokenize,
    mask_field,
    infill,
)
from dataengine.flexcad_masking import (
    enumerate_fields,
    fields_at_level,
    sample_level,
    sample_target,
    mask_all_faces_of_sketch,
    mask_all_loops_of_face,
    mask_curves_of_loop,
    SAMPLING_LEVELS,
)
from dataengine.flexcad_infill_pairs import (
    pair_from_target,
    unconditional_pair,
    sample_pair,
    build_epoch,
    verify_pair,
    INSTRUCTIONS,
)
from bench.flexcad_controllability import (
    model_validity,
    text_validity,
    prediction_validity,
    controllability,
    controllability_rate,
)


def _single_se_model() -> CADModel:
    # A square face (4 lines) + one circular hole, extruded.
    outer = loop(
        curve("line", 0, 0), curve("line", 10, 0),
        curve("line", 10, 10), curve("line", 0, 10),
    )
    hole = loop(curve("circle", 5, 5, 7, 5, 5, 7, 3, 5))
    f = face(outer, hole)
    sk = sketch(f)
    ex = extrusion(ADD, 1, 0, 0, 0, 0, 5)
    return model(se(sk, ex))


def _two_se_model() -> CADModel:
    m1 = _single_se_model().ses[0]
    tri = loop(curve("line", 0, 0), curve("line", 4, 0), curve("line", 2, 3))
    m2 = se(sketch(face(tri)), extrusion(CUT, 2, 0, 0, 0, 0, 3))
    return model(m1, m2)


class TestSerializeRoundTrip(unittest.TestCase):
    def test_round_trip_single(self):
        m = _single_se_model()
        self.assertEqual(parse(serialize(m)), m)

    def test_round_trip_multi(self):
        m = _two_se_model()
        self.assertEqual(parse(serialize(m)), m)

    def test_coords_are_decimal_integers(self):
        m = _single_se_model()
        toks = tokenize(m)
        self.assertIn("31", tokenize(model(se(sketch(face(loop(curve("line", 31, 31)))),
                                              extrusion(ADD, 1, 0, 0, 0, 0, 5)))))
        # curve type tokens appear directly
        self.assertIn("line", toks)
        self.assertIn("circle", toks)

    def test_end_tokens_present(self):
        toks = tokenize(_single_se_model())
        for t in ("<curve_end>", "<loop_end>", "<face_end>",
                  "<sketch_end>", "<extrusion_end>"):
            self.assertIn(t, toks)

    def test_negative_coords_round_trip(self):
        m = model(se(sketch(face(loop(curve("line", -3, -7)))),
                     extrusion(ADD, 1, 0, 0, 0, 0, 5)))
        self.assertEqual(parse(serialize(m)), m)

    def test_bad_type_rejected(self):
        with self.assertRaises(ValueError):
            curve("spline", 0, 0)

    def test_parse_error_on_garbage(self):
        with self.assertRaises(ParseError):
            parse("line 0 0 nonsense")


class TestMasking(unittest.TestCase):
    def test_mask_infill_round_trips_every_level(self):
        m = _two_se_model()
        for target in enumerate_fields(m):
            r = mask_field(m, target)
            rebuilt = infill(r.instruction, r.answer, r.mask)
            self.assertEqual(rebuilt, tokenize(m), f"level={target.level}")

    def test_cad_level_masks_all_ses(self):
        m = _two_se_model()
        r = mask_field(m, MaskTarget(LEVEL_CAD))
        self.assertEqual(r.mask, (SE_MASK, SE_MASK))
        self.assertEqual(r.instruction, (SE_MASK, SE_MASK))

    def test_curve_mask_is_typed(self):
        m = _single_se_model()
        # outer loop has 4 lines
        r = mask_field(m, MaskTarget(LEVEL_CURVE, se=0, face=0, loop=0))
        self.assertEqual(r.mask, tuple(CURVE_MASK["line"] for _ in range(4)))
        # loop_end preserved in instruction (curve level keeps loop structure)
        self.assertIn("<loop_end>", r.instruction)

    def test_loop_mask_single_token(self):
        m = _single_se_model()
        r = mask_field(m, MaskTarget(LEVEL_LOOP, se=0, face=0, loop=1))
        self.assertEqual(r.mask, (LOOP_MASK,))

    def test_enumerate_fields_has_every_level(self):
        m = _two_se_model()
        levels = {t.level for t in enumerate_fields(m)}
        self.assertEqual(levels, set(SAMPLING_LEVELS))

    def test_fields_at_level(self):
        m = _two_se_model()
        self.assertEqual(len(fields_at_level(m, LEVEL_SE)), 2)
        self.assertEqual(len(fields_at_level(m, LEVEL_CAD)), 1)

    def test_multi_face_and_loop_masks_round_trip(self):
        m = _single_se_model()
        r = mask_all_loops_of_face(m, 0, 0)
        self.assertEqual(len(r.mask), 2)  # outer + hole
        self.assertEqual(infill(r.instruction, r.answer, r.mask), tokenize(m))
        r2 = mask_all_faces_of_sketch(m, 0)
        self.assertEqual(len(r2.mask), 1)
        self.assertEqual(infill(r2.instruction, r2.answer, r2.mask), tokenize(m))

    def test_sampling_deterministic(self):
        rng1 = random.Random(7)
        rng2 = random.Random(7)
        m = _two_se_model()
        seq1 = [sample_target(m, rng1).level for _ in range(20)]
        seq2 = [sample_target(m, rng2).level for _ in range(20)]
        self.assertEqual(seq1, seq2)

    def test_sample_level_in_pool(self):
        rng = random.Random(1)
        for _ in range(30):
            self.assertIn(sample_level(rng), SAMPLING_LEVELS)


class TestInfillPairs(unittest.TestCase):
    def test_pair_from_target_verifies(self):
        m = _two_se_model()
        for target in enumerate_fields(m):
            pair = pair_from_target(m, target)
            self.assertTrue(verify_pair(m, pair), target.level)
            self.assertTrue(pair.instruction.startswith(INSTRUCTIONS[target.level]))

    def test_unconditional_pair(self):
        m = _single_se_model()
        pair = unconditional_pair(m)
        self.assertEqual(pair.answer, serialize(m))
        self.assertTrue(verify_pair(m, pair))

    def test_build_epoch_deterministic(self):
        models = [_single_se_model(), _two_se_model()]
        e1 = build_epoch(models, seed=3)
        e2 = build_epoch(models, seed=3)
        self.assertEqual([(p.instruction, p.answer) for p in e1],
                         [(p.instruction, p.answer) for p in e2])
        self.assertEqual(len(e1), 2)

    def test_build_epoch_with_unconditional(self):
        models = [_single_se_model()]
        e = build_epoch(models, seed=1, include_unconditional=True)
        self.assertEqual(len(e), 2)
        self.assertTrue(any(p.level == "unconditional" for p in e))

    def test_sample_pair_verifies(self):
        m = _two_se_model()
        rng = random.Random(5)
        for _ in range(10):
            pair = sample_pair(m, rng)
            self.assertTrue(verify_pair(m, pair))


class TestControllabilityMetrics(unittest.TestCase):
    def test_valid_model_renders(self):
        self.assertTrue(model_validity(_single_se_model()).valid)

    def test_degenerate_extrusion_invalid(self):
        m = model(se(sketch(face(loop(curve("line", 0, 0)))),
                     extrusion(ADD, 0, 0, 0, 0, 0, 0)))
        rep = model_validity(m)
        self.assertFalse(rep.valid)
        self.assertIn("degenerate", rep.reason)

    def test_text_validity_parse_fail(self):
        self.assertFalse(text_validity("line 0 0 broken").valid)

    def test_prediction_validity_fraction(self):
        good = serialize(_single_se_model())
        bad = "line 0 0 broken"
        self.assertAlmostEqual(prediction_validity([good, good, bad]), 2 / 3)

    def test_controllable_edit_preserves_surroundings(self):
        m = _two_se_model()
        target = MaskTarget(LEVEL_EXTRUSION, se=1)
        # Edit only se1's extrusion params.
        edited_se1 = se(m.ses[1].sketch, extrusion(CUT, 9, 9, 0, 0, 0, 4))
        predicted = model(m.ses[0], edited_se1)
        rep = controllability(m, target, predicted)
        self.assertTrue(rep.preserved)
        self.assertTrue(rep.changed)
        self.assertTrue(rep.controllable)

    def test_uncontrollable_change_outside_field_detected(self):
        m = _two_se_model()
        target = MaskTarget(LEVEL_EXTRUSION, se=1)
        # Wrongly also change se0 (outside the masked field).
        bad_se0 = se(m.ses[0].sketch, extrusion(ADD, 8, 8, 0, 0, 0, 9))
        predicted = model(bad_se0, m.ses[1])
        rep = controllability(m, target, predicted)
        self.assertFalse(rep.preserved)
        self.assertFalse(rep.controllable)

    def test_no_change_is_not_controllable_edit(self):
        m = _two_se_model()
        target = MaskTarget(LEVEL_EXTRUSION, se=1)
        rep = controllability(m, target, m)  # identical
        self.assertTrue(rep.preserved)
        self.assertFalse(rep.changed)
        self.assertFalse(rep.controllable)

    def test_controllability_rate_aggregates(self):
        m = _two_se_model()
        target = MaskTarget(LEVEL_EXTRUSION, se=1)
        good = model(m.ses[0], se(m.ses[1].sketch, extrusion(CUT, 9, 0, 0, 0, 0, 4)))
        cases = [(m, target, good), (m, target, m)]
        rates = controllability_rate(cases)
        self.assertAlmostEqual(rates["controllable"], 0.5)
        self.assertAlmostEqual(rates["preserved"], 1.0)
        self.assertAlmostEqual(rates["pv"], 1.0)


if __name__ == "__main__":
    unittest.main()
