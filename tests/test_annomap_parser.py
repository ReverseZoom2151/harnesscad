import unittest

from annomap_parser import (
    CADFeature,
    DrawingEntity,
    ENTITY_ANGLE,
    ENTITY_COUNTERBORE,
    ENTITY_DATUM,
    ENTITY_DIAMETER,
    ENTITY_GDT,
    ENTITY_LINEAR,
    ENTITY_NOTE,
    ENTITY_RADIUS,
    ENTITY_SURFACE_FINISH,
    ENTITY_THREAD,
    Tolerance,
    is_known_entity_type,
    parse_callout,
    parse_entities,
    parse_gdt_frame,
)


class ToleranceTests(unittest.TestCase):
    def test_symmetric_and_bounds(self):
        t = Tolerance(0.1, 0.1)
        self.assertTrue(t.is_symmetric)
        self.assertAlmostEqual(t.width, 0.2)
        self.assertEqual(t.bounds(10.0), (9.9, 10.1))

    def test_asymmetric(self):
        t = Tolerance(0.2, 0.05)
        self.assertFalse(t.is_symmetric)
        self.assertEqual(t.bounds(10.0), (9.95, 10.2))


class CalloutParseTests(unittest.TestCase):
    def test_diameter_symbol(self):
        e = parse_callout("Ø10")
        self.assertEqual(e.entity_type, ENTITY_DIAMETER)
        self.assertEqual(e.value, 10.0)
        self.assertEqual(e.symbol, "Ø")
        self.assertEqual(e.target_feature, "hole")

    def test_diameter_text_form(self):
        e = parse_callout("DIA 12.5")
        self.assertEqual(e.entity_type, ENTITY_DIAMETER)
        self.assertEqual(e.value, 12.5)

    def test_radius(self):
        e = parse_callout("R5")
        self.assertEqual(e.entity_type, ENTITY_RADIUS)
        self.assertEqual(e.value, 5.0)
        self.assertEqual(e.target_feature, "fillet")

    def test_thread_with_pitch(self):
        e = parse_callout("M8x1.25")
        self.assertEqual(e.entity_type, ENTITY_THREAD)
        self.assertEqual(e.value, 8.0)
        self.assertEqual(e.extra["pitch"], 1.25)
        self.assertEqual(e.target_feature, "hole")

    def test_thread_no_pitch(self):
        e = parse_callout("M10")
        self.assertEqual(e.entity_type, ENTITY_THREAD)
        self.assertEqual(e.value, 10.0)
        self.assertIsNone(e.extra["pitch"])

    def test_pattern_multiplicity(self):
        e = parse_callout("4X Ø6.5")
        self.assertEqual(e.entity_type, ENTITY_DIAMETER)
        self.assertEqual(e.multiplicity, 4)
        self.assertEqual(e.value, 6.5)

    def test_symmetric_tolerance(self):
        e = parse_callout("Ø10 ±0.1")
        self.assertEqual(e.value, 10.0)
        self.assertTrue(e.tolerance.is_symmetric)
        self.assertAlmostEqual(e.tolerance.plus, 0.1)

    def test_bilateral_tolerance(self):
        e = parse_callout("20 +0.2 -0.05")
        self.assertEqual(e.entity_type, ENTITY_LINEAR)
        self.assertAlmostEqual(e.tolerance.plus, 0.2)
        self.assertAlmostEqual(e.tolerance.minus, 0.05)
        self.assertFalse(e.tolerance.is_symmetric)

    def test_angle(self):
        e = parse_callout("45°")
        self.assertEqual(e.entity_type, ENTITY_ANGLE)
        self.assertEqual(e.value, 45.0)
        self.assertEqual(e.unit, "deg")

    def test_surface_finish(self):
        e = parse_callout("Ra 3.2")
        self.assertEqual(e.entity_type, ENTITY_SURFACE_FINISH)
        self.assertEqual(e.value, 3.2)
        self.assertEqual(e.extra["parameter"], "ra")

    def test_counterbore(self):
        e = parse_callout("CBORE 12")
        self.assertEqual(e.entity_type, ENTITY_COUNTERBORE)
        self.assertEqual(e.value, 12.0)

    def test_datum(self):
        e = parse_callout("DATUM A")
        self.assertEqual(e.entity_type, ENTITY_DATUM)
        self.assertEqual(e.symbol, "A")

    def test_plain_linear(self):
        e = parse_callout("25")
        self.assertEqual(e.entity_type, ENTITY_LINEAR)
        self.assertEqual(e.value, 25.0)
        self.assertIsNone(e.target_feature)

    def test_free_note(self):
        e = parse_callout("BREAK ALL EDGES")
        self.assertEqual(e.entity_type, ENTITY_NOTE)
        self.assertIsNone(e.value)

    def test_context_and_id_preserved(self):
        e = parse_callout("Ø10", entity_id="E7", context={"view": "front"})
        self.assertEqual(e.entity_id, "E7")
        self.assertEqual(e.context["view"], "front")

    def test_to_dict_roundtrips_keys(self):
        e = parse_callout("Ø10 ±0.1", entity_id="E1")
        d = e.to_dict()
        self.assertEqual(d["entity_type"], ENTITY_DIAMETER)
        self.assertEqual(d["tolerance"], {"plus": 0.1, "minus": 0.1})


class GdtFrameTests(unittest.TestCase):
    def test_position_frame_text(self):
        f = parse_gdt_frame("POSITION Ø0.2 M A B C")
        self.assertEqual(f["symbol"], "position")
        self.assertAlmostEqual(f["tolerance"], 0.2)
        self.assertTrue(f["diametral_zone"])
        self.assertEqual(f["modifier"], "MMC")
        self.assertEqual(f["datums"], ["A", "B", "C"])

    def test_flatness_datumless(self):
        f = parse_gdt_frame("FLATNESS 0.05")
        self.assertEqual(f["symbol"], "flatness")
        self.assertEqual(f["datums"], [])

    def test_total_runout_beats_runout(self):
        f = parse_gdt_frame("TOTAL RUNOUT 0.1 A")
        self.assertEqual(f["symbol"], "total_runout")

    def test_non_gdt_returns_none(self):
        self.assertIsNone(parse_gdt_frame("Ø10"))


class BatchTests(unittest.TestCase):
    def test_parse_entities_ids(self):
        ents = parse_entities(["Ø10", "R5", "M8"])
        self.assertEqual([e.entity_id for e in ents], ["E1", "E2", "E3"])
        self.assertEqual(ents[2].entity_type, ENTITY_THREAD)

    def test_gdt_entity_type(self):
        e = parse_callout("POSITION 0.1 A")
        self.assertEqual(e.entity_type, ENTITY_GDT)

    def test_known_type(self):
        self.assertTrue(is_known_entity_type(ENTITY_DIAMETER))
        self.assertFalse(is_known_entity_type("bogus"))


class FeatureSchemaTests(unittest.TestCase):
    def test_feature_to_dict(self):
        f = CADFeature("hole", {"diameter": 10.0}, 0.9, feature_id="F1",
                       centroid=(1.0, 2.0, 3.0))
        d = f.to_dict()
        self.assertEqual(d["feature_type"], "hole")
        self.assertEqual(d["centroid"], [1.0, 2.0, 3.0])


if __name__ == "__main__":
    unittest.main()
