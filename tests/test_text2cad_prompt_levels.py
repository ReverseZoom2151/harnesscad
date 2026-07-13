import unittest

from harnesscad.data.dataengine.annotation.prompt_levels import (
    ASPECT_EXTRUSION,
    ASPECT_SHAPE,
    ASPECT_SKETCH,
    LEVELS,
    N_LEVELS,
    PRECISION_NONE,
    PRECISION_PRECISE,
    PromptLevelError,
    classify_prompt_level,
    extract_signals,
    is_more_detailed,
    level,
    level_by_index,
    level_matches,
    ordered_codes,
)


class TaxonomyStructureTests(unittest.TestCase):
    def test_four_ordered_levels(self):
        self.assertEqual(N_LEVELS, 4)
        self.assertEqual(ordered_codes(), ("L0", "L1", "L2", "L3"))
        self.assertEqual([lv.index for lv in LEVELS], [0, 1, 2, 3])

    def test_lookup_by_code_and_index(self):
        self.assertIs(level("l2"), level_by_index(2))
        self.assertEqual(level("L0").name, "Abstract")
        self.assertEqual(level("L3").name, "Expert")

    def test_unknown_lookups_raise(self):
        with self.assertRaises(PromptLevelError):
            level("L9")
        with self.assertRaises(PromptLevelError):
            level_by_index(7)

    def test_abstract_is_vlm_shape_only(self):
        l0 = level("L0")
        self.assertEqual(l0.source, "vlm")
        self.assertEqual(l0.aspects, (ASPECT_SHAPE,))
        self.assertFalse(l0.includes(ASPECT_SKETCH))
        self.assertEqual(l0.precision, PRECISION_NONE)

    def test_expert_carries_full_detail(self):
        l3 = level("L3")
        self.assertTrue(l3.includes(ASPECT_SHAPE))
        self.assertTrue(l3.includes(ASPECT_SKETCH))
        self.assertTrue(l3.includes(ASPECT_EXTRUSION))
        self.assertEqual(l3.precision, PRECISION_PRECISE)
        self.assertTrue(l3.relative_values)
        self.assertTrue(l3.uses_jargon)

    def test_beginner_has_no_jargon_or_measurements(self):
        l1 = level("L1")
        self.assertFalse(l1.uses_jargon)
        self.assertFalse(l1.relative_values)
        self.assertEqual(l1.precision, PRECISION_NONE)

    def test_precision_is_monotone_with_detail(self):
        ranks = [lv.precision_rank for lv in LEVELS]
        self.assertEqual(ranks, sorted(ranks))

    def test_intermediate_adds_sketch_and_extrusion(self):
        l1, l2 = level("L1"), level("L2")
        self.assertNotIn(ASPECT_SKETCH, l1.aspects)
        self.assertIn(ASPECT_SKETCH, l2.aspects)
        self.assertIn(ASPECT_EXTRUSION, l2.aspects)

    def test_is_more_detailed(self):
        self.assertTrue(is_more_detailed("L3", "L0"))
        self.assertFalse(is_more_detailed("L1", "L2"))
        self.assertFalse(is_more_detailed("L2", "L2"))


class SignalExtractionTests(unittest.TestCase):
    def test_coordinates_counted(self):
        s = extract_signals("draw a circle at (0.10, 0.25) and a line to (0.5, 0.5)")
        self.assertEqual(s.n_coordinates, 2)
        self.assertTrue(s.has_coordinates)

    def test_numbers_outside_coords(self):
        s = extract_signals("radius 0.25 units")
        self.assertGreaterEqual(s.n_numbers, 1)
        self.assertFalse(s.has_coordinates)

    def test_jargon_and_relative_terms(self):
        s = extract_signals("extrude the sketch along the normal by 0.4 units")
        self.assertTrue(s.has_jargon)
        self.assertIn("extrude", s.jargon_terms)
        self.assertTrue(s.has_relative)


class ClassificationTests(unittest.TestCase):
    def test_abstract_shape_phrase(self):
        self.assertEqual(classify_prompt_level("a long rectangular shape"), "L0")
        self.assertEqual(classify_prompt_level("a thin S-shaped object"), "L0")

    def test_beginner_simple_steps(self):
        text = "Draw a circle and then make it into a solid block."
        self.assertEqual(classify_prompt_level(text), "L1")

    def test_intermediate_jargon_no_values(self):
        text = ("Sketch a circular profile and a loop, then extrude the sketch "
                "along the normal to form a solid body.")
        self.assertEqual(classify_prompt_level(text), "L2")

    def test_expert_precise(self):
        text = ("Set up a coordinate system, draw a circle with center (0.5, 0.5) "
                "and radius 0.25, then extrude along the normal by 0.4 units.")
        self.assertEqual(classify_prompt_level(text), "L3")

    def test_level_matches_roundtrip(self):
        self.assertTrue(level_matches("a ring-like structure", "L0"))
        self.assertFalse(level_matches("a ring-like structure", "L3"))


if __name__ == "__main__":
    unittest.main()
