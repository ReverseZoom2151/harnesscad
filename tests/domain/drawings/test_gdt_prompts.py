import itertools
import random
import unittest

from harnesscad.domain.drawings.gdt_prompts import (
    BASELINE_SYSTEM_PROMPT,
    DETAILED_SYSTEM_PROMPT,
    GDT_SYSTEM_PROMPT,
    build_focused_requery_prompt,
    main,
)

SEED = 20260719

ANNOTATION_TYPES = ("dimension", "fcf", "datum", "surface_finish", "note")

GEOMETRIC_CHARACTERISTICS = (
    "position",
    "flatness",
    "straightness",
    "circularity",
    "cylindricity",
    "perpendicularity",
    "parallelism",
    "angularity",
    "profileOfLine",
    "profileOfSurface",
    "circularRunout",
    "totalRunout",
    "symmetry",
    "concentricity",
)


class GdtSystemPromptTests(unittest.TestCase):
    def test_non_empty_string(self):
        self.assertIsInstance(GDT_SYSTEM_PROMPT, str)
        self.assertTrue(len(GDT_SYSTEM_PROMPT) > 500)

    def test_mentions_all_five_annotation_types(self):
        for t in ANNOTATION_TYPES:
            self.assertIn('"%s"' % t, GDT_SYSTEM_PROMPT, t)

    def test_mentions_confidence_scoring(self):
        self.assertIn("confidence score between 0.0 and 1.0", GDT_SYSTEM_PROMPT)
        self.assertIn("< 0.6", GDT_SYSTEM_PROMPT)

    def test_mentions_bounding_box_percentage_coordinates(self):
        self.assertIn("percentages (0-100)", GDT_SYSTEM_PROMPT)
        self.assertIn("top-left corner", GDT_SYSTEM_PROMPT)

    def test_lists_all_14_geometric_characteristics(self):
        for gc in GEOMETRIC_CHARACTERISTICS:
            self.assertIn('"%s"' % gc, GDT_SYSTEM_PROMPT, gc)

    def test_lists_dimension_types(self):
        for dt in ("linear", "angular", "radius", "diameter"):
            self.assertIn('"%s"' % dt, GDT_SYSTEM_PROMPT, dt)

    def test_lists_material_conditions(self):
        for mc in ("MMC", "LMC", "RFS"):
            self.assertIn('"%s"' % mc, GDT_SYSTEM_PROMPT, mc)
        self.assertIn('materialCondition must be one of: "MMC", "LMC", "RFS", or null',
                      GDT_SYSTEM_PROMPT)

    def test_datum_reference_and_letter_rules(self):
        self.assertIn(
            "datumReferences is an ordered array of up to 3 uppercase letters",
            GDT_SYSTEM_PROMPT,
        )
        self.assertIn("datumLetter must be a single uppercase letter A-Z",
                      GDT_SYSTEM_PROMPT)

    def test_declares_top_level_response_keys(self):
        for key in ("annotations", "views", "description"):
            self.assertIn('"%s"' % key, GDT_SYSTEM_PROMPT, key)

    def test_demands_json_only(self):
        self.assertIn("Only return valid JSON, no other text", GDT_SYSTEM_PROMPT)

    def test_is_ascii_only(self):
        # harness constraint: prompts are pure ASCII (the TS original used
        # the same plain text, no unicode GD&T glyphs).
        GDT_SYSTEM_PROMPT.encode("ascii")

    def test_mentions_color_convention(self):
        self.assertIn("green for dimensions", GDT_SYSTEM_PROMPT)
        self.assertIn("purple for notes", GDT_SYSTEM_PROMPT)


class GenericPromptTests(unittest.TestCase):
    def test_baseline_is_the_simpler_prompt(self):
        self.assertIn("most prominent dimensions", BASELINE_SYSTEM_PROMPT)
        self.assertIn("Only return valid JSON", BASELINE_SYSTEM_PROMPT)
        self.assertTrue(len(BASELINE_SYSTEM_PROMPT) < len(DETAILED_SYSTEM_PROMPT))

    def test_detailed_asks_for_more_annotations(self):
        self.assertIn("6-12 annotations", DETAILED_SYSTEM_PROMPT)
        self.assertIn("Group annotations by which view", DETAILED_SYSTEM_PROMPT)

    def test_generic_prompts_are_not_gdt_specific(self):
        # divergence check: only GDT_SYSTEM_PROMPT carries the enriched
        # type-specific schema; the analyze-route prompts stay generic.
        for p in (BASELINE_SYSTEM_PROMPT, DETAILED_SYSTEM_PROMPT):
            self.assertNotIn("geometricCharacteristic", p)
            self.assertNotIn("datumLetter", p)
            self.assertNotIn('"fcf"', p)

    def test_generic_prompts_share_bounding_box_contract(self):
        for p in (BASELINE_SYSTEM_PROMPT, DETAILED_SYSTEM_PROMPT):
            self.assertIn("boundingBox", p)
            self.assertIn("percentages of the image dimensions (0-100)", p)

    def test_all_prompts_distinct_and_ascii(self):
        prompts = [GDT_SYSTEM_PROMPT, BASELINE_SYSTEM_PROMPT, DETAILED_SYSTEM_PROMPT]
        self.assertEqual(len(set(prompts)), 3)
        for p in prompts:
            p.encode("ascii")
            self.assertFalse(p.startswith("\n"))


class BuildFocusedRequeryPromptTests(unittest.TestCase):
    def test_includes_type_hint(self):
        p = build_focused_requery_prompt("fcf", "Position 0.05 A B", "0.05")
        self.assertIn("- Type: fcf", p)
        self.assertIn('"type": "fcf"', p)

    def test_includes_label_and_value(self):
        p = build_focused_requery_prompt("datum", "Datum A", "A")
        self.assertIn('- Label: "Datum A"', p)
        self.assertIn('- Value: "A"', p)

    def test_dimension_instructions(self):
        p = build_focused_requery_prompt("dimension", "40.2", "40.2")
        self.assertIn("DIMENSION annotation", p)
        self.assertIn("dimensionType (linear|angular|radius|diameter)", p)
        self.assertIn("nominalValue", p)

    def test_fcf_instructions(self):
        p = build_focused_requery_prompt("fcf", "x", "y")
        self.assertIn("FEATURE CONTROL FRAME", p)
        self.assertIn("geometricCharacteristic", p)
        self.assertIn("materialCondition (MMC|LMC|RFS|null)", p)
        self.assertIn("datumReferences (array of up to 3 uppercase letters)", p)
        for gc in GEOMETRIC_CHARACTERISTICS:
            self.assertIn(gc, p, gc)

    def test_datum_instructions(self):
        p = build_focused_requery_prompt("datum", "x", "y")
        self.assertIn("DATUM annotation", p)
        self.assertIn("datumLetter (a single uppercase letter A-Z)", p)

    def test_surface_finish_instructions(self):
        p = build_focused_requery_prompt("surface_finish", "x", "y")
        self.assertIn("SURFACE FINISH annotation", p)
        self.assertIn("roughnessValue (number)", p)
        self.assertIn("processNote", p)

    def test_note_instructions(self):
        p = build_focused_requery_prompt("note", "x", "y")
        self.assertIn("NOTE annotation", p)
        self.assertIn("Extract the text content of the note", p)

    def test_confidence_rules_present_for_every_type(self):
        for t in ANNOTATION_TYPES:
            p = build_focused_requery_prompt(t, "x", "y")
            self.assertIn("confidence must be between 0.0 and 1.0", p, t)
            self.assertIn("set confidence below 0.5", p, t)
            self.assertIn("Only return valid JSON, no other text", p, t)

    def test_unknown_type_hint_yields_empty_type_section(self):
        # known-bad vector: the harness does NOT raise on an unknown type
        # hint -- _TYPE_SPECIFIC_INSTRUCTIONS.get(...) falls back to "" and
        # the generic scaffold is still emitted with the bad hint echoed.
        p = build_focused_requery_prompt("bogus", "x", "y")
        self.assertIn("- Type: bogus", p)
        self.assertIn('"type": "bogus"', p)
        self.assertNotIn("This appears to be a", p)
        self.assertIn("Only return valid JSON, no other text", p)

    def test_empty_type_hint(self):
        p = build_focused_requery_prompt("", "", "")
        self.assertIn("- Type: \n", p)
        self.assertIn('- Label: ""', p)
        self.assertIn('- Value: ""', p)
        self.assertNotIn("This appears to be a", p)

    def test_case_sensitive_type_hint(self):
        # "FCF" is not a key -> no fcf-specific instructions
        p = build_focused_requery_prompt("FCF", "x", "y")
        self.assertNotIn("FEATURE CONTROL FRAME (FCF) annotation", p)

    def test_each_type_gets_a_distinct_prompt(self):
        prompts = {t: build_focused_requery_prompt(t, "L", "V") for t in ANNOTATION_TYPES}
        self.assertEqual(len(set(prompts.values())), len(ANNOTATION_TYPES))

    def test_type_specific_sections_do_not_leak(self):
        # each type's marker must appear ONLY in its own prompt
        markers = {
            "dimension": "DIMENSION annotation",
            "fcf": "FEATURE CONTROL FRAME",
            "datum": "DATUM annotation",
            "surface_finish": "SURFACE FINISH annotation",
            "note": "NOTE annotation",
        }
        for owner, other in itertools.permutations(ANNOTATION_TYPES, 2):
            p = build_focused_requery_prompt(other, "L", "V")
            self.assertNotIn(markers[owner], p, (owner, other))

    def test_label_and_value_are_interpolated_verbatim(self):
        # No escaping is performed: an embedded quote lands in the prompt as-is.
        p = build_focused_requery_prompt("note", 'a"b', "c\\d")
        self.assertIn('- Label: "a"b"', p)
        self.assertIn('- Value: "c\\d"', p)

    def test_deterministic(self):
        for t in ANNOTATION_TYPES:
            self.assertEqual(
                build_focused_requery_prompt(t, "L", "V"),
                build_focused_requery_prompt(t, "L", "V"),
            )


class RequeryPromptPropertyTests(unittest.TestCase):
    # Substitute for the fast-check property tests in the TS suite: hypothesis
    # is unavailable, so we enumerate the full type domain exhaustively with
    # itertools and draw label/value strings from random.Random(FIXED SEED).

    def _labels(self, n):
        rng = random.Random(SEED)
        alphabet = "ABCabc019 .-+/x"
        return [
            "".join(rng.choice(alphabet) for _ in range(rng.randint(0, 12)))
            for _ in range(n)
        ]

    def test_label_value_always_embedded_for_every_type(self):
        labels = self._labels(12)
        for t, (label, value) in itertools.product(
            ANNOTATION_TYPES + ("bogus", ""), itertools.product(labels, labels[:4])
        ):
            p = build_focused_requery_prompt(t, label, value)
            self.assertIn('- Label: "%s"' % label, p)
            self.assertIn('- Value: "%s"' % value, p)
            self.assertIn("- Type: %s\n" % t, p)

    def test_prompt_is_pure_function_of_its_three_arguments(self):
        labels = self._labels(8)
        seen = {}
        for t, label, value in itertools.product(ANNOTATION_TYPES, labels, labels[:3]):
            key = (t, label, value)
            p = build_focused_requery_prompt(*key)
            if key in seen:
                self.assertEqual(seen[key], p)
            seen[key] = p
            # distinct keys must not collide either
            self.assertEqual(build_focused_requery_prompt(*key), p)
        self.assertEqual(len(set(seen.values())), len(seen))

    def test_ascii_inputs_produce_ascii_prompts(self):
        for t, label in itertools.product(ANNOTATION_TYPES, self._labels(6)):
            build_focused_requery_prompt(t, label, "v").encode("ascii")


class MainTests(unittest.TestCase):
    def test_selfcheck_passes(self):
        self.assertEqual(main(["--selfcheck"]), 0)

    def test_selfcheck_json_passes(self):
        self.assertEqual(main(["--selfcheck", "--json"]), 0)

    def test_no_args_prints_help(self):
        self.assertEqual(main([]), 0)

    def test_bad_flag_exits(self):
        with self.assertRaises(SystemExit):
            main(["--nope"])


if __name__ == "__main__":
    unittest.main()
