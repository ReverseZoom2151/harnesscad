import unittest

from programs.t2cmain_cad_ref_selectors import (
    build_cad_token,
    common_occurrence_prefix,
    is_descendant_occurrence,
    normalize_cad_path,
    normalize_selector_list,
    occurrence_depth,
    occurrence_segments,
    parse_cad_tokens,
    parse_selector,
    parse_selector_list,
    selector_type_for_kind,
)


class SelectorKindTest(unittest.TestCase):
    def test_kind_names(self):
        self.assertEqual(selector_type_for_kind("s"), "shape")
        self.assertEqual(selector_type_for_kind("f"), "face")
        self.assertEqual(selector_type_for_kind("e"), "edge")
        self.assertEqual(selector_type_for_kind("v"), "vertex")


class ParseSelectorTest(unittest.TestCase):
    def test_bare_occurrence(self):
        selector = parse_selector("o1.2")
        self.assertEqual(selector.selector_type, "occurrence")
        self.assertEqual(selector.occurrence_id, "o1.2")
        self.assertIsNone(selector.ordinal)
        self.assertEqual(selector.canonical, "o1.2")

    def test_occurrence_entity(self):
        selector = parse_selector("o3.1.f12")
        self.assertEqual(selector.selector_type, "face")
        self.assertEqual(selector.occurrence_id, "o3.1")
        self.assertEqual(selector.ordinal, 12)
        self.assertEqual(selector.canonical, "o3.1.f12")

    def test_bare_entity_without_inheritance(self):
        selector = parse_selector("e7")
        self.assertEqual(selector.selector_type, "edge")
        self.assertEqual(selector.occurrence_id, "")
        self.assertEqual(selector.canonical, "e7")

    def test_bare_entity_inherits_occurrence(self):
        selector = parse_selector("v2", inherited_occurrence_id="o9")
        self.assertEqual(selector.canonical, "o9.v2")
        self.assertEqual(selector.occurrence_id, "o9")

    def test_leading_hash_is_stripped(self):
        self.assertEqual(parse_selector("#f1").canonical, "f1")

    def test_empty_is_none(self):
        self.assertIsNone(parse_selector(""))
        self.assertIsNone(parse_selector("   "))

    def test_unknown_syntax_is_opaque(self):
        selector = parse_selector("body::top")
        self.assertEqual(selector.selector_type, "opaque")
        self.assertEqual(selector.canonical, "body::top")

    def test_unknown_kind_letter_is_opaque(self):
        self.assertEqual(parse_selector("q4").selector_type, "opaque")


class SelectorListTest(unittest.TestCase):
    def test_inheritance_threads_left_to_right(self):
        self.assertEqual(
            normalize_selector_list("o1.2.f3,f4,e5"),
            ("o1.2.f3", "o1.2.f4", "o1.2.e5"),
        )

    def test_new_occurrence_rebinds_inheritance(self):
        self.assertEqual(
            normalize_selector_list("o1.f1,o5,e1"),
            ("o1.f1", "o5", "o5.e1"),
        )

    def test_leading_bare_entity_stays_relative(self):
        self.assertEqual(normalize_selector_list("f1,o2.f2,f3"), ("f1", "o2.f2", "o2.f3"))

    def test_empty_entries_are_dropped(self):
        self.assertEqual(normalize_selector_list("o1,,f2, "), ("o1", "o1.f2"))

    def test_parse_selector_list_returns_records(self):
        selectors = parse_selector_list("o1.2.f3,f4")
        self.assertEqual([s.ordinal for s in selectors], [3, 4])
        self.assertEqual([s.occurrence_id for s in selectors], ["o1.2", "o1.2"])

    def test_normalization_is_idempotent(self):
        once = normalize_selector_list("o1.2.f3,f4")
        twice = normalize_selector_list(",".join(once))
        self.assertEqual(once, twice)


class TokenTest(unittest.TestCase):
    def test_tokens_are_line_numbered(self):
        tokens = parse_cad_tokens("first\nmill #o1.f2 flat\n\nchamfer #o2.e3")
        self.assertEqual([token.line for token in tokens], [2, 4])
        self.assertEqual(tokens[0].selectors, ("o1.f2",))
        self.assertEqual(tokens[1].selectors, ("o2.e3",))

    def test_multiple_tokens_on_one_line(self):
        tokens = parse_cad_tokens("weld #o1.f1 to #o2.f1")
        self.assertEqual(len(tokens), 2)
        self.assertEqual(tokens[0].token, "#o1.f1")
        self.assertEqual(tokens[1].token, "#o2.f1")

    def test_token_selectors_are_canonicalised(self):
        tokens = parse_cad_tokens("#o1.2.f3,f4")
        self.assertEqual(tokens[0].selectors, ("o1.2.f3", "o1.2.f4"))

    def test_text_without_tokens(self):
        self.assertEqual(parse_cad_tokens("no markers here"), ())

    def test_build_token_round_trip(self):
        self.assertEqual(build_cad_token("o1.2.f3,f4"), "#o1.2.f3,o1.2.f4")
        self.assertEqual(build_cad_token(["o1", "o1.f2"]), "#o1,o1.f2")
        self.assertEqual(build_cad_token([]), "#")


class CadPathTest(unittest.TestCase):
    def test_suffix_is_dropped(self):
        self.assertEqual(normalize_cad_path("parts/Bracket.STEP"), "parts/Bracket")
        self.assertEqual(normalize_cad_path("parts/bracket.stp"), "parts/bracket")

    def test_backslashes_are_folded(self):
        self.assertEqual(normalize_cad_path("parts\\sub\\x.step"), "parts/sub/x")

    def test_traversal_is_rejected(self):
        self.assertIsNone(normalize_cad_path("../secrets/x.step"))
        self.assertIsNone(normalize_cad_path("a/./b.step"))
        self.assertIsNone(normalize_cad_path("a//b.step"))

    def test_empty_is_none(self):
        self.assertIsNone(normalize_cad_path(""))
        self.assertIsNone(normalize_cad_path("///"))

    def test_leading_and_trailing_slashes_are_trimmed(self):
        self.assertEqual(normalize_cad_path("/parts/x.step/"), "parts/x")


class OccurrenceTreeTest(unittest.TestCase):
    def test_segments(self):
        self.assertEqual(occurrence_segments("o1.2.3"), ("1", "2", "3"))
        self.assertEqual(occurrence_segments("o4"), ("4",))
        self.assertEqual(occurrence_segments("nope"), ())

    def test_depth(self):
        self.assertEqual(occurrence_depth("o1"), 1)
        self.assertEqual(occurrence_depth("o1.2.3"), 3)
        self.assertEqual(occurrence_depth(""), 0)

    def test_descendant(self):
        self.assertTrue(is_descendant_occurrence("o1.2.3", "o1.2"))
        self.assertFalse(is_descendant_occurrence("o1.2", "o1.2"))
        self.assertFalse(is_descendant_occurrence("o1.3", "o1.2"))
        self.assertFalse(is_descendant_occurrence("o1", "o1.2"))

    def test_common_prefix_of_shared_root(self):
        self.assertEqual(common_occurrence_prefix(["o1.1", "o1.2", "o1.3.4"]), ("1",))

    def test_common_prefix_of_deeper_tree(self):
        self.assertEqual(
            common_occurrence_prefix(["o1.2.1", "o1.2.2", "o1.2.3.4"]), ("1", "2")
        )

    def test_common_prefix_never_consumes_the_whole_shortest_path(self):
        # Both ids are fully "1.2"; keeping the whole prefix would leave nothing
        # to group by, so the last segment is given back.
        self.assertEqual(common_occurrence_prefix(["o1.2", "o1.2.3"]), ("1",))
        self.assertEqual(common_occurrence_prefix(["o1.2", "o1.2"]), ("1",))

    def test_common_prefix_ignores_top_level_occurrences(self):
        self.assertEqual(common_occurrence_prefix(["o1", "o2"]), ())

    def test_common_prefix_of_disjoint_roots(self):
        self.assertEqual(common_occurrence_prefix(["o1.1", "o2.1"]), ())


if __name__ == "__main__":
    unittest.main()
