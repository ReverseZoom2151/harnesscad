"""Tests for agents/idea2cad_artifacts.py (paper 86: From Idea to CAD)."""

import unittest

from harnesscad.agents.agents.idea2cad_artifacts import (
    SEVEN_VIEWS,
    default_view_set,
    parse_summary,
    parse_addendum,
    top_issues,
    detect_ambiguities,
    QAReport,
    RequirementsAddendum,
)


class TestSevenViews(unittest.TestCase):
    def test_canonical_seven_views(self):
        self.assertEqual(len(SEVEN_VIEWS), 7)
        self.assertEqual(default_view_set(), SEVEN_VIEWS)
        for v in ("top", "bottom", "right", "left", "front", "back", "isometric"):
            self.assertIn(v, SEVEN_VIEWS)

    def test_isometric_last(self):
        self.assertEqual(SEVEN_VIEWS[-1], "isometric")


class TestParseSummary(unittest.TestCase):
    def test_extracts_block(self):
        raw = "some chatter <SUMMARY>length=10 width=5</SUMMARY> trailing"
        self.assertEqual(parse_summary(raw), "length=10 width=5")

    def test_multiline_block(self):
        raw = "<SUMMARY>\nline one\nline two\n</SUMMARY>"
        self.assertEqual(parse_summary(raw), "line one\nline two")

    def test_no_block_returns_none(self):
        self.assertIsNone(parse_summary("just asking a clarifying question?"))
        self.assertIsNone(parse_summary("<SUMMARY>unclosed"))
        self.assertIsNone(parse_summary(""))
        self.assertIsNone(parse_summary(None))

    def test_empty_block_returns_none(self):
        self.assertIsNone(parse_summary("<SUMMARY>   </SUMMARY>"))

    def test_case_insensitive(self):
        self.assertEqual(parse_summary("<summary>x</summary>"), "x")

    def test_first_block_wins(self):
        self.assertEqual(parse_summary("<SUMMARY>a</SUMMARY><SUMMARY>b</SUMMARY>"), "a")


class TestParseAddendum(unittest.TestCase):
    def test_converged(self):
        add = parse_addendum("<SUMMARY>done</SUMMARY>")
        self.assertIsInstance(add, RequirementsAddendum)
        self.assertTrue(add.converged)
        self.assertTrue(add.ok)
        self.assertEqual(add.text, "done")

    def test_not_converged(self):
        add = parse_addendum("what diameter?")
        self.assertFalse(add.converged)
        self.assertFalse(add.ok)
        self.assertIsNone(add.text)


class TestTopIssues(unittest.TestCase):
    def test_bounds_to_two(self):
        issues = ["a", "b", "c", "d"]
        self.assertEqual(top_issues(issues, 2), ["a", "b"])

    def test_default_k_is_two(self):
        self.assertEqual(top_issues(["a", "b", "c"]), ["a", "b"])

    def test_drops_blank(self):
        self.assertEqual(top_issues(["", "  ", "real"], 2), ["real"])

    def test_fewer_than_k(self):
        self.assertEqual(top_issues(["only"], 2), ["only"])

    def test_empty(self):
        self.assertEqual(top_issues([], 2), [])

    def test_priority_ranking_stable(self):
        issues = ["low", "high", "mid"]
        prios = [3, 1, 2]
        self.assertEqual(top_issues(issues, 2, priorities=prios), ["high", "mid"])

    def test_priority_length_mismatch_raises(self):
        with self.assertRaises(ValueError):
            top_issues(["a", "b"], 2, priorities=[1])

    def test_negative_k_raises(self):
        with self.assertRaises(ValueError):
            top_issues(["a"], -1)


class TestQAReport(unittest.TestCase):
    def test_acceptable_when_empty(self):
        rep = QAReport(issues=[], acceptable=True)
        d = rep.to_dict()
        self.assertTrue(d["acceptable"])
        self.assertEqual(d["issues"], [])
        self.assertEqual(len(d["views"]), 7)


class TestDetectAmbiguities(unittest.TestCase):
    def test_empty_spec(self):
        amb = detect_ambiguities("")
        self.assertTrue(amb)

    def test_no_dimensions(self):
        amb = detect_ambiguities("a plastic block")
        self.assertTrue(any("dimension" in a for a in amb))

    def test_fully_specified_bracket(self):
        # the paper's angle-bracket spec has every dimension pinned down
        spec = ("short leg length = 3cm, long leg length = 5cm. leg width = 1cm. "
                "thickness = .2cm. angle is 90 degrees. holes diameter 0.5cm 1cm apart.")
        self.assertEqual(detect_ambiguities(spec), [])

    def test_hedged_spec_flagged(self):
        spec = "make reasonable assumptions for all dimensions. length=5cm"
        amb = detect_ambiguities(spec)
        self.assertTrue(any("assumption" in a.lower() for a in amb))


if __name__ == "__main__":
    unittest.main()
