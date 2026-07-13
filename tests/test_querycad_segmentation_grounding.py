"""Tests for rag/querycad_segmentation_grounding.py."""

from __future__ import annotations

import unittest

from harnesscad.agents.rag.querycad_segmentation_grounding import (
    Part, ground, ground_ids, matches_description, visible_from,
    DEFAULT_MAX_COVERAGE,
)


def _model():
    return [
        Part(id="h1", feature="hole", attrs={"diameter": 5.0},
             visible_views={"top"}, coverage=0.05),
        Part(id="h2", feature="hole", attrs={"diameter": 8.0},
             visible_views={"front"}, coverage=0.05),
        Part(id="b1", feature="boss", attrs={"diameter": 10.0},
             visible_views={"top"}, coverage=0.10),
        Part(id="whole", feature="hole", attrs={},
             visible_views={"top"}, coverage=0.90),
    ]


class TestPartValidation(unittest.TestCase):
    def test_bad_coverage(self):
        with self.assertRaises(ValueError):
            Part(id="x", feature="hole", coverage=1.5)

    def test_views_normalised(self):
        p = Part(id="x", feature="hole", visible_views={"TOP", "Front"})
        self.assertEqual(p.visible_views, frozenset({"top", "front"}))


class TestMatching(unittest.TestCase):
    def test_open_vocab_alias(self):
        p = Part(id="h1", feature="hole")
        self.assertTrue(matches_description(p, "thru hole"))
        self.assertTrue(matches_description(p, "bore"))
        self.assertTrue(matches_description(p, "drilled hole"))

    def test_direct_id(self):
        p = Part(id="shaft_tip", feature="boss")
        self.assertTrue(matches_description(p, "shaft tip"))

    def test_explicit_alias(self):
        p = Part(id="p", feature="boss", aliases=("gear",))
        self.assertTrue(matches_description(p, "gear"))

    def test_no_match(self):
        p = Part(id="h1", feature="hole")
        self.assertFalse(matches_description(p, "slot"))

    def test_model_word_no_match(self):
        p = Part(id="h1", feature="hole")
        self.assertFalse(matches_description(p, "the model"))


class TestViewFilter(unittest.TestCase):
    def test_no_constraint(self):
        p = Part(id="h1", feature="hole", visible_views={"top"})
        self.assertTrue(visible_from(p, ()))

    def test_visible(self):
        p = Part(id="h1", feature="hole", visible_views={"top", "front"})
        self.assertTrue(visible_from(p, ("front",)))

    def test_not_visible(self):
        p = Part(id="h1", feature="hole", visible_views={"top"})
        self.assertFalse(visible_from(p, ("bottom",)))


class TestGround(unittest.TestCase):
    def test_selects_holes_excludes_whole_model(self):
        got = ground_ids(_model(), "hole")
        # whole (coverage 0.90 > 0.45) is discarded; only h1, h2 remain.
        self.assertEqual(got, ("h1", "h2"))

    def test_view_specific(self):
        got = ground_ids(_model(), "hole", views=("top",))
        self.assertEqual(got, ("h1",))

    def test_boss(self):
        got = ground_ids(_model(), "boss")
        self.assertEqual(got, ("b1",))

    def test_coverage_cap_configurable(self):
        got = ground_ids(_model(), "hole", max_coverage=0.95)
        self.assertEqual(got, ("h1", "h2", "whole"))

    def test_deterministic_order(self):
        parts = list(reversed(_model()))
        got = ground_ids(parts, "hole")
        # input order preserved: reversed model -> whole(dropped), h2, h1
        self.assertEqual(got, ("h2", "h1"))

    def test_default_cap_value(self):
        self.assertAlmostEqual(DEFAULT_MAX_COVERAGE, 0.45)

    def test_type_error(self):
        with self.assertRaises(TypeError):
            ground(["not a part"], "hole")

    def test_returns_parts(self):
        parts = ground(_model(), "hole")
        self.assertTrue(all(isinstance(p, Part) for p in parts))
        self.assertEqual(parts[0].attrs["diameter"], 5.0)


if __name__ == "__main__":
    unittest.main()
