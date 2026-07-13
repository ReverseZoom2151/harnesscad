"""Tests for chatcadplus_prob2text -- graded numeric-to-text verbalisation."""

from __future__ import annotations

import unittest

from harnesscad.eval.quality.report.chatcadplus_prob2text import (
    Band,
    BandScheme,
    SCHEMES,
    verbalise,
    verbalise_scores,
)


class TestProb2Text(unittest.TestCase):
    def test_illustrative_four_bands(self):
        self.assertEqual(verbalise("crack", 0.05), "No sign of crack")
        self.assertEqual(verbalise("crack", 0.35), "Small possibility of crack")
        self.assertEqual(verbalise("crack", 0.7), "Likely to have crack")
        self.assertEqual(verbalise("crack", 0.95), "Definitely has crack")

    def test_band_boundaries_are_half_open_low_inclusive(self):
        self.assertEqual(verbalise("x", 0.2), "Small possibility of x")
        self.assertEqual(verbalise("x", 0.5), "Likely to have x")
        self.assertEqual(verbalise("x", 0.9), "Definitely has x")

    def test_endpoints_covered(self):
        self.assertEqual(verbalise("x", 0.0), "No sign of x")
        self.assertEqual(verbalise("x", 1.0), "Definitely has x")

    def test_simplistic_two_bands(self):
        self.assertEqual(verbalise("y", 0.49, scheme="simplistic"), "No y")
        self.assertEqual(verbalise("y", 0.5, scheme="simplistic"), "The prediction is y")

    def test_direct_echoes_number(self):
        self.assertEqual(verbalise("z", 0.873, scheme="direct"), "z score: 0.87")

    def test_out_of_range_raises(self):
        with self.assertRaises(ValueError):
            verbalise("x", 1.5)
        with self.assertRaises(ValueError):
            verbalise("x", -0.1)

    def test_unknown_scheme_raises(self):
        with self.assertRaises(ValueError):
            verbalise("x", 0.5, scheme="nope")

    def test_all_named_schemes_present(self):
        self.assertEqual(set(SCHEMES), {"direct", "simplistic", "illustrative"})

    def test_verbalise_scores_sorts_by_descending_score(self):
        scores = {"edema": 0.3, "cardiomegaly": 0.95, "effusion": 0.6}
        lines = verbalise_scores(scores).splitlines()
        self.assertEqual(lines[0], "- Definitely has cardiomegaly")
        self.assertEqual(lines[1], "- Likely to have effusion")
        self.assertEqual(lines[2], "- Small possibility of edema")

    def test_verbalise_scores_tie_break_is_label_ascending(self):
        scores = {"b": 0.7, "a": 0.7}
        self.assertEqual(
            verbalise_scores(scores).splitlines(),
            ["- Likely to have a", "- Likely to have b"],
        )

    def test_verbalise_scores_sort_by_label(self):
        scores = {"b": 0.95, "a": 0.1}
        out = verbalise_scores(scores, sort="label")
        self.assertEqual(out.splitlines()[0], "- No sign of a")

    def test_verbalise_scores_deterministic(self):
        scores = {"p": 0.2, "q": 0.8, "r": 0.55}
        self.assertEqual(verbalise_scores(scores), verbalise_scores(dict(scores)))

    def test_custom_band_scheme(self):
        scheme = BandScheme(
            "risk",
            [Band(0.0, 0.5, "{label}: safe"), Band(0.5, 1.0, "{label}: RISK")],
        )
        self.assertEqual(verbalise("tol", 0.9, scheme=scheme), "tol: RISK")
        self.assertEqual(verbalise("tol", 0.1, scheme=scheme), "tol: safe")

    def test_non_contiguous_bands_rejected(self):
        with self.assertRaises(ValueError):
            BandScheme("bad", [Band(0.0, 0.4, "a"), Band(0.5, 1.0, "b")])

    def test_empty_scheme_rejected(self):
        with self.assertRaises(ValueError):
            BandScheme("empty", [])


if __name__ == "__main__":
    unittest.main()
