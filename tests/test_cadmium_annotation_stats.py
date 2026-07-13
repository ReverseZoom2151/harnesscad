"""Tests for dataengine.cadmium_annotation_stats."""

import unittest

from harnesscad.data.dataengine.cadmium_annotation_stats import (
    compare_corpora,
    corpus_stats,
    decimal_places,
    unique_word_ratio,
    vocabulary_growth,
    word_count,
    words,
)


class TokenTest(unittest.TestCase):
    def test_words_lowercase_alpha_only(self):
        self.assertEqual(words("A Circle, 3.5 mm wide."),
                         ("a", "circle", "mm", "wide"))

    def test_word_count(self):
        self.assertEqual(word_count("one two two three"), 4)

    def test_unique_word_ratio(self):
        self.assertAlmostEqual(unique_word_ratio("one two two three"), 3 / 4)
        self.assertEqual(unique_word_ratio(""), 0.0)

    def test_decimal_places(self):
        self.assertEqual(decimal_places("3.14 and 2 and 0.5000"), (2, 0, 4))
        self.assertEqual(decimal_places("no numbers here"), ())


class GrowthTest(unittest.TestCase):
    def test_vocabulary_growth_monotone(self):
        curve = vocabulary_growth(["alpha beta", "beta gamma", "delta"])
        self.assertEqual(curve, ((2, 2), (4, 3), (5, 4)))
        tokens = [t for t, _ in curve]
        vocab = [u for _, u in curve]
        self.assertEqual(tokens, sorted(tokens))
        self.assertEqual(vocab, sorted(vocab))

    def test_deterministic(self):
        c = ["a b c", "c d e", "f"]
        self.assertEqual(vocabulary_growth(c), vocabulary_growth(list(c)))


class CorpusStatsTest(unittest.TestCase):
    def test_basic_summary(self):
        stats = corpus_stats(["one two three", "one two"])
        self.assertEqual(stats.annotations, 2)
        self.assertEqual(stats.vocabulary_size, 3)
        self.assertEqual(stats.total_words, 5)
        self.assertAlmostEqual(stats.mean_words, 2.5)
        self.assertAlmostEqual(stats.median_words, 2.5)

    def test_concise_band(self):
        long_text = " ".join(["word"] * 150)
        short_text = "tiny note"
        stats = corpus_stats([long_text, short_text])
        self.assertAlmostEqual(stats.fraction_in_concise_band, 0.5)

    def test_decimal_stats(self):
        stats = corpus_stats(["value 1.25 and 3"])
        self.assertEqual(stats.numbers, 2)
        self.assertAlmostEqual(stats.mean_decimal_places, 1.0)
        self.assertEqual(stats.max_decimal_places, 2)

    def test_empty_corpus_rejected(self):
        with self.assertRaises(ValueError):
            corpus_stats([])


class CompareCorporaTest(unittest.TestCase):
    def test_cadmium_vs_verbose_template(self):
        # CADmium-like: concise, diverse vocabulary, natural precision.
        cadmium = [
            "A rounded wedge with a smooth curved top and flat base.",
            "A clamp-like ring with four evenly spaced holes.",
        ]
        # Text2CAD-like: longer, repetitive, excessively precise decimals.
        verbose = [
            "Create face_1 with line line line line at 12.345678 units and "
            "then extrude face_1 with line line line line repeated repeated.",
            "Create face_1 with line line line line at 45.987654 units and "
            "then extrude face_1 with line line line line repeated repeated.",
        ]
        cmp = compare_corpora(cadmium, verbose)
        self.assertEqual(cmp.more_concise, "a")
        self.assertEqual(cmp.more_natural_precision, "a")
        self.assertEqual(cmp.more_diverse, "a")

    def test_tie(self):
        cmp = compare_corpora(["a b", "c d"], ["e f", "g h"])
        self.assertEqual(cmp.more_concise, "tie")


if __name__ == "__main__":
    unittest.main()
