"""Tests for bench.t2cadbench_seq_edit_distance."""

import unittest

from harnesscad.eval.bench.t2cadbench_seq_edit_distance import (
    levenshtein,
    mean_sequence_similarity,
    sequence_edit_distance,
)


class LevenshteinTests(unittest.TestCase):
    def test_identical(self):
        self.assertEqual(levenshtein(["box", "hole"], ["box", "hole"]), 0)

    def test_one_substitution(self):
        self.assertEqual(levenshtein(["box", "hole"], ["box", "chamfer"]), 1)

    def test_insertion_deletion(self):
        self.assertEqual(levenshtein(["box"], ["box", "hole"]), 1)
        self.assertEqual(levenshtein(["a", "b", "c"], []), 3)


class SequenceEditDistanceTests(unittest.TestCase):
    def test_exact_match_normalisation_and_tokens(self):
        r = sequence_edit_distance(
            [".box()", "cutThruAll", ".Chamfer(2)"],
            ["box", "cutthruall", "chamfer"])
        self.assertEqual(r["distance"], 0)
        self.assertEqual(r["normalized_distance"], 0.0)
        self.assertEqual(r["similarity"], 1.0)

    def test_partial(self):
        r = sequence_edit_distance(
            ["box", "hole", "fillet"], ["box", "hole", "chamfer"])
        self.assertEqual(r["distance"], 1)
        self.assertAlmostEqual(r["normalized_distance"], 1 / 3)
        self.assertAlmostEqual(r["similarity"], 2 / 3)

    def test_length_bookkeeping(self):
        r = sequence_edit_distance(["box", "hole"], ["box"])
        self.assertEqual(r["pred_len"], 2)
        self.assertEqual(r["truth_len"], 1)
        self.assertEqual(r["api_call_delta"], 1)

    def test_both_empty(self):
        r = sequence_edit_distance([], [])
        self.assertEqual(r["normalized_distance"], 0.0)
        self.assertEqual(r["similarity"], 1.0)


class MeanSimilarityTests(unittest.TestCase):
    def test_aggregate(self):
        ex = [
            (["box"], ["box"]),                       # sim 1.0
            (["box", "cut"], ["box", "union"]),       # sim 0.5
        ]
        r = mean_sequence_similarity(ex)
        self.assertEqual(r["n"], 2)
        self.assertAlmostEqual(r["mean_similarity"], 0.75)
        self.assertAlmostEqual(r["exact_match_rate"], 0.5)

    def test_empty_raises(self):
        with self.assertRaises(ValueError):
            mean_sequence_similarity([])


if __name__ == "__main__":
    unittest.main()
