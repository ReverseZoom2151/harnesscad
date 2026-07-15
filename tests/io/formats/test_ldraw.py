"""Tests for io.formats.ldraw."""

import unittest

from harnesscad.io.formats.ldraw import (
    PartLine,
    drawing_accuracy,
    instructional_coherence,
    parse_line,
    parse_model,
    structural_validity,
)

VALID = "1 4 10 0 20 1 0 0 0 1 0 0 0 1 3001.dat"


class ParseLineTest(unittest.TestCase):
    def test_valid_type1(self):
        pl = parse_line(VALID)
        self.assertIsInstance(pl, PartLine)
        self.assertEqual(pl.colour, 4)
        self.assertEqual(pl.position, (10.0, 0.0, 20.0))
        self.assertEqual(len(pl.rotation), 9)
        self.assertEqual(pl.part, "3001.dat")

    def test_comment_returns_none(self):
        self.assertIsNone(parse_line("0 // a comment"))

    def test_blank_returns_none(self):
        self.assertIsNone(parse_line("   "))

    def test_too_few_tokens(self):
        with self.assertRaises(ValueError):
            parse_line("1 4 10 0 20 1 0 0")

    def test_non_numeric(self):
        with self.assertRaises(ValueError):
            parse_line("1 4 x 0 20 1 0 0 0 1 0 0 0 1 3001.dat")


class ParseModelTest(unittest.TestCase):
    def test_counts_and_bad_lines(self):
        text = "\n".join([
            "0 Castle",
            VALID,
            "1 4 bad line here",  # malformed
            VALID,
        ])
        parts, bad = parse_model(text)
        self.assertEqual(len(parts), 2)
        self.assertEqual(bad, [3])


class MetricsTest(unittest.TestCase):
    def test_drawing_accuracy(self):
        text = "\n".join([VALID, VALID, "1 4 broken"])
        self.assertAlmostEqual(drawing_accuracy(text), 2 / 3)

    def test_structural_validity(self):
        parts, _ = parse_model("\n".join([VALID, VALID]))
        self.assertEqual(structural_validity(parts, ["3001.dat"]), 1.0)
        self.assertEqual(structural_validity(parts, ["9999.dat"]), 0.0)

    def test_instructional_coherence_perfect(self):
        parts, _ = parse_model("\n".join([VALID, VALID, VALID, VALID]))
        steps = [parts[:2], parts[2:]]
        self.assertEqual(instructional_coherence(steps, total_parts=4), 1.0)

    def test_instructional_coherence_empty_step_penalised(self):
        parts, _ = parse_model("\n".join([VALID, VALID]))
        steps = [parts, []]
        self.assertLess(instructional_coherence(steps, total_parts=2), 1.0)

    def test_coherence_bad_total(self):
        with self.assertRaises(ValueError):
            instructional_coherence([], 0)


if __name__ == "__main__":
    unittest.main()
