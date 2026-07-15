"""Tests for eval.bench.data.operation_coverage."""

import math
import unittest

from harnesscad.eval.bench.data.operation_coverage import (
    RICH_OPERATIONS,
    beyond_sketch_extrude,
    operation_coverage,
    operation_diversity,
    readability,
)


class CoverageTest(unittest.TestCase):
    def test_sketch_extrude_only_low_coverage(self):
        seqs = [["sketch", "extrude"], ["sketch", "extrude"]]
        cov = operation_coverage(seqs)
        self.assertAlmostEqual(cov, 2 / len(RICH_OPERATIONS))

    def test_broad_coverage(self):
        seqs = [list(RICH_OPERATIONS)]
        self.assertEqual(operation_coverage(seqs), 1.0)

    def test_empty_vocab(self):
        with self.assertRaises(ValueError):
            operation_coverage([["sketch"]], vocabulary=[])


class DiversityTest(unittest.TestCase):
    def test_single_op_zero(self):
        self.assertEqual(operation_diversity([["extrude", "extrude"]]), 0.0)

    def test_uniform_is_one(self):
        self.assertAlmostEqual(operation_diversity([["sketch", "extrude"]]), 1.0)

    def test_empty_raises(self):
        with self.assertRaises(ValueError):
            operation_diversity([])


class BeyondSketchExtrudeTest(unittest.TestCase):
    def test_all_plain(self):
        self.assertEqual(beyond_sketch_extrude([["sketch", "extrude"]]), 0.0)

    def test_mixed(self):
        seq = [["sketch", "extrude", "fillet", "chamfer"]]
        self.assertAlmostEqual(beyond_sketch_extrude(seq), 0.5)


class ReadabilityTest(unittest.TestCase):
    def test_fraction(self):
        self.assertAlmostEqual(readability([True, True, False, False]), 0.5)

    def test_empty(self):
        with self.assertRaises(ValueError):
            readability([])


if __name__ == "__main__":
    unittest.main()
