"""Tests for eval.bench.geometry.edit_taxonomy."""

import unittest

from harnesscad.eval.bench.geometry.edit_taxonomy import (
    EDIT_OPERATIONS,
    MODALITY_COMBINATIONS,
    classify_sequence,
    edit_operation_fscore,
    modality_information_rate,
    normalise_operation,
)


class NormaliseTest(unittest.TestCase):
    def test_alias_folding(self):
        self.assertEqual(normalise_operation("Sketch"), "edit_sketch")
        self.assertEqual(normalise_operation("bevel"), "chamfer")
        self.assertEqual(normalise_operation("Subtract"), "boolean")

    def test_unknown_raises(self):
        with self.assertRaises(ValueError):
            normalise_operation("frobnicate")

    def test_classify_preserves_order(self):
        self.assertEqual(
            classify_sequence(["select", "Sketch", "extrusion"]),
            ["select", "edit_sketch", "extrude"],
        )


class FscoreTest(unittest.TestCase):
    def test_perfect_match(self):
        out = edit_operation_fscore(["select", "mirror"], ["mirror", "select"])
        self.assertEqual(out["f1"], 1.0)

    def test_partial(self):
        # pred {select, chamfer}; ref {select, mirror}
        out = edit_operation_fscore(["select", "chamfer"], ["select", "mirror"])
        self.assertAlmostEqual(out["precision"], 0.5)
        self.assertAlmostEqual(out["recall"], 0.5)
        self.assertAlmostEqual(out["f1"], 0.5)

    def test_both_empty(self):
        self.assertEqual(edit_operation_fscore([], [])["f1"], 1.0)

    def test_disjoint(self):
        out = edit_operation_fscore(["fillet"], ["chamfer"])
        self.assertEqual(out["f1"], 0.0)


class ModalityTest(unittest.TestCase):
    def test_four_combinations(self):
        self.assertEqual(len(MODALITY_COMBINATIONS), 4)

    def test_information_rate(self):
        self.assertEqual(modality_information_rate(("text",)), 1.0)
        self.assertEqual(
            modality_information_rate(("video", "speech", "interaction")), 3.0
        )

    def test_unknown_combo_raises(self):
        with self.assertRaises(ValueError):
            modality_information_rate(("smell",))

    def test_operations_nonempty(self):
        self.assertIn("chamfer", EDIT_OPERATIONS)


if __name__ == "__main__":
    unittest.main()
