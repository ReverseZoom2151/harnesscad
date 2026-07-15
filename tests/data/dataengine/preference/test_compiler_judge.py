"""Tests for data.dataengine.preference.compiler_judge."""

import unittest

from harnesscad.data.dataengine.preference.compiler_judge import (
    cjm_label,
    label_batch,
)


class CJMLabelTest(unittest.TestCase):
    def test_compiled_within_threshold_desirable(self):
        self.assertTrue(cjm_label(True, 0.05, 0.1))

    def test_compiled_at_threshold_desirable(self):
        self.assertTrue(cjm_label(True, 0.1, 0.1))

    def test_compiled_over_threshold_undesirable(self):
        self.assertFalse(cjm_label(True, 0.2, 0.1))

    def test_not_compiled_undesirable(self):
        self.assertFalse(cjm_label(False, None, 0.1))

    def test_compiled_without_cd_raises(self):
        with self.assertRaises(ValueError):
            cjm_label(True, None, 0.1)

    def test_negative_threshold_raises(self):
        with self.assertRaises(ValueError):
            cjm_label(True, 0.05, -1.0)


class LabelBatchTest(unittest.TestCase):
    def test_counts_and_ir(self):
        records = [
            {"compiled": True, "cd": 0.05},
            {"compiled": True, "cd": 0.5},
            {"compiled": False},
            {"compiled": True, "cd": 0.08},
        ]
        out = label_batch(records, threshold=0.1)
        self.assertEqual(out["labels"], (True, False, False, True))
        self.assertEqual(out["desirable"], 2)
        self.assertEqual(out["undesirable"], 2)
        self.assertAlmostEqual(out["invalidity_ratio"], 0.25)
        self.assertEqual(out["total"], 4)

    def test_empty_batch(self):
        out = label_batch([], threshold=0.1)
        self.assertEqual(out["total"], 0)
        self.assertEqual(out["invalidity_ratio"], 0.0)


if __name__ == "__main__":
    unittest.main()
