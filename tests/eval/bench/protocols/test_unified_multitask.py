"""Tests for eval.bench.protocols.unified_multitask."""

import unittest

from harnesscad.eval.bench.protocols.unified_multitask import (
    UNICAD_TASKS,
    cad_qa_accuracy,
    normalise_score,
    unified_score,
)


class TasksTest(unittest.TestCase):
    def test_five_tasks(self):
        self.assertEqual(len(UNICAD_TASKS), 5)
        self.assertIn("cadqa", [t.name for t in UNICAD_TASKS])


class NormaliseTest(unittest.TestCase):
    def test_lower_is_better_inverted(self):
        # chamfer distance 0 -> perfect 1.0
        self.assertAlmostEqual(normalise_score("textcad", 0.0, worst=2.0), 1.0)
        self.assertAlmostEqual(normalise_score("textcad", 2.0, worst=2.0), 0.0)

    def test_higher_is_better(self):
        self.assertAlmostEqual(normalise_score("cadqa", 0.8, worst=1.0), 0.8)

    def test_unknown_task(self):
        with self.assertRaises(ValueError):
            normalise_score("bogus", 1.0)

    def test_clamps(self):
        self.assertAlmostEqual(normalise_score("imagecad", 5.0, worst=2.0), 0.0)


class UnifiedTest(unittest.TestCase):
    def test_mean(self):
        raw = {"textcad": 0.0, "cadqa": 1.0}
        worst = {"textcad": 2.0, "cadqa": 1.0}
        self.assertAlmostEqual(unified_score(raw, worst), 1.0)

    def test_missing_cap(self):
        with self.assertRaises(ValueError):
            unified_score({"textcad": 0.5}, {})

    def test_empty(self):
        with self.assertRaises(ValueError):
            unified_score({}, {})


class QATest(unittest.TestCase):
    def test_exact_match_normalised(self):
        self.assertEqual(cad_qa_accuracy(["Two Holes"], ["two  holes"]), 1.0)

    def test_partial(self):
        acc = cad_qa_accuracy(["a", "b"], ["a", "c"])
        self.assertAlmostEqual(acc, 0.5)

    def test_length_mismatch(self):
        with self.assertRaises(ValueError):
            cad_qa_accuracy(["a"], ["a", "b"])


if __name__ == "__main__":
    unittest.main()
