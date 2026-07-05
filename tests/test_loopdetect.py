"""Tests for the LoopDetector oscillation guard."""

import unittest

from cisp.ops import Extrude, Fillet, NewSketch, Boolean
from reliability.loopdetect import LoopDetector, signature


class TestSignature(unittest.TestCase):
    def test_identical_ops_share_signature(self):
        self.assertEqual(
            signature(Extrude(sketch="sk1", distance=5.0)),
            signature(Extrude(sketch="sk1", distance=5.0)),
        )

    def test_differing_field_changes_signature(self):
        self.assertNotEqual(
            signature(Extrude(sketch="sk1", distance=5.0)),
            signature(Extrude(sketch="sk1", distance=6.0)),
        )

    def test_method_matches_function(self):
        det = LoopDetector()
        op = Fillet(edges=(1, 2), radius=2.0)
        self.assertEqual(det.signature(op), signature(op))


class TestLoopDetection(unittest.TestCase):
    def test_repeated_op_flagged_at_threshold(self):
        det = LoopDetector(window=5, threshold=3)
        op = Extrude(sketch="sk1", distance=5.0)
        self.assertFalse(det.observe(op))  # 1st occurrence
        self.assertFalse(det.observe(op))  # 2nd occurrence
        self.assertTrue(det.observe(op))   # 3rd == threshold -> loop

    def test_distinct_ops_not_flagged(self):
        det = LoopDetector(window=6, threshold=3)
        ops = [
            NewSketch(),
            Extrude(sketch="sk1", distance=1.0),
            Extrude(sketch="sk1", distance=2.0),
            Fillet(edges=(1,), radius=1.0),
            Boolean(kind="union", target="f1", tool="f2"),
        ]
        for op in ops:
            self.assertFalse(det.observe(op))

    def test_window_ages_out_old_repeats(self):
        # threshold 3 but the repeats are spread beyond the window, so the count
        # inside the window never reaches 3.
        det = LoopDetector(window=2, threshold=3)
        loop_op = Extrude(sketch="sk1", distance=5.0)
        other = NewSketch()
        self.assertFalse(det.observe(loop_op))
        self.assertFalse(det.observe(other))   # evicts nothing yet (window=2)
        self.assertFalse(det.observe(loop_op))  # window=[other, loop] -> count 1
        self.assertFalse(det.observe(other))
        self.assertFalse(det.observe(loop_op))  # still count 1 within window

    def test_reset_clears_history(self):
        det = LoopDetector(window=5, threshold=2)
        op = Extrude(sketch="sk1", distance=5.0)
        det.observe(op)
        self.assertTrue(det.observe(op))  # 2nd -> loop
        det.reset()
        self.assertFalse(det.observe(op))  # history cleared, 1st again


class TestConstruction(unittest.TestCase):
    def test_invalid_window(self):
        with self.assertRaises(ValueError):
            LoopDetector(window=0, threshold=2)

    def test_invalid_threshold(self):
        with self.assertRaises(ValueError):
            LoopDetector(window=5, threshold=1)


if __name__ == "__main__":
    unittest.main()
