"""Tests for Text2CAD token accuracy (bench.t2c3_token_accuracy)."""

import unittest

from harnesscad.eval.bench.sequence.token_accuracy import (
    DEFAULT_TOLERANCE,
    DISCARD_TOKEN,
    TokenAccuracyError,
    accuracy_from_logits,
    align,
    batch_token_accuracy,
    token_accuracy,
    value_mask,
)


class TestMaskAndAlign(unittest.TestCase):
    def test_structural_positions_dropped(self):
        target = [(1, 0), (5, 0), (6, 0), (20, 30)]
        self.assertEqual(value_mask(target), [False, False, False, True])

    def test_boolean_token_is_kept(self):
        # boolean ids 7..10 sit above the discard token in the first slot
        self.assertEqual(value_mask([(7, 0), (10, 0)]), [True, True])

    def test_discard_token_constant(self):
        self.assertEqual(DISCARD_TOKEN, 6)

    def test_align_truncates_to_shorter(self):
        p, t = align([(1, 1), (2, 2), (3, 3)], [(1, 1)])
        self.assertEqual(len(p), 1)
        self.assertEqual(len(t), 1)

    def test_bad_token_rejected(self):
        with self.assertRaises(TokenAccuracyError):
            token_accuracy([(1, 2, 3)], [(1, 2)])


class TestAccuracy(unittest.TestCase):
    def test_perfect(self):
        stream = [(20, 30), (40, 50), (5, 0)]
        score = token_accuracy(stream, stream)
        self.assertEqual((score.correct, score.total), (4, 4))   # 2 value positions
        self.assertEqual(score.accuracy, 1.0)

    def test_tolerance_is_strictly_less_than(self):
        target = [(20, 20)]
        self.assertEqual(token_accuracy([(22, 20)], target).correct, 2)  # |2| < 3
        self.assertEqual(token_accuracy([(23, 20)], target).correct, 1)  # |3| not < 3
        self.assertEqual(DEFAULT_TOLERANCE, 3)

    def test_slots_scored_independently(self):
        score = token_accuracy([(20, 99)], [(20, 20)])
        self.assertEqual((score.correct, score.total), (1, 2))
        self.assertEqual(score.accuracy, 0.5)

    def test_structural_targets_do_not_count(self):
        score = token_accuracy([(99, 99), (20, 20)], [(4, 0), (20, 20)])
        self.assertEqual(score.total, 2)
        self.assertEqual(score.accuracy, 1.0)

    def test_early_stop_only_scores_common_prefix(self):
        target = [(20, 20), (30, 30), (40, 40)]
        score = token_accuracy([(20, 20)], target)
        self.assertEqual(score.total, 2)
        self.assertEqual(score.accuracy, 1.0)

    def test_all_structural_gives_zero_total(self):
        score = token_accuracy([(1, 0)], [(1, 0)])
        self.assertEqual(score.total, 0)
        self.assertEqual(score.accuracy, 0.0)

    def test_custom_tolerance(self):
        self.assertEqual(token_accuracy([(25, 20)], [(20, 20)], tolerance=6).correct, 2)

    def test_non_positive_tolerance_rejected(self):
        with self.assertRaises(TokenAccuracyError):
            token_accuracy([(20, 20)], [(20, 20)], tolerance=0)


class TestBatch(unittest.TestCase):
    def test_counts_are_pooled_not_averaged(self):
        # sample A: 1 value position, both slots right; sample B: 2 positions, all wrong
        preds = [[(20, 20)], [(99, 99), (99, 99)]]
        targets = [[(20, 20)], [(20, 20), (30, 30)]]
        score = batch_token_accuracy(preds, targets)
        self.assertEqual((score.correct, score.total), (2, 6))
        self.assertAlmostEqual(score.accuracy, 1 / 3)

    def test_batch_size_mismatch(self):
        with self.assertRaises(TokenAccuracyError):
            batch_token_accuracy([[(20, 20)]], [])


class TestLogits(unittest.TestCase):
    def test_argmax_then_score(self):
        # 13 classes; the position's argmax is class 12 for slot x and 11 for slot y
        slot_x = [0.0] * 13
        slot_x[12] = 1.0
        slot_y = [0.0] * 13
        slot_y[11] = 1.0
        score = accuracy_from_logits([[slot_x, slot_y]], [(12, 11)])
        self.assertEqual((score.correct, score.total), (2, 2))

    def test_ties_break_low(self):
        logits = [[[1.0, 1.0, 0.0], [1.0, 1.0, 0.0]]]
        score = accuracy_from_logits(logits, [(0, 8)])
        self.assertEqual(score.total, 2)
        self.assertEqual(score.correct, 1)   # slot x within tolerance of 0, slot y not

    def test_malformed_position(self):
        with self.assertRaises(TokenAccuracyError):
            accuracy_from_logits([[[0.1, 0.2]]], [(20, 20)])


if __name__ == "__main__":
    unittest.main()
