"""Tests for bench.gencad2_loss_masks."""

import math
import unittest

from harnesscad.eval.bench.sequence.loss_masks import (
    ARC_IDX,
    ARGS_DIM,
    CIRCLE_IDX,
    CMD_ARGS_MASK,
    EOS_IDX,
    EXT_IDX,
    LINE_IDX,
    N_ARGS,
    N_ARGS_EXT,
    PAD_VAL,
    SOL_IDX,
    arg_loss_mask,
    args_mask,
    args_vocab_size,
    autoregressive_pairs,
    cad_loss,
    ccip_ground_truth,
    ccip_loss,
    clamped_logit_scale,
    command_loss_mask,
    cross_entropy,
    log_softmax,
    logit_matrix,
    mean_cross_entropy,
    padding_mask,
    selected_arg_tokens,
    selected_command_tokens,
    shift_arg_target,
    unshift_arg_target,
    used_arg_slots,
    visibility_mask,
)


class TestMaskTable(unittest.TestCase):
    def test_shape(self):
        self.assertEqual(len(CMD_ARGS_MASK), 6)
        for row in CMD_ARGS_MASK:
            self.assertEqual(len(row), N_ARGS)
        self.assertEqual(N_ARGS, 16)
        self.assertEqual(N_ARGS_EXT, 11)

    def test_per_command_slots(self):
        self.assertEqual(used_arg_slots(LINE_IDX), (0, 1))
        self.assertEqual(used_arg_slots(ARC_IDX), (0, 1, 2, 3))
        self.assertEqual(used_arg_slots(CIRCLE_IDX), (0, 1, 4))
        self.assertEqual(used_arg_slots(SOL_IDX), ())
        self.assertEqual(used_arg_slots(EOS_IDX), ())
        self.assertEqual(used_arg_slots(EXT_IDX), tuple(range(5, 16)))

    def test_unknown_command(self):
        with self.assertRaises(ValueError):
            args_mask(9)

    def test_args_vocab_and_shift(self):
        self.assertEqual(args_vocab_size(), ARGS_DIM + 1)
        self.assertEqual(shift_arg_target(PAD_VAL), 0)
        self.assertEqual(shift_arg_target(255), 256)
        self.assertEqual(unshift_arg_target(shift_arg_target(17)), 17)


class TestMasks(unittest.TestCase):
    def test_padding_mask_stops_at_eos(self):
        cmds = [SOL_IDX, LINE_IDX, LINE_IDX, EOS_IDX, EOS_IDX, EOS_IDX]
        self.assertEqual(padding_mask(cmds), [1, 1, 1, 1, 0, 0])

    def test_extended_mask_covers_more(self):
        cmds = [SOL_IDX, LINE_IDX, LINE_IDX, EOS_IDX, EOS_IDX, EOS_IDX]
        base = padding_mask(cmds)
        ext = padding_mask(cmds, extended=True)
        self.assertEqual(ext[:4], base[:4])
        self.assertEqual(ext[4], 1)  # pulled in by the shift-by-3 OR
        self.assertTrue(all(v in (0, 1) for v in ext))

    def test_extended_uses_unshifted_source(self):
        cmds = [SOL_IDX, LINE_IDX, EOS_IDX] + [EOS_IDX] * 5
        ext = padding_mask(cmds, extended=True)
        self.assertEqual(ext, [1, 1, 1, 1, 1, 1, 0, 0])

    def test_visibility_mask(self):
        self.assertEqual(visibility_mask([SOL_IDX, LINE_IDX, EOS_IDX, EOS_IDX]), 1)
        self.assertEqual(visibility_mask([EOS_IDX] * 4), 0)
        self.assertEqual(visibility_mask([LINE_IDX] + [EOS_IDX] * 3), 0)

    def test_command_loss_mask_zeroed_when_invisible(self):
        self.assertEqual(command_loss_mask([EOS_IDX] * 5), [0] * 5)

    def test_arg_loss_mask_rows(self):
        rows = arg_loss_mask([SOL_IDX, LINE_IDX])
        self.assertEqual(sum(rows[0]), 0)
        self.assertEqual(sum(rows[1]), 2)


class TestAutoregressiveSelection(unittest.TestCase):
    def setUp(self):
        self.cmds = [SOL_IDX, LINE_IDX, CIRCLE_IDX, EXT_IDX, EOS_IDX, EOS_IDX]

    def test_pairs(self):
        targets, positions = autoregressive_pairs(self.cmds)
        self.assertEqual(targets, self.cmds[1:])
        self.assertEqual(positions, [0, 1, 2, 3, 4])

    def test_short_sequence_raises(self):
        with self.assertRaises(ValueError):
            autoregressive_pairs([EOS_IDX])

    def test_selected_command_tokens_include_terminal_eos(self):
        sel = selected_command_tokens(self.cmds)
        # targets = [Line, Circle, Ext, EOS, EOS]; the base padding mask covers
        # through the first EOS (0..3) and the shift-by-3 extension pulls in 4.
        self.assertEqual(sel, [0, 1, 2, 3, 4])

    def test_selected_arg_tokens(self):
        sel = selected_arg_tokens(self.cmds)
        # Line -> 2 slots, Circle -> 3, Ext -> 11, EOS -> 0
        self.assertEqual(len(sel), 2 + 3 + 11)
        self.assertIn((0, 0), sel)
        self.assertIn((1, 4), sel)   # circle radius slot
        self.assertIn((2, 15), sel)  # extrude's last slot

    def test_invisible_sequence_selects_no_commands(self):
        self.assertEqual(selected_command_tokens([EOS_IDX] * 6), [])


class TestCrossEntropy(unittest.TestCase):
    def test_log_softmax_sums_to_one(self):
        ls = log_softmax([1.0, 2.0, 3.0])
        self.assertAlmostEqual(sum(math.exp(v) for v in ls), 1.0, places=12)

    def test_uniform_logits(self):
        self.assertAlmostEqual(cross_entropy([0.0, 0.0, 0.0, 0.0], 2),
                               math.log(4), places=12)

    def test_confident_correct_is_near_zero(self):
        self.assertLess(cross_entropy([10.0, 0.0], 0), 1e-4)

    def test_bad_target_raises(self):
        with self.assertRaises(ValueError):
            cross_entropy([0.0, 1.0], 5)

    def test_mean_and_mismatch(self):
        self.assertAlmostEqual(
            mean_cross_entropy([[0.0, 0.0], [0.0, 0.0]], [0, 1]), math.log(2), places=12)
        with self.assertRaises(ValueError):
            mean_cross_entropy([[0.0, 0.0]], [0, 1])


class TestCadLoss(unittest.TestCase):
    def _fixture(self):
        cmds = [SOL_IDX, LINE_IDX, EXT_IDX, EOS_IDX]
        args = [[PAD_VAL] * N_ARGS,
                [10, 20] + [PAD_VAL] * 14,
                [PAD_VAL] * 5 + list(range(11)),
                [PAD_VAL] * N_ARGS]
        cmd_logits = [[0.0] * 6 for _ in cmds]
        arg_logits = [[[0.0] * args_vocab_size() for _ in range(N_ARGS)]
                      for _ in cmds]
        return cmds, args, cmd_logits, arg_logits

    def test_uniform_logits_give_log_n(self):
        cmds, args, cl, al = self._fixture()
        out = cad_loss(cmds, args, cl, al, loss_cmd_weight=1.0, loss_args_weight=1.0)
        self.assertAlmostEqual(out["loss_cmd"], math.log(6), places=9)
        self.assertAlmostEqual(out["loss_args"], math.log(257), places=9)

    def test_weights_applied(self):
        cmds, args, cl, al = self._fixture()
        out = cad_loss(cmds, args, cl, al, loss_cmd_weight=1.0, loss_args_weight=2.0)
        self.assertAlmostEqual(out["loss_args"], 2 * math.log(257), places=9)

    def test_correct_predictions_lower_loss(self):
        cmds, args, cl, al = self._fixture()
        targets = cmds[1:]
        for t, c in enumerate(targets):
            cl[t] = [0.0] * 6
            cl[t][c] = 20.0
        for t, slot in selected_arg_tokens(cmds):
            row = [0.0] * args_vocab_size()
            row[shift_arg_target(args[t + 1][slot])] = 20.0
            al[t][slot] = row
        out = cad_loss(cmds, args, cl, al)
        self.assertLess(out["loss_cmd"], 1e-5)
        self.assertLess(out["loss_args"], 1e-5)

    def test_padding_argument_target_is_class_zero(self):
        # an Ext slot holding PAD_VAL must be scored against class 0
        cmds, args, cl, al = self._fixture()
        self.assertEqual(shift_arg_target(args[0][0]), 0)

    def test_logit_length_mismatch_raises(self):
        cmds, args, cl, al = self._fixture()
        with self.assertRaises(ValueError):
            cad_loss(cmds, args, cl[:-1], al)

    def test_deterministic(self):
        cmds, args, cl, al = self._fixture()
        self.assertEqual(cad_loss(cmds, args, cl, al), cad_loss(cmds, args, cl, al))


class TestCCIP(unittest.TestCase):
    def test_ground_truth(self):
        self.assertEqual(ccip_ground_truth(3), [0, 1, 2])
        with self.assertRaises(ValueError):
            ccip_ground_truth(0)

    def test_logit_matrix(self):
        m = logit_matrix([[1.0, 0.0], [0.0, 1.0]], [[1.0, 0.0], [0.0, 1.0]], 2.0)
        self.assertEqual(m, [[2.0, 0.0], [0.0, 2.0]])

    def test_perfectly_aligned_batch_has_low_loss(self):
        feats = [[1.0, 0.0], [0.0, 1.0]]
        self.assertLess(ccip_loss(feats, feats, 50.0), 1e-6)

    def test_orthogonal_batch_is_log_n(self):
        feats = [[1.0, 0.0], [0.0, 1.0]]
        self.assertAlmostEqual(ccip_loss(feats, feats, 0.0), math.log(2), places=12)

    def test_symmetric_in_its_arguments(self):
        a = [[1.0, 0.2], [0.1, 1.0]]
        b = [[0.9, 0.3], [0.2, 0.8]]
        self.assertAlmostEqual(ccip_loss(a, b, 5.0), ccip_loss(b, a, 5.0), places=12)

    def test_mismatched_batch_raises(self):
        with self.assertRaises(ValueError):
            ccip_loss([[1.0]], [[1.0], [0.0]], 1.0)

    def test_empty_batch_raises(self):
        with self.assertRaises(ValueError):
            ccip_loss([], [], 1.0)

    def test_clamped_logit_scale(self):
        self.assertAlmostEqual(clamped_logit_scale(0.0), 1.0, places=12)
        self.assertEqual(clamped_logit_scale(100.0), 100.0)


if __name__ == "__main__":
    unittest.main()
