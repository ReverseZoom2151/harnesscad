"""Tests for DeepCAD's official autoencoder accuracy metric."""

import unittest

from harnesscad.eval.bench.sequence import autoencoder_accuracy as acc
from harnesscad.domain.reconstruction.tokens import deepcad_vector_layout as vl


def _ext(**over):
    args = dict(theta=128, phi=128, gamma=128, px=100, py=100, pz=100,
                s=120, e1=140, e2=128, b=1, u=0)
    args.update(over)
    return vl.ext_row(**args)


def _model():
    """SOL, Line, Arc, Circle, Ext, EOS -- one row of every command type."""
    return [vl.sol_row(), vl.line_row(10, 20), vl.arc_row(30, 40, 64, 1),
            vl.circle_row(50, 60, 12), _ext(), vl.eos_row()]


class TestCommandAccuracy(unittest.TestCase):
    def test_perfect(self):
        gt = _model()
        self.assertEqual(acc.command_accuracy(gt, gt), 1.0)

    def test_one_wrong_type(self):
        gt = _model()
        out = list(gt)
        out[1] = vl.circle_row(10, 20, 5)
        self.assertAlmostEqual(acc.command_accuracy(out, gt), 5 / 6)

    def test_length_mismatch_raises(self):
        gt = _model()
        with self.assertRaises(ValueError):
            acc.command_accuracy(gt[:-1], gt)


class TestParameterAccuracy(unittest.TestCase):
    def test_perfect(self):
        gt = _model()
        self.assertEqual(acc.parameter_accuracy(gt, gt), 1.0)

    def test_within_tolerance_counts_as_hit(self):
        gt = _model()
        out = list(gt)
        out[1] = vl.line_row(12, 22)   # +2, +2 -> both < 3
        self.assertEqual(acc.parameter_accuracy(out, gt), 1.0)

    def test_tolerance_is_strictly_less_than_three(self):
        gt = _model()
        out = list(gt)
        out[1] = vl.line_row(13, 20)   # +3 -> miss
        # scored args: Line 2 + Arc 4 + Circle 3 + Ext 11 = 20, one miss
        self.assertAlmostEqual(acc.parameter_accuracy(out, gt), 19 / 20)

    def test_arc_flag_needs_exact_match(self):
        gt = _model()
        out = list(gt)
        out[2] = vl.arc_row(30, 40, 64, 0)   # f differs by 1 -- within tolerance,
        self.assertAlmostEqual(acc.parameter_accuracy(out, gt), 19 / 20)  # but strict

    def test_arc_sweep_angle_is_tolerant(self):
        gt = _model()
        out = list(gt)
        out[2] = vl.arc_row(30, 40, 66, 1)   # alpha +2 -> tolerated
        self.assertEqual(acc.parameter_accuracy(out, gt), 1.0)

    def test_ext_boolean_and_extent_need_exact_match(self):
        gt = _model()
        out = list(gt)
        out[4] = _ext(b=2, u=2)     # both differ by <3 but are categorical
        self.assertAlmostEqual(acc.parameter_accuracy(out, gt), 18 / 20)

    def test_ext_extent_distance_is_tolerant(self):
        gt = _model()
        out = list(gt)
        out[4] = _ext(e1=142)
        self.assertEqual(acc.parameter_accuracy(out, gt), 1.0)

    def test_mistyped_command_contributes_no_params(self):
        gt = _model()
        out = list(gt)
        out[1] = vl.circle_row(10, 20, 5)     # Line predicted as Circle
        # its 2 args drop out of the denominator entirely: 18/18
        self.assertEqual(acc.parameter_accuracy(out, gt), 1.0)

    def test_sol_and_eos_are_not_scored(self):
        gt = [vl.sol_row(), vl.eos_row()]
        self.assertEqual(acc.parameter_accuracy(gt, gt), 0.0)  # nothing scorable

    def test_padded_slots_never_scored(self):
        gt = [vl.sol_row(), vl.line_row(10, 20), vl.eos_row()]
        out = [vl.sol_row(), vl.line_row(10, 20), vl.eos_row()]
        self.assertEqual(acc.parameter_accuracy(out, gt), 1.0)
        # only the 2 masked Line slots are scored
        self.assertEqual(len(vl.used_args(gt[1])), 2)


class TestSlotHits(unittest.TestCase):
    def test_mask_width(self):
        gt = _model()
        self.assertEqual(len(acc.slot_hits(gt[1], gt[1])), 16)

    def test_hits_include_padded_slots_before_masking(self):
        gt = _model()
        hits = acc.slot_hits(gt[1], gt[1])
        self.assertEqual(sum(hits), 16)  # -1 vs -1 also "hits"; the mask removes them


class TestDataset(unittest.TestCase):
    def test_macro_average_over_models(self):
        gt = _model()
        bad = list(gt)
        bad[1] = vl.circle_row(10, 20, 5)
        res = acc.evaluate_dataset([(gt, gt), (bad, gt)])
        self.assertAlmostEqual(res["acc_cmd"], (1.0 + 5 / 6) / 2)
        self.assertAlmostEqual(res["acc_param"], 1.0)

    def test_per_command_breakdown(self):
        gt = _model()
        bad = list(gt)
        bad[1] = vl.circle_row(10, 20, 5)
        res = acc.evaluate_dataset([(gt, gt), (bad, gt)])
        self.assertEqual(res["each_cmd_count"]["Line"], 2)
        self.assertAlmostEqual(res["each_cmd_acc"]["Line"], 0.5)
        self.assertAlmostEqual(res["each_cmd_acc"]["Arc"], 1.0)

    def test_per_param_breakdown_lists_only_masked_slots(self):
        gt = _model()
        res = acc.evaluate_dataset([(gt, gt)])
        self.assertEqual(set(res["each_param_acc"]["Line"]), {"x", "y"})
        self.assertEqual(set(res["each_param_acc"]["Arc"]), {"x", "y", "alpha", "f"})
        self.assertEqual(set(res["each_param_acc"]["Circle"]), {"x", "y", "r"})
        self.assertEqual(len(res["each_param_acc"]["Ext"]), 11)
        self.assertNotIn("SOL", res["each_param_acc"])
        self.assertNotIn("EOS", res["each_param_acc"])

    def test_per_param_accuracy_values(self):
        gt = _model()
        out = list(gt)
        out[2] = vl.arc_row(30, 40, 64, 0)   # flag wrong
        res = acc.evaluate_dataset([(out, gt)])
        self.assertAlmostEqual(res["each_param_acc"]["Arc"]["f"], 0.0)
        self.assertAlmostEqual(res["each_param_acc"]["Arc"]["x"], 1.0)

    def test_evaluate_model_shape(self):
        gt = _model()
        res = acc.evaluate_model(gt, gt)
        self.assertEqual(set(res), {"acc_cmd", "acc_param"})

    def test_empty_dataset_raises(self):
        with self.assertRaises(ValueError):
            acc.evaluate_dataset([])


if __name__ == "__main__":
    unittest.main()
