"""Tests for bench/contrastcad_recon_accuracy.py — ACC_cmd / ACC_param (Eq. 8)."""

import unittest

from bench.contrastcad_recon_accuracy import (
    N_PARAM_SLOTS,
    command_accuracy,
    parameter_accuracy,
    reconstruction_accuracy,
)


def _seq():
    return [
        {"type": "SOL"},
        {"type": "L", "x": 10, "y": 20},
        {"type": "L", "x": 30, "y": 40},
        {"type": "E", "w": 0, "delta1": 100, "delta2": 0},
        {"type": "EOS"},
    ]


class TestCommandAccuracy(unittest.TestCase):
    def test_perfect(self):
        self.assertEqual(command_accuracy(_seq(), _seq()), 1.0)

    def test_one_mismatch(self):
        actual = _seq()
        actual[1] = {"type": "A", "x": 10, "y": 20}
        self.assertAlmostEqual(command_accuracy(actual, _seq()), 4 / 5)

    def test_length_mismatch(self):
        with self.assertRaises(ValueError):
            command_accuracy(_seq(), _seq()[:-1])


class TestParameterAccuracy(unittest.TestCase):
    def test_perfect(self):
        self.assertEqual(parameter_accuracy(_seq(), _seq()), 1.0)

    def test_tolerance_boundary(self):
        actual = _seq()
        actual[1] = {"type": "L", "x": 12, "y": 20}  # x off by 2 < eta(3)
        self.assertAlmostEqual(parameter_accuracy(actual, _seq(), eta=3), 1.0)

    def test_outside_tolerance(self):
        actual = _seq()
        actual[1] = {"type": "L", "x": 20, "y": 20}  # x off by 10
        # one of the 16 slots for this command is now wrong
        acc = parameter_accuracy(actual, _seq(), eta=3)
        expected_wrong = 1
        # All 5 positions keep their type, each contributing 16 fixed slots (T).
        total = 5 * N_PARAM_SLOTS
        self.assertAlmostEqual(acc, (total - expected_wrong) / total)

    def test_type_gated(self):
        # A mistyped command contributes nothing to T (denominator).
        actual = _seq()
        actual[1] = {"type": "A", "x": 10, "y": 20}
        # remaining correctly-typed param-bearing commands: L(idx2), E(idx3)
        self.assertAlmostEqual(parameter_accuracy(actual, _seq()), 1.0)

    def test_no_type_match_is_zero(self):
        actual = [{"type": "L", "x": 1, "y": 1}]
        expected = [{"type": "C", "x": 1, "y": 1, "r": 1}]
        self.assertEqual(parameter_accuracy(actual, expected), 0.0)

    def test_negative_eta_raises(self):
        with self.assertRaises(ValueError):
            parameter_accuracy(_seq(), _seq(), eta=-1)


class TestReconstructionAccuracy(unittest.TestCase):
    def test_bundle(self):
        out = reconstruction_accuracy(_seq(), _seq())
        self.assertEqual(out, {"acc_cmd": 1.0, "acc_param": 1.0})


if __name__ == "__main__":
    unittest.main()
