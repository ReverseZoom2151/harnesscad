"""Tests for datagen.gencad3d_synthbal (SynthBal dataset balancing)."""

import random
import unittest

from harnesscad.data.datagen.synthetic_balancing import (
    BalanceReport,
    balance_dataset,
    balanced_target,
    class_shares,
    imbalance_ratio,
    imbalance_report,
    length_histogram,
    perturb_noise,
    reduction_balance,
    replace_sketch,
    structural_valid,
    synthbal_augment,
    synthbal_dataset,
)


def _prog(sketch_len, cx=50, cy=60):
    """A small sketch-and-extrude program of a controllable command count."""
    cmds = [{"type": "SOL"}]
    for i in range(sketch_len):
        cmds.append({"type": "L", "x": cx + i, "y": cy})
    cmds.append({"type": "E", "w": 0, "delta1": 30, "delta2": 40, "orientation": 90})
    return cmds


def _skewed_dataset():
    # Length imbalance: many short, few long.
    data = []
    for _ in range(20):
        data.append(_prog(1))   # length 3
    for _ in range(6):
        data.append(_prog(2))   # length 4
    for _ in range(2):
        data.append(_prog(4))   # length 6
    return data


class PerturbNoiseTests(unittest.TestCase):
    def test_deterministic(self):
        p = _prog(3)
        a = perturb_noise(p, 7)
        b = perturb_noise(p, 7)
        self.assertEqual(a, b)

    def test_discrete_values_preserved(self):
        p = [{"type": "E", "w": 2, "delta1": 100, "orientation": 90}]
        out = perturb_noise(p, 1, magnitude=0.5, prob=1.0)
        self.assertEqual(out[0]["w"], 2)          # discrete type untouched
        self.assertEqual(out[0]["orientation"], 90)  # near-discrete untouched
        self.assertEqual(out[0]["type"], "E")

    def test_continuous_clipped_to_range(self):
        p = [{"type": "L", "x": 250, "y": 3}]
        out = perturb_noise(p, 2, magnitude=1.0, prob=1.0)
        self.assertTrue(1 <= out[0]["x"] <= 255)
        self.assertTrue(1 <= out[0]["y"] <= 255)

    def test_zero_prob_no_change_to_continuous(self):
        p = _prog(3)
        out = perturb_noise(p, 3, prob=0.0)
        self.assertEqual(out, p)

    def test_bad_prob(self):
        with self.assertRaises(ValueError):
            perturb_noise(_prog(1), 0, prob=1.5)


class ReplaceSketchTests(unittest.TestCase):
    def test_deterministic(self):
        a = replace_sketch(_prog(2, cx=10), _prog(3, cx=200), 5, replace_prob=1.0)
        b = replace_sketch(_prog(2, cx=10), _prog(3, cx=200), 5, replace_prob=1.0)
        self.assertEqual(a, b)

    def test_keeps_own_extrude_swaps_sketch(self):
        mine = _prog(1, cx=10)
        donor = _prog(3, cx=200)
        out = replace_sketch(mine, donor, 1, replace_prob=1.0)
        # extrude preserved from mine
        self.assertEqual(out[-1]["type"], "E")
        # sketch now has donor's line count (3) rather than mine's (1)
        n_lines = sum(1 for c in out if c["type"] == "L")
        self.assertEqual(n_lines, 3)

    def test_donor_without_extrude_returns_unchanged(self):
        mine = _prog(2)
        donor = [{"type": "SOL"}, {"type": "L", "x": 1, "y": 2}]  # no extrude
        out = replace_sketch(mine, donor, 0, replace_prob=1.0)
        self.assertEqual(out, mine)


class SynthbalAugmentTests(unittest.TestCase):
    def test_deterministic(self):
        base, donor = _prog(3, cx=10), _prog(2, cx=100)
        self.assertEqual(synthbal_augment(base, donor, 9),
                         synthbal_augment(base, donor, 9))

    def test_produces_valid_program(self):
        base, donor = _prog(3, cx=10), _prog(2, cx=100)
        out = synthbal_augment(base, donor, 4)
        self.assertTrue(structural_valid(out))


class StructuralValidTests(unittest.TestCase):
    def test_valid_program(self):
        self.assertTrue(structural_valid(_prog(2)))

    def test_missing_extrude_invalid(self):
        self.assertFalse(structural_valid([{"type": "SOL"}, {"type": "L", "x": 1, "y": 2}]))

    def test_out_of_range_invalid(self):
        self.assertFalse(structural_valid([{"type": "L", "x": 900, "y": 2},
                                           {"type": "E", "delta1": 5}]))

    def test_empty_invalid(self):
        self.assertFalse(structural_valid([]))


class ImbalanceMetricTests(unittest.TestCase):
    def test_histogram(self):
        hist = length_histogram(_skewed_dataset())
        self.assertEqual(hist[3], 20)
        self.assertEqual(hist[4], 6)
        self.assertEqual(hist[6], 2)

    def test_shares_sum_to_one(self):
        shares = class_shares(length_histogram(_skewed_dataset()))
        self.assertAlmostEqual(sum(shares.values()), 1.0)

    def test_imbalance_ratio(self):
        hist = length_histogram(_skewed_dataset())
        self.assertEqual(imbalance_ratio(hist), 20 / 2)

    def test_balanced_target_uniform(self):
        tgt = balanced_target([3, 4, 6, 6])
        self.assertEqual(set(tgt), {3, 4, 6})
        for v in tgt.values():
            self.assertAlmostEqual(v, 1 / 3)

    def test_imbalance_report(self):
        rep = imbalance_report(_skewed_dataset())
        self.assertEqual(rep["n_items"], 28)
        self.assertEqual(rep["n_lengths"], 3)


class BalanceDatasetTests(unittest.TestCase):
    def test_deterministic(self):
        data = _skewed_dataset()
        b1, r1 = balance_dataset(data, target_size=30, real_ratio=0.2, seed=0)
        b2, r2 = balance_dataset(data, target_size=30, real_ratio=0.2, seed=0)
        self.assertEqual(b1, b2)
        self.assertEqual(r1.to_dict(), r2.to_dict())

    def test_balances_across_lengths(self):
        data = _skewed_dataset()
        balanced, report = balance_dataset(data, target_size=30, real_ratio=0.2, seed=1)
        # 3 lengths, target 30 -> 10 per length
        self.assertEqual(report.per_length_target, 10)
        hist = length_histogram(balanced)
        for ell in (3, 4, 6):
            self.assertEqual(hist[ell], 10)
        # balanced dataset is far more even than the original
        self.assertLess(imbalance_ratio(hist), imbalance_ratio(length_histogram(data)))

    def test_real_ratio_caps_real_data(self):
        data = _skewed_dataset()
        # r=0.2, n_per=10 -> at most 2 real per length; length 6 has 2 real available.
        balanced, report = balance_dataset(data, target_size=30, real_ratio=0.2, seed=2)
        for ell in (3, 4, 6):
            self.assertLessEqual(report.real_counts[ell], 2)
        self.assertGreater(report.total_synthetic, 0)

    def test_all_produced_valid(self):
        data = _skewed_dataset()
        balanced, _ = balance_dataset(data, target_size=30, real_ratio=0.2, seed=3)
        self.assertTrue(all(structural_valid(p) for p in balanced))

    def test_synthetic_fraction(self):
        data = _skewed_dataset()
        _, report = balance_dataset(data, target_size=30, real_ratio=0.2, seed=4)
        self.assertIsInstance(report, BalanceReport)
        self.assertGreater(report.synthetic_fraction(), 0.5)

    def test_invalid_augmenter_terminates(self):
        data = _skewed_dataset()
        bad = lambda prog, donor, seed: []  # always structurally invalid
        balanced, report = balance_dataset(data, target_size=30, real_ratio=0.2,
                                           seed=5, augment=bad, max_tries_per_slot=3)
        # only real programs admitted; no synthetic
        self.assertEqual(report.total_synthetic, 0)
        self.assertGreater(report.rejected, 0)

    def test_bad_ratio(self):
        with self.assertRaises(ValueError):
            balance_dataset(_skewed_dataset(), 10, 2.0, seed=0)

    def test_empty_dataset(self):
        balanced, report = balance_dataset([], 10, 0.2, seed=0)
        self.assertEqual(balanced, [])
        self.assertEqual(report.per_length_target, 0)

    def test_synthbal_dataset_wrapper(self):
        data = _skewed_dataset()
        balanced, report = synthbal_dataset(data, target_size=30, seed=7)
        self.assertEqual(len(length_histogram(balanced)), 3)


class ReductionBalanceTests(unittest.TestCase):
    def test_purely_real_and_balanced(self):
        data = _skewed_dataset()
        reduced = reduction_balance(data, seed=0)
        hist = length_histogram(reduced)
        # smallest bucket = 2, so every length reduced to 2
        for ell in (3, 4, 6):
            self.assertEqual(hist[ell], 2)

    def test_deterministic(self):
        data = _skewed_dataset()
        self.assertEqual(reduction_balance(data, 1), reduction_balance(data, 1))

    def test_explicit_per_length(self):
        data = _skewed_dataset()
        reduced = reduction_balance(data, seed=0, per_length=1)
        hist = length_histogram(reduced)
        for ell in (3, 4, 6):
            self.assertEqual(hist[ell], 1)

    def test_empty(self):
        self.assertEqual(reduction_balance([], 0), [])


if __name__ == "__main__":
    unittest.main()
