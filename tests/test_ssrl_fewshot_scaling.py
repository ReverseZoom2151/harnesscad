"""Tests for the SSRL few-shot scaling-curve protocol."""

from __future__ import annotations

import random
import unittest

from bench.ssrl_fewshot_scaling import (
    stratified_subset,
    make_evaluator,
    scaling_curve,
    few_shot_advantage,
    ScalingPoint,
)


def _blob(cx, cy, n, spread=0.05):
    rng = random.Random("blob:%s:%s:%d" % (cx, cy, n))
    return [(cx + rng.uniform(-spread, spread), cy + rng.uniform(-spread, spread))
            for _ in range(n)]


# Two separated classes, plenty of samples for subset draws.
X = _blob(0.0, 0.0, 30) + _blob(5.0, 5.0, 30)
Y = ["a"] * 30 + ["b"] * 30
TEST_X = [(0.0, 0.0), (0.1, 0.0), (5.0, 5.0), (4.9, 5.1)]
TEST_Y = ["a", "a", "b", "b"]


class StratifiedSubsetTests(unittest.TestCase):
    def test_size_and_determinism(self):
        r1 = stratified_subset(Y, 10, random.Random(3))
        r2 = stratified_subset(Y, 10, random.Random(3))
        self.assertEqual(len(r1), 10)
        self.assertEqual(r1, r2)

    def test_stratified_covers_all_classes_when_possible(self):
        idx = stratified_subset(Y, 4, random.Random(1))
        labs = {Y[i] for i in idx}
        self.assertEqual(labs, {"a", "b"})

    def test_clamped_to_dataset_size(self):
        idx = stratified_subset(Y, 999, random.Random(1))
        self.assertEqual(len(idx), len(Y))

    def test_zero_size(self):
        self.assertEqual(stratified_subset(Y, 0, random.Random(1)), [])

    def test_sorted_indices(self):
        idx = stratified_subset(Y, 12, random.Random(9))
        self.assertEqual(idx, sorted(idx))


class ScalingCurveTests(unittest.TestCase):
    def test_curve_length_and_types(self):
        curve = scaling_curve(X, Y, TEST_X, TEST_Y, [2, 6, 20], repeats=3, seed=0)
        self.assertEqual(len(curve), 3)
        self.assertTrue(all(isinstance(p, ScalingPoint) for p in curve))
        self.assertEqual([p.size for p in curve], [2, 6, 20])

    def test_deterministic(self):
        c1 = scaling_curve(X, Y, TEST_X, TEST_Y, [4, 10], repeats=4, seed=7)
        c2 = scaling_curve(X, Y, TEST_X, TEST_Y, [4, 10], repeats=4, seed=7)
        self.assertEqual([p.accuracies for p in c1], [p.accuracies for p in c2])

    def test_accuracy_in_unit_interval(self):
        curve = scaling_curve(X, Y, TEST_X, TEST_Y, [2, 8], repeats=3, seed=1)
        for p in curve:
            self.assertGreaterEqual(p.mean_accuracy, 0.0)
            self.assertLessEqual(p.mean_accuracy, 1.0)
            self.assertEqual(len(p.accuracies), 3)

    def test_larger_sets_help_on_separable_data(self):
        # With separable classes, even 2 stratified samples should classify
        # the test set perfectly with a linear probe.
        curve = scaling_curve(X, Y, TEST_X, TEST_Y, [2, 30], repeats=5, seed=2,
                              evaluator=make_evaluator("linear", ridge=0.1))
        self.assertEqual(curve[-1].mean_accuracy, 1.0)

    def test_knn_evaluator(self):
        curve = scaling_curve(X, Y, TEST_X, TEST_Y, [4, 10], repeats=3, seed=5,
                              evaluator=make_evaluator("knn", k=1))
        self.assertEqual(curve[-1].mean_accuracy, 1.0)

    def test_spread_property(self):
        p = ScalingPoint(10, 0.8, (0.7, 0.9, 0.8))
        self.assertAlmostEqual(p.spread, 0.2, places=6)

    def test_validation(self):
        with self.assertRaises(ValueError):
            scaling_curve(X, Y[:3], TEST_X, TEST_Y, [2])
        with self.assertRaises(ValueError):
            scaling_curve(X, Y, TEST_X, TEST_Y, [2], repeats=0)
        with self.assertRaises(ValueError):
            scaling_curve(X, Y, TEST_X, TEST_Y, [0])
        with self.assertRaises(ValueError):
            scaling_curve([], [], TEST_X, TEST_Y, [2])

    def test_unknown_evaluator(self):
        with self.assertRaises(ValueError):
            make_evaluator("random-forest")


class FewShotAdvantageTests(unittest.TestCase):
    def test_gap_computation(self):
        a = [ScalingPoint(10, 0.9, (0.9,)), ScalingPoint(100, 0.95, (0.95,))]
        b = [ScalingPoint(10, 0.6, (0.6,)), ScalingPoint(100, 0.94, (0.94,))]
        gaps = few_shot_advantage(a, b)
        self.assertAlmostEqual(gaps[0], 0.3)
        self.assertAlmostEqual(gaps[1], 0.01, places=6)
        # Paper's claim: the few-shot gap is larger than the large-data gap.
        self.assertGreater(gaps[0], gaps[1])

    def test_misaligned_curves_raise(self):
        a = [ScalingPoint(10, 0.9, (0.9,))]
        b = [ScalingPoint(20, 0.6, (0.6,))]
        with self.assertRaises(ValueError):
            few_shot_advantage(a, b)
        with self.assertRaises(ValueError):
            few_shot_advantage(a, a + a)


if __name__ == "__main__":
    unittest.main()
