"""Tests for dataengine.creft_data_engine."""

import random
import unittest

from harnesscad.data.dataengine.creft_data_engine import (
    build_cot_steps,
    corrupt_parameters,
    make_dichotomous_samples,
    make_multiple_choice_sample,
)


GT = {"a": 1, "b": 2, "c": 3, "d": 4}


class CorruptTest(unittest.TestCase):
    def test_corrupts_exactly_n(self):
        rng = random.Random(0)
        out = corrupt_parameters(GT, 2, rng)
        changed = [k for k in GT if out[k] != GT[k]]
        self.assertEqual(len(changed), 2)

    def test_zero_leaves_unchanged(self):
        out = corrupt_parameters(GT, 0, random.Random(0))
        self.assertEqual(out, GT)

    def test_domain_values(self):
        rng = random.Random(1)
        out = corrupt_parameters({"a": 1}, 1, rng, {"a": [1, 5, 6]})
        self.assertIn(out["a"], [5, 6])

    def test_out_of_range(self):
        with self.assertRaises(ValueError):
            corrupt_parameters(GT, 99, random.Random(0))


class DichotomousTest(unittest.TestCase):
    def test_balanced_split(self):
        samples = make_dichotomous_samples(GT, 10, seed=42)
        positives = [s for s in samples if s.label]
        negatives = [s for s in samples if not s.label]
        self.assertEqual(len(positives), 5)
        self.assertEqual(len(negatives), 5)

    def test_positive_is_ground_truth(self):
        samples = make_dichotomous_samples(GT, 2, seed=42)
        self.assertTrue(samples[0].label)
        self.assertEqual(samples[0].parameters, GT)

    def test_negative_has_errors(self):
        samples = make_dichotomous_samples(GT, 4, seed=7)
        for s in samples:
            if not s.label:
                self.assertGreaterEqual(s.num_errors, 1)
                diffs = [k for k in GT if s.parameters[k] != GT[k]]
                self.assertEqual(len(diffs), s.num_errors)

    def test_deterministic(self):
        a = make_dichotomous_samples(GT, 6, seed=99)
        b = make_dichotomous_samples(GT, 6, seed=99)
        self.assertEqual([s.parameters for s in a], [s.parameters for s in b])


class MultipleChoiceTest(unittest.TestCase):
    def test_masking(self):
        s = make_multiple_choice_sample(GT, p=2, q=1, num_candidates=4, seed=3)
        self.assertEqual(len(s.masked), 2)
        # candidates only carry unmasked keys
        for cand in s.candidates.values():
            for m in s.masked:
                self.assertNotIn(m, cand)

    def test_c0_always_correct(self):
        s = make_multiple_choice_sample(GT, p=1, q=2, num_candidates=4, seed=3)
        self.assertIn("c0", s.correct_ids)
        for k, v in s.candidates["c0"].items():
            self.assertEqual(v, GT[k])

    def test_incorrect_candidates_flip_q(self):
        s = make_multiple_choice_sample(GT, p=0, q=2, num_candidates=4, seed=5)
        for cid, cand in s.candidates.items():
            if cid not in s.correct_ids:
                diffs = [k for k in cand if cand[k] != GT[k]]
                self.assertEqual(len(diffs), 2)

    def test_deterministic(self):
        a = make_multiple_choice_sample(GT, p=1, q=1, num_candidates=3, seed=11)
        b = make_multiple_choice_sample(GT, p=1, q=1, num_candidates=3, seed=11)
        self.assertEqual(a.candidates, b.candidates)
        self.assertEqual(a.masked, b.masked)

    def test_invalid_p(self):
        with self.assertRaises(ValueError):
            make_multiple_choice_sample(GT, p=99, q=1, num_candidates=3, seed=1)


class CoTTest(unittest.TestCase):
    def test_steps(self):
        steps = build_cot_steps("spacing", "+", ("pier", "pile"),
                                {"pier": 3, "pile": 4})
        self.assertEqual([s.kind for s in steps],
                         ["identify", "formula", "compute"])
        self.assertIn("pier", steps[0].detail)
        self.assertIn("spacing = pier + pile", steps[1].detail)
        self.assertIn("7", steps[2].detail)

    def test_multiply(self):
        steps = build_cot_steps("area", "*", ("w", "h"), {"w": 2, "h": 5})
        self.assertIn("10", steps[2].detail)


if __name__ == "__main__":
    unittest.main()
