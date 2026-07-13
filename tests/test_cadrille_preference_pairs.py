"""Tests for cadrille DPO preference-pair construction from K samples."""

import unittest

from harnesscad.data.dataengine.cadrille_preference_pairs import (
    preference_pair,
    all_preference_pairs,
    sample_preference_pairs,
    to_dpo_records,
)


def _s(code, reward):
    return {"code": code, "reward": reward}


class PreferencePairTest(unittest.TestCase):
    def test_orders_by_reward(self):
        a, b = _s("a", 3.0), _s("b", 9.0)
        chosen, rejected = preference_pair(a, b)
        self.assertEqual(chosen["code"], "b")
        self.assertEqual(rejected["code"], "a")

    def test_tie_dropped(self):
        self.assertIsNone(preference_pair(_s("a", 5.0), _s("b", 5.0)))

    def test_all_pairs(self):
        samples = [_s("a", 1.0), _s("b", 2.0), _s("c", 2.0)]
        pairs = all_preference_pairs(samples)
        # (a,b),(a,c) valid; (b,c) tie dropped
        self.assertEqual(len(pairs), 2)
        for chosen, rejected in pairs:
            self.assertGreater(chosen["reward"], rejected["reward"])


class SamplingTest(unittest.TestCase):
    def test_deterministic_with_seed(self):
        samples = [_s(str(i), float(i)) for i in range(5)]
        p1 = sample_preference_pairs(samples, 10, seed=42)
        p2 = sample_preference_pairs(samples, 10, seed=42)
        self.assertEqual(
            [(c["code"], r["code"]) for c, r in p1],
            [(c["code"], r["code"]) for c, r in p2],
        )

    def test_chosen_always_higher(self):
        samples = [_s(str(i), float(i)) for i in range(5)]
        for chosen, rejected in sample_preference_pairs(samples, 20, seed=7):
            self.assertGreater(chosen["reward"], rejected["reward"])

    def test_needs_two_samples(self):
        with self.assertRaises(ValueError):
            sample_preference_pairs([_s("a", 1.0)], 3, seed=0)

    def test_ties_reduce_output(self):
        # all equal reward -> every draw is a tie -> no pairs emitted
        samples = [_s(str(i), 1.0) for i in range(4)]
        self.assertEqual(sample_preference_pairs(samples, 5, seed=1), [])


class RecordsTest(unittest.TestCase):
    def test_to_dpo_records(self):
        pairs = [(_s("win", 9.0), _s("lose", 1.0))]
        rows = to_dpo_records(pairs, prompt="reconstruct")
        self.assertEqual(rows[0]["prompt"], "reconstruct")
        self.assertEqual(rows[0]["chosen"], "win")
        self.assertEqual(rows[0]["rejected"], "lose")
        self.assertEqual(rows[0]["chosen_reward"], 9.0)


if __name__ == "__main__":
    unittest.main()
