"""Tests for MemoryBank-style reinforced-decay salience."""

import math
import unittest

from memory.comet_reinforced_decay import (
    Salience,
    decay_sweep,
    reinforce,
    retention,
    time_to_retention,
)


class TestRetention(unittest.TestCase):
    def test_zero_elapsed_is_full(self):
        self.assertEqual(retention(1.0, 0.0), 1.0)
        self.assertEqual(retention(1.0, -3.0), 1.0)

    def test_monotonic_decay(self):
        r1 = retention(1.0, 1.0, tau=5.0)
        r2 = retention(1.0, 5.0, tau=5.0)
        r3 = retention(1.0, 20.0, tau=5.0)
        self.assertTrue(1.0 > r1 > r2 > r3 > 0.0)

    def test_strength_slows_decay(self):
        weak = retention(1.0, 10.0)
        strong = retention(5.0, 10.0)
        self.assertGreater(strong, weak)

    def test_known_value(self):
        # elapsed = S*tau -> R = e^-1
        self.assertAlmostEqual(retention(2.0, 10.0, tau=5.0), math.exp(-1.0))

    def test_bad_params(self):
        with self.assertRaises(ValueError):
            retention(0.0, 1.0)
        with self.assertRaises(ValueError):
            retention(1.0, 1.0, tau=0.0)


class TestReinforce(unittest.TestCase):
    def test_bumps_strength_and_count_and_clock(self):
        s = Salience("n", strength=1.0, last_recall_day=0.0, recall_count=0)
        s2 = reinforce(s, now_day=10.0)
        self.assertEqual(s2.strength, 2.0)
        self.assertEqual(s2.recall_count, 1)
        self.assertEqual(s2.last_recall_day, 10.0)

    def test_clock_only_moves_forward(self):
        s = Salience("n", strength=1.0, last_recall_day=20.0)
        s2 = reinforce(s, now_day=5.0)
        self.assertEqual(s2.last_recall_day, 20.0)  # not moved back
        self.assertEqual(s2.strength, 2.0)  # but still strengthened

    def test_immutable_input(self):
        s = Salience("n")
        reinforce(s, 5.0)
        self.assertEqual(s.strength, 1.0)

    def test_recall_slows_future_decay(self):
        s = Salience("n", strength=1.0, last_recall_day=0.0)
        before = retention(s.strength, 10.0 - s.last_recall_day)
        s2 = reinforce(s, now_day=10.0)
        after = retention(s2.strength, 10.0 - s2.last_recall_day)
        self.assertGreater(after, before)


class TestTimeToRetention(unittest.TestCase):
    def test_inverts_curve(self):
        dt = time_to_retention(2.0, math.exp(-1.0), tau=5.0)
        self.assertAlmostEqual(dt, 10.0)

    def test_floor_zero_is_inf(self):
        self.assertEqual(time_to_retention(1.0, 0.0), math.inf)

    def test_floor_one_is_zero(self):
        self.assertEqual(time_to_retention(1.0, 1.0), 0.0)

    def test_roundtrip_with_retention(self):
        dt = time_to_retention(3.0, 0.3, tau=4.0)
        self.assertAlmostEqual(retention(3.0, dt, tau=4.0), 0.3)


class TestDecaySweep(unittest.TestCase):
    def _saliences(self):
        return [
            Salience("fresh", strength=1.0, last_recall_day=95.0),
            Salience("stale", strength=1.0, last_recall_day=0.0),
            Salience("strong", strength=10.0, last_recall_day=50.0),
        ]

    def test_stale_is_forgotten(self):
        res = decay_sweep(self._saliences(), now_day=100.0, forget_threshold=0.2)
        forgotten_ids = [n for n, _ in res.forgotten]
        retained_ids = [n for n, _ in res.retained]
        self.assertIn("stale", forgotten_ids)
        self.assertIn("fresh", retained_ids)

    def test_retained_sorted_strongest_first(self):
        res = decay_sweep(self._saliences(), now_day=100.0, forget_threshold=0.0)
        scores = [r for _, r in res.retained]
        self.assertEqual(scores, sorted(scores, reverse=True))

    def test_keep_min_protects_top(self):
        # threshold 1.0 would forget everything but the two strongest survive.
        res = decay_sweep(
            self._saliences(), now_day=100.0, forget_threshold=1.0, keep_min=2
        )
        self.assertEqual(len(res.retained), 2)
        self.assertEqual(len(res.forgotten), 1)

    def test_deterministic_regardless_of_order(self):
        s = self._saliences()
        a = decay_sweep(s, now_day=100.0).to_dict()
        b = decay_sweep(list(reversed(s)), now_day=100.0).to_dict()
        self.assertEqual(a, b)

    def test_bad_threshold(self):
        with self.assertRaises(ValueError):
            decay_sweep([], now_day=0.0, forget_threshold=2.0)


if __name__ == "__main__":
    unittest.main()
