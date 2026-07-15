"""Tests for data.dataengine.domain_mixing."""

import unittest

from harnesscad.data.dataengine.domain_mixing import (
    batch_composition,
    real_fraction,
    schedule_table,
)


class RealFractionTest(unittest.TestCase):
    def test_starts_at_start(self):
        self.assertAlmostEqual(real_fraction(0, 10, start=0.0, end=0.5), 0.0)

    def test_ramps_up(self):
        f_early = real_fraction(2, 10, start=0.0, end=0.5)
        f_late = real_fraction(8, 10, start=0.0, end=0.5)
        self.assertLess(f_early, f_late)

    def test_holds_at_end_after_warmup(self):
        f = real_fraction(6, 10, start=0.0, end=0.4, warmup=0.5)
        self.assertAlmostEqual(f, 0.4)

    def test_bad_step(self):
        with self.assertRaises(ValueError):
            real_fraction(10, 10)


class CompositionTest(unittest.TestCase):
    def test_sums_to_batch(self):
        c = batch_composition(64, 0.25)
        self.assertEqual(c["real"] + c["synthetic"], 64)
        self.assertEqual(c["real"], 16)

    def test_label_efficient_small_fraction(self):
        c = batch_composition(64, 0.015)  # ~1.5% -> 1 real sample
        self.assertEqual(c["real"], 1)

    def test_bad_fraction(self):
        with self.assertRaises(ValueError):
            batch_composition(10, 1.5)


class ScheduleTableTest(unittest.TestCase):
    def test_totals_consistent(self):
        t = schedule_table(10, batch_size=32, start=0.0, end=0.5)
        self.assertEqual(len(t["rows"]), 10)
        self.assertEqual(t["total_real"] + t["total_synthetic"], 10 * 32)

    def test_real_increases_over_time(self):
        t = schedule_table(10, batch_size=100, start=0.0, end=0.8)
        first = t["rows"][0]["real"]
        last = t["rows"][-1]["real"]
        self.assertLess(first, last)

    def test_deterministic(self):
        a = schedule_table(8, 16, end=0.5)
        b = schedule_table(8, 16, end=0.5)
        self.assertEqual(a, b)


if __name__ == "__main__":
    unittest.main()
