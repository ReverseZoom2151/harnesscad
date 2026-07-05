import unittest

from reconstruction.cmt_cascade_schedule import (
    Stage, cascade_stages, validate_stage_order,
    cosine_reveal_counts, reveal_order, mar_schedule,
)


class TestStages(unittest.TestCase):
    def test_cascade_is_edge_then_surface(self):
        stages = cascade_stages()
        self.assertEqual([s.name for s in stages], ["edge", "surface"])
        self.assertEqual(stages[1].depends_on, ("edge",))

    def test_valid_order(self):
        self.assertTrue(validate_stage_order(cascade_stages()))

    def test_surface_before_edge_invalid(self):
        rev = tuple(reversed(cascade_stages()))
        self.assertFalse(validate_stage_order(rev))

    def test_duplicate_stage_invalid(self):
        s = Stage("edge")
        self.assertFalse(validate_stage_order((s, s)))


class TestCosineReveal(unittest.TestCase):
    def test_counts_sum_to_n(self):
        for n, steps in ((64, 64), (64, 32), (128, 64), (10, 3), (7, 5)):
            counts = cosine_reveal_counts(n, steps)
            self.assertEqual(len(counts), steps)
            self.assertEqual(sum(counts), n)
            self.assertTrue(all(c >= 0 for c in counts))

    def test_one_token_per_step_when_steps_equal_n(self):
        counts = cosine_reveal_counts(8, 8)
        self.assertEqual(counts, (1, 1, 1, 1, 1, 1, 1, 1))

    def test_monotone_cumulative(self):
        counts = cosine_reveal_counts(100, 20)
        cumulative = 0
        for c in counts:
            cumulative += c
            self.assertLessEqual(cumulative, 100)

    def test_bad_args(self):
        with self.assertRaises(ValueError):
            cosine_reveal_counts(10, 0)
        with self.assertRaises(ValueError):
            cosine_reveal_counts(-1, 4)


class TestRevealOrder(unittest.TestCase):
    def test_permutation(self):
        order = reveal_order(16, seed=7)
        self.assertEqual(sorted(order), list(range(16)))

    def test_deterministic(self):
        self.assertEqual(reveal_order(20, seed=3), reveal_order(20, seed=3))

    def test_seed_changes_order(self):
        self.assertNotEqual(reveal_order(20, seed=1), reveal_order(20, seed=2))


class TestMarSchedule(unittest.TestCase):
    def test_covers_all_indices_once(self):
        schedule = mar_schedule(32, 16, seed=5)
        flat = [i for step in schedule for i in step]
        self.assertEqual(sorted(flat), list(range(32)))
        self.assertEqual(len(schedule), 16)

    def test_one_per_step_default(self):
        schedule = mar_schedule(5, 5, seed=0)
        self.assertTrue(all(len(step) == 1 for step in schedule))

    def test_deterministic(self):
        self.assertEqual(mar_schedule(12, 4, seed=9), mar_schedule(12, 4, seed=9))


if __name__ == "__main__":
    unittest.main()
